import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PositionalMapping(nn.Module):
    def __init__(self, input_dim, L=5, scale=1.0):
        super(PositionalMapping, self).__init__()
        self.L = L
        self.output_dim = input_dim * (L*2 + 1)
        self.scale = scale

    def forward(self, x):
        x = x * self.scale
        if self.L == 0:
            return x

        h = [x]
        PI = 3.1415927410125732
        for i in range(self.L):
            x_sin = torch.sin(2**i * PI * x)
            x_cos = torch.cos(2**i * PI * x)
            h.append(x_sin)
            h.append(x_cos)

        return torch.cat(h, dim=-1) / self.scale

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.mapping = PositionalMapping(input_dim=input_dim, L=5)
        
        self.linear1 = nn.Linear(in_features=self.mapping.output_dim, out_features=hidden_dim, bias=True)
        self.linear2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)
        self.linear3 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)
        self.linear4 = nn.Linear(in_features=hidden_dim, out_features=output_dim, bias=True)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.mapping(x)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)
        return x

class Residual(nn.Module):
    def __init__(self, input_dim, output_dim, action_ranges, hidden_dim=128):
        super().__init__()
        self.action_dim = output_dim
        
        self.action_ranges = action_ranges
        self.shared_net = MLP(input_dim=input_dim, output_dim=hidden_dim, hidden_dim=hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, output_dim)
        self.log_std = nn.Parameter(torch.ones(output_dim) * -0.5)
        self.critic_head = nn.Linear(hidden_dim, 1)
        self.optimizer = optim.RMSprop(self.parameters(), lr=5e-5)
        
    def forward(self, x):
        # 形状处理：确保输入是2D [batch_size, obs_dim]
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # 通过共享网络提取特征
        features = self.shared_net(x)
        
        mu = torch.tanh(self.mu_head(features))

        value = self.critic_head(features).squeeze(-1)
        
        # 标准差（确保数值稳定性）
        std = torch.exp(self.log_std).clamp(min=1e-6, max=1.0)
        
        return mu, std, value

    def _scale_actions(self, actions):
        """
        将动作从[-1,1]范围映射到指定范围
        actions: shape [..., action_dim]
        """
        scaled_actions = torch.zeros_like(actions)
        for i, (min_val, max_val) in enumerate(self.action_ranges):
            # 从[-1,1]映射到[min_val, max_val]
            scaled_actions[..., i] = (actions[..., i] + 1.0) * (max_val - min_val) / 2.0 + min_val
        return scaled_actions    

    def get_action(self, state, deterministic=False):
        # 收集数据时不需要计算梯度，加速运行并节省显存
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(device)
            mu, std, value = self.forward(state_tensor)
            
            if deterministic:
                raw_action = mu
            else:
                normal_dist = Normal(mu, std)
                # PPO 不需要重参数化技巧，直接 sample 即可
                raw_action = normal_dist.sample() 
            
            # 1. 计算原始动作的真实对数概率
            log_prob = normal_dist.log_prob(raw_action).sum(dim=-1)
            
            # 2. 替代 tanh 的方案：直接裁切 (Clip)
            # 这样既保证动作不会越界，又不需要进行复杂的 Jacobian 概率修正
            clipped_action = torch.clamp(raw_action, -1.0, 1.0)
            
            # 3. 映射到物理范围，仅供环境执行使用
            RL_action = self._scale_actions(clipped_action)
            
        # 注意：这里返回了两个动作
        # env_action: 给 env.step() 执行用
        # raw_action: 必须存入 PPO 的 Replay Buffer 中！
        return RL_action.detach().cpu().numpy().squeeze(0), raw_action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy()
    
    def evaluate_actions(self, states, actions_raw):
        """
        注意：这里的 actions_raw 必须是你从 Buffer 中取出的、在 get_action 时生成的原始动作
        """
        mu, std, values = self.forward(states)
        dist = Normal(mu, std)
        
        # 因为 buffer 里存的就是 raw_action，所以这里不需要任何 _unscale_actions！
        # 直接计算概率，彻底杜绝了逆映射带来的精度损失和逻辑错误
        log_probs = dist.log_prob(actions_raw).sum(dim=-1)
        
        dist_entropy = dist.entropy().mean()
        
        return log_probs, values, dist_entropy
    
    @staticmethod
    def calculate_returns(next_value, rewards, masks, gamma=0.99):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
        return returns
    
    def update_ppo(self, states, raw_action, old_log_probs, advantages, returns, 
                  clip_epsilon=0.2, ppo_epochs=4, mini_batch_size=64):
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.FloatTensor(np.array(raw_action)).to(device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(device)
        advantages = torch.FloatTensor(advantages).to(device)
        returns = torch.FloatTensor(returns).to(device)
        
        # 归一化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for epoch in range(ppo_epochs):
            # 随机打乱数据
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), mini_batch_size):
                end = start + mini_batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 评估当前策略
                new_log_probs, values, dist_entropy = self.evaluate_actions(batch_states, batch_actions)
                
                # 计算概率比
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO的clipped objective
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                critic_loss = 0.5 * (batch_returns - values).pow(2).mean()
                
                # 熵奖励（鼓励探索）
                entropy_bonus = 0.01 * dist_entropy
                
                # 总损失
                total_loss = actor_loss + critic_loss - entropy_bonus
                
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                self.optimizer.step()