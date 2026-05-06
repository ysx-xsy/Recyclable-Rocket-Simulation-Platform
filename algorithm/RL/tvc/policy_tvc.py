import random
import numpy as np
import torch
import torch.optim as optim

import torch.nn as nn

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def calculate_returns(next_value, rewards, masks, gamma=0.99):
    """
    计算折扣回报值（returns）
    
    该函数使用逆序迭代的方式，根据下一个状态的价值、即时奖励、掩码和折扣因子
    来计算每个时间步的折扣回报值。主要用于强化学习中的策略梯度算法。
    
    参数:
        next_value: 下一个状态的价值估计值
        rewards: 各时间步的即时奖励列表
        masks: 掩码列表，用于区分episode的结束状态（0表示结束，1表示继续）
        gamma: 折扣因子，默认为0.99
    
    返回:
        returns: 计算得到的各时间步折扣回报值列表
    """
    R = next_value
    returns = []
    # 逆序计算折扣回报值
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


class PositionalMapping(nn.Module):
    """
    Positional mapping Layer.
    This layer map continuous input coordinates into a higher dimensional space
    and enable the prediction to more easily approximate a higher frequency function.
    See NERF paper for more details (https://arxiv.org/pdf/2003.08934.pdf)
    """
    def __init__(self, input_dim, L=5, scale=1.0):
        """
        初始化位置映射层
        
        参数:
            input_dim (int): 输入维度大小
            L (int): 位置编码的频率参数，默认值为5
            scale (float): 缩放因子，默认值为1.0
            
        返回值:
            None
        """
        super(PositionalMapping, self).__init__()
        self.L = L
        self.output_dim = input_dim * (L*2 + 1)
        self.scale = scale

    def forward(self, x):
        """
        前向传播函数，对输入数据进行特征变换
        
        参数:
            x: 输入张量，形状为[..., dim]
            
        返回:
            变换后的张量，形状为[..., dim * (1 + 2 * L)]，其中L为频率编码的层数
        """
        
        # 对输入进行缩放
        x = x * self.scale

        # 如果不使用频率编码，则直接返回缩放后的输入
        if self.L == 0:
            return x

        # 初始化特征列表，包含原始输入
        h = [x]
        PI = 3.1415927410125732
        
        # 生成不同频率的正弦和余弦特征
        for i in range(self.L):
            x_sin = torch.sin(2**i * PI * x)
            x_cos = torch.cos(2**i * PI * x)
            h.append(x_sin)
            h.append(x_cos)

        # 拼接所有特征并进行反缩放
        return torch.cat(h, dim=-1) / self.scale


class MLP(nn.Module):
    """
    Multilayer perception with an embedded positional mapping
    """

    def __init__(self, input_dim, output_dim):
        """
        初始化神经网络模型
        
        参数:
            input_dim (int): 输入特征的维度
            output_dim (int): 输出特征的维度
        """
        super().__init__()

        # 创建位置映射层，将输入维度映射到更高维度空间
        self.mapping = PositionalMapping(input_dim=input_dim, L=7)

        # 定义网络层维度
        h_dim = 128
        
        # 创建全连接层序列，构建多层感知机结构
        self.linear1 = nn.Linear(in_features=self.mapping.output_dim, out_features=h_dim, bias=True)
        self.linear2 = nn.Linear(in_features=h_dim, out_features=h_dim, bias=True)
        self.linear3 = nn.Linear(in_features=h_dim, out_features=h_dim, bias=True)
        self.linear4 = nn.Linear(in_features=h_dim, out_features=output_dim, bias=True)
        
        # 创建激活函数层
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        # shape x: 1 x m_token x m_state
        x = x.view([1, -1])
        x = self.mapping(x)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)
        return x


class ActorCritic(nn.Module):
    """
    RL policy and update rules
    """

    def __init__(self, input_dim, output_dim):
        """
        初始化函数，创建一个包含actor和critic网络的模型
        
        参数:
            input_dim (int): 输入维度大小
            output_dim (int): 输出维度大小，用于actor网络的输出
            
        返回值:
            无
        """
        super().__init__()

        # 存储输出维度
        self.output_dim = output_dim
        # 创建actor网络：输入状态，输出动作概率分布
        self.actor = MLP(input_dim=input_dim, output_dim=output_dim)
        # 创建critic网络：输入状态，输出状态价值
        self.critic = MLP(input_dim=input_dim, output_dim=1)
        # 创建softmax层：用于将actor输出转换为概率分布
        self.softmax = nn.Softmax(dim=-1)

        # 创建优化器：使用RMSprop优化算法，学习率为5e-5
        self.optimizer = optim.RMSprop(self.parameters(), lr=5e-5)

    def forward(self, x):
        # shape x: batch_size x m_token x m_state
        y = self.actor(x)
        probs = self.softmax(y)
        value = self.critic(x)

        return probs, value

    def get_action(self, state, deterministic=False, exploration=0.01):
        """
        根据当前状态获取要执行的动作
        
        参数:
            state: 当前环境状态，通常为观测值
            deterministic: 是否采用确定性策略，默认为False
            exploration: 探索概率，用于epsilon-greedy策略，默认为0.01
            
        返回值:
            action_id: 选择的动作ID
            log_prob: 所选动作的对数概率
            value: 当前状态的价值估计
        """

        # 将状态转换为tensor并添加批次维度
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        # 前向传播获取动作概率分布和状态价值
        probs, value = self.forward(state)
        probs = probs[0, :]
        value = value[0]

        # 根据是否确定性策略选择动作
        if deterministic:
            # 确定性策略：选择概率最大的动作
            action_id = np.argmax(np.squeeze(probs.detach().cpu().numpy()))
        else:
            # 随机策略：以exploration概率随机探索，否则按概率分布采样
            if random.random() < exploration:  # exploration
                action_id = random.randint(0, self.output_dim - 1)
            else:
                action_id = np.random.choice(self.output_dim, p=np.squeeze(probs.detach().cpu().numpy()))

        # 计算所选动作的对数概率
        log_prob = torch.log(probs[action_id] + 1e-9)

        return action_id, log_prob, value

    @staticmethod
    def update_ac(network, rewards, log_probs, values, masks, Qval, gamma=0.99):
        """
        更新Actor-Critic网络的参数
        
        该函数通过计算优势函数和损失函数来更新网络的优化器参数，包括actor和critic两部分的损失计算和反向传播
        
        参数:
            network: 神经网络模型，包含optimizer属性用于参数更新
            rewards: 奖励序列列表
            log_probs: 对数概率列表，记录动作的对数概率
            values: 状态值函数估计列表
            masks: 掩码列表，用于区分不同episode
            Qval: 最终状态的Q值估计
            gamma: 折扣因子，默认为0.99
            
        返回值:
            无返回值
        """

        # 计算回报值Qvals
        Qvals = calculate_returns(Qval.detach(), rewards, masks, gamma=gamma)
        Qvals = torch.tensor(Qvals, dtype=torch.float32).to(device).detach()

        log_probs = torch.stack(log_probs)
        values = torch.stack(values)

        # 计算优势函数并分别计算actor和critic损失
        advantage = Qvals - values
        actor_loss = (-log_probs * advantage.detach()).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss

        # 执行反向传播和参数更新
        network.optimizer.zero_grad()
        ac_loss.backward()
        network.optimizer.step()

