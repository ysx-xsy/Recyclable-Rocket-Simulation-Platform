import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import numpy as np
import torch
import random
from hybridagent import HybridAgent
from policy_pid_RL import Residual
import matplotlib.pyplot as plt
from rocket_pid_RL import Rocket
from pid_controller_fins import PIDController
import utils
import glob

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':


    max_m_episode = 800000
    max_steps = 1000

    env = Rocket(max_steps=max_steps)
    pidcontroller = PIDController()
    ckpt_folder = os.path.join('./', 'co_landing_re' + '_ckpt')
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)

    last_episode_id = 0
    REWARDS = []
    FUEL = []
    LANDING_X = []
    SETTLING_TIME = []
    ADJUST_TIME = []
    EFFORT = []

    net = Residual(input_dim=14, output_dim=2, action_ranges=env.action_ranges).to(device)
    hybridagent = HybridAgent(pid_controller=pidcontroller, ppo_net=net, env_params=env.env_params)
    if len(glob.glob(os.path.join(ckpt_folder, '*.pt'))) > 0:
        # load the last ckpt
        torch.serialization.add_safe_globals([np._core.multiarray.scalar, np.dtype, np.dtypes.Float64DType])
        checkpoint = torch.load(glob.glob(os.path.join(ckpt_folder, '*.pt'))[-1])
        net.load_state_dict(checkpoint['model_G_state_dict'])
        last_episode_id = checkpoint['episode_id']
        REWARDS = checkpoint['REWARDS']
        FUEL = checkpoint['FUEL']
        LANDING_X = checkpoint['LANDING_X']
        SETTLING_TIME = checkpoint['SETTLING_TIME']
        ADJUST_TIME = checkpoint['ADJUST_TIME']
        EFFORT = checkpoint['EFFORT']

    for episode_id in range(last_episode_id, max_m_episode):

        # training loop
        seed_id = random.choice([2,7])
        state = env.reset(seed=seed_id)
        state_debug = env.state.copy()
        rewards, log_probs, values, masks = [], [], [], []
        states, actions = [], []
        settling_time = 0
        current_stable_steps = 0
        effort = 0
        final_adjust_step = max_steps
        env.settling_time, env.fuel, env.landing_x = 0, env.max_fuel, 0
        time = []
        for step_id in range(max_steps):
            _, raw_action, log_prob, value = net.get_action(state)
            f, v_phi, v_delta, theta_cmd, M_req = hybridagent.get_action(state, env.state)
            env_action = [f, v_phi, v_delta, v_delta, theta_cmd, M_req]# 左右舵面偏转速度相等
            state, reward, done, _ = env.step(env_action)
            rewards.append(reward)
            log_probs.append(log_prob.item())
            values.append(value)
            masks.append(1-done)
            states.append(state)
            actions.append(raw_action)
            if episode_id % 1000 == 1 or episode_id == 1:
               env.render()

            effort += np.abs(v_phi) + np.abs(v_delta)

            is_stable = np.abs(env.state['theta']) < (5/180*np.pi) and np.abs(env.state['x']) < 15
            
            if is_stable:
                if current_stable_steps == 0:
                    # 刚刚进入稳定区，记录下这一刻的 step_id
                    final_adjust_step = step_id 
                current_stable_steps += 1
            else:
                # 一旦脱离稳定区，前面记录的作废，重新计数
                current_stable_steps = 0
                final_adjust_step = max_steps

            if done or step_id == max_steps-1:
                _, _, _, last_value = net.get_action(state)
                # 计算returns和advantages
                returns = net.calculate_returns(last_value.item(), rewards, masks)
                # 计算优势函数
                advantages = [ret - val for ret, val in zip(returns, values)]
                # 标准化优势函数
                net.update_ppo(states, actions, log_probs, advantages, returns)
                step = step_id
                break

        env.settling_time = current_stable_steps * env.dt 
        
        if current_stable_steps > 0:
            env.adjust_time = final_adjust_step * env.dt
        else:
            env.adjust_time = max_steps * env.dt  

        REWARDS.append(np.sum(rewards))
        FUEL.append(env.fuel)
        LANDING_X.append(env.landing_x)
        SETTLING_TIME.append(env.settling_time)
        ADJUST_TIME.append(env.adjust_time)
        EFFORT.append(effort)

        print('episode id: %d, episode reward: %.3f, seed_id: %d'
              % (episode_id, np.sum(rewards), seed_id))
        if np.sum(rewards) < 0:
            print(state_debug)
            print('Wind Strength: %.3f' % env.env_params['wind_strength'])

        if episode_id % 1000 == 1:
            fig, axes = plt.subplots(6, 1, figsize=(10, 20))
            metrics = [
                ('Reward', REWARDS),
                ('Fuel Residual', FUEL),
                ('Landing X Error', LANDING_X),
                ('Settling Time', SETTLING_TIME),
                ('Adjust Time', ADJUST_TIME),
                ('Actuator Effort', EFFORT)
            ]
            
            for i, (name, data) in enumerate(metrics):
                ax = axes[i]
                ax.plot(data, alpha=0.3, label='Original')
                if len(data) >= 50:
                    ax.plot(utils.moving_avg(data, N=50), label='Moving Avg (50)')
                
                ax.set_ylabel(name)
                ax.grid(True, linestyle='--', alpha=0.6)

            axes[-1].set_xlabel('Episodes')
            plt.tight_layout() # 防止子图标签重叠
            
            # 2. 统一保存大图
            summary_path = os.path.join(ckpt_folder, f'summary_{str(episode_id).zfill(8)}.jpg')
            plt.savefig(summary_path, dpi=150)
            plt.close(fig) # 彻底释放内存

            torch.save({'episode_id': episode_id,
                        'REWARDS': REWARDS,
                        'FUEL': FUEL,
                        'LANDING_X': LANDING_X,
                        'SETTLING_TIME': SETTLING_TIME,
                        'ADJUST_TIME': ADJUST_TIME,
                        'EFFORT': EFFORT,
                        'model_G_state_dict': net.state_dict()},
                       os.path.join(ckpt_folder, 'ckpt_' + str(episode_id).zfill(8) + '.pt'))



