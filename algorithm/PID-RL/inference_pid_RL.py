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
import glob
import utils
import pandas as pd

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    theta = []
    x = []
    y = []
    vx = []
    vy = []
    times = []
    vphi = []
    vdelta = []
    task = 'landing'  # 'hover' or 'landing'
    max_steps = 1000
    ckpt_dir = glob.glob(os.path.join('co_landing_re_ckpt', '*.pt'))[6]  # last ckpt

    env = Rocket(max_steps=max_steps)
    pidcontroller = PIDController()

    net = Residual(input_dim=14, output_dim=2, action_ranges=env.action_ranges).to(device)
    hybridagent = HybridAgent(pid_controller=pidcontroller, ppo_net=net, env_params=env.env_params)
    if os.path.exists(ckpt_dir):
        checkpoint = torch.load(ckpt_dir, weights_only=False)
        net.load_state_dict(checkpoint['model_G_state_dict'])

    seed_id = random.choice([6])
    state = env.reset(seed=seed_id)
    for step_id in range(max_steps):
        _, raw_action, log_prob, value = net.get_action(state)
        f, v_phi, v_delta, theta_cmd, M_req = hybridagent.get_action(state, env.state)
        env_action = [f, v_phi, v_delta, v_delta, theta_cmd, M_req]# 左右舵面偏转速度相等
        state, reward, done, _ = env.step(env_action)
        x.append(env.state['x'])
        y.append(env.state['y'])
        theta.append(env.state['theta']*180/np.pi)
        vy.append(env.state['vy'])
        vx.append(env.state['vx'])
        vphi.append(v_phi)
        vdelta.append(v_delta)
        times.append(step_id * env.dt)
        env.render()
        if done or step_id == max_steps-1:
            break

    plt.figure(figsize=(10, 6))
    plt.plot(times, x, label='X Position', linewidth=2)
    plt.plot(times, y, label='Y Position', linewidth=2)
    plt.xlabel('Time Step (s)')
    plt.ylabel('Position (m)')
    plt.title('Simulation curve of Rocket Landing')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 6))
    plt.plot(times, vy, label='vy', linewidth=2)
    plt.plot(times, vx, label='vx', linewidth=2)
    plt.xlabel('Time Step (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Simulation curve of Rocket Landing')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 6))
    plt.plot(times, theta, label='Theta', linewidth=2)
    plt.xlabel('Time Step (s)')
    plt.ylabel('Theta (degrees)')
    plt.title('Simulation curve of Rocket Landing')
    plt.legend()
    plt.grid(True)

    plt.show()
    '''data_dict = {
        'Time (s)': times,
        'vphi-PID-RL (m/s)': vphi,
        'vdelta-PID-RL (m/s)': vdelta,
        # 如果需要其他数据，也可以加在这里，例如：
        # 'X Position': x,
        # 'Y Position': y,
    }
    
    # 转换为 DataFrame
    df = pd.DataFrame(data_dict)
    
    # 保存为 Excel 文件
    output_file = 'simulation_data_vphi_vdelta_PID-RL.xlsx'
    df.to_excel(output_file, index=False)
    print(f"数据已保存至: {output_file}")
    
    # 如果更喜欢 CSV 格式，可以使用下面这行代替 to_excel
    # df.to_csv('simulation_data_vphi_vdelta_PID-RL.csv', index=False)

    # --- 原有的绘图代码 ---
    plt.figure(figsize=(10, 6))
    plt.plot(times, vphi, label='vphi', linewidth=2)
    plt.plot(times, vdelta, label='vdelta', linewidth=2)
    plt.xlabel('Time Step (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Simulation curve of Rocket Landing')
    plt.legend()
    plt.grid(True)

    plt.show()

    max_steps = 1000
    env = Rocket(max_steps=max_steps)
    net = Residual(input_dim=14, output_dim=2, action_ranges=env.action_ranges).to(device)
    ckpt_folder = os.path.join('./', 'co_landing_re' + '_ckpt')
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)
    if len(glob.glob(os.path.join(ckpt_folder, '*.pt'))) > 0:
    # load the last ckpt
        torch.serialization.add_safe_globals([np._core.multiarray.scalar, np.dtype, np.dtypes.Float64DType])
        checkpoint = torch.load(glob.glob(os.path.join(ckpt_folder, '*.pt'))[6])
        net.load_state_dict(checkpoint['model_G_state_dict'])
        last_episode_id = checkpoint['episode_id']
        REWARDS = checkpoint['REWARDS']
    
    plt.figure()
    plt.plot(REWARDS), plt.plot(utils.moving_avg(REWARDS, N=50))
    plt.legend(['episode reward', 'moving avg'], loc=2)
    plt.ylim(200, 800)
    plt.xlabel('m episode')
    plt.ylabel('reward')
    plt.savefig(os.path.join(ckpt_folder, 'rewards_' + str(last_episode_id).zfill(8) + '.jpg'))
    plt.close()'''