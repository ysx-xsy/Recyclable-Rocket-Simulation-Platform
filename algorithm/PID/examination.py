from rocket_pid_RL import Rocket
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import random

if __name__ == '__main__':
    env = Rocket(task='landing', max_steps=1000)
    seed_id = random.choice([6])
    env.reset(seed=seed_id)
    x = []
    theta = []
    y = []
    vy = []
    vx = []
    times = []
    delta_L = []
    delta_R = []
    ax = []
    atheta = []
    vphi = []
    vdelta = []
    for step in range(env.max_steps):
        x.append(env.state['x'])
        y.append(env.state['y'])
        theta.append(env.state['theta']*180/np.pi)
        vy.append(env.state['vy'])
        vx.append(env.state['vx'])
        delta_L.append(env.state['delta_L']*180/np.pi)
        delta_R.append(env.state['delta_R']*180/np.pi)
        ax.append(env.ax)
        atheta.append(env.atheta*180/np.pi)
        vphi.append(env.state['vphi'])
        vdelta.append(env.state['vdelta'])
        times.append(step * env.dt)

        env.step()
        env.render()
        if env.already_crash or env.already_landing:
            #cv2.waitKey(0)
            break

    print('Fuel Consumption:', env.max_fuel-env.fuel)
    print('Seed id:', seed_id)

    plt.figure(figsize=(10, 6))
    plt.plot(times, x, label='X Position', linewidth=2)
    plt.plot(times, y, label='Y Position', linewidth=2)
    plt.xlabel('Time Step (s)')
    plt.ylabel('Position (m)')
    plt.title('Simulation curve of Rocket Landing')
    plt.legend()
    plt.grid(True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(times, env.pid_controller.f_cmd_history, linewidth=2)
    plt.xlabel('Time Step (s)')
    plt.ylabel('Thrust Command (Thrust/G)')
    plt.title('Simulation curve of Rocket Landing')
    plt.grid(True)

    plt.figure(figsize=(10, 6))
    plt.plot(times, vy, label=' vy', linewidth=2)
    #plt.plot(times, env.pid_controller.vy_des_history, label='vy_des', linewidth=2)
    plt.plot(times, vx, label='vx', linewidth=2)
    plt.xlabel('Time Step (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Simulation curve of Rocket Landing')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 6))
    plt.plot(times, env.v_body_x, label='v_body_x', linewidth=2)
    plt.xlabel('Time Step (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Simulation curve of Rocket Landing')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 6))
    plt.plot(times, env.aoa, label='theta', linewidth=2)
    plt.xlabel('Time Step (s)')
    plt.ylabel('Angle of Attack (degrees)')
    plt.title('Simulation curve of Rocket Landing')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 6))
    plt.plot(times, env.torque_fins, label='moment', linewidth=2)
    plt.xlabel('Time Step (s)')
    plt.ylabel('Moment (Nm)')
    plt.title('Simulation curve of Rocket Landing')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 6))
    plt.plot(times, env.pid_controller.theta_cmd_history, label='Theta Command', linewidth=2)
    plt.plot(times, env.pid_controller.theta_history, label='Theta Response', linewidth=2)
    plt.xlabel('Time Step (s)')
    plt.ylabel('Theta (degrees)')
    plt.title('Attitude angle tracking curve')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 6))
    plt.plot(times, env.pid_controller.M_cmd_history, label='Moment Command', linewidth=2)
    plt.xlabel('Time Step (s)')
    plt.ylabel('Moment (Nm)')
    plt.title('Simulation curve of Rocket Landing')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 6))
    plt.plot(times, env.pid_controller.a_history, label='a', linewidth=2)
    plt.xlabel('Time Step (s)')
    plt.ylabel('a (m/s^2)')
    plt.title('Simulation curve of Rocket Landing')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 6))
    plt.plot(times, env.phi_history, label='phi', linewidth=2)
    plt.xlabel('Time Step (s)')
    plt.ylabel('phi (°)')
    plt.title('Simulation curve of Rocket Landing')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 6))
    plt.plot(times, delta_L, label='delta_L', linewidth=2)
    plt.plot(times, delta_R, label='delta_R', linewidth=2)
    #plt.plot(times, env.pid_controller.delta_target, label='delta_target', linewidth=2)
    plt.xlabel('Time Step (s)')
    plt.ylabel('delta (°)')
    plt.title('Simulation curve of Rocket Landing')
    plt.legend()
    plt.grid(True)

    data_dict = {
        'Time (s)': times,
        'vphi-PID (m/s)': vphi,
        'vdelta-PID (m/s)': vdelta,
        # 如果需要其他数据，也可以加在这里，例如：
        # 'X Position': x,
        # 'Y Position': y,
    }
    
    # 转换为 DataFrame
    df = pd.DataFrame(data_dict)
    
    # 保存为 Excel 文件
    output_file = 'simulation_data_vphi_vdelta.xlsx'
    df.to_excel(output_file, index=False)
    print(f"数据已保存至: {output_file}")
    
    # 如果更喜欢 CSV 格式，可以使用下面这行代替 to_excel
    # df.to_csv('simulation_data_vphi_vdelta.csv', index=False)

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