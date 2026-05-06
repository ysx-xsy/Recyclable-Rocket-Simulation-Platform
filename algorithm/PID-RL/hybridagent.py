import numpy as np
import torch

class HybridAgent:
    def __init__(self, pid_controller, ppo_net, env_params):
        self.pid = pid_controller
        self.ppo_net = ppo_net  # 你的 PPO Actor 网络
        self.env_params = env_params

    def get_action(self, state, state_raw):
        # 先让 PID 思考，获取基线动作和内部指令
        f_cmd, v_phi_pid, v_delta_pid, v_delta_pid, theta_cmd, M_req = self.pid.get_action(state_raw, self.env_params, self.env_params['wind_strength'])# 左右舵面偏转速度相等

        # RL 策略网络输出残差动作
        RL_action, _, _, _ = self.ppo_net.get_action(state)

        v_phi_RL = RL_action[0]
        v_delta_RL = RL_action[1]

        # 残差叠加与硬限幅保护
        v_phi = np.clip(v_phi_pid + v_phi_RL, -np.deg2rad(90), np.deg2rad(90))
        v_delta = np.clip(v_delta_pid + v_delta_RL, -np.deg2rad(30), np.deg2rad(30))

        # f_pid 保持不变（推力大小依然由 PID 的高度环绝对控制，保证不砸地）
        return f_cmd, v_phi, v_delta, theta_cmd, M_req