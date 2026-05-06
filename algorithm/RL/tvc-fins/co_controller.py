import cv2
import torch
import os
import glob
import numpy as np
from rocket_co import Rocket
from policy_co import ActorCritic

class TBaseline:
    def __init__(self):

        task = 'landing'  

        ckpt_folder = os.path.join('./', task + '_ckpt')
        if not os.path.exists(ckpt_folder):
            os.mkdir(ckpt_folder)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(glob.glob(os.path.join(ckpt_folder, '*.pt'))[-1], weights_only=False)
        self.net = ActorCritic(input_dim = 14, output_dim = 9).to(device)
        self.net.load_state_dict(ckpt['model_G_state_dict'])
        self.net.eval()
        self.device = device


class ConditionalFenv:
    """
    A minimal wrapper that:
    - Accepts fin-only action from agent
    - Calls T-only baseline for thrust & gimbal
    - Returns observation = [raw_state , baseline_TVC]
    """
    def __init__(self):
        task = 'landing'  
        max_steps = 800

        self.env = Rocket(task=task, max_steps=max_steps)
        self.baseline = TBaseline()


        self.state_dims = 14
        self.action_dims = self.env.action_dims

    # ---------- 核心接口 ----------
    def act(self, state):
        state = self.env.flatten(state)
        with torch.no_grad():
            action, _, _ = self.baseline.net.get_action(state)
        return action  # 返回动作 ID
    
    def action_table(self):
        f0 = 0.2 * self.env.g  # thrust
        f1 = 1.0 * self.env.g
        f2 = 2 * self.env.g
        vphi0 = 0  # Nozzle angular velocity
        vphi1 = 30 / 180 * np.pi
        vphi2 = -30 / 180 * np.pi

        action_table = [[f0, vphi0], [f0, vphi1], [f0, vphi2],
                        [f1, vphi0], [f1, vphi1], [f1, vphi2],
                        [f2, vphi0], [f2, vphi1], [f2, vphi2]
                        ]
        return action_table
    
    def reset(self, seed=None):
        state = self.env.reset(seed=seed)
        return state

        
    def step(self, fin_action):
        # 1) 先得到 baseline 的 TVC
        env_state = self.env.state
        action = self.act(env_state)
        action_table = self.action_table()
        f, vphi = action_table[action]  # baseline 输出完整动作

        # 2) 生成完整动作 = baseline + agent 舵面
        self.env.action_table = self.env.create_action_table(a=f, b=vphi)  # 先生成一个动作表

        # 3) 送入原环境
        next_state, reward, done, info = self.env.step(fin_action)

        return next_state, reward, done, info