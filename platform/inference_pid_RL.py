import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import numpy as np
import torch
import random
from pid_RL_controller import HybridAgent
from policy import Residual
import matplotlib.pyplot as plt
from rocket_pid_RL import Rocket
from pid_controller import PIDController
import glob

class PIDRLInference:
    def __init__(self):
        self.task = 'landing'
        self.max_steps = 2000
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ckpt_dir = glob.glob(os.path.join('pid_RL_landing_ckpt', '*.pt'))[-1]

        self.env = Rocket(max_steps=self.max_steps)
        self.pidcontroller = PIDController()

        self.net = Residual(input_dim=14, output_dim=2, action_ranges=self.env.action_ranges).to(self.device)
        self.hybridagent = HybridAgent(pid_controller=self.pidcontroller, ppo_net=self.net, env_params=self.env.env_params)
        if os.path.exists(self.ckpt_dir):
            checkpoint = torch.load(self.ckpt_dir, weights_only=False)
            self.net.load_state_dict(checkpoint['model_G_state_dict'])

    def run_inference(self):
        state = self.env.flatten(self.env.state)
        f, v_phi, v_delta, theta_cmd, M_req = self.hybridagent.get_action(state, self.env.state)
        env_action = [f, v_phi, v_delta, v_delta, theta_cmd, M_req]# 左右舵面偏转速度相等
        state, reward, done, _ = self.env.step(env_action)
        frame_0, frame_1 = self.env.render()
        return frame_0, frame_1, self.env.state, done