import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import torch
from re_controller import ResidualWrapper
from policy_re import Residual
import glob
import random
import numpy as np

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

    env = ResidualWrapper(max_steps=max_steps)

    net = Residual(input_dim=14, output_dim=2, action_ranges=env.env.action_ranges).to(device)
    if os.path.exists(ckpt_dir):
        checkpoint = torch.load(ckpt_dir, weights_only=False)
        net.load_state_dict(checkpoint['model_G_state_dict'])

    seed_id = random.choice([6])
    state = env.reset(seed=seed_id)
    for step_id in range(max_steps):
        residual_action, raw_action, log_prob, value = net.get_action(state)
        state, reward, done, _ = env.step(residual_action)
        x.append(env.env.state['x'])
        y.append(env.env.state['y'])
        theta.append(env.env.state['theta']*180/np.pi)
        vy.append(env.env.state['vy'])
        vx.append(env.env.state['vx'])
        vphi.append(env.env.state['v_phi'])
        vdelta.append(env.env.state['v_delta'])
        times.append(step_id * env.dt)
        env.env.render()
        if done or step_id == max_steps-1:
            break