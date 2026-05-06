import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import torch
from co_controller import ConditionalFenv
from policy_co import ActorCritic
import matplotlib.pyplot as plt
import glob
import random

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

    env = ConditionalFenv()
    net = ActorCritic(input_dim=env.state_dims, output_dim=env.action_dims).to(device)
    if os.path.exists(ckpt_dir):
        checkpoint = torch.load(ckpt_dir, weights_only=False)
        net.load_state_dict(checkpoint['model_G_state_dict'])

    seed_id = random.choice([6])
    state = env.reset(seed=seed_id)
    time = []
    for step_id in range(max_steps):
        action, log_prob, value = net.get_action(state)
        state, reward, done, _ = env.step(action)
        env.env.render(window_name='test')
        time.append(step_id*env.env.dt)
        if done:
            break
