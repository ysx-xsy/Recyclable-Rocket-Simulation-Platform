# eval/monte_carlo.py
import numpy as np
import pandas as pd
import os
import glob
import torch
from rocket_tvc import Rocket  # adapt to your repo
from policy_tvc import ActorCritic  # your loader

class monte_carlo:
    def __init__(self):
        self.env = Rocket(task='landing', max_steps=800)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = ActorCritic(input_dim=self.env.state_dims, output_dim=self.env.action_dims).to(self.device)
        ckpt_dir = glob.glob(os.path.join('landing_ckpt', '*.pt'))[-1]  # last ckpt
        if os.path.exists(ckpt_dir):
            checkpoint = torch.load(ckpt_dir, weights_only=False)
            self.net.load_state_dict(checkpoint['model_G_state_dict'])

    
    def run_trial(self, state_range, max_steps=800):
        obs = self.env.reset(state_range)
        done = False
        wind_strengths = []
        time = []
        self.env.landing_x = np.nan
        settling_time = 0
        key = 0

        for step_id in range(max_steps):
            self.env.get_wind_info(state_range)
            action, _, _ = self.net.get_action(obs)  # full action for T-only policy
            state, _, done, _ = self.env.step(action)
            # record proxies
            wind_strengths.append(self.env.wind_strength)
            obs = state
            if key == 0:
                time.append(settling_time)
                settling_time = 0
            if np.abs(self.env.state['theta']) < 5/180*np.pi and np.abs(self.env.state['x']) < 50:
                key = 1
                settling_time += 1
            else:
                key = 0
            if done or step_id == max_steps-1:
                break
        if len(time) != (step_id+1):
            time.append(0)
        self.env.settling_time = max(time) if len(time) > 0 else 0
        max_index = np.argmax(time) if len(time) > 0 else np.nan
        self.env.adjust_time = (max_index+1) * self.env.dt
        # summary
        if self.env.already_crash:
            done = False
        wind_strength = np.mean(wind_strengths) if len(wind_strengths)>0 else np.nan
        return {
            'success': int(done),'wind_strength': wind_strength,
            'terminal_x': self.env.landing_x, 'adjust_time': self.env.adjust_time,
            'settling_time': self.env.settling_time,'fuel_consumption': self.env.max_fuel - self.env.fuel
        }

    def run_batch(self, n, out_csv, state_range):
        rows=[]
        n = int(n)
        for i in range(n):
            res = self.run_trial(state_range)
            res['init_x'] = self.env.init_x
            res['init_y'] = self.env.init_y
            res['init_vx'] = self.env.init_vx
            res['init_vy'] = self.env.init_vy
            res['init_theta'] = self.env.init_theta * 180/np.pi
            if i == 0:
                for key, (min_val, max_val) in state_range.items():
                    res[f'{key}_min'] = min_val
                    res[f'{key}_max'] = max_val
            rows.append(res)
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, mode='a', header=not os.path.exists(out_csv), index=False)
        print("saved", out_csv)

if __name__ == "__main__":
    mc = monte_carlo()
    mc.run_batch(n=20, out_csv='T_only_results.csv')