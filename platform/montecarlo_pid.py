import numpy as np
import pandas as pd
import os
import json
import itertools
from rocket_pid import Rocket
from rocket_pid import RocketScenario

class monte_carlo_pid:
    def __init__(self):
        self.env = Rocket(task='landing', max_steps=2000)
        self.scenario = RocketScenario()
    
    def run_trial(self, max_steps=2000, state_range=None, seed=None):
        x = []
        y = []
        theta = []
        vx = []
        vy = []
        v_TVC = []
        v_Fins = []
        times = []
        
        landing_x, adjust_time, settling_time, fuel_consumption, effort = np.nan, 0, 0, 0, 0
        self.env.reset(seed=seed, state_range=state_range)
        current_stable_steps = 0
        final_adjust_step = max_steps
        res = {}
        res['x'] = self.env.state['x']
        res['y'] = self.env.state['y']
        res['vx'] = self.env.state['vx']
        res['vy'] = self.env.state['vy']
        res['theta'] = self.env.state['theta'] * 180/np.pi
        res['wind'] = self.env.base_wind
        for step_id in range(max_steps):
            done = self.env.step()
            x.append(self.env.state['x'])
            y.append(self.env.state['y'])
            theta.append(self.env.state['theta'] * 180/np.pi)
            vx.append(self.env.state['vx'])
            vy.append(self.env.state['vy'])
            v_phi, v_delta = self.env.state['vphi'], self.env.state['vdelta']
            v_TVC.append(v_phi)
            v_Fins.append(v_delta)
            times.append(step_id * self.env.dt)
            effort += np.abs(self.env.state['vphi']) + np.abs(self.env.state['vdelta'])

            is_stable = np.abs(self.env.state['theta']) < (5/180*np.pi) and np.abs(self.env.state['x']) < 15
            
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
                landing_x = np.abs(self.env.state['x'])
                fuel_consumption = self.env.max_fuel - self.env.fuel
                break
        settling_time = current_stable_steps * self.env.dt 
        
        if current_stable_steps > 0:
            adjust_time = final_adjust_step * self.env.dt
        else:
            adjust_time = max_steps * self.env.dt
        res['success'] = int(done)
        res['terminal_x'] = landing_x
        res['adjust_time'] = adjust_time
        res['settling_time'] = settling_time
        res['fuel_consumption'] = fuel_consumption
        res['effort'] = effort

        trajectory_data = {
            'times': times,
            'x': x,
            'y': y,
            'theta': theta,
            'vx': vx,
            'vy': vy,
            'v_TVC': v_TVC,
            'v_Fins': v_Fins
        }
        return res, trajectory_data

    def _get_unique_path(self, base_path):
        """
        如果文件存在，则在文件名后添加 (2), (3) 等序号，直到找到不存在的文件名。
        例如: test.csv -> test(2).csv -> test(3).csv
        """
        if not os.path.exists(base_path):
            return base_path
        
        directory = os.path.dirname(base_path)
        filename = os.path.basename(base_path)
        name, ext = os.path.splitext(filename)
        
        counter = 2
        while True:
            new_filename = f"{name}({counter}){ext}"
            new_path = os.path.join(directory, new_filename)
            if not os.path.exists(new_path):
                return new_path
            counter += 1
    def run_batch(self, n, out_csv, state_range=None, seed=None, scenario_name=None):
        unique_csv_path = self._get_unique_path(out_csv)
        if unique_csv_path != out_csv:
            print(f"File '{os.path.basename(out_csv)}' already exists. Saving to '{os.path.basename(unique_csv_path)}'")
        rows = []
        # 1. 运行模拟，只收集纯数据
        for i in range(n):
            res, trajectory_data = self.run_trial(state_range=state_range, seed=seed)
            rows.append(res)
        
            if n == 1:
                base_name = os.path.splitext(unique_csv_path)[0]
                traj_csv_path = f"{base_name}_trajectory.csv"
                
                df_traj = pd.DataFrame(trajectory_data)
                df_traj.to_csv(traj_csv_path, index=False)
                print(f"Saved detailed trajectory to {traj_csv_path}")

        # 2. 构建元数据字典
        meta_data = {
            "scenario_name": scenario_name if scenario_name else "Custom",
            "seed": seed,
            "num_trials": n,
            "state_range": {},
            "strategy": "PID"
        }

        if seed is None:
            if state_range is not None:
                # 将 tuple 转换为 list 以便 JSON 序列化
                meta_data["state_range"] = {k: list(v) for k, v in state_range.items()}
        else:
            scenario = self.scenario.get_config(seed)
            # 跳过第一个键，获取剩余范围
            range_dict = {}
            for key, (min_val, max_val) in itertools.islice(scenario.items(), 1, None):
                range_dict[key] = [min_val, max_val]
            meta_data["state_range"] = range_dict

        # 3. 保存数据到 CSV (纯数据，无元数据行)
        df = pd.DataFrame(rows)
        df.to_csv(unique_csv_path, index=False)
        
        # 4. 保存元数据到 JSON
        # 生成对应的 json 文件名，例如 results.csv -> results_meta.json
        base_name = os.path.splitext(unique_csv_path)[0]
        json_path = f"{base_name}_meta.json"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, indent=4, ensure_ascii=False)
            
        print(f"Saved data to {unique_csv_path}")
        print(f"Saved metadata to {json_path}")