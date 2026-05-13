import numpy as np
import random
import cv2
import utils
from pid_controller import PIDController

class RocketScenario:
    def __init__(self):
        # 种子值映射：描述, 初始位置(x, y), 初始速度(vx, vy), 初始姿态(theta, vtheta), 风速(wind_v)
        self.SCENARIOS = {
            1: {"name": "Center point", 
                "x": (-100, 100), "y": (480, 520), "vx": (-5, 5), "vy": (-130, -120), "theta": (-45, 45), "wind": (-5, 5)},
            2: {"name": "Left offset", 
                "x": (-150, -100), "y": (480, 520), "vx": (5, 10), "vy": (-130, -120), "theta": (-45, 45), "wind": (-5, 5)},
            3: {"name": "Right offset", 
                "x": (100, 150), "y": (480, 520), "vx": (-10, -5), "vy": (-130, -120), "theta": (-45, 45), "wind": (-5, 5)},
            4: {"name": "Strong crosswind", 
                "x": (-100, 100), "y": (480, 520), "vx": (-5, 5), "vy": (-130, -120), "theta": (-45, 45), "wind": ((-10, -5), (5, 10))},
            5: {"name": "High-speed descent", 
                "x": (-100, 100), "y": (480, 520), "vx": (-5, 5), "vy": (-140, -130), "theta": (-45, 45), "wind": (-5, 5)},
            6: {"name": "Large angle tilt", 
                "x": (-90), "y": (480, 520), "vx": (-5, 5), "vy": (-130, -120), "theta": (62), "wind": (-5, 5)},
            7: {"name": "Extreme conditions", 
                "x": (-20, 20), "y": (400, 450), "vx": (-15, 15), "vy": (-140, -130), "theta": (-90, 90), "wind": ((-15, -10),(10, 15))},
        }

    def get_config(self, seed):
        # 如果 seed 不在 1-7，则随机选一个
        scenario = self.SCENARIOS.get(seed, self.SCENARIOS[random.randint(1, 7)])
        return scenario

class Rocket(object):

    def __init__(self, max_steps, task='landing', rocket_type='falcon',
                 viewport_h=768, path_to_bg_img=None):

        self.task = task
        self.rocket_type = rocket_type

        self.atheta = 0  # angular acceleration

        self.max_fuel = 395700.0
        self.fuel = 395700.0
        self.m_true = 410000.0
        self.m_simu = 500.0
        self.Isp = 282.0
        self.ax = 0.0
        self.debug = 0

        self.g = 9.8
        self.H = 50  # rocket height (meters)
        self.I = 1/12*self.H*self.H*self.m_simu  # Moment of inertia
        self.fin_area = 2.2
        self.dt = 1/60

        self.world_x_min = -300  # meters
        self.world_x_max = 300
        self.world_y_min = -30
        self.world_y_max = 570

        self.Scenarios = RocketScenario()
        self.wind_v = 0
        self.wind_strength = 0
        self.base_wind = 0

        self.landing_x = np.nan 
        self.settling_time = 0
        self.adjust_time = 0

        self.rho_air = 1.15
        self.q_ref = 5000.0
        # target point
        if self.task == 'hover':
            self.target_x, self.target_y, self.target_r = 0, 200, 50
        elif self.task == 'landing':
            self.target_x, self.target_y, self.target_r = 0, self.H/2.0, 50

        self.already_landing = False
        self.already_crash = False
        self.max_steps = max_steps

        # viewport height x width (pixels)
        self.viewport_h = int(viewport_h)
        self.viewport_w = int(viewport_h * (self.world_x_max-self.world_x_min) \
                          / (self.world_y_max - self.world_y_min))
        self.step_id = 0

        self.state = self.create_random_state()
        self.action_table = self.create_action_table()

        self.state_dims = 10
        self.action_dims = len(self.action_table)
        self.env_params = {'rho': self.m_simu / (125/(self.g/2.0))**0.5, 'q': 0, 'fin_area': self.fin_area, 'H': self.H, 'W': self.H/10, 'I': self.I, 'count': 0, 'm': self.m_simu, 'g': self.g, 'wind_strength': self.wind_strength}
        self.pid_controller = PIDController()

        if path_to_bg_img is None:
            path_to_bg_img = 'landing.jpg'
        self.bg_img = utils.load_bg_img(path_to_bg_img, w=self.viewport_w, h=self.viewport_h)

        self.action_ranges = [(-15/180*np.pi, 15/180*np.pi), (-10/180*np.pi, 10/180*np.pi)]

        self.state_buffer = []
        self.q_history = []
        self.phi_history = []
        self.cl_history = []
        self.ay = []

        
    def reset(self, state_dict=None, seed=None):
        
        if state_dict and seed is None:
            self.state = self.create_random_state()
        else:
            # 随机种子需要任意传入一个非空的state_dict，同时seed保持为None
            current_seed = seed if seed is not None else random.randint(1, 7)
            scenario = self.Scenarios.get_config(current_seed)

            x = random.uniform(*scenario["x"]) if isinstance(scenario["x"], tuple) else scenario["x"]
            y = random.uniform(*scenario["y"]) if isinstance(scenario["y"], tuple) else scenario["y"]
            vx = random.uniform(*scenario["vx"]) if isinstance(scenario["vx"], tuple) else scenario["vx"]
            vy = random.uniform(*scenario["vy"]) if isinstance(scenario["vy"], tuple) else scenario["vy"]
            
            theta_config = scenario["theta"]

            if isinstance(theta_config, (int, float)):
                theta = theta_config*np.pi/180

            elif isinstance(theta_config, tuple) and len(theta_config) == 2 and isinstance(theta_config[0], (int, float)):
                theta = random.uniform(*theta_config)
                theta = np.pi/180*theta

            elif isinstance(theta_config, tuple) and len(theta_config) == 2 and isinstance(theta_config[0], tuple):
                # theta 是嵌套元组，先随机选一个范围，再在该范围内随机取值
                selected_theta_range = random.choice(theta_config)
                theta = random.uniform(*selected_theta_range)
                theta = np.pi/180*theta
            else:
                raise ValueError("Invalid theta configuration")
            
            wind_config = scenario["wind"]
            
            # 情况1: wind 是数字 (例如 0)
            if isinstance(wind_config, (int, float)):
                self.base_wind = wind_config
            
            # 情况2: wind 是简单元组 (例如 (-15, 15)) -> 直接在范围内随机
            elif isinstance(wind_config, tuple) and len(wind_config) == 2 and isinstance(wind_config[0], (int, float)):
                self.base_wind = random.uniform(*wind_config)
                
            # 情况3: wind 是嵌套元组 (例如 ((-10, -5), (5, 10))) -> 先随机选一个范围，再在该范围内随机取值
            elif isinstance(wind_config, tuple) and len(wind_config) == 2 and isinstance(wind_config[0], tuple):
                # 随机选择其中一个范围
                selected_range = random.choice(wind_config)
                # 在选中的范围内随机取值
                self.base_wind = random.uniform(*selected_range)
            else:
                raise ValueError("Invalid wind configuration")

            self.state = {
            'x': x, 'y': y, 'vx': vx, 'vy': vy,
            'theta': theta, 'vtheta': 0,
            'phi': 0, 'f': 0,
            't': 0, 'a_': 0,
            'delta_L': 0, 'delta_R': 0,
            'f_cmd': 0, 'theta_cmd': 0, 'M_req': 0, 'q': 0,
            'vphi': 0,'vdelta': 0
            }

        self.fuel = self.max_fuel
        self.state_buffer = []
        self.step_id = 0
        self.already_landing = False
        self.already_crash = False
        cv2.destroyAllWindows()
        return self.flatten(self.state)
 
    def create_action_table(self):
        f0 = 0.2 * self.g * self.m_simu  # thrust
        f1 = 1.0 * self.g * self.m_simu
        f2 = 2.0 * self.g * self.m_simu
        vphi0 = 0  # Nozzle angular velocity
        vphi1 = 15 / 180 * np.pi
        vphi2 = -15 / 180 * np.pi
        vdelta0 = 0
        vdelta1 = 30 / 180 * np.pi
        vdelta2 = -30 / 180 * np.pi

        action_table = {
            'f': [f0, f1, f2],
            'vphi': [vphi0, vphi1, vphi2],
            'vdelta_L': [vdelta0, vdelta1, vdelta2],
            'vdelta_R': [vdelta0, vdelta1, vdelta2]
        }
        return action_table

    def get_random_action(self):
        return random.randint(0, len(self.action_table)-1)

    def create_random_state(self):
        # predefined locations
        x_range = self.world_x_max - self.world_x_min
        y_range = self.world_y_max - self.world_y_min
        xc = (self.world_x_max + self.world_x_min) / 2.0
        yc = (self.world_y_max + self.world_y_min) / 2.0

        if self.task == 'landing':
            x = random.uniform(xc - x_range / 4.0, xc + x_range / 4.0)
            y = yc + 0.4*y_range
            theta = random.uniform(-30, 30) / 180 * np.pi
            vy = -140
            vx = 0

        if self.task == 'hover':
            x = xc
            y = yc + 0.2 * y_range
            theta = random.uniform(-45, 45) / 180 * np.pi
            vy = -10
            vx = 0
       
        state = {
            'x': x, 'y': y, 'vx': vx, 'vy': vy,
            'theta': theta, 'vtheta': 0,
            'phi': 0, 'f': 0,
            't': 0, 'a_': 0,
            'delta_L': 0, 'delta_R': 0,
            'f_cmd': 0, 'theta_cmd': 0, 'M_req': 0, 'q': 0
        }

        return state

    def check_crash(self, state):
        if self.task == 'hover':
            x, y = state['x'], state['y']
            theta = state['theta']
            crash = False
            if y <= self.H / 2.0:
                crash = True
            if y >= self.world_y_max - self.H / 2.0:
                crash = True
            return crash

        elif self.task == 'landing':
            x, y = state['x'], state['y']
            vx, vy = state['vx'], state['vy']
            theta = state['theta']
            vtheta = state['vtheta']
            v = (vx**2 + vy**2)**0.5

            crash = False
            if y >= self.world_y_max - self.H / 2.0:
                crash = True
            if y <= 10 + self.H / 2.0 and v >= 10.0:
                crash = True
            if y <= 10 + self.H / 2.0 and abs(x) >= 15.0:
                crash = True
            if y <= 35 + self.H / 2.0 and abs(x) >= 30.0:
                crash = True
            if y <= 10 + self.H / 2.0 and abs(theta) >= 5/180*np.pi:
                crash = True
            if y <= 10 + self.H / 2.0 and abs(vtheta) >= 5/180*np.pi:
                crash = True
            if self.fuel <= 0:
                crash = True
            # 在 check_crash 方法的 landing 任务部分添加
            if x <= self.world_x_min + self.H / 10 or x >= self.world_x_max - self.H / 10:
                crash = True
            return crash

    def check_landing_success(self, state):
        if self.task == 'hover':
            return False
        elif self.task == 'landing':
            x, y = state['x'], state['y']
            vx, vy = state['vx'], state['vy']
            theta = state['theta']
            vtheta = state['vtheta']
            v = (vx**2 + vy**2)**0.5
            return True if y <= 10 + self.H / 2.0 and v < 10.0 and abs(x) < 15.0 \
                           and abs(theta) < 5/180*np.pi and abs(vtheta) < 5/180*np.pi and self.fuel > 0 else False

    def calculate_reward(self, state):

        x_range = self.world_x_max - self.world_x_min
        y_range = self.world_y_max - self.world_y_min

        # dist between agent and target point
        dist_x = abs(state['x'] - self.target_x)
        dist_y = abs(state['y'] - self.target_y)
        dist_norm = dist_x / x_range + dist_y / y_range

        dist_reward = 1.0*(1.0 - dist_norm)

        theta_penalty = -0.5 * (abs(state['theta']) / np.pi)

        f_ratio = state['f_cmd'] / (2.0*self.env_params['g']*self.m_simu)
        fuel_penalty = -0.2 * f_ratio

        # 动压计算（越高，舵面越有效；但TVC不应主导）
        q_dyn_factor = np.clip(self.env_params['q'] / self.q_ref, 0.0, 1.0)

        # 惩罚推力摆角：不论动压如何，摆动喷管都要扣分；高动压下扣得更狠！
        # 如果你的状态里有 v_phi ，最好惩罚速度，如果没有，惩罚绝对角度。
        phi_penalty = -0.01 * (1.0 + 2.0 * q_dyn_factor) * (abs(state['vphi']) / (90*np.pi/180))
        
        # 栅格舵几乎免费（或者给极小的惩罚，防止无意义乱摆）
        fin_penalty = -0.01 * abs(state['vdelta']) / (30*np.pi/180)

        #抗扰动/防侧滑惩罚 (替代你之前的风力奖励)
        # 不要奖励风，而是惩罚“速度矢量与目标不一致”。
        # 也就是如果火箭横向速度 vx 很大，扣分。这自然增强了鲁棒性。
        velocity_penalty = -0.1 * abs(state['vx']) / 50.0

        reward = dist_reward + theta_penalty + fuel_penalty + phi_penalty + fin_penalty + velocity_penalty

        v = (state['vx'] ** 2 + state['vy'] ** 2) ** 0.5
        if self.task == 'landing':
            if self.already_crash:
                # 坠毁：一次性扣除巨大的分数，不要乘剩余步数！
                # 速度越快坠毁，扣得越狠
                crash_penalty = -50.0 - 2.0 * v 
                reward = crash_penalty
                
            elif self.already_landing:
                # 成功着陆：一次性给予巨大奖励
                # 着陆时姿态越正、速度越小、越靠近中心，得分越高
                vel_quality = np.exp(-v / 10.0)
                land_quality = np.exp(-dist_norm) * np.exp(-abs(state['theta'])) * vel_quality
                reward = 100.0 * land_quality
                
                # 只有成功着陆了，才可以把“剩余时间”作为省油奖励加回去
                reward += 0.5 * (self.max_steps - self.step_id)

        return reward

    def step(self, action=None):

        x, y, vx, vy = self.state['x'], self.state['y'], self.state['vx'], self.state['vy']
        theta, vtheta = self.state['theta'], self.state['vtheta']
        phi, delta_L, delta_R = self.state['phi'], self.state['delta_L'], self.state['delta_R']

        self.wind_strength = self.base_wind + 2.0 * np.sin(self.step_id * 0.05) if self.base_wind != 0 else 0
        rel_vx, rel_vy = self.wind_strength - vx, 0 - vy

        self.env_params['q'] = 0.5 * self.rho_air * (rel_vx**2 + rel_vy**2)
        self.q_history.append(self.env_params['q'])

        f_cmd, vphi, vdelta_L, vdelta_R, theta_cmd, M_req = action
        v_body_x = rel_vx * np.cos(theta) + rel_vy * np.sin(theta)
        v_body_y = -rel_vx * np.sin(theta) + rel_vy * np.cos(theta)
        alpha_base = np.arctan2(v_body_x, v_body_y + 1e-6)
        alpha_base = np.clip(alpha_base, -np.pi/4, np.pi/4)

        alpha_total_L = alpha_base + delta_L
        alpha_total_R = alpha_base + vdelta_R

        Cl_L, Cd_L = 2*alpha_total_L, 0.5 + 0.8*(alpha_total_L**2)
        Cl_R, Cd_R = 2*alpha_total_R, 0.5 + 0.8*(alpha_total_R**2)

        fl_L = self.env_params['q'] * Cl_L * self.fin_area
        fd_L = self.env_params['q'] * Cd_L * self.fin_area
        fl_R = self.env_params['q'] * Cl_R * self.fin_area
        fd_R = self.env_params['q'] * Cd_R * self.fin_area

        if delta_L - np.pi/2 == 0:
            fl_L = 0
            fd_L = 0
        if delta_R + np.pi/2 == 0:
            fl_R = 0
            fd_R = 0

        v_mag = np.sqrt(v_body_x**2 + v_body_y**2) + 1e-6

        ft_L = (fl_L * (v_body_y / v_mag))
        fr_L = (fd_L * (v_body_y / v_mag))

        # 右舵贡献
        ft_R = (fl_R * (v_body_y / v_mag))
        fr_R = (fd_R * (v_body_y / v_mag))

        ft_fin_total = ft_L + ft_R
        fr_fin_total = fr_L + fr_R

        ft = -f_cmd * np.sin(phi) + ft_fin_total
        fr =  f_cmd * np.cos(phi) + fr_fin_total
        fx = ft*np.cos(theta) - fr*np.sin(theta)
        fy = ft*np.sin(theta) + fr*np.cos(theta)

        ax, ay = (fx+self.env_params['rho']*rel_vx)/self.m_simu, (fy-self.g*self.m_simu+self.env_params['rho']*rel_vy)/self.m_simu
        self.ax = ax  # update acceleration
        self.ay.append(ay)
        
        torque_fins = (-ft_L * self.env_params['H'] / 2.0 - fr_L * self.env_params['W'] / 2.0) + (-ft_R * self.env_params['H'] / 2.0 + fr_R * self.env_params['W'] / 2.0)
        torque_thrust = -f_cmd * np.sin(phi) * self.env_params['H'] / 2.0

        atheta = (torque_fins + torque_thrust) / self.I
        self.atheta = atheta  # update angular acceleration

        # update agent
        if self.already_landing:
            vx, vy, ax, ay, theta, vtheta, atheta = 0, 0, 0, 0, 0, 0, 0
            phi, f = 0, 0
            delta_L, delta_R = 0, 0
            action, f_cmd, theta_cmd, M_req, self.env_params['q'] = 0, 0, 0, 0, 0
            vphi, vdelta_L, vdelta_R = 0, 0, 0

        self.step_id += 1
        x_new = x + vx*self.dt + 0.5 * ax * (self.dt**2)
        y_new = y + vy*self.dt + 0.5 * ay * (self.dt**2)
        vx_new, vy_new = vx + ax * self.dt, vy + ay * self.dt
        theta_new = theta + vtheta*self.dt + 0.5 * atheta * (self.dt**2)
        vtheta_new = vtheta + atheta * self.dt
        phi = phi + self.dt*vphi
        delta_L = delta_L + vdelta_L*self.dt
        delta_R = delta_R + vdelta_R*self.dt
        delta_L = np.clip(delta_L, -np.deg2rad(20), np.deg2rad(20))
        delta_R = np.clip(delta_R, -np.deg2rad(20), np.deg2rad(20))
        phi = np.clip(phi, -np.deg2rad(15), np.deg2rad(15))
        self.rho_air = 1.225*np.exp(-y_new/8500)
        dm = f_cmd/self.m_simu*self.m_true*4*self.dt/(self.Isp*self.g)
        self.fuel = max(self.fuel - dm, 0)  

        theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))   # 归一化到 [-π, π]

        self.phi_history.append(phi/np.pi*180)
        self.state = {
            'x': x_new, 'y': y_new, 'vx': vx_new, 'vy': vy_new,
            'theta': theta_new, 'vtheta': vtheta_new,
            'phi': phi, 'f': 0,
            't': self.step_id, 'action_': action, 'delta_L': delta_L, 'delta_R': delta_R,
            'f_cmd': f_cmd, 'theta_cmd': theta_cmd, 'M_req': M_req, 'q': self.env_params['q'],
            'vphi': vphi, 'vdelta': vdelta_L
        }
        self.state_buffer.append(self.state)

        self.already_landing = self.check_landing_success(self.state)
        self.already_crash = self.check_crash(self.state)
        reward = self.calculate_reward(self.state)

        if self.already_landing:
            self.landing_x = np.abs(x_new)

        if self.already_crash or self.already_landing:
            done = True
        else:
            done = False

        return self.flatten(self.state), reward, done, {}

    def flatten(self, state):
        x = [state['x']/150, state['y']/510, state['vx']/50, state['vy']/100,
             state['theta']/(np.pi/2), state['vtheta']/np.deg2rad(15), state['t']/self.max_steps,
             state['phi']/np.deg2rad(20), state['delta_L']/np.deg2rad(30), state['delta_R']/np.deg2rad(30),
             state['f_cmd']/(2*self.g*self.m_simu), state['theta_cmd']/(np.pi/2), state['M_req']/5e5, state['q']/self.q_ref]## s
        return np.array(x, dtype=np.float32)

    def render(self, window_name='env', wait_time=20,
               with_trajectory=True, with_camera_tracking=True,
               crop_scale=0.4):

        canvas = np.copy(self.bg_img)
        polys = self.create_polygons()

        # draw target region
        for poly in polys['target_region']:
            self.draw_a_polygon(canvas, poly)
        # draw rocket
        for poly in polys['rocket']:
            self.draw_a_polygon(canvas, poly)
        # draw grid fins
        for poly in polys['grid_fins']:
            self.draw_a_polygon(canvas, poly)
        frame_0 = canvas.copy()

        # draw engine work
        for poly in polys['engine_work']:
            self.draw_a_polygon(canvas, poly)
        frame_1 = canvas.copy()

        if with_camera_tracking:
            frame_0 = self.crop_alongwith_camera(frame_0, crop_scale=crop_scale)
            frame_1 = self.crop_alongwith_camera(frame_1, crop_scale=crop_scale)

        return frame_0, frame_1

    def create_polygons(self):

        polys = {'rocket': [], 'engine_work': [], 'target_region': [], 'grid_fins': []}

        if self.rocket_type == 'falcon':

            H, W = self.H, self.H/10
            dl = self.H / 30

            # rocket main body
            pts = [[-W/2, H/2], [W/2, H/2], [W/2, -H/2], [-W/2, -H/2]]
            polys['rocket'].append({'pts': pts, 'face_color': (242, 242, 242), 'edge_color': None})
            # rocket paint
            pts = utils.create_rectangle_poly(center=(0, -0.35*H), w=W, h=0.1*H)
            polys['rocket'].append({'pts': pts, 'face_color': (42, 42, 42), 'edge_color': None})
            pts = utils.create_rectangle_poly(center=(0, -0.46*H), w=W, h=0.02*H)
            polys['rocket'].append({'pts': pts, 'face_color': (42, 42, 42), 'edge_color': None})
            # rocket landing rack
            pts = [[-W/2, -H/2], [-W/2-H/10, -H/2-H/20], [-W/2, -H/2+H/20]]
            polys['rocket'].append({'pts': pts, 'face_color': None, 'edge_color': (0, 0, 0)})
            pts = [[W/2, -H/2], [W/2+H/10, -H/2-H/20], [W/2, -H/2+H/20]]
            polys['rocket'].append({'pts': pts, 'face_color': None, 'edge_color': (0, 0, 0)})
    
        elif self.rocket_type == 'starship':

            H, W = self.H, self.H / 2.6
            dl = self.H / 30

            # rocket main body (right half)
            pts = np.array([[ 0.        ,  0.5006878 ],
                           [ 0.03125   ,  0.49243465],
                           [ 0.0625    ,  0.48143053],
                           [ 0.11458334,  0.43878955],
                           [ 0.15277778,  0.3933975 ],
                           [ 0.2326389 ,  0.23796424],
                           [ 0.2326389 , -0.49931225],
                           [ 0.        , -0.49931225]], dtype=np.float32)
            pts[:, 0] = pts[:, 0] * W
            pts[:, 1] = pts[:, 1] * H
            polys['rocket'].append({'pts': pts, 'face_color': (242, 242, 242), 'edge_color': None})

            # rocket main body (left half)
            pts = np.array([[-0.        ,  0.5006878 ],
                           [-0.03125   ,  0.49243465],
                           [-0.0625    ,  0.48143053],
                           [-0.11458334,  0.43878955],
                           [-0.15277778,  0.3933975 ],
                           [-0.2326389 ,  0.23796424],
                           [-0.2326389 , -0.49931225],
                           [-0.        , -0.49931225]], dtype=np.float32)
            pts[:, 0] = pts[:, 0] * W
            pts[:, 1] = pts[:, 1] * H
            polys['rocket'].append({'pts': pts, 'face_color': (212, 212, 232), 'edge_color': None})

            # upper wing (right)
            pts = np.array([[0.15972222, 0.3933975 ],
                           [0.3784722 , 0.303989  ],
                           [0.3784722 , 0.2352132 ],
                           [0.22916667, 0.23658872]], dtype=np.float32)
            pts[:, 0] = pts[:, 0] * W
            pts[:, 1] = pts[:, 1] * H
            polys['rocket'].append({'pts': pts, 'face_color': (42, 42, 42), 'edge_color': None})

            # upper wing (left)
            pts = np.array([[-0.15972222,  0.3933975 ],
                           [-0.3784722 ,  0.303989  ],
                           [-0.3784722 ,  0.2352132 ],
                           [-0.22916667,  0.23658872]], dtype=np.float32)
            pts[:, 0] = pts[:, 0] * W
            pts[:, 1] = pts[:, 1] * H
            polys['rocket'].append({'pts': pts, 'face_color': (42, 42, 42), 'edge_color': None})

            # lower wing (right)
            pts = np.array([[ 0.2326389 , -0.16368638],
                           [ 0.4548611 , -0.33562586],
                           [ 0.4548611 , -0.48555708],
                           [ 0.2638889 , -0.48555708]], dtype=np.float32)
            pts[:, 0] = pts[:, 0] * W
            pts[:, 1] = pts[:, 1] * H
            polys['rocket'].append({'pts': pts, 'face_color': (100, 100, 100), 'edge_color': None})

            # lower wing (left)
            pts = np.array([[-0.2326389 , -0.16368638],
                           [-0.4548611 , -0.33562586],
                           [-0.4548611 , -0.48555708],
                           [-0.2638889 , -0.48555708]], dtype=np.float32)
            pts[:, 0] = pts[:, 0] * W
            pts[:, 1] = pts[:, 1] * H
            polys['rocket'].append({'pts': pts, 'face_color': (100, 100, 100), 'edge_color': None})

        else:
            raise NotImplementedError('rocket type [%s] is not found, please choose one '
                                      'from (falcon, starship)' % self.rocket_type)

        # engine work
        f, phi = self.state['f_cmd'], self.state['phi']
        c, s = np.cos(phi), np.sin(phi)

        if f > 0 and f <= 0.6 * self.g * self.m_simu:
            pts1 = utils.create_rectangle_poly(center=(2 * dl * s, -H / 2 - 2 * dl * c), w=dl, h=dl)
            pts2 = utils.create_rectangle_poly(center=(5 * dl * s, -H / 2 - 5 * dl * c), w=1.5 * dl, h=1.5 * dl)
            polys['engine_work'].append({'pts': pts1, 'face_color': (255, 255, 255), 'edge_color': None})
            polys['engine_work'].append({'pts': pts2, 'face_color': (255, 255, 255), 'edge_color': None})
        elif f > 0.6 * self.g * self.m_simu and f < 1.2 * self.g * self.m_simu:
            pts1 = utils.create_rectangle_poly(center=(2 * dl * s, -H / 2 - 2 * dl * c), w=dl, h=dl)
            pts2 = utils.create_rectangle_poly(center=(5 * dl * s, -H / 2 - 5 * dl * c), w=1.5 * dl, h=1.5 * dl)
            pts3 = utils.create_rectangle_poly(center=(8 * dl * s, -H / 2 - 8 * dl * c), w=2 * dl, h=2 * dl)
            polys['engine_work'].append({'pts': pts1, 'face_color': (255, 255, 255), 'edge_color': None})
            polys['engine_work'].append({'pts': pts2, 'face_color': (255, 255, 255), 'edge_color': None})
            polys['engine_work'].append({'pts': pts3, 'face_color': (255, 255, 255), 'edge_color': None})
        elif f >= 1.2 * self.g * self.m_simu:
            pts1 = utils.create_rectangle_poly(center=(2 * dl * s, -H / 2 - 2 * dl * c), w=dl, h=dl)
            pts2 = utils.create_rectangle_poly(center=(5 * dl * s, -H / 2 - 5 * dl * c), w=1.5 * dl, h=1.5 * dl)
            pts3 = utils.create_rectangle_poly(center=(8 * dl * s, -H / 2 - 8 * dl * c), w=2 * dl, h=2 * dl)
            pts4 = utils.create_rectangle_poly(center=(12 * dl * s, -H / 2 - 12 * dl * c), w=3 * dl, h=3 * dl)
            polys['engine_work'].append({'pts': pts1, 'face_color': (255, 255, 255), 'edge_color': None})
            polys['engine_work'].append({'pts': pts2, 'face_color': (255, 255, 255), 'edge_color': None})
            polys['engine_work'].append({'pts': pts3, 'face_color': (255, 255, 255), 'edge_color': None})
            polys['engine_work'].append({'pts': pts4, 'face_color': (255, 255, 255), 'edge_color': None})
        # target region
        if self.task == 'hover':
            pts1 = utils.create_rectangle_poly(center=(self.target_x, self.target_y), w=0, h=self.target_r/3.0)
            pts2 = utils.create_rectangle_poly(center=(self.target_x, self.target_y), w=self.target_r/3.0, h=0)
            polys['target_region'].append({'pts': pts1, 'face_color': None, 'edge_color': (242, 242, 242)})
            polys['target_region'].append({'pts': pts2, 'face_color': None, 'edge_color': (242, 242, 242)})
        else:
            pts1 = utils.create_ellipse_poly(center=(0, 0), rx=self.target_r, ry=self.target_r/4.0)
            pts2 = utils.create_rectangle_poly(center=(0, 0), w=self.target_r/3.0, h=0)
            pts3 = utils.create_rectangle_poly(center=(0, 0), w=0, h=self.target_r/6.0)
            polys['target_region'].append({'pts': pts1, 'face_color': None, 'edge_color': (242, 242, 242)})
            polys['target_region'].append({'pts': pts2, 'face_color': None, 'edge_color': (242, 242, 242)})
            polys['target_region'].append({'pts': pts3, 'face_color': None, 'edge_color': (242, 242, 242)})

        # grid fins
        delta_L, delta_R = self.state['delta_L'], self.state['delta_R']
        pts1 = utils.create_grid_fin_R(delta_R, W, H)
        pts2 = utils.create_grid_fin_L(delta_L, W, H)
        polys['grid_fins'].append({'pts': pts1, 'face_color': (0, 0, 0), 'edge_color': None})
        polys['grid_fins'].append({'pts': pts2, 'face_color': (0, 0, 0), 'edge_color': None})

        # apply transformation
        for poly in polys['rocket'] + polys['engine_work'] + polys['grid_fins']:
            M = utils.create_pose_matrix(tx=self.state['x'], ty=self.state['y'], rz=self.state['theta'])
            pts = np.array(poly['pts'])
            pts = np.concatenate([pts, np.ones_like(pts)], axis=-1)  # attach z=1, w=1
            pts = np.matmul(M, pts.T).T
            poly['pts'] = pts[:, 0:2]

        return polys


    def draw_a_polygon(self, canvas, poly):

        pts, face_color, edge_color = poly['pts'], poly['face_color'], poly['edge_color']
        pts_px = self.wd2pxl(pts)
        if face_color is not None:
            cv2.fillPoly(canvas, [pts_px], color=face_color, lineType=cv2.LINE_AA)
        if edge_color is not None:
            cv2.polylines(canvas, [pts_px], isClosed=True, color=edge_color, thickness=1, lineType=cv2.LINE_AA)

        return canvas


    def wd2pxl(self, pts, to_int=True):

        pts_px = np.zeros_like(pts)

        scale = self.viewport_w / (self.world_x_max - self.world_x_min)
        for i in range(len(pts)):
            pt = pts[i]
            x_p = (pt[0] - self.world_x_min) * scale
            y_p = (pt[1] - self.world_y_min) * scale
            y_p = self.viewport_h - y_p
            pts_px[i] = [x_p, y_p]

        if to_int:
            return pts_px.astype(int)
        else:
            return pts_px

    def draw_text(self, canvas, color=(255, 255, 0)):

        def put_text(vis, text, pt):
            cv2.putText(vis, text=text, org=pt, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=color, thickness=1, lineType=cv2.LINE_AA)

        pt = (10, 20)
        text = "simulation time: %.2fs" % (self.step_id * self.dt)
        put_text(canvas, text, pt)

        pt = (10, 40)
        text = "simulation steps: %d" % (self.step_id)
        put_text(canvas, text, pt)

        pt = (10, 60)
        text = "x: %.2f m, y: %.2f m" % \
               (self.state['x'], self.state['y'])
        put_text(canvas, text, pt)

        pt = (10, 80)
        text = "vx: %.2f m/s, vy: %.2f m/s" % \
               (self.state['vx'], self.state['vy'])
        put_text(canvas, text, pt)

        pt = (10, 100)
        text = "theta: %.2f degree, vtheta: %.2f degree/s" % \
               (self.state['theta'] * 180 / np.pi, self.state['vtheta'] * 180 / np.pi)
        put_text(canvas, text, pt)

        pt = (10, 120)
        text = "fuel-consumption: %.2f kg" % (self.max_fuel - self.fuel)
        put_text(canvas, text, pt)

        pt = (10, 140)
        text = "wind: %.2f m/s" % (self.wind_strength)
        put_text(canvas, text, pt)

        pt = (10, 160)
        text = "delta_L: %.2f, delta_R: %.2f, q: %.2f" % (self.state['delta_L'] * 180 / np.pi, self.state['delta_R'] * 180 / np.pi, self.env_params['q'])
        put_text(canvas, text, pt)

        pt = (10, 180)
        text = "f: %.2f, phi: %.2f" % (self.state['f_cmd'], self.state['phi'])
        put_text(canvas, text, pt)

    def draw_trajectory(self, canvas, color=(255, 0, 0)):

        pannel_w, pannel_h = 256, 256
        traj_pannel = 255 * np.ones([pannel_h, pannel_w, 3], dtype=np.uint8)

        sw, sh = pannel_w/self.viewport_w, pannel_h/self.viewport_h  # scale factors

        # draw horizon line
        range_x, range_y = self.world_x_max - self.world_x_min, self.world_y_max - self.world_y_min
        pts = [[self.world_x_min + range_x/3, self.H/2], [self.world_x_max - range_x/3, self.H/2]]
        pts_px = self.wd2pxl(pts)
        x1, y1 = int(pts_px[0][0]*sw), int(pts_px[0][1]*sh)
        x2, y2 = int(pts_px[1][0]*sw), int(pts_px[1][1]*sh)
        cv2.line(traj_pannel, pt1=(x1, y1), pt2=(x2, y2),
                 color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        # draw vertical line
        pts = [[0, self.H/2], [0, self.H/2+range_y/20]]
        pts_px = self.wd2pxl(pts)
        x1, y1 = int(pts_px[0][0]*sw), int(pts_px[0][1]*sh)
        x2, y2 = int(pts_px[1][0]*sw), int(pts_px[1][1]*sh)
        cv2.line(traj_pannel, pt1=(x1, y1), pt2=(x2, y2),
                 color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        if len(self.state_buffer) < 2:
            return

        # draw traj
        pts = []
        for state in self.state_buffer:
            pts.append([state['x'], state['y']])
        pts_px = self.wd2pxl(pts)

        dn = 5
        for i in range(0, len(pts_px)-dn, dn):

            x1, y1 = int(pts_px[i][0]*sw), int(pts_px[i][1]*sh)
            x1_, y1_ = int(pts_px[i+dn][0]*sw), int(pts_px[i+dn][1]*sh)

            cv2.line(traj_pannel, pt1=(x1, y1), pt2=(x1_, y1_), color=color, thickness=2, lineType=cv2.LINE_AA)

        roi_x1, roi_x2 = self.viewport_w - 10 - pannel_w, self.viewport_w - 10
        roi_y1, roi_y2 = 10, 10 + pannel_h
        canvas[roi_y1:roi_y2, roi_x1:roi_x2, :] = 0.6*canvas[roi_y1:roi_y2, roi_x1:roi_x2, :] + 0.4*traj_pannel



    def crop_alongwith_camera(self, vis, crop_scale=0.4):
        x, y = self.state['x'], self.state['y']
        xp, yp = self.wd2pxl([[x, y]])[0]
        crop_w_half, crop_h_half = int(self.viewport_w*crop_scale), int(self.viewport_h*crop_scale)
        # check boundary
        if xp <= crop_w_half + 1:
            xp = crop_w_half + 1
        if xp >= self.viewport_w - crop_w_half - 1:
            xp = self.viewport_w - crop_w_half - 1
        if yp <= crop_h_half + 1:
            yp = crop_h_half + 1
        if yp >= self.viewport_h - crop_h_half - 1:
            yp = self.viewport_h - crop_h_half - 1

        x1, x2, y1, y2 = xp-crop_w_half, xp+crop_w_half, yp-crop_h_half, yp+crop_h_half
        vis = vis[y1:y2, x1:x2, :]

        vis = cv2.resize(vis, (self.viewport_w, self.viewport_h))
        return vis