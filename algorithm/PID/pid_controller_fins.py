import numpy as np

class PID:
    """标准PID控制器"""
    def __init__(self, kp, ki, kd, output_limits=(None, None)):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.limits = output_limits
        self.integral = 0
        self.last_error = 0
        self.last_derivative = 0.0

    def compute(self, target, current, dt):
        error = target - current
        self.integral += error * dt
        # 简单的积分抗饱和
        if self.limits[0] is not None:
            self.integral = np.clip(self.integral, -50, 50) 
            
        raw_derivative = (error - self.last_error) / dt
        # 简单的微分平滑
        alpha_d = 0.2
        derivative = alpha_d * raw_derivative + (1 - alpha_d) * self.last_derivative
        self.last_derivative = derivative
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.last_error = error
        
        if self.limits[0] is not None:
            output = np.clip(output, self.limits[0], self.limits[1])
        return output

class PIDController:
    def __init__(self, dt=1/60, action_table=None, max_steps = None):
        self.dt = dt
        self.action_table = action_table
        self.M_req = 0.0

        # 高度控制 PID（之前用于位置环，现在高度环改用闭环位置+速度，但保留原来的 PID 以备后用）
        #self.alt_pid = PID(kp=1.2, ki=0, kd=0.6, output_limits=(-0.8 * env_params['g'], env_params['g']))

        # 水平位置控制器
        self.pos_x_pid = PID(kp=0.005, ki=0.0, kd=0.015)

        # 姿态控制器（可保留，但这里不再使用单独的 att_pid，而是直接在 get_action 中计算）
        # self.att_pid = PID(kp=35, ki=0, kd=70)

        # 状态记录
        self.prev_theta_cmd = 0.0
        self.smoothed_theta_cmd = 0.0
        self.last_f_cmd = 0.0
        self.last_error_v = 0.0
        self.vel_integral = 0.0
        self.theta_cmd_history = []
        self.theta_history = []
        self.f_cmd_history = []
        self.M_cmd_history = []
        self.a_history = []
        self.drag_history = []
        self.vy_des_history = []
        self.vtheta_history = []
        self.m_fin = []
        self.delta_target = []

        # 新增：速度环积分项
        self.vel_integral = 0.0

        # 新增：用于剩余时间计算的仿真步数（需要从外部传入或后续设置）
        self.max_steps = max_steps        # 将在 reset 时从 env 获取
        self.step_id = 0          # 将在每次 step 时更新
        self.w_smooth = 0.0
        self.alpha_w = 0.1
        self.alpha_vtheta = 0.2
        self.last_phi_cmd = 0.0
        self.H_switch = 150.0
        self.torque_integral = 0.0

    def compute_fin_forces(self, delta, alpha_base, q_dyn, fin_area, eps):
        alpha_total = delta + alpha_base
        Cl = 2 * alpha_total
        Cd = 0.5 + 0.8 * alpha_total**2
        fl = q_dyn * Cl * fin_area
        fd = q_dyn * Cd * fin_area
        if abs(delta) - np.pi / 2 == 0:  
            fl = 0.0
            fd = 0.0
        return fl, fd
    
    def compute_fin_moment(self, v_body_x, v_body_y, delta_L, delta_R, q_dyn, fin_area, W, H, eps):
        alpha_base = np.arctan2(v_body_x, v_body_y + 1e-6)
        alpha_base = np.clip(alpha_base, -np.pi/4, np.pi/4)
        fl_L, fd_L = self.compute_fin_forces(delta_L, alpha_base, q_dyn, fin_area, eps)
        fl_R, fd_R = self.compute_fin_forces(delta_R, alpha_base, q_dyn, fin_area, eps)
        if abs(delta_L) - np.pi / 2 == 0:  
            fl_L = 0.0
            fd_L = 0.0
        if abs(delta_R) - np.pi / 2 == 0:  
            fl_R = 0.0
            fd_R = 0.0
        v_mag = np.sqrt(v_body_x**2 + v_body_y**2) + 1e-6

        ft_L = (fd_L * (v_body_x / v_mag))*0 + (fl_L * (v_body_y / v_mag))
        fr_L = (fd_L * (v_body_y / v_mag)) - 0*(fl_L * (v_body_x / v_mag))

        # 右舵贡献
        ft_R = (fd_R * (v_body_x / v_mag))*0 + (fl_R * (v_body_y / v_mag))
        fr_R = (fd_R * (v_body_y / v_mag)) - 0*(fl_R * (v_body_x / v_mag))

        torque_fins = (-ft_L * H / 2.0 - fr_L * W / 2.0) + (-ft_R * H / 2.0 + fr_R * W / 2.0)
        return torque_fins
    
    def get_action(self, state, env_params, wind_strength):

        x, y = state['x'], state['y']
        vx, vy = state['vx'], state['vy']
        theta, vtheta = state['theta'], state['vtheta']
        delta_L, delta_R = state['delta_L'], state['delta_R']

        # --------------------------------------------------------------
        # 气动力估算（用于高度环推力补偿）
        # --------------------------------------------------------------

        q_dyn = env_params['q']
        fin_area = env_params['fin_area']

        rel_vx, rel_vy = wind_strength - vx, 0 - vy
        v_body_x = rel_vx * np.cos(theta) + rel_vy * np.sin(theta)
        v_body_y = -rel_vx * np.sin(theta) + rel_vy * np.cos(theta)
        alpha_base = np.arctan2(v_body_x, v_body_y + 1e-6)
        alpha_base = np.clip(alpha_base, -np.pi/4, np.pi/4)
        eps = np.deg2rad(1.0)                  
        fl_L, fd_L = self.compute_fin_forces(delta_L, alpha_base, q_dyn, fin_area, eps)
        fl_R, fd_R = self.compute_fin_forces(delta_R, alpha_base, q_dyn, fin_area, eps)

        v_mag = np.sqrt(v_body_x**2 + v_body_y**2) + 1e-6

        ft_L = (fd_L * (v_body_x / v_mag))*0 + (fl_L * (v_body_y / v_mag))
        fr_L = (fd_L * (v_body_y / v_mag)) - 0*(fl_L * (v_body_x / v_mag))

        # 右舵贡献
        ft_R = (fd_R * (v_body_x / v_mag))*0 + (fl_R * (v_body_y / v_mag))
        fr_R = (fd_R * (v_body_y / v_mag)) - 0*(fl_R * (v_body_x / v_mag))

        ft_fin_total = ft_L + ft_R
        fr_fin_total = fr_L + fr_R
        ft, fr = ft_fin_total, fr_fin_total
        fy = ft*np.sin(theta) + fr*np.cos(theta)
        drag_env = env_params['rho'] * (0-vy)
        self.drag_history.append(drag_env)
        # --------------------------------------------------------------
        # 3. 高度控制
        # --------------------------------------------------------------
        # 3.1 速度剖面设计（根据高度设定期望下降速度）
        # 设置切换高度（例如 200 米）

        if y > self.H_switch:
            a_des = 0.0
            self.a_history.append(a_des)
            self.vy_des_history.append(vy)
            f_cmd = 0.0 * env_params['g'] * env_params['m'] 
        else:
            h_des = 25.0 
            h_rem = y - h_des
            if h_rem > 0:
                # 1. 速度剖面制导：设计一个平滑的期望下落速度 (基于 v^2 = 2as)
                vy_target = -np.sqrt(2 * 0.8 * env_params['g'] * h_rem)
                # 限制最大下降速度和触地前最小速度
                vy_target = np.clip(vy_target, -140.0, -2.0)
                self.vy_des_history.append(vy_target)
                
                # 2. 闭环速度追踪 (P 控制)
                error_vy = vy_target - vy
                kp_v = 0.8  # 速度环比例增益
                a_des = kp_v * error_vy
                
                # 限制加速度指令
                a_des = np.clip(a_des, 0, 1.5 * env_params['g'])
            else:
                a_des = 0
                self.vy_des_history.append(0)
                
            self.a_history.append(a_des)

            # 推力计算
            total_force_vert = env_params['m'] * (env_params['g'] + a_des)
            # 防止大倾角时 cos(theta) 过小导致推力指令发散
            f_cmd = total_force_vert / np.clip(np.cos(theta), 0.75, 1.0)
            
            f_cmd = np.clip(f_cmd, 0.2 * env_params['m'] * env_params['g'],
                                    2.0 * env_params['m'] * env_params['g'])

        # --------------------------------------------------------------
        # 水平位置控制（输出期望姿态角 theta_cmd）
        # --------------------------------------------------------------
        # 外环：位置 PID
        theta_cmd = -self.pos_x_pid.compute(0, x, self.dt)    # 目标 x=0

        # 动态限幅（基于横向位移）
        if y < 100 :
            # 高度从 100m 降到 25m，最大允许倾角从 30° 线性减到 0
            max_tilt_ground = np.deg2rad(30) * max(0, (y - 30) / (100 - 30))
            max_tilt_ground = np.clip(max_tilt_ground, np.deg2rad(0), np.deg2rad(35))
            theta_cmd = np.clip(theta_cmd, -max_tilt_ground, max_tilt_ground)
        else:
            # 正常动态限幅（基于横向位移）
            dist_limit = np.deg2rad(0.3 * abs(x))
            tilt_limit = np.clip(dist_limit, np.deg2rad(0), np.deg2rad(35))
            theta_cmd = np.clip(theta_cmd, -tilt_limit, tilt_limit)

        if y < 50 and abs(x) < 50 and abs(vx) < 2.0:
            theta_cmd = 0.0
        max_rate = np.deg2rad(15)
        dtheta = theta_cmd - self.prev_theta_cmd
        dtheta = np.clip(dtheta, -max_rate * self.dt, max_rate * self.dt)
        theta_cmd_limited = self.prev_theta_cmd + dtheta
        self.prev_theta_cmd = theta_cmd_limited
        #theta_cmd_limited = np.deg2rad(5)

        # --------------------------------------------------------------
        # 姿态控制（PD 控制，输出总力矩 M_req）
        # --------------------------------------------------------------
        kp_att = 10
        kd_att = 6
        error_att = theta_cmd_limited - theta
        error_att = np.arctan2(np.sin(error_att), np.cos(error_att))   # 归一化
        M_req = (kp_att * error_att - kd_att * vtheta) * env_params['I']
        #alpha_phi = 0.6
        #self.M_req_filtered = alpha_phi * M_req + (1 - alpha_phi) * getattr(self, 'M_req_filtered', 0.0)
        #M_req = self.M_req_filtered
        M_req = np.clip(M_req, -5e5, 5e5)
        self.M_req = M_req   # 供 get_v 使用

        # --------------------------------------------------------------
        # 执行器分配（TVC 和栅格舵速率指令）
        # --------------------------------------------------------------
        # 调用 get_v 计算 TVC 和舵面角速度指令（注意 get_v 内部会使用 self.M_req）
        v_phi, v_delta_L, v_delta_R = self.get_v(state, f_cmd, env_params, wind_strength)

        self.step_id += 1
        # 可选：记录数据用于调试
        self.f_cmd_history.append(f_cmd / (env_params['m'] * env_params['g']))
        self.theta_cmd_history.append(theta_cmd_limited * 180 / np.pi)
        self.theta_history.append(theta * 180 / np.pi)
        self.M_cmd_history.append(self.M_req)
        self.vtheta_history.append(vtheta * 180 / np.pi)

        return f_cmd, v_phi, v_delta_L, v_delta_R
    
    def get_v(self, state, f, env_params, wind_strength):
        current_phi = state['phi']
        q_dyn = env_params['q']                     # 动压
        fin_area = env_params['fin_area']
        W, H = env_params['W'], env_params['H']
        delta_L = state['delta_L']
        delta_R = state['delta_R']
        theta = state['theta']
        vx, vy = state['vx'], state['vy']
        y = state['y']

        rel_vx, rel_vy = wind_strength - vx, 0 - vy
        v_body_x = rel_vx * np.cos(theta) + rel_vy * np.sin(theta)
        v_body_y = -rel_vx * np.sin(theta) + rel_vy * np.cos(theta)
              
        
        # 1. 力矩分配权重
        if y > self.H_switch:
            m_fin_target = self.M_req
        else:
            w_target = np.clip((q_dyn) / 5000, 0.0, 1.0)
            self.w_smooth = self.alpha_w * w_target + (1 - self.alpha_w) * self.w_smooth
            m_fin_target = self.w_smooth * self.M_req
        #m_fin_target = self.M_req
        # 2. 基于当前舵偏计算实际力矩（用于 TVC 补偿）   
        #moment_fin = self.compute_fin_moment(v_body_x, v_body_y, delta_L, delta_R, q_dyn, fin_area, W, H, 0)
        #self.m_fin.append(moment_fin)

        # 3. 求解所需舵偏（使得两个舵总力矩等于 m_fin_target）
        # 使用数值微分求导，然后线性求解
        #eps = np.deg2rad(1.0)            
        #moment_plus = self.compute_fin_moment(v_body_x, v_body_y, delta_L+eps, delta_R+eps, q_dyn, fin_area, W, H, 0)
        #deriv = (moment_plus - moment_fin) / eps   # 力矩对舵偏的导数
        k_fin = 2 * fin_area * (H/2)   # 升力系数斜率 2，力臂 H/2
        alpha_base = np.arctan2(v_body_x, v_body_y + 1e-6)
        alpha_base = np.clip(alpha_base, -np.pi/4, np.pi/4)
        if q_dyn > 100:
            delta_target = -m_fin_target / (k_fin * q_dyn) - alpha_base
            #aoa_inc = (m_fin_target - moment_fin) / deriv
            #aoa_inc = np.clip(aoa_inc, -np.deg2rad(2), np.deg2rad(2))
            #delta_target = delta_L + aoa_inc
            #delta_inc = (m_fin_target - moment_fin) / deriv
            #delta_inc = np.clip(delta_inc, -np.deg2rad(2), np.deg2rad(2))  # 限制单步增量
            #delta_target = delta_L + delta_inc
        else:
            delta_target = delta_L
        delta_target = np.clip(delta_target, -np.deg2rad(20), np.deg2rad(20))
        self.delta_target.append(delta_target*180/np.pi)
    
        # 4. TVC 补偿剩余力矩
        m_req_tvc = self.M_req - m_fin_target
        if f > 0.0 * env_params['m'] * env_params['g']:
            target_phi = -m_req_tvc / (f * (H / 2)* np.clip(np.cos(theta), 0.75, 1.0))
        else:
            target_phi = 0.0
        target_phi = np.clip(target_phi, -np.deg2rad(15), np.deg2rad(15))

        #alpha_phi = 0.1
        #self.target_phi_filtered = alpha_phi * target_phi + (1 - alpha_phi) * getattr(self, 'target_phi_filtered', 0.0)
        #target_phi = self.target_phi_filtered

        dphi = target_phi - self.last_phi_cmd
        dphi = np.clip(dphi, -np.deg2rad(5), np.deg2rad(5))
        target_phi = self.last_phi_cmd + dphi
        self.last_phi_cmd = target_phi
        v_phi_req = (target_phi - current_phi) * 60.0
        
        # 5. 舵面速率指令（与原逻辑相同）
        error_L = delta_target - delta_L
        error_R = delta_target - delta_R
        deadband = np.deg2rad(0.05)
        v_delta_L_req = 0.0 if abs(error_L) < deadband or abs(delta_L) == np.pi/2 else error_L * 3.0
        v_delta_R_req = 0.0 if abs(error_R) < deadband or abs(delta_R) == np.pi/2 else error_R * 3.0
        
        return (np.clip(v_phi_req, -np.deg2rad(90), np.deg2rad(90)), 
                np.clip(v_delta_L_req, -np.deg2rad(30), np.deg2rad(30)), 
                np.clip(v_delta_R_req, -np.deg2rad(30), np.deg2rad(30)))
