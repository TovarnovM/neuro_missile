import numpy as np
import sys 
sys.path.append(r'D:\neuro_missile\src\cydrone')
from drone import Drone2d, Missile2d
from easyvec import Vec2, Mat2
from scipy.optimize import minimize, Bounds, minimize_scalar, root_scalar
from gym.spaces.box import Box


class DroneGym3:
    @classmethod
    def make(cls, **kwargs):
        def drone_foo():
            drone = Drone2d.get_DJI()
            d = drone.to_dict()
            if vel := kwargs.get('vel'):
                d['vel'] = vel
            else:
                d['vel'] = Vec2.random((-1, 1), (-0.5,2))
            
            if alpha := kwargs.get('alpha'):
                d['alpha'] = alpha
            else:
                d['alpha'] = np.random.uniform(-10, 10) * 3.14 / 180
            
            if omega := kwargs.get('omega'):
                d['omega'] = omega
            else:
                d['omega'] = np.random.uniform(-0.01, 0.01)

            drone.from_dict(d)
            return drone

        def pos0_pos_trg_foo():
            pos0 = kwargs.get('pos0',       Vec2.random((-4000, 0),(4000, 2500)))
            pos_trg = kwargs.get('pos_trg', Vec2.random((-4000, 1000),(4000, 1500)))
            return pos0, pos_trg

        def vel_trg_foo():
            if vel_trg := kwargs.get('vel_trg'):
                return vel_trg
            vel_trg_alpha = kwargs.get('vel_trg_alpha', np.random.uniform(-1, 1) * np.pi)
            vel_trg_len = kwargs.get('vel_trg_len', np.random.uniform(1, 13))
            return Vec2(vel_trg_len, 0).rotate(vel_trg_alpha)

        def missile_pos0_vel_ay_alpha_foo(pos_drone, t_predict):
            vel_missile = kwargs.get('vel_missile', np.random.uniform(15, 30))
            ay_missile = kwargs.get('ay_missile', np.random.uniform(5, 15))
            alpha_missile = kwargs.get('alpha_missile', np.random.uniform(0, 1))
            r = vel_missile * kwargs.get('t_missile_prop', np.random.uniform(0.4, 0.9)) * t_predict
            angle = kwargs.get('missile_angle', np.random.uniform(-1, 1) * np.pi)
            return Vec2.from_list(pos_drone) + Vec2(r, 0).rotate(angle), vel_missile, ay_missile, alpha_missile


        return cls(
            drone_foo=drone_foo, 
            pos0_pos_trg_foo=pos0_pos_trg_foo,
            vel_trg_foo=vel_trg_foo,
            missile_pos0_vel_ay_alpha_foo=missile_pos0_vel_ay_alpha_foo,
            xy_bounds = ((-5000, 5000), (-10, 3000))
        )

    def __init__(self, drone_foo, pos0_pos_trg_foo, vel_trg_foo, missile_pos0_vel_ay_alpha_foo, **kwargs):
        self.time_curr = 0.0
        self.drone = None 
        self.drone_foo = drone_foo

        self.pos_trg = None
        self.pos0_pos_trg_foo = pos0_pos_trg_foo

        self.vel_trg_foo = vel_trg_foo
        self.vel_trg = None

        self.missile_pos0_vel_ay_alpha_foo = missile_pos0_vel_ay_alpha_foo
        self.missile_pos0 = None
        self.missile = None

        self.tau = kwargs.get('tau', 0.1)
        self.n_4_step = kwargs.get('n_4_step', 8)
        self.vel_max = kwargs.get('vel_max', 33.0)
        self.omega_max = kwargs.get('omega_max', 23.0)
        self.a_max = kwargs.get('a_max', 7.0)
        self.trg_radius = kwargs.get('trg_radius', 13)
        self.xy_bounds = kwargs.get('xy_bounds', ((-5000, 5000), (-10, 3000)))
        self.obs_min = np.array(kwargs.get('obs_min', 
            (self.xy_bounds[1][0],          # 0   H      - высота
            -np.pi,                         # 1   alpha  - угол наклона дрона к горизонту
            -np.pi,                         # 2   beta_v - угол между осью дрона и его скоростью
            0,                              # 3   D      - длина линии визирования
            -np.pi,                         # 4   phi    - угол медлу осью дрона и линией визинования
            0,                              # 5   V      - модуль скорости дрона
            -self.omega_max,                # 6   omega  - угловая скорость дрона
            0,                              # 7   V_trg  - модуль скорости, финального положениея
            -np.pi,                         # 8   gamma  - угол между финальной скоростью и линией визирования
            0,                              # 9   D_missile - длина линии визирования до преследующей ракеты
            0,                              # 10  V_missile - модуль вектора скорости ракеты
            -np.pi,                         # 11  gamma_missile - угол между скоростью ракеты и линией визирования
            -np.pi                          # 12  phi_missile    - угол медлу осью дрона и линией визинования до ракеты
                ) ))
        self.obs_max = np.array(kwargs.get('obs_max', 
            (self.xy_bounds[1][1],          # 0   H      - высота
            np.pi,                          # 1   alpha  - угол наклона дрона к горизонту 
            np.pi,                          # 2   beta_v - угол между осью дрона и его скоростью
            self.xy_bounds[0][1] - self.xy_bounds[0][0], # 3   D      - длина линии визирования
            np.pi,                          # 4   phi    - угол медлу осью дрона и линией визинования
            self.vel_max,                   # 5   V      - модуль скорости дрона
            self.omega_max,                 # 6   omega  - угловая скорость дрона
            self.vel_max,                   # 7   V_trg  - модуль скорости, финального положениея
            np.pi,                          # 8   gamma  - угол между финальной скоростью и линией визирования
            self.xy_bounds[0][1] - self.xy_bounds[0][0],  # 9   D_missile - длина линии визирования до преследующей ракеты
            self.vel_max,                  # 10  V_missile - модуль вектора скорости ракеты
            np.pi,                         # 11  gamma_missile - угол между скоростью ракеты и линией визирования
            np.pi                          # 12  phi_missile    - угол медлу осью дрона и линией визинования до ракеты
                ) ))
        self.history = []
        self.record_history = False

        self.action_space = Box(0,1, shape=(2,))
        self.antiflip = kwargs.get('antiflip', True)


    def set_state(self, state):
        self.drone.from_numpy(state[:self.drone.n])
        self.missile.from_numpy(state[self.drone.n:])
        self.time_curr = self.drone.t

    def reset(self):
        self.drone = self.drone_foo()
        self.pos0, self.pos_trg = self.pos0_pos_trg_foo()

        self.pos0 = Vec2.from_list(self.pos0)
        self.pos_trg = Vec2.from_list(self.pos_trg)

        self.vel_trg = Vec2.from_list(self.vel_trg_foo())

        d = self.drone.to_dict()
        d['pos'] = self.pos0
        self.drone.from_dict(d)

        t_predict = self.get_delta_t()
        mpos0, mvel_len, ay, alpha = self.missile_pos0_vel_ay_alpha_foo(self.pos0, t_predict)
        self.missile = Missile2d(mpos0, self.pos0, mvel_len, ay, alpha)

        self.history = []
        return self.get_observ()

    def get_state(self):
        return np.hstack((self.drone.to_numpy(), self.missile.to_numpy()))

    def get_observ(self, state=None, normalize=True):
        """np.ndarray
        [0, 1,     2,      3, 4,   5, 6,     7,     8]
        [H, alpha, beta_v, D, phi, V, omega, V_trg, gamma]

        0   H      - высота
        1   alpha  - угол наклона дрона к горизонту
        2   beta_v - угол между осью дрона и его скоростью
        3   D      - длина линии визирования
        4   phi    - угол медлу осью дрона и линией визинования
        5   V      - модуль скорости дрона
        6   omega  - угловая скорость дрона
        7   V_trg  - модуль скорости, финального положениея
        8   gamma  - угол между финальной скоростью и линией визирования

        # 9   D_missile - длина линии визирования до преследующей ракеты
        # 10  V_missile - модуль вектора скорости ракеты
        # 11  gamma_missile - угол между скоростью ракеты и линией визирования
        # 12  phi_missile    - угол медлу осью дрона и линией визинования до ракеты
        

        :param state: state np.ndarray, defaults to None
        :type state: [type], optional
        """
        state0 = None
        if state:
            state0 = self.get_state()
            self.set_state(state)
        
        # drone_dict = self.drone.to_dict()
        H = self.drone.pos.y
        alpha = self.drone.alpha
        D_vec = self.pos_trg - self.drone.pos
        D = D_vec.len()
        V = self.drone.vel.len()
        
        beta_v = Vec2(1,0).angle_to(self.drone.vel.rotate(-alpha))
        phi =  Vec2(1,0).angle_to(D_vec.rotate(-alpha))
        omega = self.drone.omega

        V_trg = self.vel_trg.len()
        gamma = D_vec.angle_to(self.vel_trg)

        D_missile_vec = self.drone.pos - self.missile.pos
        D_missile = D_missile_vec.len()

        V_missile_vec = self.missile.vel
        V_missile = self.missile.vel_len

        gamma_missile = V_missile_vec.angle_to(D_missile_vec)

        phi_missile = Vec2(1,0).angle_to(D_missile_vec.rotate(-alpha))

        if state0:
            self.set_state(state0)
        
        res = np.array([H, alpha, beta_v, D, phi, V, omega, V_trg, gamma, D_missile, V_missile, gamma_missile, phi_missile])
        if normalize:
            res = (res - self.obs_min) / (self.obs_max - self.obs_min)
        return res


    def get_delta_t(self):
        try:
            delta_t = self.drone.get_delta_t_minimum(self.pos_trg, self.vel_trg, self.vel_max, self.a_max, 1e-4)

            return delta_t
        except Exception as e:
            print(f'При подсчете get_delta_t была ошибка {e}')
            return 999


    def is_done(self):
        drone_pos = self.drone.pos
        r = drone_pos - self.pos_trg
        if r.len() < self.trg_radius:
            if r * self.vel_trg > 0:
                pos_diff = r.len()/self.trg_radius   # 0..1
                drone_vel = self.drone.vel
                dir_diff = -(drone_vel.norm() * self.vel_trg.norm() -1)  # 0 .. 2                
                vel_diff = (self.vel_trg - drone_vel).len() / self.vel_trg.len() # 0 .. 2+

                r_pos = 5 * (1-pos_diff)
                r_dir = 5 * (2-dir_diff)
                r_vel = 2 * (3 - vel_diff)
                final_reward = r_pos + r_dir + r_vel
                return True, final_reward, {
                    'result': 'success', 
                    'final_reward': final_reward, 
                    'pos_diff': pos_diff,
                    'dir_diff': dir_diff,
                    'vel_diff': vel_diff
                    }
        if self.antiflip and abs(self.drone.alpha) > np.pi * 0.75:
            return True, -20, {'result': 'flip over'}
        if drone_pos.y < self.xy_bounds[1][0] or drone_pos.y > self.xy_bounds[1][1]:
            return True, -20, {'result': 'out of Y bounds'}
        if drone_pos.x < self.xy_bounds[0][0] or drone_pos.x > self.xy_bounds[0][1]:
            return True, -20, {'result': 'out of X bounds'}
        if abs(self.drone.omega) > self.omega_max:
            return True, -20, {'result': f'omega too mutch {self.drone.omega}'}

        mr = drone_pos - self.missile.pos
        if mr.len() < self.trg_radius:
            return True, -30, {'result': f'missie win'}
        
        return False, 0, {}


    def step(self, actions):
        """observation_, reward, done, info



        :param action: [description]
        :type action: [type]
        """
        delta_t0 = self.get_delta_t()
        F1 = actions[0]
        F2 = actions[1]
        F1 = 1 if F1 > 1 else 0 if F1 < 0 else F1
        F2 = 1 if F2 > 1 else 0 if F2 < 0 else F2
        self.missile.step(self.drone.pos, self.drone.vel, self.tau, self.n_4_step)
        self.drone.step(F1, F2, self.tau, self.n_4_step)
        self.time_curr = self.drone.t
        delta_t1 = self.get_delta_t()

        reward = (delta_t0 - delta_t1)
        
        if reward > 7:
            reward = 7
        if reward < -7:
            reward = -7
        observation_ = self.get_observ()
        done, add_reward, info = self.is_done()
        reward += add_reward
        if self.record_history:
            self.history.append({
                'state': self.get_state(),
                'actions': np.array(actions),
                'delta_t1': delta_t1,
                'observation_': observation_,
                'reward': reward,
                'done': done,
                'info': info
            })

        return observation_, reward, done, info

    def plot(self, ax, state=None, drone_mashtb=None, vec_mashtb=None,
            actions=None, ideal_traj=False, missile_vec_mshtb=None, missile_d=False, **kwargs):
        state0 = None
        if state:
            state0 = self.get_state()
            self.set_state(state)

        if drone_mashtb:
            L = self.drone.L * drone_mashtb
            points = np.array([
                (-0.3, 0), (-0.3, -0.1), (-1.7, -0.1), (-1.7, 0.1), (-0.3, 0.1), (0,0),
                (0.3, 0.1),(1.7, 0.1), (1.7, -0.1), (0.3, -0.1), (0.3, 0)
            ]) * L
            pos = self.drone.pos_np

            M_rot = np.asarray(Mat2.from_angle(self.drone.alpha).as_np())
            points =  points @ M_rot+ pos

            drone_kw = {'lw': 1, 'color': 'darkblue'}
            drone_kw = dict(drone_kw, **kwargs.get('drone_kw', {}))
            ax.plot(points[:,0], points[:,1], **drone_kw)

            if actions is not None:
                F1 = np.array([0, L * (actions[0] * 0.5 + 0.5)])
                F1 =  F1 @ M_rot
                F01 =  np.array([-L, 0]) @M_rot + pos
                actions_kw = {'width': 0.05 * L, 'color': 'red'}
                actions_kw = dict(actions_kw, **kwargs.get('actions_kw', {}))
                ax.arrow(F01[0], F01[1], F1[0], F1[1], **actions_kw)

                F2 = np.array([0, L * (actions[1]* 0.5 + 0.5)])
                F2 =  F2 @ M_rot
                F02 = np.array([L, 0]) @ M_rot + pos
                ax.arrow(F02[0], F02[1], F2[0], F2[1], **actions_kw)     

        if vec_mashtb:
            vel = self.drone.vel * vec_mashtb
            pos = self.drone.pos

            vel_kw = {'width': 0.05 * vel.len(), 'color': 'green'}
            vel_kw = dict(vel_kw, **kwargs.get('vel_kw', {}))
            ax.arrow(pos.x, pos.y, vel.x, vel.y, **vel_kw)
        
        if ideal_traj:
            delta_t = self.get_delta_t()
            traject = self.drone.get_traject(delta_t, self.pos_trg, self.vel_trg)

            ideal_traj_kw = {'ls': ':', 'color': 'green'}
            ideal_traj_kw = dict(ideal_traj_kw, **kwargs.get('ideal_traj_kw', {}))
            ax.plot(traject[:,0], traject[:,1], **ideal_traj_kw)

        if missile_vec_mshtb:
            vel = self.missile.vel * vec_mashtb
            pos = self.missile.pos

            vel_kw = {'width': 0.05 * vel.len(), 'color': 'red'}
            vel_kw = dict(vel_kw, **kwargs.get('mis_vel_kw', {}))
            ax.arrow(pos.x, pos.y, vel.x, vel.y, **vel_kw)

        if missile_d:
            pos_drone = self.drone.pos
            mis_pos = self.missile.pos
            ideal_traj_kw = {'ls': '--', 'color': 'red'}
            ideal_traj_kw = dict(ideal_traj_kw, **kwargs.get('missile_d_kw', {}))
            ax.plot((pos_drone.x, mis_pos.x), (pos_drone.y, mis_pos.y), **ideal_traj_kw)

        if state0:
            self.set_state(state0)
        
