import numpy as np
from cydrone.drone import Drone2d
from easyvec import Vec2, Mat2
from scipy.optimize import minimize, Bounds, minimize_scalar
from gym.spaces.box import Box




class DroneGym:
    @classmethod
    def make(cls, name):
        def drone_foo(**kwargs):
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

        def pos0_pos_trg_foo(**kwargs):
            pos0 = kwargs.get('pos0',       Vec2.random((-4000, 0),(4000, 2500)))
            pos_trg = kwargs.get('pos_trg', Vec2.random((-4000, 1000),(4000, 1500)))
            return pos0, pos_trg


        return cls(
            drone_foo=drone_foo, 
            pos0_pos_trg_foo=pos0_pos_trg_foo,
            vel_trg_len=1.0,
            xy_bounds = ((-5000, 5000), (-10, 3000))
        )

    def __init__(self, drone_foo, pos0_pos_trg_foo, vel_trg_len: float, **kwargs):
        self.time_curr = 0.0
        self.drone = None 
        self.drone_foo = drone_foo

        self.pos_trg = None
        self.pos0_pos_trg_foo = pos0_pos_trg_foo

        self.vel_trg_len = vel_trg_len

        self.tau = kwargs.get('tau', 0.25)
        self.n_4_step = kwargs.get('n_4_step', 10)
        self.vel_max = kwargs.get('vel_max', 33.0)
        self.omega_max = kwargs.get('omega_max', 23.0)
        self.a_trg_max = kwargs.get('a_trg_max', 3.0)
        self.trg_radius =  kwargs.get('trg_radius', 5.7)
        self.xy_bounds = kwargs.get('xy_bounds', ((-5000, 5000), (-10, 3000)))
        self.obs_min = np.array(kwargs.get('obs_min', 
            (self.xy_bounds[1][0], -np.pi, -np.pi, 0, -np.pi, 0, -self.omega_max) ))
        self.obs_max = np.array(kwargs.get('obs_max', 
            (self.xy_bounds[1][1], np.pi, np.pi, self.xy_bounds[0][1] - self.xy_bounds[0][0], np.pi, self.vel_max, self.omega_max) ))
        self.history = []
        self.record_history = False

        self.action_space = Box(0,1, shape=(2,))
        self.antiflip = kwargs.get('antiflip', True)


    def set_state(self, state):
        self.drone.from_numpy(state)
        self.time_curr = self.drone.t

    def reset(self, **kwargs):
        self.drone = self.drone_foo(**kwargs)
        self.pos0, self.pos_trg = self.pos0_pos_trg_foo(**kwargs)

        self.pos0 = Vec2.from_list(self.pos0)
        self.pos_trg = Vec2.from_list(self.pos_trg)

        d = self.drone.to_dict()
        d['pos'] = self.pos0
        self.drone.from_dict(d)

        self.history = []
        return self.get_observ()

    def get_state(self):
        return self.drone.to_numpy()

    def get_observ(self, state=None, normalize=True):
        """np.ndarray
        [0, 1,     2,      3, 4, 5, 6]
        [H, alpha, beta_v, D, phi, V, omega]

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

        if state0:
            self.set_state(state0)
        
        res = np.array([H, alpha, beta_v, D, phi, V, omega])
        if normalize:
            res = (res - self.obs_min) / (self.obs_max - self.obs_min)
        return res

    def get_delta_t_alpha(self):
        try:
            D = self.pos_trg-self.drone.pos
            some = (self.drone.vel.len() + self.drone.vel * D.norm()) / 2 + self.vel_trg_len

            delta_t = (self.pos_trg-self.drone.pos).len() / some

            def minim_t_alpha(alpha_trg):
                vel_trg = Vec2(self.vel_trg_len, 0).rotate(alpha_trg)
                vmin ,vmax, amax = self.drone.get_vmin_vmax_amax(delta_t, self.pos_trg, vel_trg)
                shtraf = 0
                if vmax > self.vel_max:
                    shtraf += 10000
                if amax > self.a_trg_max:
                    shtraf += 10000
                return amax + shtraf
            
            res = minimize_scalar(minim_t_alpha, bounds=(-np.pi, np.pi), method='bounded')

            alpha = res.x
            vel_trg = Vec2(self.vel_trg_len, 0).rotate(alpha)

            def minim_t(t):
                vmin ,vmax, amax = self.drone.get_vmin_vmax_amax(t, self.pos_trg, vel_trg)
                shtraf = 0
                if vmax > self.vel_max:
                    shtraf += 10000
                if amax > self.a_trg_max:
                    shtraf += 10000
                return t + shtraf
            
            res = minimize_scalar(minim_t, bounds=(0.01, 3000), method='bounded')
            delta_t = res.x
            # def minim_foo(x, drone, pos_trg, vel_trg_len, a_max, v_max):
            #     delta_t, alpha_trg = denorm(x, drone, pos_trg, vel_trg_len, a_max, v_max)
            #     if delta_t < 1e-6:
            #         return np.nan
            #     vA = drone.vel
            #     vel_trg = Vec2(vel_trg_len, 0).rotate(alpha_trg)
            #     vmin ,vmax, amax = drone.get_vmin_vmax_amax(delta_t, pos_trg, vel_trg)
            #     shtraf = 0
            #     vminAD = min(vA.len(), vel_trg_len) * 0.5
            #     vmaxAD = max(vA.len(), vel_trg_len, v_max) 
            #     if vmin < vminAD:
            #         shtraf += 300
            #         return np.nan
            #     if vmax > vmaxAD:
            #         shtraf += 300
            #         return np.nan
            #     if amax > a_max:
            #         shtraf += 300
            #         return np.nan
            #     return delta_t
            
            # initial_simplex = []
            # while len(initial_simplex)<3:
            #     x = np.random.uniform((0.1,-np.pi),(10,np.pi))
            #     if not np.isnan(minim_foo(x, self.drone, self.pos_trg, self.vel_trg_len, self.a_trg_max, self.vel_max)):
            #         break
            #     initial_simplex.append(x)

            # res = minimize(
            #     minim_foo, 
            #     (1,0), 
            #     args=(self.drone, self.pos_trg, self.vel_trg_len, self.a_trg_max, self.vel_max),
            #     method='Nelder-Mead', 
            #     options={
            #         'maxfev': 200,
            #         'initial_simplex': initial_simplex
            #         }
            #     )

            # delta_t, alpha = denorm(res.x, self.drone, self.pos_trg, self.vel_trg_len, self.a_trg_max, self.vel_max)
            return delta_t, alpha
        except Exception as e:
            print(f'При подсчете get_delta_t была ошибка {e}')
            return 999, 0


    def is_done(self):
        drone_pos = self.drone.pos
        if (drone_pos - self.pos_trg).len() < self.trg_radius:
            if self.drone.vel.len() <= self.vel_trg_len:
                return True, 100, {'result': 'success'}
        alpha = self.drone.alpha
        if self.antiflip and abs(alpha) > np.pi * 0.75:
            return True, -100, {'result': 'flip over'}
        if drone_pos.y < self.xy_bounds[1][0] or drone_pos.y > self.xy_bounds[1][1]:
            return True, -100, {'result': 'out of Y bounds'}
        if drone_pos.x < self.xy_bounds[0][0] or drone_pos.x > self.xy_bounds[0][1]:
            return True, -100, {'result': 'out of X bounds'}
        if abs(self.drone.omega) > self.omega_max:
            return True, -100, {'result': f'omega too mutch {self.drone.omega}'}
        return False, 0, {}


    def step(self, actions):
        """observation_, reward, done, info



        :param action: [description]
        :type action: [type]
        """
        delta_t0, alpha = self.get_delta_t_alpha()
        F1 = actions[0]
        F2 = actions[1]
        F1 = 1 if F1 > 1 else 0 if F1 < 0 else F1
        F2 = 1 if F2 > 1 else 0 if F2 < 0 else F2
        self.drone.step(F1, F2, self.tau, self.n_4_step)
        self.time_curr = self.drone.t
        delta_t1, alpha = self.get_delta_t_alpha()

        reward = delta_t0 - delta_t1
        if reward > 7:
            reward = 1
        if reward < -7:
            reward = -1
        observation_ = self.get_observ()
        done, add_reward, info = self.is_done()
        reward += add_reward
        if self.record_history:
            self.history.append({
                'state': self.get_state(),
                'actions': np.array(actions),
                'delta_t1': delta_t1,
                'alpha': alpha,
                'observation_': observation_,
                'reward': reward,
                'done': done,
                'info': info
            })

        return observation_, reward, done, info

    def plot(self, ax, state=None, drone_mashtb=None, vec_mashtb=None,
            actions=None, ideal_traj=False, **kwargs):
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
            delta_t, alpha = self.get_delta_t_alpha()
            vel_trg = Vec2(self.vel_trg_len, 0).rotate(alpha)
            traject = self.drone.get_traject(delta_t, self.pos_trg, vel_trg)

            ideal_traj_kw = {'ls': ':', 'color': 'green'}
            ideal_traj_kw = dict(ideal_traj_kw, **kwargs.get('ideal_traj_kw', {}))
            ax.plot(traject[:,0], traject[:,1], **ideal_traj_kw)

        if state0:
            self.set_state(state0)
        
        
if __name__ == '__main__':
    gym = DroneGym.make('ha')
    a = gym.reset()
    a1 = gym.step([1,1])
    a2 = gym.step([1,0.5])
    a3 = gym.step([1,0.5])
    for i in range(100):
        a4 = gym.step([1,1])

 

        
        


        


    

    


    