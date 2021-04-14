import numpy as np
from cydrone.drone import Drone2d
from easyvec import Vec2, Mat2
from scipy.optimize import minimize




class DroneGym:
    @classmethod
    def make(cls, name):
        drone = Drone2d.get_DJI()
        return cls(
            drone=drone, 
            pos0=Vec2(-2000, 20), 
            pos_trg=Vec2(2000,1500), 
            vel_trg_len=1.0
        )

    def __init__(self, drone: Drone2d, pos0: Vec2, pos_trg: Vec2, vel_trg_len: float, **kwargs):
        self.time_curr = 0.0
        self.drone = drone
        self.drone.from_dict(dict(
            t=self.time_curr,
            pos = pos0,
            vel=(0,0),
            alpha=0,
            omega=0
        ))
        self.pos_trg = pos_trg.copy()
        self.vel_trg_len = vel_trg_len
        self.state0 = self.drone.to_numpy()

        self.tau = kwargs.get('tau', 0.25)
        self.n_4_step = kwargs.get('n_4_step', 10)
        self.vel_max = kwargs.get('vel_max', 33.0)
        self.omega_max = kwargs.get('omega_max', 23.0)
        self.a_trg_max = kwargs.get('a_trg_max', 13.0)
        self.trg_radius =  kwargs.get('trg_radius', 0.7)
        self.xy_bounds = kwargs.get('xy_bounds', ((-5000, 5000), (-10, 3000)))
        self.obs_min = np.array(kwargs.get('obs_max', 
            (self.xy_bounds[1][0], -np.pi, -np.pi, 0, -np.pi, 0, -self.omega_max) ))
        self.obs_max = np.array(kwargs.get('obs_max', 
            (self.xy_bounds[1][1], np.pi, np.pi, self.xy_bounds[0][1] - self.xy_bounds[0][0], np.pi, self.vel_max, self.omega_max) ))
        self.history = []
        self.record_history = False


    def set_state(self, state):
        self.drone.from_numpy(state)
        self.time_curr = self.drone.t

    def reset(self):
        self.set_state(self.state0)
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
            def denorm(x, drone, pos_trg, vel_trg_len, a_max):
                delta_t, alpha_trg = x
                vA = drone.vel
                t1 = (pos_trg-drone.pos).len() / (0.5 * (vel_trg_len + vA.len()))
                delta_t = delta_t * t1
                return delta_t, alpha_trg

            def minim_foo(x, drone, pos_trg, vel_trg_len, a_max, v_max):
                delta_t, alpha_trg = denorm(x, drone, pos_trg, vel_trg_len, a_max)
                if delta_t < 1e-6:
                    return 1e10
                vA = drone.vel
                vel_trg = Vec2(vel_trg_len, 0).rotate(alpha_trg)
                vmin ,vmax, amax = drone.get_vmin_vmax_amax(delta_t, pos_trg, vel_trg)
                shtraf = 0
                vminAD = min(vA.len(), vel_trg_len) * 0.5
                vmaxAD = max(vA.len(), vel_trg_len, v_max) 
                if vmin < vminAD:
                    shtraf += 10
                if vmax > vmaxAD:
                    shtraf += 10
                if amax > a_max:
                    shtraf += 30
                return delta_t * (shtraf + 1) + shtraf
            
            res = minimize(
                minim_foo, 
                (1,0), 
                args=(self.drone, self.pos_trg, self.vel_trg_len, self.a_trg_max, self.vel_max),
                method='Nelder-Mead', 
                options={
                    'maxfev': 100,
                    'initial_simplex': np.array([[1,0], [2.5, -1], [0.5, 1]])
                    }
                )
            delta_t, alpha = denorm(res.x, self.drone, self.pos_trg, self.vel_trg_len, self.a_trg_max)
            return delta_t, alpha
        except Exception as e:
            print(f'При подсчете get_delta_t была ошибка {e}')
            return 999, 0


    def is_done(self):
        drone_pos = self.drone.pos
        if (drone_pos - self.pos_trg).len() < self.trg_radius:
            if self.drone.vel.len() <= self.vel_trg_len:
                return True, 100, {'result': 'success'}
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
                F1 = np.array([0, L * actions[0]])
                F1 =  F1 @ M_rot
                F01 =  np.array([-L, 0]) @M_rot + pos
                actions_kw = {'width': 0.05 * L, 'color': 'red'}
                actions_kw = dict(actions_kw, **kwargs.get('actions_kw', {}))
                ax.arrow(F01[0], F01[1], F1[0], F1[1], **actions_kw)

                F2 = np.array([0, L * actions[1]])
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

 

        
        


        


    

    


    