import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from torch.utils import data
from missile_gym import MissileGym
from gymtree import GymTree
from tqdm import tqdm
from torch.utils.data import DataLoader
from gymtree import Node, GymTree

class DatasetAllScenarios(data.Dataset):
    trees = {}

    @classmethod
    def load(cls):
        state_memory = []
        new_state_memory = []
        action_mem = []
        reward_memory = []
        terminal_memory = []

        saves_dir = os.path.join(os.path.dirname(__file__), 'saves')
        for scenario_name in MissileGym.scenario_names:
            tree_file = os.path.join(saves_dir, f'{scenario_name}.bin')
            if os.path.isfile(tree_file):
                gym = MissileGym.make(scenario_name)
                tree = GymTree(gym)
                tree.reset()
                print(f'Loading {tree_file}...')
                tree.load_from_file(tree_file)
                cls.trees[scenario_name] = {
                    'tree': tree,
                    'file': tree_file
                }
                for state, action, reward, state_, done in tree.transaction_iter():
                    state_memory.append(state)
                    new_state_memory.append(state_)
                    action_mem.append(action)
                    reward_memory.append(reward)
                    terminal_memory.append(done)
        
        state_memory = np.array(state_memory, dtype=np.float32)
        new_state_memory = np.array(new_state_memory, dtype=np.float32)
        action_mem = np.array(action_mem, dtype=np.int64)
        reward_memory = np.array(reward_memory, dtype=np.float32)
        terminal_memory = np.array(terminal_memory, dtype=np.bool)

        return cls(state_memory, new_state_memory, action_mem, 
                   reward_memory, terminal_memory)

    @classmethod
    def save(cls):
        for d in cls.trees.values():
            d['tree'].save_to_file(d['file'])
               
    def __init__(self, 
                state_memory,
                new_state_memory,
                action_mem,
                reward_memory,
                terminal_memory):

        self.state_memory = T.tensor(state_memory)
        self.new_state_memory = T.tensor(new_state_memory)
        self.action_mem = T.tensor(action_mem, dtype=T.long)
        self.reward_memory = T.tensor(reward_memory)
        self.terminal_memory = T.tensor(terminal_memory)
        print('Dataset loaded!')

    def __getitem__(self, index):
        return self.state_memory[index], self.action_mem[index], self.reward_memory[index], self.new_state_memory[index], self.terminal_memory[index]

    def __len__(self):
        return len(self.reward_memory)

class NeuroMissileNet2(nn.Module):
    GPU_ACTIVATE = True

    @classmethod
    def from_dict(cls, d):
        net = cls(**d['constructor'])
        if 'state_dict' in d:
            net.load_state_dict(d['state_dict'])
        if 'optimizer_dict' in d:
            net.optimizer.load_state_dict(d['optimizer_dict'])
        return net

    def __init__(self, lr, 
                n_actions,  
                state_env_shape,
                fc1_n,
                fc2_n,
                fc3_n,
                fc4_n):
        super().__init__()
        self._lr = lr
        self.n_actions = n_actions
        self.state_env_shape = state_env_shape
        self.fc1_n = fc1_n
        self.fc2_n = fc2_n
        self.fc3_n = fc3_n
        self.fc4_n = fc4_n

        self.fc1 = nn.Linear(state_env_shape, fc1_n)
        self.fc2 = nn.Linear(fc1_n, fc2_n)
        self.fc3 = nn.Linear(fc2_n, fc3_n)
        self.fc4 = nn.Linear(fc3_n, fc4_n)
        self.fc_act = nn.Linear(fc4_n, n_actions)
        self.fc_val = nn.Linear(fc4_n, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=self._lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() and self.GPU_ACTIVATE else 'cpu')
        self.to(self.device)
        

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, value):
        for g in self.optimizer.param_groups:
            g['lr'] = value
        self._lr = value

    def forward(self, x):
        x = F.relu(self.fc1(x.to(self.device)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        act = self.fc_act(x)
        val = self.fc_val(x)
        return val, act
    
    def to_dict(self):
        return {
            'constructor': dict(
                lr=self.lr, 
                n_actions=self.n_actions,  
                state_env_shape=self.state_env_shape,
                fc1_n=self.fc1_n,
                fc2_n=self.fc2_n,
                fc3_n=self.fc3_n,
                fc4_n=self.fc4_n),
            'state_dict': self.state_dict(),
            'optimizer_dict': self.optimizer.state_dict() }

class Agent2(object):
    @classmethod
    def from_dict(cls, d):
        q_eval = NeuroMissileNet2.from_dict(d['q_eval_dict'])
        q_next = NeuroMissileNet2.from_dict(d.get('q_next_dict', d['q_eval_dict']))

        return cls(gamma=d['gamma'], replace=d['replace'], q_eval=q_eval, q_next=q_next)

    def __init__(self, gamma, replace, q_eval, q_next):
        self.gamma = gamma
        self.replace_target_cnt = replace
        self.learn_step_counter = 0

        self.q_eval = q_eval # NeuroMissileNet2(self._lr, self.n_actions, self.state_env_shape[0], fc1_n_eval, fc2_n_eval, fc3_n_eval)
        self.q_next = q_next # NeuroMissileNet2(self._lr, self.n_actions, self.state_env_shape[0], fc1_n_next, fc2_n_next, fc3_n_next)
        self.replace_target_network(forced=True)

    def to_dict(self):
        return {
            'gamma': self.gamma,
            'replace': self.replace_target_cnt,
            'q_eval_dict': self.q_eval.to_dict(),
            'q_next_dict': self.q_next.to_dict() }

    @property
    def lr(self):
        return self.q_eval.lr

    @lr.setter
    def lr(self, value):
        self.q_eval.lr = value
        self.q_next.lr = value

    def choose_action(self, observation):
        observation = T.tensor(observation, dtype=T.float32)
        _, advantage = self.q_eval.forward(observation)
        action = T.argmax(advantage).item()
        action_space = [-1, 0, 1]
        action = action_space[action]
        return action

    def replace_target_network(self, forced=False):
        if self.learn_step_counter % self.replace_target_cnt == 0 or forced:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def learn(self, states, actions, rewards, states_, dones, return_errors=True):
        self.q_eval.optimizer.zero_grad()
        self.replace_target_network() 
        batch_size = len(dones)

        actions = actions.to(self.q_eval.device)
        rewards = rewards.to(self.q_eval.device)
        dones = dones.to(self.q_eval.device)

        indeces = T.arange(batch_size, dtype=T.long)

        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)

        V_s_eval, A_s_eval = self.q_eval.forward(states_)

        q_pred = T.add(V_s, 
                       (A_s - A_s.mean(dim=1, keepdim=True)))[indeces, actions]
        q_next = T.add(V_s_,
                       (A_s_ - A_s_.mean(dim=1, keepdim=True)))
        q_eval = T.add(V_s_eval,
                       (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))

        max_actions = T.argmax(q_eval, dim=1)

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next[indeces, max_actions]

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        if return_errors:
            errors = ((q_target-q_pred)**2).cpu().detach().numpy()
            return errors


class MissileCoach:
    __dataset = None
    num_workers = 4
    
    @classmethod
    def get_dataset(cls):
        if cls.__dataset is None:
            print('loading dataset')
            cls.__dataset = DatasetAllScenarios.load()
        return cls.__dataset

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    @classmethod
    def create_new(cls, fc1_n, fc2_n, fc3_n, fc4_n, gamma, replace, batch_size, lr_min, lr_max, lr_epoch_cycle):
        agent_dict = {
            'gamma': gamma,
            'replace': replace,
            'q_eval_dict': {
                'constructor': {
                    'lr': lr_max,
                    'n_actions': 3,
                    'state_env_shape': cls.get_dataset()[0][0].shape[0], # 10
                    'fc1_n': fc1_n,
                    'fc2_n': fc2_n,
                    'fc3_n': fc2_n,
                    'fc4_n': fc3_n 
                }
            }
        }
        agent = Agent2.from_dict(agent_dict)
        agent_dict = agent.to_dict()
        return cls(agent_dict, batch_size, lr_min, lr_max, lr_epoch_cycle, [], agent_dict, 0)
        

    def __init__(self, agent_dict, batch_size, lr_min, lr_max, lr_epoch_cycle,
                 history, best_agent_dict, epoch):
        self.agent: Agent2 = Agent2.from_dict(agent_dict)
        self.batch_size = batch_size
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_epoch_cycle = lr_epoch_cycle
        self.dataloader = DataLoader(self.get_dataset(), batch_size=batch_size, 
            shuffle=True, pin_memory=True, num_workers=self.num_workers)
        self.history = history
        self.best_agent_dict = best_agent_dict
        self.epoch = epoch

    def calc_lr(self):
        t = (self.epoch % self.lr_epoch_cycle) / self.lr_epoch_cycle
        self.agent.lr = (1-t) * self.lr_max + t * self.lr_min

    def make_traect(self, node: Node):
        ts = []
        alphas = []
        xs_missile = []
        ys_missile = []
        xs_target = []
        ys_target = []
        while not node.done:
            node.make_current()
            ts.append(node.owner.gym.missile.t)
            alphas.append(node.owner.gym.missile.alpha)
            x, y = node.owner.gym.missile.pos
            xs_missile.append(x)
            ys_missile.append(y)
            x, y = node.owner.gym.target.pos
            xs_target.append(x)
            ys_target.append(y)

            action = self.agent.choose_action(node.obs)
            node = node.produce_node(action)
        is_hit = node.reward > 600
        return is_hit, np.asarray(ts), np.asarray(alphas), np.asarray(xs_missile), \
                np.asarray(ys_missile), np.asarray(xs_target), np.asarray(ys_target)  
    
    def make_epoch(self, tqdm_=True):
        self.calc_lr()
        iterator = tqdm(self.dataloader) if tqdm_ else self.dataloader
        err_sum = 0
        for states, actions, rewards, states_, dones in iterator:
            errors = self.agent.learn(states, actions, rewards, states_, dones, return_errors=True)
            err_sum += np.sum(errors)
        error = err_sum / len(self.get_dataset())
        self.history.append(
            {
                'epoch': self.epoch,
                'error': error,
                'lr': self.agent.lr
            })
        self.epoch += 1

    def to_dict(self):
        return {
            'agent_dict': self.agent.to_dict(),
            'batch_size': self.batch_size,
            'lr_min': self.lr_min,
            'lr_max': self.lr_max,
            'lr_epoch_cycle': self.lr_epoch_cycle,
            'history': self.history,
            'best_agent_dict': self.best_agent_dict,
            'epoch': self.epoch }


def main2():
    mc = MissileCoach()
    print(mc.get_dataset())
    print(mc.get_dataset())
    mc2 = MissileCoach()
    print(mc2.get_dataset())
    

def main():
    
    from tqdm import tqdm

    dataset = DatasetAllScenarios.train_test(-1)

    d = {
        'gamma': 0.9,
        'replace': 500,
        'q_eval_dict': {
            'constructor': {
                'lr': 0.00025,
                'n_actions': 3,
                'state_env_shape': dataset[0][0].shape[0], # 10
                'fc1_n': 64,
                'fc2_n': 128,
                'fc3_n': 128,
                'fc4_n': 64 
            }
        }
    }
    agent = Agent2.from_dict(d)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, pin_memory=True, num_workers=4)

    for states, actions, rewards, states_, dones in tqdm(dataloader):
        agent.learn(states, actions, rewards, states_, dones)
        


if __name__ == "__main__":
    main2()

