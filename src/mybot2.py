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

class DatasetAllScenarios(data.Dataset):
    @classmethod
    def train_test(cls, test_portion=-1):
        state_memory = []
        new_state_memory = []
        action_mem = []
        reward_memory = []
        terminal_memory = []

        for scenario_name in MissileGym.scenario_names:
            tree_file = f'saves/{scenario_name}.bin'
            if os.path.isfile(tree_file):
                gym = MissileGym.make(scenario_name)
                tree = GymTree(gym)
                tree.reset()
                print(f'Loading {tree_file}')
                tree.load_from_file(tree_file)
                for state, action, reward, state_, done in tqdm(list(tree.transaction_iter())):
                    state_memory.append(state)
                    new_state_memory.append(state_)
                    action_mem.append(action)
                    reward_memory.append(reward)
                    terminal_memory.append(done)
                break
        
        state_memory = np.array(state_memory, dtype=np.float32)
        new_state_memory = np.array(new_state_memory, dtype=np.float32)
        action_mem = np.array(action_mem, dtype=np.int32)
        reward_memory = np.array(reward_memory, dtype=np.float32)
        terminal_memory = np.array(terminal_memory, dtype=np.bool)

        if test_portion < 0:
            return cls(state_memory, new_state_memory, action_mem, 
                   reward_memory, terminal_memory)


        n_all = len(state_memory)
        indeces = np.random.permutation(n_all)
        n_test = int(n_all*test_portion)
        i_test = indeces[:n_test]
        i_train = indeces[n_test:]

        train_set = cls(state_memory[i_train], new_state_memory[i_train], action_mem[i_train], 
                   reward_memory[i_train], terminal_memory[i_train])
        test_set  = cls(state_memory[i_test], new_state_memory[i_test], action_mem[i_test], 
                   reward_memory[i_test], terminal_memory[i_test])

        return train_set, test_set
               
    def __init__(self, 
                state_memory,
                new_state_memory,
                action_mem,
                reward_memory,
                terminal_memory):

        self.state_memory = T.tensor(state_memory)
        self.new_state_memory = T.tensor(new_state_memory)
        self.action_mem = T.tensor(action_mem)
        self.reward_memory = T.tensor(reward_memory)
        self.terminal_memory = T.tensor(terminal_memory)

    def __getitem__(self, index):
        return self.state_memory[index], self.action_mem[index], self.reward_memory[index], self.new_state_memory[index], self.terminal_memory[index]

    def __len__(self):
        return len(self.reward_memory)

class NeuroMissileNet2(nn.Module):
    GPU_ACTIVATE = False

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
                fc3_n):
        super().__init__()
        self._lr = lr
        self.n_actions = n_actions
        self.state_env_shape = state_env_shape
        self.fc1_n = fc1_n
        self.fc2_n = fc2_n
        self.fc3_n = fc3_n

        self.fc1 = nn.Linear(state_env_shape, fc1_n)
        self.fc2 = nn.Linear(fc1_n, fc2_n)
        self.fc3 = nn.Linear(fc2_n, fc3_n)
        self.fc_act = nn.Linear(fc3_n, n_actions)
        self.fc_val = nn.Linear(fc3_n, 1)
        
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
                fc3_n=self.fc3_n),
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
        self.q_eval.lr = lr
        self.q_next.lr = lr

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
    
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
                  else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self, states, actions, rewards, states_, dones):
        self.q_eval.optimizer.zero_grad()
        self.replace_target_network() 
        batch_size = len(dones)

        actions = actions.to(self.q_eval.device)
        rewards = rewards.to(self.q_eval.device)
        dones = dones.to(self.q_eval.device)

        indeces = T.arange(batch_size, dtype=T.int32)
        print(indeces)
        print(actions)

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
        errors = ((q_target-q_pred)**2).cpu().detach().numpy()
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        return errors