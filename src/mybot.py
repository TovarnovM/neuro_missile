import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle


class ReplayBuffer(object):
    def __init__(self, max_size, state_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *state_shape), dtype=np.float64)
        self.new_state_memory = np.zeros_like(self.state_memory)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float64)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.prioritetes = np.zeros(self.mem_size, dtype=np.float64)
        self.max_error_val = 1

        total_size = 0
        for attr, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                total_size += value.nbytes
        total_size /= 1024*1024
        print(f'Memory buffer get {total_size:.1f} MB')


    def store_transaction(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        # self.nextfig_memory[index] = nextfig
        self.new_state_memory[index] = state_
        # self.new_nextfig_memory[index] = nextfig_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.prioritetes[index] = self.max_error_val
        self.mem_cntr += 1


    def sample_buffer(self, batch_size, prioritized=True):
        max_mem = min(self.mem_cntr, self.mem_size)

        if prioritized:
            ps = self.prioritetes[:max_mem] / np.sum(self.prioritetes[:max_mem])
            batch = np.random.choice(max_mem, batch_size, replace=False, p=ps)
        else:
            batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        # nextfigs = self.nextfig_memory[batch]
        state_ = self.new_state_memory[batch]
        # nextfigs_ = self.new_nextfig_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, state_, dones, batch

    def change_priorities(self, indeces, values):
        self.prioritetes[indeces] = values
        self.max_error_val = max(np.max(values), self.max_error_val)

    
    def ready_for_batch(self, batch_size):
        return self.mem_cntr >= batch_size



class NeuroMissileNet(nn.Module):
    GPU_ACTIVATE = True

    def __init__(self, lr, n_actions, name, chkpt_dir, 
                state_env_shape,
                fc1=32,
                fc2=64,
                fc3=32
                ):
        super().__init__()
        self.chkpt_dir = chkpt_dir
        self.name = name
        self.chkpt_file = os.path.join(self.chkpt_dir, name)

        self.state_env_shape = state_env_shape
        
        self.fc1 = nn.Linear(state_env_shape, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, fc3)
        self.fc_act = nn.Linear(fc3, n_actions)
        self.fc_val = nn.Linear(fc3, 1)
        self._lr = lr
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

    def forward(self, x):
        x = T.tensor(x, dtype=T.float32).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        act = self.fc_act(x)
        val = self.fc_val(x)
        return val, act
        

    def save_checkpoint(self):
        if not os.path.exists(self.chkpt_dir):
            print(f'... {self.chkpt_dir} не существует ...')
            os.mkdir(self.chkpt_dir)
            print(f'... {self.chkpt_dir} создана ...')
        print(f'... saving {self.name} ....')
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        print(f'... loading {self.name} ....')
        self.load_state_dict(T.load(self.chkpt_file))
    

class Agent(object):
    def __init__(self, state_env_shape, gamma, epsilon, lr, n_actions, mem_size, batch_size, random_action_foo, eps_min=0.005, eps_dec=5e-8,
                replace=500, chkpt_dir='./models/'):
        self.gamma = gamma
        self.epsilon = epsilon
        self._lr = lr
        self.n_actions = n_actions
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec 
        self.replace_target_cnt = replace
        self.chkpt_dir = chkpt_dir
        self.learn_step_counter = 0
        self.state_env_shape = state_env_shape

        self.q_eval = NeuroMissileNet(self._lr, self.n_actions, 'tetris_bot_eval',  self.chkpt_dir, self.state_env_shape[0])
        self.q_next = NeuroMissileNet(self._lr, self.n_actions, 'tetris_bot_next',  self.chkpt_dir, self.state_env_shape[0])
        self.replace_target_network(forced=True)

        self.memory = ReplayBuffer(self.mem_size, state_env_shape)
        self.random_action_foo = random_action_foo

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            _, advantage = self.q_eval.forward(observation)
            action = T.argmax(advantage).item()
            action_space = [-1, 0, 1]
            action = action_space[action]
        else:
            action = self.random_action_foo()
        return action

    def store_transaction(self, state, action, reward, state_, done):
        self.memory.store_transaction(state, action, reward, state_, done)

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

    def learn(self):
        if not self.memory.ready_for_batch(self.batch_size):
            return
        
        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones, batch_indeces = \
                    self.memory.sample_buffer(self.batch_size)

        actions = T.tensor(actions).to(self.q_eval.device)
        rewards = T.tensor(rewards).to(self.q_eval.device)
        dones = T.tensor(dones).to(self.q_eval.device)

        indeces = np.arange(self.batch_size)
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
        self.memory.change_priorities(batch_indeces, errors)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()




    
# if __name__ == "__main__":
#     tb = TetrisBot(0.001, 12, 'as', './models/')
#     tb.save_checkpoint()