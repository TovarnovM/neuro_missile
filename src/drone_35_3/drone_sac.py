import os
import torch as T
import torch.nn.functional as F
import numpy as np
from sac_networks3layers import ActorNetwork3layers, CriticNetwork3layers, ValueNetwork3layers




class AgentParallel3layers:
    def __init__(self, input_dims, n_actions, action_space_high, **kwargs):
        self.alpha = kwargs.get('alpha', 0.0003)
        self.beta = kwargs.get('beta', 0.0003)   
        self.tau = kwargs.get('tau', 0.005)   
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.action_space_high = action_space_high
        self.reward_scale=kwargs.get('reward_scale', 10)
        self.gamma = kwargs.get('gamma', 0.99)

        self.actor_fc1_dims = kwargs.get('actor_fc1_dims', 512)
        self.actor_fc2_dims = kwargs.get('actor_fc2_dims', 256)
        self.actor_fc3_dims = kwargs.get('actor_fc3_dims', 256)
        
        self.critic_fc1_dims = kwargs.get('critic_fc1_dims', 512)
        self.critic_fc2_dims = kwargs.get('critic_fc2_dims', 256)
        self.critic_fc3_dims = kwargs.get('critic_fc3_dims', 256)

        self.value_fc1_dims = kwargs.get('value_fc1_dims', 512)
        self.value_fc2_dims = kwargs.get('value_fc2_dims', 256)
        self.value_fc3_dims = kwargs.get('value_fc3_dims', 256)

        self.device_name = kwargs.get('device_name', 'cuda:0' if T.cuda.is_available() else 'cpu')

        self.actor = ActorNetwork3layers(self.alpha, self.input_dims, n_actions=self.n_actions,
                    name='actor', max_action=self.action_space_high,
                    fc1_dims=self.actor_fc1_dims, fc2_dims=self.actor_fc2_dims, fc3_dims=self.actor_fc3_dims,
                    device_name=self.device_name)
        
        self.critic_1 = CriticNetwork3layers(self.beta, self.input_dims, n_actions=self.n_actions,
                    name='critic_1', fc1_dims=self.critic_fc1_dims, fc2_dims=self.critic_fc2_dims, 
                    fc3_dims=self.critic_fc3_dims, device_name=self.device_name)
        self.critic_2 = CriticNetwork3layers(self.beta, self.input_dims, n_actions=self.n_actions,
                    name='critic_2', fc1_dims=self.critic_fc2_dims, fc2_dims=self.critic_fc2_dims, 
                    fc3_dims=self.critic_fc3_dims,device_name=self.device_name)
       
        self.value = ValueNetwork3layers(self.beta, self.input_dims, name='value', 
                    fc1_dims=self.value_fc1_dims, fc2_dims=self.value_fc2_dims, 
                    fc3_dims=self.value_fc3_dims, device_name=self.device_name)
        self.target_value = ValueNetwork3layers(self.beta, self.input_dims, name='target_value',
                    fc1_dims=self.value_fc1_dims, fc2_dims=self.value_fc2_dims, 
                    fc3_dims=self.value_fc3_dims, device_name=self.device_name)
        
        self.update_network_parameters(tau=1)


    def change_lr(self, alpha, beta):
        self.actor.change_lr(alpha)

        self.critic_1.change_lr(beta)
        self.critic_2.change_lr(beta)

        self.value.change_lr(beta)
        self.target_value.change_lr(beta)


    def choose_action(self, observation):
        state = T.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]

    def choose_action_determ(self, observation):
        state = T.Tensor([observation]).to(self.actor.device)
        mu, sigma = self.actor.forward(state)
        return mu.cpu().detach().numpy()[0]

    
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                    (1-tau)*target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    

    def learn(self, state, action, reward, new_state, done):
        state_ = new_state
        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        
        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.reward_scale*reward + self.gamma*value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()
        return np.array([
            value_loss.cpu().detach().numpy(), 
            actor_loss.cpu().detach().numpy(), 
            critic_loss.cpu().detach().numpy() ])


    def to_dict(self):
        res = {
            'actor': self.actor.state_dict(),
            'value': self.value.state_dict(),
            'target_value': self.target_value.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict()
        }
        for ss in res:
            res[ss] = {k: v.cpu() for k, v in res[ss].items()}
        return res

    def from_dict(self, d):
        self.actor.load_state_dict(d['actor'],)
        self.value.load_state_dict(d['value'])
        self.target_value.load_state_dict(d['target_value'])
        self.critic_1.load_state_dict(d['critic_1'])
        self.critic_2.load_state_dict(d['critic_2'])



class BufferParallel:
    def __init__(self, agent_paralell):
        self.device = agent_paralell.actor.device
        self.state_memory_list = []
        self.new_state_memory_list = []
        self.action_memory_list = []
        self.reward_memory_list = []
        self.terminal_memory_list = []

        self.state_memory_torch = None
        self.new_state_memory_torch = None
        self.action_memory_torch = None
        self.reward_memory_torch = None
        self.terminal_memory_torch = None

        self.rnd_inds_torch = None

        self.mem_cntr = 0
        self.n = 0

    def store_transition(self, state, action, reward, state_, done):
        self.state_memory_list.append(state)
        self.new_state_memory_list.append(state_)
        self.action_memory_list.append(action)
        self.reward_memory_list.append(reward)
        self.terminal_memory_list.append(done)

    def get_list_len(self):
        return len(self.state_memory_list)

    def trim_oldest(self, n_remain, refresh=True):
        if self.get_list_len() > n_remain:
            self.state_memory_list = list(self.state_memory_list[-n_remain:])
            self.new_state_memory_list = list(self.new_state_memory_list[-n_remain:])
            self.action_memory_list = list(self.action_memory_list[-n_remain:])
            self.reward_memory_list = list(self.reward_memory_list[-n_remain:])
            self.terminal_memory_list = list(self.terminal_memory_list[-n_remain:])
        if refresh:
            self.refresh()

    def refresh(self, permut=True):
        self.n = len(self.state_memory_list)
        self.mem_cntr = 0

        self.state_memory_torch = T.tensor(self.state_memory_list, dtype=T.float).to(self.device)
        self.new_state_memory_torch = T.tensor(self.new_state_memory_list, dtype=T.float).to(self.device)
        self.action_memory_torch = T.tensor(self.action_memory_list, dtype=T.float).to(self.device)
        self.reward_memory_torch = T.tensor(self.reward_memory_list, dtype=T.float).to(self.device)
        self.terminal_memory_torch = T.tensor(self.terminal_memory_list, dtype=T.bool).to(self.device)

        if permut:
            self.rnd_inds_torch = T.randperm(self.n).to(self.device)
        else:
            self.rnd_inds_torch = T.arange(0, self.n-1, 1).to(self.device)
        self.rnd_inds_torch = T.hstack([self.rnd_inds_torch, self.rnd_inds_torch])

        return self.n
    
    def to_dict(self):
        return dict(
            state_memory_list = self.state_memory_list,
            new_state_memory_list = self.new_state_memory_list,
            action_memory_list = self.action_memory_list,
            reward_memory_list = self.reward_memory_list,
            terminal_memory_list = self.terminal_memory_list       
        )

    def extend_from_dict(self, d):
        self.state_memory_list.extend(d['state_memory_list'])
        self.new_state_memory_list.extend(d['new_state_memory_list'])
        self.action_memory_list.extend(d['action_memory_list'])
        self.reward_memory_list.extend(d['reward_memory_list'])
        self.terminal_memory_list.extend(d['terminal_memory_list'])
    

    def get_batches(self, batchsize):
        i2 = self.mem_cntr + batchsize
        inds = self.rnd_inds_torch[self.mem_cntr:i2]
        self.mem_cntr += batchsize
        if self.mem_cntr >= (self.n-1):
            self.mem_cntr = 0

        return self.state_memory_torch[inds], \
            self.action_memory_torch[inds], \
            self.reward_memory_torch[inds], \
            self.new_state_memory_torch[inds], \
            self.terminal_memory_torch[inds]
    
    def iter_batches(self, batchsize):
        self.mem_cntr = 0
        yield self.get_batches(batchsize)
        while self.mem_cntr > 0:
            yield self.get_batches(batchsize)


