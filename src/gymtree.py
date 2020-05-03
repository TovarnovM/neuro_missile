from uuid import uuid1
import numpy as np
import pickle

class GymTree:
    def __init__(self, gym):
        self.gym = gym
        self.root = None
        self.nodes = {}

    def reset(self):
        self.nodes = {}
        self.root = Node(
            owner=self,
            parent=None,
            gym_state=self.gym.reset())
        self.nodes[self.root.uuid] = self.root 

    def to_dict(self):
        return {
            'root_uuid': None if self.root is None else self.root.uuid,
            'nodes': {
                uuid: node.to_dict() for uuid, node in self.nodes.items()
            }
        }
    
    def from_dict(self, d):
        self.nodes = {
            uuid: Node(owner=self, parent=None, uuid = uuid)
            for uuid in d['nodes'] }

        self.root = self.nodes[d['root_uuid']]

        for uuid, node in self.nodes.items():
            node.from_dict(d[uuid])

    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.to_dict(), f)    

    def load_from_file(self, filename):
        with open(filename, 'rb') as f:
            d = pickle.load(filename)
        self.from_dict(d)
    
    def produce_node(self, node_parent: 'Node', action: int):
        if node_parent.done:
            return None
        if action in node_parent.children:
            return None
            raise AttributeError(f'Мы уже считали это!! ')
        self.gym.set_state(node_parent.gym_state)
        obs, reward, done, info = self.gym.step(action)
        child = Node(
            owner = self,
            parent = node_parent,
            gym_state = self.gym.get_state(),
            action_from = action,
            obs = obs,
            reward = reward,
            done = done,
            info = info)
        node_parent.children[action] = child
        self.nodes[child.uuid] = child
        return child
        
    def get_done_nodes(self):
        return [node for node in self.nodes.values() if node.done]

    def get_not_full_nodes(self, forced=True):
        def foo(node: 'Node'):
            return node.is_full_children()
        self.calc_info('is_full', foo, forced)
        return [node for node in self.nodes.values() if not node.info['is_full']]

    def get_full_nodes(self, forced=True):
        def foo(node: 'Node'):
            return node.is_full_children()
        self.calc_info('is_full', foo, forced)
        return [node for node in self.nodes.values() if node.info['is_full']]

    def calc_info(self, name, foo, forced=False):
        for node in self.nodes.values():
            if forced or (name not in node.info):
                node.info[name] = foo(node)

    def fill_info_step(self):
        name = 'step'
        self.root.info[name] = 0
        def foo(node: 'Node'):
            if name not in node.info:
                node.info[name] = foo(node.parent) + 1
            return node.info[name]
        self.calc_info(name, foo)
    

    def fill_node_pos(self):
        name = 'pos'
        def foo(node):
            self.gym.set_state(node.gym_state)
            return self.gym.missile.pos
        self.calc_info('pos', foo)

    # def fill_node_colors(self):


    def plot(self, fig, ax):
        self.fill_node_pos()
        col1 = (0, 0, 0.9, 0.7)
        for node in self.nodes.values():
            if node.parent:
                p1 = node.info['pos']
                p2 = node.parent.info['pos']
                ax.plot((p1[0], p2[0]), (p1[1], p2[1]), 'o-', color=col1)


class Node:
    actions_full = [-1,0,1]

    def __init__(self, 
                 owner, 
                 parent, 
                 gym_state=None, 
                 action_from=None,  
                 obs=None, 
                 reward=None, 
                 done=False, 
                 info=None,
                 uuid=None):
        self.owner = owner
        self.parent = parent
        self.gym_state = np.array(gym_state) if gym_state is not None else None
        self.action_from = action_from
        self.obs = np.array(obs) if obs is not None else None
        self.reward = reward
        self.done = done
        self.info = {} if info is None else info
        self.uuid = uuid if uuid is not None else uuid1()        
        self.children = {}

    def produce_node(self, action):
        return self.owner.produce_node(self, action)

    def to_dict(self):
        return {
            'parent_uuid': None if self.parent is None else self.parent.uuid,
            'gym_state': self.gym_state,
            'action_from': self.action_from,
            'obs': self.obs,
            'reward': self.reward,
            'done': self.done,
            'info': self.info,
            'uuid': self.uuid,        
            'children_uuids': {action: child.uuid for action, child in self.children.items() } 
        }

    def from_dict(self, d):
        self.parent = self.owner.nodes[d['parent_uuid']] if d['parent_uuid'] else None
        self.gym_state = d['gym_state']
        self.action_from = d['action_from']
        self.obs = d['obs']
        self.reward = d['reward']
        self.done = d['done']
        self.info = d['info']
        self.uuid = d['uuid']
        self.children = {
            action: self.owner.nodes[child_uuid]
            for action, child_uuid in d['children_uuids'].items() }


    def is_full_children(self, actions=None):
        if self.done:
            return True
        if actions is None:
            actions = self.actions_full
        for a in actions:
            if a not in self.children:
                return False
        return True
    
    def get_missing_actions(self, actions_full=None):
        if actions_full is None:
            actions_full = self.actions_full
        return [a for a in actions_full if a not in self.children]
    
    def get_high_aver_low_rewards(self):
        rewards = np.array([n.reward for n in self.children.values()])
        return np.max(rewards), np.average(rewards), np.min(rewards)
    
    def get_best_node(self, foo=None):
        children_list = list(self.children.values())
        if foo:
            scores = [foo(node) for node in children_list]
        else:
            scores = [node.reward for node in children_list]
        return children_list[np.argmax(scores)]
    
    def get_nodes_sorted(self, foo=None):
        children_list = list(self.children.values())
        if foo:
            scores = [foo(node) for node in children_list]
        else:
            scores = [node.reward for node in children_list]
        return [children_list[i] for i in np.argsort(scores)]
        
        
    