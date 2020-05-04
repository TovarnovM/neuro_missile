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
        obs = self.gym.reset()
        self.root = Node(
            owner=self,
            parent=None,
            gym_state=self.gym.get_state(),
            obs = obs)
        self.nodes[self.root.uuid] = self.root 

    def __len__(self):
        return len(self.nodes)-1

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
            node.from_dict(d['nodes'][uuid])

    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.to_dict(), f)    

    def load_from_file(self, filename):
        with open(filename, 'rb') as f:
            d = pickle.load(f)
        self.from_dict(d)
    
    def produce_node(self, node_current: 'Node', action: int):
        if node_current.done:
            return None
        if action in node_current.children:
            return node_current.children[action]
            # raise AttributeError(f'Мы уже считали это!! ')
        self.gym.set_state(node_current.gym_state)
        obs, reward, done, info = self.gym.step(action)
        child = Node(
            owner = self,
            parent = node_current,
            gym_state = self.gym.get_state(),
            action_from = action,
            obs = obs,
            reward = reward,
            done = done,
            info = info)
        node_current.children[action] = child
        self.nodes[child.uuid] = child
        return child

    def transaction_iter(self):
        for node in self.nodes.values():
            trans = node.get_transaction()
            if trans:
                yield trans


    def walk(self, node_current: 'Node', p_random=0.5):
        if node_current.done:
            return None
        missing_actions = node_current.get_missing_actions()
        if len(missing_actions) == 1:
            return self.produce_node(node_current, missing_actions[0])
        if np.random.random() < p_random:
            if len(missing_actions) == 0:
                return np.random.choice(list(node_current.children.values()))
            return self.produce_node(node_current, np.random.choice(missing_actions))
        
        good_action = self.get_action_parallel_guidance(node_current)
        if good_action in missing_actions:
            return self.produce_node(node_current, good_action)
        return node_current.children[good_action]

    def get_action_parallel_guidance(self, node=None):
        if node:
            self.gym.set_state(node.gym_state)
        if self.gym.missile.v > self.gym.target.v:
            action_parallel_guidance = self.gym.missile.get_action_parallel_guidance(self.gym.target)
        else:
            action_parallel_guidance = self.gym.missile.get_action_chaise_guidance(self.gym.target)
        if -0.5 <= action_parallel_guidance <= 0.5:
            action_parallel_guidance = 0
        elif action_parallel_guidance < -0.5:
            action_parallel_guidance = -1
        else:
            action_parallel_guidance = 1
        return int(action_parallel_guidance) 

    def get_line_nodes(self, node: 'Node'):
        res = [node]
        while node.parent and not node.parent.children:
            node = node.parent
            res.append(node)
        return res

    def get_perspective_node(self):
        nodes_done = self.get_done_nodes()
        if nodes_done:
            tail = np.random.choice(nodes_done)
            line = self.get_line_nodes(tail)
            if len(line) > 10:
                return line[int(len(line)/1.618)]
        return np.random.choice(self.get_not_full_nodes())

    def get_perspective_node2(self):
        nodes_done = self.get_done_nodes()
        if nodes_done:
            tails = [self.get_line_nodes(end_node) for end_node in nodes_done]
            tails_lengths = [len(tail) for tail in tails]
            line = tails[np.argmax(tails_lengths)]
            if len(line) > 10:
                return line[int(len(line)/1.618)]
        return np.random.choice(self.get_not_full_nodes())

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


    def plot(self, ax=None, figsize=(10,7)):
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=figsize)
            self.plot(ax=ax)
            ax.grid()
            ax.axis('equal')
            plt.show()
            return

        from matplotlib import collections  as mc
        self.fill_node_pos()
        col1 = (0, 0, 0.9, 0.1)
        segments = []
        colors = []
        for node in self.nodes.values():
            if node.parent:
                p1 = node.info['pos']
                p2 = node.parent.info['pos']
                segments.append((p1, p2))
                colors.append(col1)
        lc = mc.LineCollection(segments, colors=colors)
        ax.add_collection(lc)
        ax.autoscale()

    def plot_scatter(self, ax=None, figsize=(10,7)):
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=figsize)
            self.plot_scatter(ax=ax)
            ax.grid()
            ax.axis('equal')
            plt.show()
            return

        self.fill_node_pos()
        xs = []
        ys = []
        cs = []
        ss = []
        # als = []/
        for node in self.nodes.values():
            p1 = node.info['pos']
            xs.append(p1[0])
            ys.append(p1[1])
            r = 0 if node.reward is None else node.reward
            if node.done:
                col = (0.1,1.0,0.1,0.9) if r > 0 else (1.0,0.1,0.1,0.9)
            else:
                col = (0.4, 0.4, 0.9, 0.3)
            cs.append(col)
            ss.append(10 if node.done else 0.4 )
            # als.append(0.5 if node.done else 1)
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        cs = np.asarray(cs)
        ss = np.asarray(ss)
        # als = np.asarray(als)
        ax.scatter(xs, ys, c=cs, s=ss, alpha=0.5)


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

    def get_action4transaction(self):
        return self.action_from + 1

    def get_transaction(self):
        if self.parent:
            return self.parent.obs, self.reward, self.get_action4transaction(), self.obs, self.done

    def produce_node(self, action):
        return self.owner.produce_node(self, action)

    def walk(self, p_random=0.5):
        return self.owner.walk(self, p_random)
    
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

    def make_current(self):
        self.owner.gym.set_state(self.gym_state)
        
    def get_distance_to_trg(self):
        self.make_current()
        return np.linalg.norm(self.owner.gym.missile.pos - self.owner.gym.target.pos)  
    