from game import Game_2048
import numpy as np
import math
import time
class Node():
    def __init__(self, state):
        self.children = []
        self.state = state
        self.visits = 0
        self.avg_reward = 0
class player_Node(Node):
    def __init__(self, state, transition_reward):
        super(probabilistic_Node, self).__init__(state)
        self.trainsition_reward = transition_reward
class probabilistic_Node(Node):
    def __init__(self, state, probability):
        super(probabilistic_Node, self).__init__(state)
        self.probability = probability
class MCTS():
    def __init__(self,game:Game_2048):
        self.game = game
        self.exploration_constant = 700
        self.root = Node(game.state)
    def tree_policy_UCT(self,node:Node,parent:Node):
        if node.visits==0:
            return np.inf
        return node.avg_reward+self.exploration_constant*math.sqrt((2*math.log(parent.visits))/node.visits)
    def prob_node_tree_policy(self,node:probabilistic_Node,parent:Node):
        if node.visits==0:
            return np.inf
        return node.probability+math.sqrt((2*math.log(parent.visits))/node.visits)
    def choose_child(self,node:Node):
        if len(node.children)==0:
            return None
        max_val = -np.inf
        best_child = None
        tree_policy = self.tree_policy_UCT if isinstance(node,probabilistic_Node) else self.prob_node_tree_policy
        for child in node.children:
            val = tree_policy(child,node)
            if val > max_val:
                max_val = val
                best_child = child
        return best_child
    def playout(self,node:Node):
        game.state = node.state
        cum_reward = 0
        done = False
        if not isinstance(node,probabilistic_Node):
            game.spawn_number()
        while not done:
            _, reward, _ = game.random_step()
            cum_reward += reward
        return cum_reward
    def backtrack(self,node_path,reward):
        while len(node_path)>0:
            node = node_path.pop()
            if isinstance(node,player_Node):
                reward+=node.transition_reward
            node.avg_reward=(reward+node.avg_reward*node.visits)/(node.visits+1)
            node.visits+=1
    def select_most_promising(self):
        node = self.root
        path = [node]
        while 1:
            child = self.choose_child(node)
            if child is None:
                return path
            else:
                node = child
                path.append(node)
    def expand(self,path):
        node = path[-1]
        if isinstance(node,probabilistic_Node):
            for a in range(self.game.action_space.n):
                next_state,reward,done = self.game.check_update(node.state,a)
                if not done:
                    child = player_Node(next_state, reward)
                    node.children.append()
        else:
            states,probs = self.game.get_state_expectations(node.state)
            for i in range(len(states)):
                child = probabilistic_Node(states[i],probs[i])
                node.children.append(child)
        for i in range(len(path)-1,-1,-1)
    def mcts(self,max_time):
        start = time.time()
        while time.time()- start<max_time:
