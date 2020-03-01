from AI_2048.env.game import Game_2048
import numpy as np
import math
import time
import random
from AI_2048.agents.base import Agent
from AI_2048.util.get_policy_func import get_policy_func
class Node():
    def __init__(self, state):
        self.children = []
        self.state = state
        self.visits = 0
        self.avg_reward = 0
class player_Node(Node):
    def __init__(self, state, transition_reward):
        super(player_Node, self).__init__(state)
        self.transition_reward = transition_reward
class probabilistic_Node(Node):
    def __init__(self, state, probability):
        super(probabilistic_Node, self).__init__(state)
        self.probability = probability
        self.done = False
class MCTS(Agent):
    def __init__(self,game=Game_2048(),playout_policy=get_policy_func("Random_agent")):
        self.game = game
        self.exploration_constant = 100
        self.playout_policy = playout_policy
        self.root = probabilistic_Node(self.game.state,1)
    def set_state(self, state):
        self.root = probabilistic_Node(state,1)
    def tree_policy_UCT(self,node:Node,parent:Node):
        if node.visits==0:
            return 1<<20+node.transition_reward
        return node.avg_reward+self.exploration_constant*math.sqrt((2*math.log(parent.visits))/node.visits)
    def prob_node_tree_policy(self,node:probabilistic_Node,parent:Node):
        return node.probability-node.visits/parent.visits
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
        self.game.state = node.state.copy()
        cum_reward = 0
        done = False
        if not isinstance(node,probabilistic_Node):
            self.game.spawn_number()
        while not done:
            action = self.playout_policy(self.game.state)
            _, reward, done = self.game.step(action)
            cum_reward += reward
        return cum_reward
    def backtrack(self,node_path,reward):
        visit_power = 1
        while len(node_path)>0:
            node = node_path.pop()
            if isinstance(node,player_Node):
                reward+=node.transition_reward
            node.avg_reward=(reward+node.avg_reward*node.visits)/(node.visits+visit_power)
            node.visits+=visit_power
            """else:
                reward*=node.probability
                visit_power*=node.probability"""

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
    def expand(self,node):
        if isinstance(node,probabilistic_Node):
            if node.done:
                return None
            for a in range(self.game.action_space.n):
                next_state,reward,done = self.game.check_update(node.state,a)
                if not done:
                    child = player_Node(next_state, reward)
                    node.children.append(child)
            if len(node.children)>0:
                return random.choice(node.children)
            else:
                node.done = True
                return None
        else:
            states,probs = self.game.get_state_expectations(node.state)
            for i in range(len(states)):
                child = probabilistic_Node(states[i],probs[i])
                node.children.append(child)
            return self.choose_child(node)
    def get_action_vals(self):
        action_vals = []
        for a in range(self.game.action_space.n):
            state,reward,done=self.game.check_update(self.root.state,a)
            for child in self.root.children:
                if np.array_equal(child.state,state):
                    action_vals.append({"action":a,"visits":child.visits,"reward":child.avg_reward})
                    break
        return action_vals
    def get_action(self,state,max_time=1):
        start = time.time()
        while time.time()- start<max_time:
            path = self.select_most_promising()
            node = path[-1]
            if node.visits==0:
                playout_node = node
            else:
                playout_node = self.expand(node)
                if playout_node is not None:
                    path.append(playout_node)
            if playout_node is None:
                reward = self.game.done_reward
            else:
                reward=self.playout(playout_node)
            self.backtrack(path,reward)
        action_vals = self.get_action_vals()
        self.game.state = self.root.state
        if len(action_vals)==0:
            return None
        return sorted(action_vals,key=lambda x:x["visits"])[-1]["action"]

if __name__ == "__main__":
    game = Game_2048()
    for ec in [10,40,70,100,175,250,500]:
        mcts = MCTS(game,playout_policy=get_policy_func("Random_agent"))
        mcts.exploration_constant = ec
        print(game)
        cum_reward = 0
        while 1:
            mcts.set_state(game.state)
            action = mcts.get_action(game.state,1)
            if action is None:
                break
            _,reward,done = game.step(action)
            cum_reward+=reward
            print(game)
        log = f"{ec}:{cum_reward}\n"
        print(log)
        with open("hyperparameter_tuning.log","a") as f:
            f.write(log)