import numpy as np
import os,sys
from shutil import rmtree
from AI_2048.neural_nets.mlp import mlp
import tensorflow as tf
from AI_2048.env.game import Game_2048
from AI_2048.util.constants import *
import random
import time
from collections import deque
import math

class Node():
    def __init__(self,state):
        self.visit_count = 0
        self.state = state
        self.children = []
class player_Node(Node):
    def __init__(self, state, prior, transition_reward):
        super(player_Node, self).__init__(state)
        self.transition_reward = transition_reward
        self.prior_value = prior
        self.action_value = 0
        self.total_value = 0
class probabilistic_Node(Node):
    def __init__(self, state, probability):
        super(probabilistic_Node, self).__init__(state)
        self.probability = probability
        self.done = False

class MCTS_NN():
    def __init__(self,game=Game_2048()):
        self.game = game
        self.memory = deque(maxlen=100000)
        self.learning_rate = 0.01
        self.batch_size = 1024
        self.model = mlp(self.game.space_1d.n,5,256,self.game.action_space.n+1,lr=self.learning_rate)
        self.exploration_constant = 100
    def player_node_policy(self,node:Node,parent:Node):
        return node.action_value+self.exploration_constant*node.prior*(math.sqrt(parent.visit_count)/(node.visit_count+1))
    def prob_node_tree_policy(self,node:probabilistic_Node,parent:Node):
        return node.probability-node.visits/parent.visits
    def choose_child(self,node:Node):
        max_val = -np.inf
        best_child = None
        tree_policy = self.player_node_policy if isinstance(node,probabilistic_Node) else self.prob_node_tree_policy
        for child in node.children:
            val = tree_policy(child,node)
            if val > max_val:
                max_val = val
                best_child = child
        return best_child
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
        nn_input = self.game.convert_to_nn_input(node.state)
        nn_output = self.model.predict()