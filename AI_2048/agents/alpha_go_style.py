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
from multiprocessing import Queue

class Node():
    def __init__(self,state):
        self.visits = 0
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
        # First model output is the value, the rest are the move probabilities
        self.model = mlp(self.game.space_1d.n,5,256,self.game.action_space.n+1,lr=self.learning_rate)
        self.exploration_constant = 100
        self.root = probabilistic_Node(self.game.state,1)
    def player_node_policy(self,node:Node,parent:Node):
        return node.action_value+self.exploration_constant*node.prior*(math.sqrt(parent.visits)/(node.visits+1))
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
    def expand(self,node,move_props):
        for a in range(self.game.action_space.n):
            new_state,reward,done = self.game.check_update(node.state,a)
            if not done:
                child = player_Node(new_state,move_props[a],reward)
                prob_states,probs = self.game.get_state_expectations(child.state)
                for i in range(len(prob_states)):
                    child_child = probabilistic_Node(prob_states[i],probs[i])
                    child.children.append(child_child)
        if len(node.children)==0:
            node.done = True
    def backtrack(self,node_path,value,compensate_virtual_loss):
        while len(node_path)>0:
            node = node_path.pop()
            node.visits+=1-compensate_virtual_loss
            if isinstance(node,player_Node):
                value+=node.transition_reward
                node.total_value += value
                node.action_value = node.total_value/node.visits
    def virtual_loss(self,node_path):
        for node in node_path:
            node.visits+=1
    def nn_evaluation_worker(self,input_queues,output_queues):
        while len(input_queues)>0:
            tasks = []
            return_queues = []
            i=0
            while i<len(input_queues):
                q = input_queues[i]
                while not q.empty():
                    queue_val = q.get()
                    if queue_val is None:
                        del input_queues[i]
                        del output_queues[i]
                        i-=1
                        break
                    task, indicator = queue_val
                    tasks.append()
                    return_queues.append([i,indicator])
                i+=1
            prediction = self.model.predict(np.array(tasks))
            for i,return_stuff in enumerate(return_queues):
                q_num,indicator = return_stuff
                output_queues[q_num].put([prediction[i],indicator])
    def get_move_props(self):
        move_props = np.zeros(4)
        for a in range(self.game.action_space.n):
            state,reward,done=self.game.check_update(self.root.state,a)
            for child in self.root.children:
                if np.array_equal(child.state,state):
                    move_props[a]=child.visits/self.root.visits
                    break
        return move_props
    def monte_carlo_worker(self,nn_input_queue:Queue,nn_output_queue:Queue,training_examples_queue:Queue,move_time,game_count):
        path_store_num = 0
        max_path_store_num = 10000
        store_paths = {}
        for _ in range(game_count):
            self.game.reset()
            self.root = probabilistic_Node(self.game.state,1)
            done = False
            rewardlist = []
            incomplete_training_examples = []
            recent_expands = deque(maxlen=10)
            next_move_time = time.time()+move_time
            no_more_new = False
            while not done:
                if not no_more_new:
                    path = self.select_most_promising()
                    node = path[-1]
                    if not node in recent_expands:
                        recent_expands.append(node)
                        nn_input = self.game.convert_to_nn_input(node.state)
                        store_paths[path_store_num]=path
                        nn_output_queue.put([nn_input,path_store_num])
                        path_store_num=path_store_num+1 if path_store_num<max_path_store_num else 0
                        self.virtual_loss(path)
                while not nn_input_queue.empty():
                    nn_res,path_num = nn_input_queue.get()
                    back_path = store_paths[path_num]
                    node = back_path[-1]
                    del store_paths[path_num]
                    value = nn_res[0]
                    move_props = nn_res[1:]
                    self.expand(node,move_props)
                    self.backtrack(back_path,value,True)
                if time.time()>next_move_time:
                    no_more_new = True
                if no_more_new and len(store_paths)==0:
                    move_props=self.get_move_props()
                    nn_state = self.game.convert_to_nn_input(self.root.state)
                    action = np.argmax(move_props)
                    _,reward,done = self.game.step(action)
                    incomplete_training_examples.append([nn_state,move_props])
                    rewardlist.append(reward)
                    self.root = probabilistic_Node(self.game.state,1)
                    next_move_time = time.time()+move_time
            rew_sum = 0
            for i in range(len(incomplete_training_examples)-1,-1,-1):
                training_example = incomplete_training_examples[i]
                rew_sum += rewardlist[i]
                training_example.append(rew_sum)
            training_examples_queue.put(incomplete_training_examples)
        nn_output_queue.put(None)
    def do_training(self,iterations):
