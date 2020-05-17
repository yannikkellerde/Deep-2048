import numpy as np
import os,sys
from shutil import rmtree
from AI_2048.neural_nets.mlp import mlp
import tensorflow as tf
from AI_2048.env.game import Game_2048
from AI_2048.util.constants import *
import random
from shutil import rmtree
import time
from collections import deque
import math
from multiprocessing import Queue,Process,Pool,Manager
from AI_2048.util.generators import RL_sequence
from tensorflow.keras import Model
from tensorflow.keras.callbacks import CSVLogger
import tensorflow.keras as keras
from functools import reduce
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

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
    def __init__(self,game=Game_2048(), batch_size=1024, lr=0.01):
        self.game = game
        self.memory = deque(maxlen=100000)
        self.learning_rate = lr
        self.batch_size = batch_size
        self.model = mlp(self.game.space_1d.n,5,256,self.game.action_space.n+1,lr=self.learning_rate)
        self.experimental_model = mlp(self.game.space_1d.n,5,256,self.game.action_space.n+1,lr=self.learning_rate)
        rmtree(os.path.abspath(os.path.dirname(__file__)) + "/tensorboard_logs",ignore_errors=True)
        self.tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./tensorboard_logs")
        self.generator = RL_sequence(self.memory,self.batch_size)
        self.best_net_avg_score = 0
        self.workers = MCTS_workers(game)

    def save_model(self,model):
        folder = "model_save"
        os.makedirs(folder,exist_ok=True)
        model_json = model.to_json()
        with open(os.path.join(folder,"model.json"), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(os.path.join(folder,"model.hf5"))
        print("Saved model to disk")

    def nn_evaluation_worker(self,input_queues,output_queues):
        print("EVALUATION WORKER READY, process id:",os.getpid(),input_queues)
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
                    tasks.append(task)
                    return_queues.append([i,indicator])
                i+=1
            if len(tasks)==0:
                continue
            before = time.time()
            prediction = self.model.predict(np.array(tasks))
            print(f"Evaluated {len(tasks)} tasks in {time.time()-before} seconds. Workers left: {len(input_queues)}")
            for i,return_stuff in enumerate(return_queues):
                q_num,indicator = return_stuff
                output_queues[q_num].put([prediction[i],indicator])
        print("Ending evaluation worker")
    def do_training(self,model:Model,steps_per_epoch:int,epochs:int,validation_data):
        model.fit_generator(self.generator,validation_data=validation_data,steps_per_epoch=steps_per_epoch,epochs=epochs,callbacks=[self.tensorboard_callback])
    def learn(self):
        mc_workers = 5
        move_time = 0.1
        games_per_monte_carlo = 1
        training_iterations = 2000
        validation_set_size = 32
        self.batch_size = 32
        train_epochs = 100
        nn_input_queues = [Queue() for _ in range(mc_workers)]
        nn_output_queues = [Queue() for _ in range(mc_workers)]
        training_examples_queue = Queue()
        args = list(zip(nn_input_queues,nn_output_queues,[training_examples_queue]*mc_workers,[move_time]*mc_workers,[games_per_monte_carlo]*mc_workers))
        for i in range(training_iterations):
            logger.info(f"Starting monte carlo rollouts for iteration {i}")
            self.workers.my_own_pool(mc_workers,self.workers.monte_carlo_worker,args)
            self.nn_evaluation_worker(nn_input_queues,nn_output_queues)
            while not training_examples_queue.empty():
                self.memory.append(training_examples_queue.get())
            for queue in nn_input_queues+nn_output_queues+[training_examples_queue]:
                queue.close()
                queue.join_thread()
            validation_set = self.generator.extract_validation_set(validation_set_size)
            self.experimental_model.set_weights(self.model.get_weights())
            logger.info(f"Starting training for iteration {i}")
            self.do_training(self.experimental_model,len(self.generator),train_epochs,validation_set)
            logger.info(f"Evaluating new model")
            exp_score = self.workers.evaluate_net(self.experimental_model,game_batch_len=256,time_per_move=1,batch_num=1)
            if exp_score > self.best_net_avg_score:
                self.model.set_weights(self.experimental_model.get_weights())
                self.save_model(self.model)
                print(f"New best model {exp_score}>{self.best_net_avg_score}")
                self.best_net_avg_score = exp_score
            else:
                print(f"Model did not improve {exp_score}<{self.best_net_avg_score}")

class MCTS_workers():
    def __init__(self,game):
        self.game = game
        # First model output is the value, the rest are the move probabilities
        self.exploration_constant = 100
        self.root = probabilistic_Node(self.game.state,1)

    def player_node_policy(self,node:player_Node,parent:Node):
        return node.action_value+self.exploration_constant*node.prior_value*(math.sqrt(parent.visits)/(node.visits+1))
    def prob_node_tree_policy(self,node:probabilistic_Node,parent:Node):
        return node.probability-(node.visits/parent.visits if parent.visits>0 else 0)
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
    def select_most_promising(self,root):
        node = root
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
                node.children.append(child)
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
    def get_move_props(self, root):
        move_props = np.zeros(4)
        for a in range(self.game.action_space.n):
            state,_reward,_done=self.game.check_update(root.state,a)
            for child in root.children:
                if np.array_equal(child.state,state):
                    move_props[a]=child.visits/root.visits
                    break
        return move_props
    def my_own_pool(self,num_processes,target,args):
        processes = []
        for i in range(num_processes):
            p = Process(target=target, args=args[i])
            processes.append(p)
            p.start()
            processes.append(p)
        return processes
    def monte_carlo_worker(self,nn_input_queue:Queue,nn_output_queue:Queue,examples_queue:Queue,move_time,game_count):
        print('READY: process id:', os.getpid())
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
                    path = self.select_most_promising(self.root)
                    node = path[-1]
                    if not node in recent_expands:
                        recent_expands.append(node)
                        nn_input = self.game.convert_to_nn_input(node.state)
                        store_paths[path_store_num]=path
                        nn_input_queue.put([nn_input,path_store_num])
                        path_store_num=path_store_num+1 if path_store_num<max_path_store_num else 0
                        self.virtual_loss(path)
                while not nn_output_queue.empty():
                    nn_res,path_num = nn_output_queue.get()
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
                    move_props=self.get_move_props(self.root)
                    nn_state = self.game.convert_to_nn_input(self.root.state)
                    action = np.argmax(move_props)
                    _,reward,done = self.game.step(action)
                    incomplete_training_examples.append([nn_state,move_props])
                    rewardlist.append(reward)
                    self.root = probabilistic_Node(self.game.state,1)
                    next_move_time = time.time()+move_time
                    no_more_new = False
            rew_sum = 0
            for i in range(len(incomplete_training_examples)-1,-1,-1):
                training_example = incomplete_training_examples[i]
                rew_sum += rewardlist[i]
                training_example[1]=[rew_sum]+training_example[1]
            for training_example in incomplete_training_examples:
                examples_queue.put(training_example)
        nn_input_queue.put(None)
    def evaluate_net(self,new_net,game_batch_len,time_per_move,batch_num=1):
        score_sum = 0
        for i in range(batch_num):
            games = [Game_2048() for _ in range(game_batch_len)]
            scores = [0]*game_batch_len
            paths = [None]*game_batch_len
            nn_inputs = [None]*game_batch_len
            while not reduce(lambda a,b:a.done*b.done,games,True):
                roots = [probabilistic_Node(game.state,1) for game in games]
                next_move_time = time.time()+time_per_move
                while time.time()<next_move_time:
                    for i,game in enumerate(games):
                        if game.done:
                            continue
                        path = self.select_most_promising(roots[i])
                        node = path[-1]
                        nn_input = self.game.convert_to_nn_input(node.state)
                        paths[i] = path
                        nn_inputs[i] = nn_input
                    prediction = new_net.predict(np.array(nn_inputs))
                    for i,game in enumerate(games):
                        if game.done:
                            continue
                        value = prediction[i][0]
                        move_props = prediction[i][1:]
                        back_path = paths[i]
                        node = back_path[-1]
                        self.expand(node,move_props)
                        self.backtrack(back_path,value,True)
                for i,game in enumerate(games):
                    if game.done:
                        continue
                    move_props=self.get_move_props(roots[i])
                    action = np.argmax(move_props)
                    _,reward,_done = game.step(action)
                    scores[i]+=reward
            score_sum+=sum(scores)/len(scores)
        return score_sum/batch_num

if __name__ == "__main__":
    agent = MCTS_NN()
    agent.learn()