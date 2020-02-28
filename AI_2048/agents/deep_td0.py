import numpy as np
import os,sys
from shutil import rmtree
from tensorflow.keras.layers import Dense,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
import tensorflow as tf
from AI_2048.env.game import Game_2048
from AI_2048.util.constants import *
import random
import time
from collections import deque
from AI_2048.agents.base import Agent

class DeepTD0(Agent):
    def __init__(self,game:Game_2048):
        self.game = game
        self.memory = deque(maxlen=100000)
        self.learning_rate = 0.003
        self.discount_factor = 1
        self.batch_size = 4096
        self.rollout_batch_size = 100
        self.train_per_it = 100
        self.train_start = 10000
        self.model = Sequential()
        self.model.add(Dense(256, input_dim=self.game.observation_space.n,
                        activation='relu',kernel_initializer='he_uniform'))
        self.model.add(Dense(256, activation='relu',
                        kernel_initializer='he_uniform'))
        self.model.add(Dense(256, activation='relu',
                        kernel_initializer='he_uniform'))
        self.model.add(Dense(256, activation='relu',
                        kernel_initializer='he_uniform'))
        self.model.add(Dense(256, activation='relu',
                        kernel_initializer='he_uniform'))
        self.model.add(Dense(256, activation='relu',
                        kernel_initializer='he_uniform'))
        self.model.add(Dense(256, activation='relu',
                        kernel_initializer='he_uniform'))
        self.model.add(Dense(256, activation='relu',
                        kernel_initializer='he_uniform'))
        self.model.add(Dense(1, activation='linear',
                        kernel_initializer='he_uniform'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
    def greedy_rollout_value_func(self,it):
        with open("greedy_rollout.log","a") as f:
            f.write(f"==========Iteration {it}==========\n")
            self.game.reset()
            done = False
            while not done:
                action = None
                best_val = -np.inf
                for a in range(self.game.action_space.n):
                    _,reward,_ = self.game.check_update(self.game.state,a)
                    if reward>best_val:
                        best_val = reward
                        action = a
                new_state,reward,done = self.game.check_update(self.game.state,action)
                self.game.state = new_state
                nn_input = self.game.convert_to_nn_input(self.game.state).reshape(1,-1)
                f.write(f"State: {self.game}, Value: {self.model.predict(nn_input).item()}\n")
                if not done:
                    self.game.spawn_number()
    def rollout_batch(self,batch_size):
        games = [Game_2048() for _ in range(batch_size)]
        states = [[game.state] for game in games]
        still_running = list(range(batch_size))
        rewardlist = [[] for _ in games]
        while len(still_running)>0:
            """eval_batch = []
            evaluations = np.zeros((len(still_running),self.game.action_space.n))
            for sn in range(len(still_running)):
                i = still_running[sn]
                for a in range(games[i].action_space.n):
                    new_state,reward,done = games[i].check_update(games[i].state,a)
                    if done:
                        evaluations[sn,a]=reward
                    else:
                        nn_input = self.game.convert_to_nn_input(new_state)
                        eval_batch.append(nn_input)
                        evaluations[sn,a] = reward
            if len(eval_batch)>0:
                predictions = self.model.predict(np.array(eval_batch)).reshape(len(eval_batch))
            pindex = 0
            for e in range(evaluations.shape[0]):
                for a in range(evaluations.shape[1]):
                    if evaluations[e,a]!=self.game.done_reward:
                        #if e==0 and 0 in still_running:
                        #    log_file.write(f"action {BACKMAP[a]}, reward: {evaluations[e,a]}, value: {predictions[pindex]}\n")
                        evaluations[e,a] += predictions[pindex]
                        pindex+=1
                    #else:
                        #if e==0 and 0 in still_running:
                        #    log_file.write(f"action {BACKMAP[a]}, reward: {evaluations[e,a]}, value: done\n")"""
            for sn in range(len(still_running)-1,-1,-1):
                i = still_running[sn]
                action = None
                best_val = -np.inf
                for a in range(games[i].action_space.n):
                    _,reward,_ = games[i].check_update(games[i].state,a)
                    if reward>best_val:
                        best_val = reward
                        action = a
                new_state,reward,done = games[i].check_update(games[i].state,action)
                if done:
                    del still_running[sn]
                    continue
                games[i].state = new_state
                states[i].append(new_state.copy())
                games[i].spawn_number()
                rewardlist[i].append(reward)
        cum_rewards = [sum(x) for x in rewardlist]
        for i in range(len(states)):
            for j in range(len(states[i])):
                if j<len(states[i])-1:
                    self.memory.append((states[i][j],rewardlist[i][j],states[i][j+1]))
        return cum_rewards
    def get_action(self,state):
        action_vals = []
        for a in range(self.game.action_space.n):
            new_state,reward,done = self.game.check_update(state,a)
            if done:
                action_vals.append(reward)
            else:
                nn_input = self.game.convert_to_nn_input(new_state).reshape(1,-1)
                action_vals.append(self.model.predict(nn_input).item()+reward)
        return np.argmax(action_vals)
    def train_one_batch(self):
        if len(self.memory) < self.train_start:
            return 0
        mini_batch = np.array(random.sample(self.memory, self.batch_size))
        update_in = np.array([self.game.convert_to_nn_input(x[0]) for x in mini_batch])
        nn_input = np.array([self.game.convert_to_nn_input(x[2]) for x in mini_batch])
        evals = self.model.predict(nn_input).reshape(len(mini_batch))
        target = np.array([mini_batch[i][1]+self.discount_factor*evals[i] for i in range(len(mini_batch))])
        loss = self.model.train_on_batch(update_in, target)
        return loss
    def train_iterations(self,iterations):
        try:
            rmtree(os.path.abspath(os.path.dirname(__file__)) + "/logdir")
        except:
            pass
        train_writer = tf.summary.create_file_writer(os.path.abspath(os.path.dirname(__file__)) + "/logdir")
        state = self.game.reset()
        print("Filling up memory to get started")
        while len(self.memory) < self.train_start:
            self.rollout_batch(500)
        print("Min samples stored, starting training now")
        #rew_avg = 0
        for i in range(iterations):
            self.greedy_rollout_value_func(i)
            print(f"Performing {self.train_per_it} batch trainings")
            loss_avg = 0
            for _ in range(self.train_per_it):
                loss = self.train_one_batch()
                loss_avg+=loss
            loss_avg = loss_avg/self.train_per_it
            print(f"Rolling out policy {self.rollout_batch_size} times")
            rewards = self.rollout_batch(self.rollout_batch_size)
            rew_avg = sum(rewards)/len(rewards)
            print(f"Iteration: {i}, Reward avg: {rew_avg}, avg loss: {loss_avg:.3f}")
            with train_writer.as_default():
                tf.summary.scalar('reward', rew_avg, step=i)
                tf.summary.scalar('avg loss', loss_avg, step=i)

if __name__ == "__main__":
    game = Game_2048()
    learner = DeepTD0(game)
    learner.train_iterations(100000000)