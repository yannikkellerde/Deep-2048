import numpy as np
import os,sys
from shutil import rmtree
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
import tensorflow as tf
from game import Game_2048
from constants import *
import random
import time
from collections import deque
import math

class MC_state_value():
    def __init__(self,game:Game_2048):
        self.game = game
        self.memory = deque(maxlen=100000)
        self.learning_rate = 0.01
        self.batch_size = 4096
        self.rollout_batch_size = 400
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
        #log_file = open("rollout.log","a")
        #log_file.write("\n==============STARTING NEW BATCH=============\n")
        games = [Game_2048() for _ in range(batch_size)]
        #log_file.write(str(games[0])+"\n")
        states = [[game.state] for game in games]
        still_running = list(range(batch_size))
        rewardlist = [[] for _ in games]
        while len(still_running)>0:
            eval_batch = []
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
                        #    log_file.write(f"action {BACKMAP[a]}, reward: {evaluations[e,a]}, value: done\n")
            for sn in range(len(still_running)-1,-1,-1):
                i = still_running[sn]
                action_vals = evaluations[sn]
                action = np.argmax(action_vals)
                """action = None
                best_val = -np.inf
                for a in range(games[i].action_space.n):
                    _,reward,_ = games[i].check_update(games[i].state,a)
                    if reward>best_val:
                        best_val = reward
                        action = a"""
                new_state,reward,done = games[i].check_update(games[i].state,action)
                if done:
                    del still_running[sn]
                    continue
                games[i].state = new_state
                #if i==0:
                #    log_file.write(f"chosen action: {BACKMAP[action]}, new state before spawn: {games[i]}\n")
                states[i].append(new_state.copy())
                games[i].spawn_number()
                #if i==0:
                #    log_file.write(f"state after spawn: {games[i]}")
                rewardlist[i].append(reward)
        #log_file.write("\n==========MEMORY STORE AHEAD===========\n")
        cum_rewards = [sum(x) for x in rewardlist]
        samples = []
        for i in range(len(states)):
            store_reward = cum_rewards[i]
            for j in range(len(states[i])):
                state = states[i][j]
                #if i==0:
                #    temp_game = Game_2048()
                #    temp_game.state = state
                #    log_file.write(f"REWARD: {store_reward}, STATE: {temp_game}\n")
                samples.append((store_reward,state))
                if j<len(states[i])-1:
                    store_reward-=rewardlist[i][j]
        return cum_rewards, samples
    def train_one_batch(self,samples):
        if len(samples) < self.train_start:
            print("WARNING: NOT enough samples to train")
            return 0
        mini_batch = np.array(random.sample(samples, self.batch_size))
        update_in = np.array([self.game.convert_to_nn_input(x[1]) for x in mini_batch])
        target = np.array([x[0] for x in mini_batch])
        loss = self.model.train_on_batch(update_in, target)
        return loss
    def calc_testing_error(self,new_samples):
        update_in = np.array([self.game.convert_to_nn_input(x[1]) for x in new_samples])
        target = np.array([x[0] for x in new_samples])
        predicted = self.model.predict(update_in).reshape(len(target))
        abs_error = sum([abs(target[i]-predicted[i]) for i in range(len(target))])/len(target)
        mse_error = sum([(target[i]-predicted[i])**2 for i in range(len(target))])/len(target)
        return abs_error,mse_error
    def compare_to_monte_carlo_mean(self,rollouts,samples):
        deviation_tot = 0
        sudo_test_errors = []
        for sample in samples:
            state = sample[1]
            cum_rews = []
            for _ in range(rollouts):
                self.game.state = state.copy()
                cum_rew = 0
                done=False
                while not done:
                    action = None
                    best_val = -np.inf
                    for a in range(self.game.action_space.n):
                        _,reward,_ = self.game.check_update(self.game.state,a)
                        if reward>best_val:
                            best_val = reward
                            action = a
                    _,reward,done = self.game.step(action)
                    cum_rew+=reward
                cum_rews.append(cum_rew)
            mean = sum(cum_rews)/len(cum_rews)
            prediction = self.model.predict(self.game.convert_to_nn_input(state).reshape(1,-1)).item()
            sudo_test_error = sum([(prediction-cum_rews[i])**2 for i in range(len(cum_rews))])/len(cum_rews)
            sudo_test_errors.append(sudo_test_error)
            std = math.sqrt((1/len(cum_rews))*sum([(x-mean)**2 for x in cum_rews]))
            deviation_tot+=abs(mean-prediction)
            print(f"Monte carlo mean: {mean}, prediction: {prediction}, sample val: {sample[0]}, sudo test error: {sudo_test_error}, std:{std}")
        print(f"Average deviation: {deviation_tot/len(samples)}, sudo test error avg: {sum(sudo_test_errors)/len(sudo_test_errors)}")
    def train_iterations(self,iterations):
        try:
            rmtree(os.path.abspath(os.path.dirname(__file__)) + "/logdir")
        except:
            pass
        train_writer = tf.summary.create_file_writer(os.path.abspath(os.path.dirname(__file__)) + "/logdir")
        state = self.game.reset()
        print("Filling up memory to get started")
        rewards,samples = self.rollout_batch(400)
        print("Min samples stored, starting training now")
        #rew_avg = 0
        for i in range(iterations):
            print(f"Performing {self.train_per_it} batch trainings")
            loss_avg = 0
            for j in range(self.train_per_it):
                loss = self.train_one_batch(samples)
                loss_avg+=loss
            loss_avg = loss_avg/self.train_per_it
            print(f"Rolling out policy {self.rollout_batch_size} times")
            rewards,samples = self.rollout_batch(self.rollout_batch_size)
            rew_avg = sum(rewards)/len(rewards)
            abs_error,mse_error = self.calc_testing_error(samples[:4096])
            #self.compare_to_monte_carlo_mean(1000,random.sample(samples,5))
            print(f"Iteration: {i}, Reward avg: {rew_avg}, train error: {loss_avg:.3f}, test MSE_error:{mse_error}, test abs_error: {abs_error}")
            with train_writer.as_default():
                tf.summary.scalar('reward', rew_avg, step=i)
                tf.summary.scalar('train loss', loss_avg, step=i)
                tf.summary.scalar('test loss', mse_error, step=i)
if __name__ == "__main__":
    game = Game_2048()
    learner = MC_state_value(game)
    learner.train_iterations(100000000)