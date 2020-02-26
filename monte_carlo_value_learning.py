import numpy as np
import os,sys
from shutil import rmtree
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
import tensorflow as tf
from game import Game_2048
import random
from collections import deque

class MC_state_value():
    def __init__(self,game:Game_2048):
        self.game = game
        self.memory = deque(maxlen=100000)
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.batch_size = 64
        self.train_start = 1000
        self.model = Sequential()
        self.model.add(Dense(32, input_dim=self.game.observation_space.n,
                        activation='relu',kernel_initializer='he_uniform'))
        self.model.add(Dense(32, activation='relu',
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
                nn_input = self.game.convert_to_nn_input(state).reshape(1,-1)
                action_vals.append(self.model.predict(nn_input).item()+reward)
        return np.argmax(action_vals)
    def rollout_policy(self):
        self.game.reset()
        done = False
        states = [self.game.state]
        rewardlist = []
        cum_reward = 0
        while not done:
            action = self.get_action(self.game.state)
            new_state,reward,done = self.game.check_update(self.game.state,action)
            self.game.state = new_state
            states.append(new_state)
            if not done:
                self.game.spawn_number()
            cum_reward+=reward
            rewardlist.append(reward)
        store_rew = cum_reward
        for i in range(len(states)):
            self.memory.append((store_rew,states[i]))
            store_rew-=rewardlist[i]
        return cum_reward
    def train_one_batch(self):
        if len(self.memory) < self.train_start:
            return 0
        mini_batch = np.array(random.sample(self.memory, self.batch_size))
        update_in = np.array([self.game.convert_to_nn_input(x[1]) for x in mini_batch])
        target = np.array([x[0] for x in mini_batch])
        loss = self.model.train_on_batch(update_in, target)
        return loss
    def train_iterations(self,iterations):
        try:
            rmtree(os.path.abspath(os.path.dirname(__file__)) + "/logdir")
        except:
            pass
        train_writer = tf.summary.create_file_writer(os.path.abspath(os.path.dirname(__file__)) + "/logdir")
        state = self.game.reset()
        rew_sum = 0
        loss_sum = 0
        cnt = 0
        print("Filling up memory to get started")
        while len(self.memory) < self.train_start:
            self.rollout_policy()
        print("Min samples stored, starting training now")
        for i in range(iterations):
            loss = self.train_one_batch()
            reward = self.rollout_policy()
            loss_sum += loss
            rew_sum += reward
            cnt+=1
            if i%10==0:
                print(f"Iteration: {i}, Reward sum: {rew_sum/cnt}, cnt: {cnt}, avg loss: {loss_sum/cnt:.3f}")
                with train_writer.as_default():
                    tf.summary.scalar('reward', rew_sum/cnt, step=i)
                    tf.summary.scalar('avg loss', loss_sum/cnt, step=i)
                rew_sum = 0
                cnt = 0
                loss_sum = 0

if __name__ == "__main__":
    game = Game_2048()
    learner = MC_state_value(game)
    learner.train_iterations(100000000)