import tensorflow as tf
import os,sys
import numpy as np
import random
import time
from shutil import rmtree
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential,clone_model
from game import Game_2048
class DeepTD0():
    def __init__(self,game:Game_2048):
        #Hyperparams
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 0
        self.epsilon_decay = 0
        self.epsilon_min = 0.05
        self.batch_size = 64
        self.soft_update_rate = 0.01
        self.train_start = 1000
        #most simple replay memory
        self.memory = deque(maxlen=10000)

        self.state_size = game.observation_space.n
        self.action_size = game.action_space.n
        self.game = game
        self.model = self.build_model()
        self.target_model = clone_model(self.model)
    def build_model(self):
        model = Sequential()
        model.add(Dense(124, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(124, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(1, activation='linear',
                        kernel_initializer='he_uniform'))
        print(model.summary())
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
    def hard_target_update(self):
        self.target_model.set_weights(self.model.get_weights())
    def soft_target_update(self):
        W = self.model.get_weights()
        tgt_W = self.target_model.get_weights()
        for i in range(len(W)):
            tgt_W[i] = self.soft_update_rate * W[i] + (1 - self.soft_update_rate) * tgt_W[i]
        self.target_model.set_weights(tgt_W)
    def epsilon_greedy(self, state):
        if np.random.rand() <= self.epsilon:
            avaliable =  []
            for a in range(self.action_size):
                _,_,done = self.game.check_update(state.copy(),a)
                if not done:
                    avaliable.append(a)
            if len(avaliable)>0:
                return random.choice(avaliable)
            else:
                return random.randrange(self.action_size)
        else:
            best_val = -np.inf
            best_a = None
            for a in range(self.action_size):
                new_state,reward,done = self.game.check_update(state.copy(),a)
                if done:
                    val = reward
                else:
                    val = self._get_state_expected_value(new_state,self.model)
                    val = reward+self.discount_factor*val
                if  val > best_val:
                    best_val = val
                    best_a = a
            return best_a
    def _get_state_expected_value(self,state,model):
        value = 0
        states,probs = self.game.get_state_expectations(state)
        nn_input = np.array([self.game.convert_to_nn_input(s) for s in states])
        predictions = model.predict(nn_input).reshape(len(nn_input))
        weighted = predictions*probs
        return weighted.sum()
    def _get_target_value(self,state):
        best_value = -np.inf
        for a in range(self.action_size):
            state,reward,invalid = self.game.check_update(state.copy(),a)
            if invalid:
                value = reward
            else:
                value = self._get_state_expected_value(state,self.target_model)*self.discount_factor+reward
            if value>best_value:
                best_value = value
        return best_value
    def append_sample(self,state):
        self.memory.append(state)
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def train_one_batch(self):
        if len(self.memory) < self.train_start:
            return 0
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = np.array(random.sample(self.memory, batch_size))
        update_in = np.array([self.game.convert_to_nn_input(state) for state in mini_batch])
        target = np.array([self._get_target_value(state) for state in mini_batch])
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
        for i in range(iterations):
            action = self.epsilon_greedy(state)
            print(self.game)
            new_state,reward,done,_info = self.game.step(action)
            rew_sum += reward
            self.append_sample(new_state)
            loss = self.train_one_batch()
            self.soft_target_update()
            loss_sum += loss
            cnt+=1
            if done:
                state=self.game.reset()
                nn_input = self.game.convert_to_nn_input(state).reshape(1,-1)
                predict_mod = self._get_state_expected_value(state,self.model)
                predict_targ = self._get_state_expected_value(state,self.target_model)
                self.append_sample(state)
                if len(self.memory) > self.train_start:
                    self.decay_epsilon()
                    print(f"Iteration: {i}, Reward sum: {rew_sum}, cnt: {cnt}, avg loss: {loss_sum/cnt:.3f}, eps: {self.epsilon:.3f}")
                    with train_writer.as_default():
                        tf.summary.scalar('reward', rew_sum, step=i)
                        tf.summary.scalar('avg loss', loss_sum/cnt, step=i)
                        tf.summary.scalar('predict_mod', predict_mod, step=i)
                        tf.summary.scalar('predict_targ', predict_targ, step=i)
                rew_sum = 0
                loss_sum = 0
                cnt = 0
if __name__ == '__main__':
    learner = DeepTD0(Game_2048())
    #learner.train_iterations(100000)