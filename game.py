import numpy as np
import random
import gym
from constants import *
class Game_2048(gym.Env):
    def __init__(self):
        self.max_power = 16
        self.action_space = gym.spaces.Discrete(4)
        self.state_space = self.observation_space = gym.spaces.Discrete((16*self.max_power))
        self.reset()
    def reset(self):
        self.state=np.zeros(16,dtype=np.int)
        self.spawn_number()
        self.spawn_number()
        return self.state
    def _get_free_squares(self):
        return list(filter(lambda x:self.state[x]==0,range(len(self.state))))
    def _merge_rows(self,state,from_row_indicies,to_row_indicies,merge_avaliable):
        reward = 0
        for i in range(len(to_row_indicies)):
            if state[to_row_indicies[i]]==0:
                state[to_row_indicies[i]]=state[from_row_indicies[i]]
                state[from_row_indicies[i]]=0
            elif merge_avaliable[i] and state[to_row_indicies[i]]==state[from_row_indicies[i]]:
                state[to_row_indicies[i]]=state[to_row_indicies[i]]+1
                reward+=2<<(state[to_row_indicies[i]]-1)
                state[from_row_indicies[i]]=0
                merge_avaliable[i]=False
        return reward
    def convert_to_nn_input(self,state):
        out_state = np.zeros(self.observation_space.n,dtype=np.int)
        for i in range(self.max_power):
            for j in range(16):
                out_state[i*16+j]=state[j]==i
        return out_state
    def spawn_number(self):
        free_squares = self._get_free_squares()
        choice = np.random.choice(free_squares)
        self.state[choice]=1 if random.random()<0.9 else 2
    def check_update(self,state,action):
        new_state=state.copy()
        rows = INDICES[action]
        merge_avaliable = [True]*4
        reward = 0
        for n in range(3,0,-1):
            for row_num in range(n):
                reward+=self._merge_rows(new_state,rows[row_num+1],rows[row_num],merge_avaliable)
        done = np.array_equal(new_state,state)
        return new_state,reward-done,done
    def step(self,action):
        state,reward,done=self.check_update(self.state,action)
        self.state = state
        if not done:
            self.spawn_number()
        return state,reward,done
    def random_step(self):
        aval_actions = list(range(self.action_space.n))
        while len(aval_actions)>0:
            action=random.choice(aval_actions)
            state,reward,done=self.check_update(self.state,action)
            if done:
                aval_actions.remove(action)
            else:
                self.state = state
                self.spawn_number()
                return state, reward, done
        return state, reward, done
    def get_state_expectations(self,state):
        expectations = []
        states = []
        free_squares = list(filter(lambda x:state[x]==0,range(len(state))))
        for square in free_squares:
            store_state = state.copy()
            store_state[square] = 1
            states.append(store_state)
            expectations.append(0.9/len(free_squares))
            store_state = state.copy()
            store_state[square] = 2
            states.append(store_state)
            expectations.append(0.1/len(free_squares))
        return np.array(states),np.array(expectations)
    def __str__(self):
        outstr = ""
        maxlen = max([len(str(1<<x)) for x in self.state])
        for i in range(len(self.state)):
            if i%4==0:
                outstr += "\n"
            myself = str(1<<(self.state[i])) if self.state[i]>0 else "0"
            outstr += myself+" "*(maxlen-len(myself)+1)
        return outstr

if __name__=="__main__":
    game = Game_2048()
    print(game.get_state_expectations(game.state))
    """while 1:
        direction = int(input("Enter your move"))
        state,reward,done = game.step(direction)
        print(game,reward,done)"""