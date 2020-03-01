from AI_2048.env.game import Game_2048
import random
import math
import numpy as np
from AI_2048.agents.base import Agent
class Random_agent(Agent):
    def __init__(self,game=Game_2048()):
        self.game = game
    def get_action(self,state):
        return self.game.action_space.sample()
    def __str__(self):
        return "Random agent"
class Random_avoid_done(Agent):
    def __init__(self,game=Game_2048()):
        self.game = game
    def get_action(self,state):
        aval_actions = []
        for a in range(self.game.action_space.n):
            _,_,done = self.game.check_update(self.game.state,a)
            if not done:
                aval_actions.append(a)
        if len(aval_actions)>0:
            return random.choice(aval_actions)
        else:
            return self.game.action_space.sample()
    def __str__(self):
        return "Random avoid done agent"
class Greedy_biased_agent(Agent):
    def __init__(self,game=Game_2048()):
        self.game = game
    def get_action(self,state):
        best_a = None
        best_val = -np.inf
        for a in range(self.game.action_space.n):
            _,reward,_ = self.game.check_update(self.game.state,a)
            if reward>best_val:
                best_val = reward
                best_a = a
        return best_a
    def __str__(self):
        return "Greedy biased agent"
class Greedy_agent(Agent):
    def __init__(self,game=Game_2048()):
        self.game = game
    def get_action(self,state):
        best_a = []
        best_val = -np.inf
        for a in range(self.game.action_space.n):
            _,reward,_ = self.game.check_update(self.game.state,a)
            if reward>=best_val:
                if reward>best_val:
                    best_a = [a]
                else:
                    best_a.append(a)
                best_val = reward
        return random.choice(best_a)
    def __str__(self):
        return "Greedy agent"

def rollout_agent(agent,rollouts):
    rewlist = []
    step_list = []
    for i in range(rollouts):
        done = False
        rew = 0
        step_count = 0
        state = agent.game.reset()
        while not done:
            a = agent.get_action(state)
            state,reward,done = agent.game.step(a)
            rew+=reward
            step_count+=1
        rewlist.append(rew)
        step_list.append(step_count)
    return sum(rewlist)/len(rewlist), sum(step_list)/len(step_list)
def calc_rollout_variances(agent,rollouts,step_size):
    done = False
    states = []
    state = agent.game.reset()
    while not done:
        a = agent.get_action(state)
        state,_,done = agent.game.step(a)
        states.append(state)
    for i in range(0,len(states),step_size):
        cum_rews = []
        for _ in range(rollouts):
            agent.game.state = states[i].copy()
            cum_rew = 0
            done=False
            while not done:
                a = agent.get_action(state)
                state,reward,done = agent.game.step(a)
                cum_rew+=reward
            cum_rews.append(cum_rew)
        mean = sum(cum_rews)/len(cum_rews)
        std = math.sqrt((1/len(cum_rews))*sum([(x-mean)**2 for x in cum_rews]))
        print(f"State number: {i}, Mean reward: {mean}, STD: {std}")
if __name__=="__main__":
    game = Game_2048()
    greedy = Greedy_agent(game)
    biased_greedy = Greedy_biased_agent(game)
    random_agent = Random_agent(game)
    avoid_done = Random_avoid_done(game)
    agents = [greedy,biased_greedy,random_agent,avoid_done]
    """for agent in agents:
        reward,steps = rollout_agent(agent,1000)
        print(f"Agent: {agent}, avg reward:{reward}, avg steps:{steps}")"""
    calc_rollout_variances(biased_greedy,100,100)