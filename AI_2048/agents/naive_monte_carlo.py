from AI_2048.env.game import Game_2048
import random
import math
import numpy as np
from AI_2048.agents.base import Agent
from AI_2048.heuristics.mc_rollouts import mc_rollout
class Naive_monte_carlo(Agent):
    def __init__(self,game=Game_2048()):
        self.game = game
        self.rollout_game = Game_2048()
    def get_action(self,state,rollouts = 50):
        aval_actions = []
        best_val = -np.inf
        best_a = None
        for a in range(self.game.action_space.n):
            new_state,reward,done = self.game.check_update(self.game.state,a)
            if not done:
                val = reward+mc_rollout(new_state,self.rollout_game,rollouts)
            else:
                val = reward
            if val>best_val:
                best_a = a
                best_val = val
        return best_a
    def __str__(self):
        return "Naive monte carlo agent"
if __name__ == "__main__":
    agent = Naive_monte_carlo()
    done = False
    state = agent.game.reset()
    print(agent.game)
    while not done:
        action = agent.get_action(state)
        _,_,done = agent.game.step(action)
        print(agent.game)