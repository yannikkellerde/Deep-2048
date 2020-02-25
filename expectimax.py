import time
import math
import numpy as np
from constants import *
from game import Game_2048
from human_heuristic import human_heuristic
class Expectimax_human_heuristic():
	def __init__(self,game:Game_2048,heuristic=human_heuristic):
		self.game = game
		self.heuristic = heuristic
	def set_state(self, state):
		self.game.state = state
	def get_action(self,maxdepth=1):
		self.maxdepth = maxdepth
		return self.eval_state(self.game.state,0)[1]
	def eval_state(self,state,depth):
		if depth>self.maxdepth:
			return self.heuristic(state),None
		bestval = -1001
		best_action = None
		for a in range(self.game.action_space.n):
			new_state,reward,done = self.game.check_update(state,a)
			if not done:
				val = 0
				new_states,probs = self.game.get_state_expectations(new_state)
				for i in range(len(new_states)):
					val += self.eval_state(new_states[i],depth+1)[0]*probs[i]
				if val > bestval:
					bestval = val
					best_action = a
		return bestval,best_action

if __name__ == "__main__":
    game = Game_2048()
    expectimax = Expectimax_human_heuristic(game,human_heuristic)
    print(game)
    while 1:
        expectimax.set_state(game.state)
        action = expectimax.get_action(1)
        if action is None:
            break
        game.step(action)
        print(game)