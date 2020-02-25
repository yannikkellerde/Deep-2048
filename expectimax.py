import time
import math
import numpy as np
from constants import *
class Expectimax_human_heuristic():
	def __init__(self,game):
		self.game = game
		self.maxdepth = 0
	def set_state(self, state):
		self.game.state = state
	def sort_in(self,derray,einsorter,size,state):
		for g in range(len(derray)):
			if size>state[derray[g]]:
				derray.insert(g,einsorter)
				return derray
		derray.append(einsorter)
		return derray
	def eval_position(self,state):
		punkte=2000
		sizeliste=[]
		abstandh=0
		abstandv=0
		ecken=[0,3,12,15]
		obenunten=[0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15]
		bisvier=0
		hochrun=0
		bevor=0
		for a in range(len(state)):
			if state[a]==0:
				punkte+=10
			if a%4!=3 and state[a]!=0 and state[a+1]!=0:
				abstandh=state[a]-state[a+1]
				if abstandh<0:
					abstandh=-abstandh
				punkte-=abstandh*5
			if a<12 and state[a]!=0 and state[a+4]!=0:
				abstandv=state[a]-state[a+4]
				if abstandv<0:
					abstandv=-abstandv
				punkte-=abstandh*5
			
			if state[a]!=0:
				sizeliste=self.sort_in(sizeliste,a,state[a],state)
				if hochrun==0:
					if state[a]>bevor:
						hochrun=1
					if state[a]<bevor:
						hochrun=-1
				elif hochrun==1:
					if state[a]<bevor:
						punkte-=10
						hochrun=0
				elif hochrun==-1:
					if state[a]>bevor:
						punkte-=10
						hochrun=0
				bevor=state[a]
			bisvier+=1
			if bisvier>=4:
				bisvier=0
				hochrun=0
				bevor=0
		punktedafuer=10
		for a in range(4):
			if a+1>=len(sizeliste):
				break
			punkte-=(abs(sizeliste[a]%4-sizeliste[a+1]%4)+abs(int(sizeliste[a]/4)-int(sizeliste[a+1]/4)))*punktedafuer
			punktedafuer-=2
		if sizeliste[0] in ecken:
			punkte+=1000
		tome=""
		for a in sizeliste:
			tome+=str(state[a])+", "
		return punkte
	def action_node(self,depth,state):
		bestval=-1000
		if depth==self.maxdepth:
			return self.eval_position(state)
		for a in range(self.game.action_space.n):
			new_state,reward,done = self.game.check_update(state,a)
			if done:
				continue
			val=self.probabilistic_node(depth+1,new_state)
			if val>bestval:
				bestval=val
		return bestval
	def probabilistic_node(self,depth,state):
		val=0
		nullen=0
		if depth==self.maxdepth:
			return self.eval_position(state)
		for a in state:
			if a==0:
				nullen+=1
		for a in range(len(state)):
			if state[a]==0:
				new_state=state[:]
				new_state[a]=2
				val+=(self.action_node(depth+1,new_state)*0.9)/nullen
				new_state=state[:]
				new_state[a]=4
				val+=(self.action_node(depth+1,new_state)*0.1)/nullen
		return val
	def get_action(self,maxdepth=4):
		bestval=-np.inf
		bestmove=None
		self.maxdepth = maxdepth
		for a in range(self.game.action_space.n):
			new_state,reward,done = self.game.check_update(self.game.state,a)
			if done:
				continue
			val=self.probabilistic_node(0,new_state)
			if val>bestval:
				bestval=val
				bestmove=a
		return bestmove