from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from AI_2048.agents.base import Agent
from AI_2048.agents.mcts import MCTS
import math
import numpy as np
import time
from AI_2048.util.constants import *
class Web_interface():
    def __init__(self,agent:Agent):
        self.driver = webdriver.Firefox()
        self.driver.get('https://gabrielecirulli.github.io/2048/')
        self.body = self.driver.find_element_by_tag_name("body")
        self.agent = agent
    def get_board(self):
        self.board=np.zeros(16,dtype=int)
        boardelems = self.driver.find_element_by_class_name('tile-container').get_attribute('innerHTML')
        splited=boardelems.split('"')
        for a in splited:
            if a.startswith("tile "):
                innersplit=a.split(" ")
                possplit=innersplit[2].split("-")
                numbersplit=innersplit[1].split("-")
                self.board[(int(possplit[3])-1)*4+int(possplit[2])-1]=int(math.log(int(numbersplit[1]),2))
    def step(self):
        self.agent.set_state(self.board)
        action=self.agent.get_action(self.agent.game.state,0.1)
        if action==RIGHT:
            self.body.send_keys(Keys.ARROW_RIGHT)
        if action==BOTTOM:
            self.body.send_keys(Keys.ARROW_DOWN)
        if action==LEFT:
            self.body.send_keys(Keys.ARROW_LEFT)
        if action==TOP:
            self.body.send_keys(Keys.ARROW_UP)
    def run(self):
        while(1):
            self.get_board()
            self.step()
            time.sleep(0.2)
if __name__ == "__main__":
    agent=MCTS()
    web = Web_interface(agent)
    web.run()