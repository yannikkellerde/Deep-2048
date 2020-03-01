from AI_2048.env.game import Game_2048
class Agent:
    def __init__(self,game=Game_2048()):
        self.game = game
    def get_action(self, state):
        return NotImplementedError("")