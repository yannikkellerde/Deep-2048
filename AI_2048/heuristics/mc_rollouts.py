from AI_2048.env.game import Game_2048
def random_policy(game:Game_2048):
    return game.action_space.sample()
def mc_rollout(state,game:Game_2048,rollouts=1,policy=random_policy):
    cum_rews = []
    for _ in range(rollouts):
        game.state = state
        done = False
        cum_rew = 0
        while not done:
            action = random_policy(game)
            _,reward,done = game.step(action)
            cum_rew+=reward
        cum_rews.append(cum_rew)
    return sum(cum_rews)/len(cum_rews)