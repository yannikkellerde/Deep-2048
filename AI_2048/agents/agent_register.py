from importlib import import_module

agents = {  "Random_agent":"basic_agents",
            "Random_avoid_done":"basic_agents",
            "Greedy_biased_agent":"basic_agents",
            "Greedy_agent":"basic_agents",
            "DeepTD0":"deep_td0",
            "Expectimax_human_heuristic":"expectimax",
            "MCTS":"mcts",
            "MC_state_value":"monte_carlo_value_learning"
}
agents_path = "AI_2048.agents"
def get_agent(agent_name):
    assert agent_name in agents
    mod = import_module(agents_path+"."+agents[agent_name])
    agent_class = mod.__dict__[agent_name]
    return agent_class