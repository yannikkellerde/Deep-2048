from AI_2048.agents.agent_register import get_agent
def get_policy_func(agent_name,**kwargs):
    agent = get_agent(agent_name)()
    return lambda state:agent.get_action(state,**kwargs)