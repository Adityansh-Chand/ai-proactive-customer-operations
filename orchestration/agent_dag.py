from agents.planner_agent import planner_agent
from agents.sentiment_agent import sentiment_agent
from agents.policy_agent import policy_agent
from agents.action_agent import action_agent

def run_dag(user_input):

    trace={}

    trace["planner"]=planner_agent(user_input)

    trace["sentiment"]=sentiment_agent(user_input)

    context={
        "intent":user_input,
        "sentiment":trace["sentiment"]
    }

    trace["policy"]=policy_agent(context)

    trace["action"]=action_agent(trace["policy"])

    return trace
