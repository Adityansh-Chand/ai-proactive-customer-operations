
from pathlib import Path
import subprocess

def run(cmd):
    subprocess.run(cmd, shell=True, check=True)

def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip()+"\n", encoding="utf-8")

BASE = Path.cwd()

write(BASE/"agents/planner_agent.py", """
def planner_agent(user_input):
    return {"tasks":["sentiment","policy","action"]}
""")

write(BASE/"agents/sentiment_agent.py", """
def sentiment_agent(text):
    if "complaint" in text.lower():
        return "negative"
    return "neutral"
""")

write(BASE/"agents/policy_agent.py", """
def policy_agent(context):
    if context["sentiment"]=="negative":
        return "escalate"
    return "auto_resolve"
""")

write(BASE/"agents/action_agent.py", """
def action_agent(policy):
    if policy=="escalate":
        return "create_ticket"
    return "knowledge_response"
""")

write(BASE/"orchestration/agent_dag.py", """
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
""")

write(BASE/"architecture/agent_dag.md", """
```mermaid
flowchart TD

User --> PlannerAgent
PlannerAgent --> SentimentAgent
SentimentAgent --> PolicyAgent
PolicyAgent --> ActionAgent
ActionAgent --> ExternalSystems
```
""")

write(BASE/"README.md", """
# AI Proactive Customer Operations

Explicit multi-agent DAG workflow system.

## Architecture

```mermaid
flowchart TD

User --> PlannerAgent
PlannerAgent --> SentimentAgent
SentimentAgent --> PolicyAgent
PolicyAgent --> ActionAgent
```

## Reasoning Trace Example

{
"sentiment":"negative",
"policy":"escalate",
"action":"create_ticket"
}
""")

run("git add .")

try:
    run('git commit -m "upgrade to multi-agent DAG architecture"')
except:
    pass

run("git push")

print("multi-agent DAG upgrade complete")
