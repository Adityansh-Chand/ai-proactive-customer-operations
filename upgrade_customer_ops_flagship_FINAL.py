
from pathlib import Path
import subprocess

def run(cmd):
    subprocess.run(cmd, shell=True, check=True)

def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip()+"\n", encoding="utf-8")

BASE = Path.cwd()

write(BASE/"README.md", """
# AI Proactive Customer Operations

Multi-agent AI orchestration system for proactive CX automation.

## Architecture

```mermaid
flowchart LR

User --> IntentClassifier
IntentClassifier --> RouterAgent
RouterAgent --> DomainAgents
DomainAgents --> ReasoningEngine
ReasoningEngine --> PolicyEngine
PolicyEngine --> ActionExecutor
ActionExecutor --> CRM
ActionExecutor --> Monitoring
```
""")

write(BASE/"agents/router.py","""
def route(intent):

    routes = {
        "refund":"returns_agent",
        "delay":"delivery_agent",
        "complaint":"support_agent",
        "track":"tracking_agent"
    }

    for k,v in routes.items():
        if k in intent.lower():
            return v

    return "support_agent"
""")

write(BASE/"agents/reasoning_engine.py","""
def decide(context):

    if context["sentiment"]=="negative":
        return "escalate"

    if context["delay_risk"]:
        return "offer_credit"

    return "auto_resolve"
""")

run("git add .")
run('git commit -m "flagship multi-agent architecture with mermaid diagram"')
run("git push")
print("customer ops upgraded")
