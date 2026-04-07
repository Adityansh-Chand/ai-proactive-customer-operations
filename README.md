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
