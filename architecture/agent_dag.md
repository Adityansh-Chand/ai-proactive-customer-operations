```mermaid
flowchart TD

User --> PlannerAgent
PlannerAgent --> SentimentAgent
SentimentAgent --> PolicyAgent
PolicyAgent --> ActionAgent
ActionAgent --> ExternalSystems
```
