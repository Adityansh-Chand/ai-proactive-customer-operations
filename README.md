
# Ai Proactive Customer Operations

![Python](https://img.shields.io/badge/python-3.11-blue)
![Docker](https://img.shields.io/badge/docker-ready-blue)
![Kubernetes](https://img.shields.io/badge/kubernetes-supported-green)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

Multi-agent AI customer operations platform with proactive monitoring and automated escalation.

Production-style AI architecture demonstrating distributed AI agents, orchestration workflows, and scalable infrastructure.

---

## Architecture

```mermaid
flowchart LR

User --> API
API --> RouterAgent
RouterAgent --> DomainAgents
DomainAgents --> DecisionEngine
DecisionEngine --> ExternalSystems
DecisionEngine --> Monitoring

Monitoring --> Dashboard
```

---

## Features

• distributed AI agents  
• containerized microservices  
• evaluation pipelines  
• Kubernetes manifests  
• CI/CD workflow  
• observability ready  

---

## Demo

![demo](demo/demo.gif)

---

## Run locally

pip install -r requirements.txt

python api/server.py

---

## Docker

docker compose up --build

---

## Kubernetes

kubectl apply -f k8s/

---

## License

MIT
