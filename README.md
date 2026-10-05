# Healthcare LLM Pipeline with Guardrails, DeepEval & LangSmith

This repository contains a role-aware clinical LLM service (supporting Physician, Nurse, and Patient personas) built with FastAPI and OpenAI. It includes:
- **Low-Standard Guardrails**: MRN regex redaction and automatic disclaimer insertion for patient queries.
- **Observability**: Real-time tracing of prompts, guardrail interventions, latency, and tokens with LangSmith.
- **Evaluation Gate (CI)**: Automated quality and relevancy evaluations using DeepEval in Azure DevOps.
- **Continuous Deployment (CD)**: Automated container build and release to Azure Container Apps.

---

## Repository Structure

```text
├── app/
│   ├── __init__.py
│   ├── hospital_assistant.py  # Guardrails, OpenAI client, LangSmith @traceable
│   └── main.py                # FastAPI production endpoint
├── tests/
│   ├── __init__.py
│   └── test_hospital_evals.py # DeepEval test suite
├── Dockerfile
├── requirements.txt
├── azure-pipelines.yml        # CI/CD multi-stage Azure DevOps pipeline
├── .env.example
└── README.md
```

---

## Local Setup

1. **Clone repository & set up virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Populate OPENAI_API_KEY and LANGCHAIN_API_KEY in .env
   ```

3. **Run DeepEval tests locally**:
   ```bash
   pytest tests/test_hospital_evals.py -v
   ```

4. **Run FastAPI locally**:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
   Visit `http://localhost:8000/docs` to test endpoints.
