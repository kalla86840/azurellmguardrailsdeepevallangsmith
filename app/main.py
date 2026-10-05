import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
from app.hospital_assistant import ask_hospital_assistant

app = FastAPI(
    title="Hospital LLM Service",
    description="Role-aware clinical assistant with guardrails and LangSmith tracing",
    version="1.0.0"
)

class AssistantRequest(BaseModel):
    prompt: str
    role: Literal["physician", "nurse", "patient"] = "patient"

class AssistantResponse(BaseModel):
    role: str
    response: str

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/api/v1/chat", response_model=AssistantResponse)
def chat_endpoint(payload: AssistantRequest):
    try:
        output = ask_hospital_assistant(prompt=payload.prompt, role=payload.role)
        return AssistantResponse(role=payload.role, response=output)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
