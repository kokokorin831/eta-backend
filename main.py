import os
import random
from datetime import datetime, timedelta
from typing import Optional
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="ETA Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---
class AgentAction(BaseModel):
    agent_id: Optional[str] = None

class ConfigUpdate(BaseModel):
    supabase_url: Optional[str] = None
    gemini_key: Optional[str] = None
    max_concurrent: Optional[int] = 5

class ReportGenerate(BaseModel):
    student_id: str

# --- State ---
agents_state = {
    "jupas_matcher": {"id": "jupas_matcher", "name": "JUPAS Matcher", "status": "running", "description": "Programme matching & scoring", "cpu": 42, "memory": 61, "last_active": "2 min ago"},
    "transcript_analyzer": {"id": "transcript_analyzer", "name": "Transcript Analyzer", "status": "running", "description": "DSE result parsing & analysis", "cpu": 28, "memory": 45, "last_active": "5 min ago"},
    "report_generator": {"id": "report_generator", "name": "Report Generator", "status": "stopped", "description": "PDF report generation", "cpu": 0, "memory": 12, "last_active": "1 hour ago"},
    "data_collector": {"id": "data_collector", "name": "Data Collector", "status": "running", "description": "University data scraping", "cpu": 15, "memory": 33, "last_active": "30 sec ago"},
    "scoring_engine": {"id": "scoring_engine", "name": "Scoring Engine", "status": "stopped", "description": "Admission probability calculation", "cpu": 0, "memory": 8, "last_active": "3 hours ago"},
    "notification_agent": {"id": "notification_agent", "name": "Notification Agent", "status": "stopped", "description": "Email & alert delivery", "cpu": 0, "memory": 5, "last_active": "6 hours ago"},
}

config_state = {"supabase_url": "", "gemini_key": "", "max_concurrent": 5}

# --- Root & Health ---
@app.get("/")
def root():
    return {"status": "ok", "message": "ETA Backend API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/api/test")
def test_endpoint():
    return {"data": "Hello from ETA Backend!", "version": "1.0.0"}

# --- Agents ---
@app.get("/api/agents")
def get_agents():
    return list(agents_state.values())

@app.post("/api/agents/start")
def start_all_agents():
    for a in agents_state.values():
        a["status"] = "running"
        a["cpu"] = random.randint(10, 60)
    return {"status": "ok", "message": "All agents started"}

@app.post("/api/agents/{agent_id}/start")
def start_agent(agent_id: str):
    if agent_id in agents_state:
        agents_state[agent_id]["status"] = "running"
        return {"status": "ok"}
    return {"status": "error", "message": "Agent not found"}

@app.post("/api/agents/{agent_id}/stop")
def stop_agent(agent_id: str):
    if agent_id in agents_state:
        agents_state[agent_id]["status"] = "stopped"
        agents_state[agent_id]["cpu"] = 0
        return {"status": "ok"}
    return {"status": "error", "message": "Agent not found"}

@app.post("/api/agents/{agent_id}/restart")
def restart_agent(agent_id: str):
    if agent_id in agents_state:
        agents_state[agent_id]["status"] = "running"
        return {"status": "ok"}
    return {"status": "error", "message": "Agent not found"}

# --- Logs ---
@app.get("/api/logs")
def get_logs(agent: Optional[str] = None, level: Optional[str] = None, search: Optional[str] = None, since: Optional[str] = None):
    levels = ["info", "warn", "error", "debug"]
    agent_names = ["JUPAS Matcher", "Transcript Analyzer", "Report Generator", "Data Collector", "Scoring Engine"]
    messages = [
        "Processing student transcript batch",
        "HTTP GET /api/universities - 200 OK",
        "Queue depth growing: 45 pending tasks",
        "Failed to parse DSE transcript: invalid format",
        "Authentication token expired for service account",
        "GC pause: 12ms (minor collection)",
        "Entering matchStudentProgrammes() with 6 candidates",
        "Cache hit ratio: 87.3%",
        "Connection timeout to upstream service (30s)",
        "Report generated successfully for student S2024001",
    ]
    logs = []
    for i in range(100):
        log_level = random.choice(levels)
        log_agent = random.choice(agent_names)
        if agent and agent.lower() not in log_agent.lower():
            continue
        if level and level.lower() != log_level:
            continue
        log = {
            "id": f"log_{i}",
            "timestamp": (datetime.now() - timedelta(minutes=i*2)).isoformat(),
            "level": log_level,
            "agent": log_agent,
            "message": random.choice(messages),
        }
        if search and search.lower() not in log["message"].lower():
            continue
        logs.append(log)
    return logs

# --- Reports ---
@app.get("/api/reports")
def get_reports():
    return [
        {"id": "r1", "student_id": "S2024001", "student_name": "Chan Tai Man", "created_at": "2026-03-04T10:00:00", "status": "completed"},
        {"id": "r2", "student_id": "S2024002", "student_name": "Wong Siu Ming", "created_at": "2026-03-04T09:30:00", "status": "completed"},
        {"id": "r3", "student_id": "S2024003", "student_name": "Lee Ka Yi", "created_at": "2026-03-04T08:00:00", "status": "pending"},
    ]

@app.post("/api/reports/generate")
def generate_report(body: ReportGenerate):
    return {"status": "ok", "report_id": "r_new", "message": f"Report generation started for {body.student_id}"}

@app.get("/api/reports/{report_id}/pdf")
def get_report_pdf(report_id: str):
    return {"url": f"https://example.com/reports/{report_id}.pdf"}

# --- Config ---
@app.get("/api/config")
def get_config():
    return config_state

@app.put("/api/config")
def update_config(body: ConfigUpdate):
    if body.supabase_url is not None:
        config_state["supabase_url"] = body.supabase_url
    if body.gemini_key is not None:
        config_state["gemini_key"] = body.gemini_key
    if body.max_concurrent is not None:
        config_state["max_concurrent"] = body.max_concurrent
    return {"status": "ok", "config": config_state}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
