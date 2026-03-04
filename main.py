import os
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

app = FastAPI(title="ETA Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Supabase Client ---
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
supabase: Client = None

def get_sb() -> Client:
    global supabase
    if supabase is None and SUPABASE_URL and SUPABASE_KEY:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return supabase

# --- Models ---
class AgentAction(BaseModel):
    agent_id: Optional[str] = None

class ConfigUpdate(BaseModel):
    supabase_url: Optional[str] = None
    gemini_key: Optional[str] = None
    max_concurrent: Optional[int] = 5

class ReportGenerate(BaseModel):
    student_id: str

# --- Root & Health ---
@app.get("/")
def root():
    return {"status": "ok", "message": "ETA Backend API is running"}

@app.get("/health")
def health():
    sb = get_sb()
    return {"status": "healthy", "supabase_connected": sb is not None}

@app.get("/api/test")
def test_endpoint():
    return {"data": "Hello from ETA Backend!", "version": "2.0.0"}

# --- Agents ---
@app.get("/api/agents")
def get_agents():
    sb = get_sb()
    if not sb:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    res = sb.table("eta_agents").select("*").execute()
    return res.data

@app.post("/api/agents/{agent_id}/start")
def start_agent(agent_id: str):
    sb = get_sb()
    if not sb:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    sb.table("eta_agents").update({"status": "running", "updated_at": datetime.utcnow().isoformat()}).eq("id", agent_id).execute()
    return {"status": "ok", "agent_id": agent_id, "new_status": "running"}

@app.post("/api/agents/{agent_id}/stop")
def stop_agent(agent_id: str):
    sb = get_sb()
    if not sb:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    sb.table("eta_agents").update({"status": "stopped", "cpu": 0, "updated_at": datetime.utcnow().isoformat()}).eq("id", agent_id).execute()
    return {"status": "ok", "agent_id": agent_id, "new_status": "stopped"}

@app.post("/api/agents/{agent_id}/restart")
def restart_agent(agent_id: str):
    sb = get_sb()
    if not sb:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    sb.table("eta_agents").update({"status": "running", "updated_at": datetime.utcnow().isoformat()}).eq("id", agent_id).execute()
    return {"status": "ok", "agent_id": agent_id, "new_status": "running"}

@app.post("/api/agents/start-all")
def start_all_agents():
    sb = get_sb()
    if not sb:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    sb.table("eta_agents").update({"status": "running", "updated_at": datetime.utcnow().isoformat()}).neq("status", "running").execute()
    return {"status": "ok", "message": "All agents started"}

# --- Logs ---
@app.get("/api/logs")
def get_logs(level: Optional[str] = None, agent: Optional[str] = None, search: Optional[str] = None):
    sb = get_sb()
    if not sb:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    query = sb.table("eta_logs").select("*").order("timestamp", desc=True).limit(100)
    if level:
        query = query.eq("level", level)
    if agent:
        query = query.eq("agent", agent)
    if search:
        query = query.ilike("message", f"%{search}%")
    res = query.execute()
    return res.data

# --- Reports ---
@app.get("/api/reports")
def get_reports():
    sb = get_sb()
    if not sb:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    res = sb.table("eta_reports").select("*").order("created_at", desc=True).execute()
    return res.data

@app.post("/api/reports/generate")
def generate_report(body: ReportGenerate):
    sb = get_sb()
    if not sb:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    import uuid
    report_id = f"r_{uuid.uuid4().hex[:8]}"
    sb.table("eta_reports").insert({"id": report_id, "student_id": body.student_id, "status": "pending"}).execute()
    return {"status": "ok", "report_id": report_id, "message": f"Report generation started for {body.student_id}"}

@app.get("/api/reports/{report_id}/pdf")
def get_report_pdf(report_id: str):
    sb = get_sb()
    if not sb:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    res = sb.table("eta_reports").select("pdf_url").eq("id", report_id).execute()
    if res.data:
        return {"url": res.data[0].get("pdf_url", f"https://example.com/reports/{report_id}.pdf")}
    return {"url": f"https://example.com/reports/{report_id}.pdf"}

# --- Config ---
@app.get("/api/config")
def get_config():
    sb = get_sb()
    if not sb:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    res = sb.table("eta_config").select("*").execute()
    config = {row["key"]: row["value"] for row in res.data}
    return config

@app.put("/api/config")
def update_config(body: ConfigUpdate):
    sb = get_sb()
    if not sb:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    if body.supabase_url is not None:
        sb.table("eta_config").upsert({"key": "supabase_url", "value": body.supabase_url}).execute()
    if body.gemini_key is not None:
        sb.table("eta_config").upsert({"key": "gemini_key", "value": body.gemini_key}).execute()
    if body.max_concurrent is not None:
        sb.table("eta_config").upsert({"key": "max_concurrent", "value": str(body.max_concurrent)}).execute()
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
