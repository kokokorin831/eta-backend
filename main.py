import os
import json
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

app = FastAPI(title="ETA Multi-Agent Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Config ---
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

def get_sb() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def get_llm():
    if HAS_GEMINI and GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        return genai.GenerativeModel("gemini-2.0-flash")
    return None

# --- Pydantic Models ---
class ChatMessage(BaseModel):
    student_id: str
    message: str

class TaskCreate(BaseModel):
    student_id: str
    agent: str
    task_type: str
    title: str
    priority: str = "normal"

class ReportGenerate(BaseModel):
    student_id: str

# ============================================
# SHARED LAYER
# ============================================

def db_query(table: str, filters: dict = None, limit: int = 100):
    sb = get_sb()
    q = sb.table(table).select("*")
    if filters:
        for k, v in filters.items():
            q = q.eq(k, v)
    return q.limit(limit).execute().data

def db_insert(table: str, data: dict):
    sb = get_sb()
    return sb.table(table).insert(data).execute().data

def db_update(table: str, id_val: str, data: dict):
    sb = get_sb()
    return sb.table(table).update(data).eq("id", id_val).execute().data

def resolve_student_uuid(student_id_code: str) -> str:
    rows = db_query("students", {"student_id": student_id_code}, limit=1)
    if rows:
        return rows[0]["id"]
    return student_id_code

def log_message(student_id: str, role: str, content: str, agent: str = None):
    db_insert("message_log", {
        "student_id": resolve_student_uuid(student_id),
        "role": role,
        "content": content,
        "agent": agent,
        "channel": "chat"
    })

def call_llm(prompt: str, system_instruction: str = "") -> str:
    model = get_llm()
    if not model:
        return "[LLM not configured - set GEMINI_API_KEY]"
    try:
        full_prompt = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"[LLM Error: {str(e)}]"

def update_agent_state(agent_name: str, student_id: str, last_action: str, extra_state: dict = None):
    sb = get_sb()
    student_uuid = resolve_student_uuid(student_id)
    state = extra_state or {}
    existing = sb.table("agent_state").select("id").eq("agent_name", agent_name).eq("student_id", student_uuid).execute().data
    if existing:
        sb.table("agent_state").update({"state": json.dumps(state), "last_action": last_action, "last_active": datetime.utcnow().isoformat()}).eq("id", existing[0]["id"]).execute()
    else:
        sb.table("agent_state").insert({"agent_name": agent_name, "student_id": student_uuid, "state": json.dumps(state), "last_action": last_action}).execute()

# ============================================
# 6 AGENTS
# ============================================

AGENT_INSTRUCTIONS = {
    "lead_strategist": "You are the Lead Strategist agent for a Hong Kong JUPAS university admissions system. You help students with university selection strategy, analyzing their DSE scores against programme requirements, and optimizing their Band A/B/C choices. Always consider the student's scores, target band, and programme competitiveness.",
    "case_manager": "You are the Case Manager agent. You track student progress, manage deadlines, send reminders, and coordinate between other agents. Provide concise status updates and action items.",
    "academic_mentor": "You are the Academic Mentor agent. You advise students on extracurricular activities, academic enrichment, competitions, and experiences that strengthen university applications.",
    "essay_coach": "You are the Essay Coach agent. You help students write and refine personal statements and supplementary essays for university applications. Provide specific, actionable feedback.",
    "interview_coach": "You are the Interview Coach agent. You prepare students for university admission interviews with mock questions, feedback on answers, and tips for specific programmes.",
    "research_analyst": "You are the Research Analyst agent. You analyze admission statistics, calculate acceptance probabilities, and provide data-driven insights about programme competitiveness."
}

def get_student_context(student_id: str) -> str:
    sb = get_sb()
    student = sb.table("students").select("*").eq("student_id", student_id).execute().data
    if not student:
        return "Student not found."
    s = student[0]
    uuid = s["id"]
    grades = sb.table("grades_scores").select("*").eq("student_id", uuid).execute().data
    apps = sb.table("applications").select("*").eq("student_id", uuid).execute().data
    context = f"Student: {s.get('name_en','')} ({s.get('name_zh','')})\n"
    context += f"Grade: {s.get('grade_level','')}, Curriculum: {s.get('curriculum','')}, Target Band: {s.get('target_band','')}\n"
    if grades:
        context += "Grades:\n"
        for g in grades:
            context += f"  {g['subject']}: {g['grade']} ({g.get('exam_type','')})\n"
    if apps:
        context += "Applications:\n"
        for a in apps:
            context += f"  {a['university']} - {a['programme']} ({a.get('jupas_code','')}) Band {a.get('band','')} P{a.get('priority_rank','')} Status:{a.get('status','')}\n"
    return context

def run_agent(agent_name: str, student_id: str, user_message: str) -> str:
    instruction = AGENT_INSTRUCTIONS.get(agent_name, "You are a helpful education advisor.")
    context = get_student_context(student_id)
    prompt = f"Student Context:\n{context}\nStudent Question: {user_message}"
    response = call_llm(prompt, instruction)
    update_agent_state(agent_name, student_id, f"Responded to: {user_message[:50]}")
    log_message(student_id, "assistant", response, agent_name)
    return response

# ============================================
# ORCHESTRATOR
# ============================================

ROUTING_KEYWORDS = {
    "lead_strategist": ["strategy", "choice", "band", "jupas", "select", "university", "programme", "rank", "priority"],
    "case_manager": ["deadline", "progress", "status", "remind", "schedule", "update", "when", "timeline"],
    "academic_mentor": ["activity", "extracurricular", "competition", "volunteer", "enrichment", "experience"],
    "essay_coach": ["essay", "personal statement", "write", "draft", "feedback", "ps", "statement"],
    "interview_coach": ["interview", "mock", "question", "prepare", "practice"],
    "research_analyst": ["probability", "chance", "statistics", "data", "score", "median", "cutoff", "admission rate"]
}

def route_message(message: str) -> str:
    msg_lower = message.lower()
    scores = {}
    for agent, keywords in ROUTING_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in msg_lower)
        if score > 0:
            scores[agent] = score
    if scores:
        return max(scores, key=scores.get)
    return "case_manager"

def orchestrate(student_id: str, message: str) -> dict:
    log_message(student_id, "user", message)
    agent_name = route_message(message)
    response = run_agent(agent_name, student_id, message)
    return {"agent": agent_name, "response": response, "student_id": student_id}

# ============================================
# API ROUTES
# ============================================

@app.get("/")
def root():
    return {"status": "ok", "message": "ETA Multi-Agent Backend v2.0"}

@app.get("/health")
def health():
    sb_ok = False
    try:
        sb = get_sb()
        sb.table("students").select("id").limit(1).execute()
        sb_ok = True
    except:
        pass
    return {
        "status": "ok",
        "supabase_connected": sb_ok,
        "gemini_configured": bool(GEMINI_API_KEY),
        "version": "2.0"
    }

# --- Chat (Orchestrator entry) ---
@app.post("/api/chat/send")
def chat_send(msg: ChatMessage):
    result = orchestrate(msg.student_id, msg.message)
    return result

@app.get("/api/chat/history")
def chat_history(student_id: str, limit: int = 50):
    sb = get_sb()
    student_uuid = resolve_student_uuid(student_id)
    data = sb.table("message_log").select("*").eq("student_id", student_uuid).order("created_at", desc=False).limit(limit).execute().data
    return data

# --- Dashboard ---
@app.get("/api/dashboard/students")
def dashboard_students():
    return db_query("students")

@app.get("/api/dashboard/stats")
def dashboard_stats():
    sb = get_sb()
    students = sb.table("students").select("id").execute().data
    apps = sb.table("applications").select("id, status").execute().data
    tasks = sb.table("task_registry").select("id, status, priority").execute().data
    agents = sb.table("agent_state").select("agent_name").execute().data
    return {
        "total_students": len(students),
        "total_applications": len(apps),
        "submitted_apps": len([a for a in apps if a.get("status") == "submitted"]),
        "pending_tasks": len([t for t in tasks if t.get("status") == "pending"]),
        "urgent_tasks": len([t for t in tasks if t.get("priority") == "urgent"]),
        "active_agents": len(set(a["agent_name"] for a in agents))
    }

# --- Students ---
@app.get("/api/students")
def list_students():
    return db_query("students")

@app.get("/api/students/{student_id}")
def get_student(student_id: str):
    sb = get_sb()
    data = sb.table("students").select("*").eq("student_id", student_id).execute().data
    if not data:
        data = sb.table("students").select("*").eq("id", student_id).execute().data
    if not data:
        raise HTTPException(404, "Student not found")
    student = data[0]
    uuid = student["id"]
    student["grades"] = db_query("grades_scores", {"student_id": uuid})
    student["applications"] = db_query("applications", {"student_id": uuid})
    student["essays"] = db_query("essay_drafts", {"student_id": uuid})
    student["interviews"] = db_query("interview_sessions", {"student_id": uuid})
    student["tasks"] = db_query("task_registry", {"student_id": uuid})
    return student

# --- Applications ---
@app.get("/api/applications")
def list_applications(student_id: str = None):
    if student_id:
        uuid = resolve_student_uuid(student_id)
        return db_query("applications", {"student_id": uuid})
    return db_query("applications")

# --- Tasks ---
@app.get("/api/tasks/urgent")
def urgent_tasks():
    sb = get_sb()
    data = sb.table("task_registry").select("*, students(name_en, student_id)").in_("status", ["pending", "in_progress"]).eq("priority", "urgent").execute().data
    return data

@app.get("/api/tasks")
def list_tasks(student_id: str = None, status: str = None):
    sb = get_sb()
    q = sb.table("task_registry").select("*")
    if student_id:
        uuid = resolve_student_uuid(student_id)
        q = q.eq("student_id", uuid)
    if status:
        q = q.eq("status", status)
    return q.order("created_at", desc=True).execute().data

@app.post("/api/tasks")
def create_task(task: TaskCreate):
    data = task.dict()
    data["student_id"] = resolve_student_uuid(data["student_id"])
    return db_insert("task_registry", data)

@app.patch("/api/tasks/{task_id}")
def update_task(task_id: str, status: str):
    return db_update("task_registry", task_id, {"status": status, "completed_at": datetime.utcnow().isoformat() if status == "completed" else None})

# --- Reports ---
@app.get("/api/reports")
def list_reports():
    return db_query("report_snapshots")

@app.post("/api/reports/generate")
def generate_report(req: ReportGenerate):
    context = get_student_context(req.student_id)
    prompt = f"Generate a comprehensive JUPAS admissions report for this student. Include strategy analysis, probability assessment, and recommended actions.\n\n{context}"
    report_content = call_llm(prompt, "You are an expert JUPAS admissions analyst. Generate a detailed student report.")
    student_uuid = resolve_student_uuid(req.student_id)
    report = db_insert("report_snapshots", {
        "student_id": student_uuid,
        "report_type": "jupas_analysis",
        "title": f"JUPAS Analysis - {datetime.utcnow().strftime('%Y-%m-%d')}",
        "content": {"text": report_content},
        "generated_by": "research_analyst"
    })
    update_agent_state("research_analyst", req.student_id, "Generated JUPAS analysis report")
    return report

# --- Agents status (for dashboard) ---
@app.get("/api/agents")
def list_agents():
    sb = get_sb()
    states = sb.table("agent_state").select("*").execute().data
    agent_map = {}
    for s in states:
        name = s["agent_name"]
        if name not in agent_map:
            agent_map[name] = {"id": name, "name": name.replace("_", " ").title(), "status": "running", "students_served": 0, "last_action": s.get("last_action", ""), "last_active": s.get("last_active", "")}
        agent_map[name]["students_served"] += 1
    all_agents = ["lead_strategist", "case_manager", "academic_mentor", "essay_coach", "interview_coach", "research_analyst"]
    result = []
    for a in all_agents:
        if a in agent_map:
            result.append(agent_map[a])
        else:
            result.append({"id": a, "name": a.replace("_", " ").title(), "status": "idle", "students_served": 0, "last_action": "", "last_active": ""})
    return result

# --- Logs ---
@app.get("/api/logs")
def list_logs(level: str = None, agent: str = None, limit: int = 100):
    sb = get_sb()
    q = sb.table("message_log").select("*")
    if agent:
        q = q.eq("agent", agent)
    return q.order("created_at", desc=True).limit(limit).execute().data

# --- Config ---
@app.get("/api/config")
def get_config():
    return db_query("system_config")

@app.put("/api/config/{key}")
def update_config(key: str, value: dict):
    sb = get_sb()
    existing = sb.table("system_config").select("key").eq("key", key).execute().data
    if existing:
        return sb.table("system_config").update({"value": json.dumps(value), "updated_at": datetime.utcnow().isoformat()}).eq("key", key).execute().data
    return db_insert("system_config", {"key": key, "value": json.dumps(value)})
