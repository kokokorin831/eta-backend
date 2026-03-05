import os
import json
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

try:
    from google import genai as google_genai
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

def get_llm_client():
    if HAS_GEMINI and GEMINI_API_KEY:
        return google_genai.Client(api_key=GEMINI_API_KEY)
    return None

# --- Load Skill Files ---
SKILLS_DIR = os.path.join(os.path.dirname(__file__), "skills")

def load_skill(agent_name: str) -> str:
    skill_path = os.path.join(SKILLS_DIR, f"{agent_name}_skill.md")
    try:
        with open(skill_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"You are the {agent_name.replace('_', ' ').title()} agent for the ETA JUPAS Admissions Advisory System."

AGENT_NAMES = ["lead_strategist", "case_manager", "academic_mentor", "essay_coach", "interview_coach", "research_analyst"]
AGENT_INSTRUCTIONS = {name: load_skill(name) for name in AGENT_NAMES}

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
    client = get_llm_client()
    if not client:
        return "[LLM not configured - set GEMINI_API_KEY]"
    try:
        full_prompt = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt
        )
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
                # Add programme matching data
    try:
        total_score = sum(g.get("score", 0) or 0 for g in grades)
        if total_score > 0:
            progs = sb.table("programmes_hk").select("jupas_code, university_name, programme_name, median_score, lq_score").execute().data
            matched = []
            for p in progs:
                median = p.get("median_score") or 0
                if median > 0 and total_score >= (p.get("lq_score") or 0):
                    chance = 80 if total_score >= median else 50
                    matched.append((p, chance))
            matched.sort(key=lambda x: -x[1])
            if matched:
                context += f"\nJUPAS Score: {total_score}\n"
                context += "Top Programme Matches (from 416 real JUPAS programmes):\n"
                for p, chance in matched[:10]:
                    context += f"  {p['jupas_code']} {p['university_name']} - {p['programme_name']} (Median:{p.get('median_score','N/A')}) Chance:{chance}%\n"
    except:
        pass

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
    return {"status": "ok", "message": "ETA Multi-Agent Backend v2.1", "skills_loaded": list(AGENT_INSTRUCTIONS.keys())}

@app.get("/health")
def health():
    sb_ok = False
    try:
        sb = get_sb()
        sb.table("students").select("id").limit(1).execute()
        sb_ok = True
    except:
        pass
    skills_loaded = {name: len(content) > 100 for name, content in AGENT_INSTRUCTIONS.items()}
    return {
        "status": "ok",
        "supabase_connected": sb_ok,
        "gemini_configured": bool(GEMINI_API_KEY),
        "skills_loaded": skills_loaded,
        "version": "2.1"
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
    sb = get_sb()
    reports = sb.table("eta_reports").select("*").order("created_at", desc=True).execute().data
    result = []
    for r in reports:
        student = sb.table("students").select("student_id, name_en, target_band").eq("student_id", r.get("student_id", "")).execute().data
        s = student[0] if student else {}
        apps = sb.table("applications").select("university, programme").eq("student_id", s.get("id")).limit(3).execute().data if student and s.get("id") else []
        top_matches = [{"university": a["university"], "programme": a["programme"], "chance": 70} for a in apps]
        band_val = s.get("target_band", "A")
        band_num = 1 if band_val in ["A", "1"] else (2 if band_val in ["B", "2"] else 3)
        result.append({
            "id": r["id"],
            "studentId": r.get("student_id", ""),
            "studentName": r.get("student_name", s.get("name_en", "")),
            "school": "",
            "band": band_num,
            "jupasScore": 0,
            "topMatches": top_matches,
            "status": "generated" if r.get("status") == "completed" else "pending",
            "generatedAt": r.get("completed_at") or r.get("created_at", "")
        })
    return result

@app.post("/api/reports/generate")
def generate_report(req: ReportGenerate):
    sb = get_sb()
    # Look up student info
    student_rows = sb.table("students").select("*").eq("student_id", req.student_id).execute().data
    if not student_rows:
        student_rows = sb.table("students").select("*").eq("id", req.student_id).execute().data
    if not student_rows:
        raise HTTPException(404, "Student not found")
    s = student_rows[0]
    context = get_student_context(req.student_id)
    prompt = f"Generate a comprehensive JUPAS admissions report for this student. Include strategy analysis, probability assessment, and recommended actions.\n\n{context}"
    report_content = call_llm(prompt, "You are an expert JUPAS admissions analyst. Generate a detailed student report.")
    # Insert into eta_reports
    report = db_insert("eta_reports", {
        "student_id": s.get("student_id", req.student_id),
        "student_name": s.get("name_en", ""),
        "status": "completed",
        "completed_at": datetime.utcnow().isoformat()
    })
    update_agent_state("research_analyst", req.student_id, "Generated JUPAS analysis report")
    report_id = report[0]["id"] if report else "unknown"
    return {"success": True, "reportId": report_id}
# --- Agents status (for dashboard) ---
@app.get("/api/agents")
def list_agents():
    sb = get_sb()
    states = sb.table("agent_state").select("*").execute().data
    agent_map = {}
    for s in states:
        name = s["agent_name"]
        if name not in agent_map:
            agent_map[name] = {"id": name, "name": name.replace("_", " ").title(), "status": "running", "students_served": 0, "last_action": s.get("last_action", ""), "last_active": s.get("last_active", ""), "skill_loaded": name in AGENT_INSTRUCTIONS and len(AGENT_INSTRUCTIONS.get(name, "")) > 100}
        agent_map[name]["students_served"] += 1
    all_agents = ["lead_strategist", "case_manager", "academic_mentor", "essay_coach", "interview_coach", "research_analyst"]
    result = []
    for a in all_agents:
        if a in agent_map:
            result.append(agent_map[a])
        else:
            result.append({"id": a, "name": a.replace("_", " ").title(), "status": "idle", "students_served": 0, "last_action": "", "last_active": "", "skill_loaded": a in AGENT_INSTRUCTIONS and len(AGENT_INSTRUCTIONS.get(a, "")) > 100})
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

# ---- Programmes (JUPAS + Global) ----
@app.get("/api/programmes/hk")
def list_programmes_hk(university: str = None, band: str = None, search: str = None, limit: int = 100):
    sb = get_sb()
    q = sb.table("programmes_hk").select("*")
    if university:
        q = q.ilike("university_name", f"%{university}%")
    if band:
        q = q.eq("band", band)
    if search:
        q = q.or_(f"programme_name.ilike.%{search}%,jupas_code.ilike.%{search}%")
    return q.order("university_name").limit(limit).execute().data

@app.get("/api/programmes/hk/university/{uni_name}")
def programmes_by_university(uni_name: str):
    sb = get_sb()
    return sb.table("programmes_hk").select("*").ilike("university_name", f"%{uni_name}%").order("jupas_code").execute().data

@app.get("/api/programmes/hk/stats")
def programmes_hk_stats():
    sb = get_sb()
    all_progs = sb.table("programmes_hk").select("university_name, median_score, lq_score").execute().data
    unis = {}
    for p in all_progs:
        uni = p["university_name"]
        if uni not in unis:
            unis[uni] = 0
        unis[uni] += 1
    return {
        "total_programmes": len(all_progs),
        "universities": len(unis),
        "by_university": unis
    }

@app.get("/api/programmes/hk/{jupas_code}")
def get_programme_hk(jupas_code: str):
    sb = get_sb()
    data = sb.table("programmes_hk").select("*").eq("jupas_code", jupas_code).execute().data
    if not data:
        raise HTTPException(404, "Programme not found")
    return data[0]


@app.get("/api/programmes/match")
def match_programmes(student_id: str, limit: int = 10):
    """Match programmes to a student based on their scores"""
    sb = get_sb()
    student = sb.table("students").select("*").eq("student_id", student_id).execute().data
    if not student:
        student = sb.table("students").select("*").eq("id", student_id).execute().data
    if not student:
        raise HTTPException(404, "Student not found")
    s = student[0]
    uuid = s["id"]
    grades = sb.table("grades_scores").select("*").eq("student_id", uuid).execute().data
    total_score = sum(g.get("score", 0) or 0 for g in grades)
    progs = sb.table("programmes_hk").select("jupas_code, university_name, programme_name, median_score, lq_score, uq_score").execute().data
    matches = []
    for p in progs:
        median = p.get("median_score") or 0
        lq = p.get("lq_score") or 0
        if median > 0:
            if total_score >= median:
                chance = 80
            elif total_score >= lq:
                chance = 50
            else:
                chance = max(10, int(100 * total_score / median) - 20)
        else:
            chance = 0
        matches.append({**p, "chance": chance, "student_score": total_score})
    matches.sort(key=lambda x: -x["chance"])
    return matches[:limit]

@app.get("/api/programmes/search")
def search_programmes_global(q: str, region: str = None, limit: int = 50):
    sb = get_sb()
    tables_map = {
        "hk": "programmes_hk",
        "hk_nonjupas": "programmes_hk_nonjupas",
        "hk_selffinanced": "programmes_hk_selffinanced",
        "uk": "programmes_uk",
        "us": "programmes_us",
        "au": "programmes_au",
        "sg": "programmes_sg",
        "eu": "programmes_eu",
        "jp": "programmes_jp",
        "kr": "programmes_kr",
        "ca": "programmes_ca",
        "mainland": "programmes_mainland",
        "macau": "programmes_macau",
        "nz": "programmes_nz",
        "asia_other": "programmes_asia_other",
    }
    results = []
    search_tables = {region: tables_map[region]} if region and region in tables_map else tables_map
    for reg, table in search_tables.items():
        try:
            if table == "programmes_hk_selffinanced":
                data = sb.table(table).select("*").or_(f"programme_name_en.ilike.%{q}%,institution.ilike.%{q}%").limit(limit).execute().data
                for d in data:
                    d["region"] = reg
                    d["programme_name"] = d.get("programme_name_en", "")
                    d["university_name"] = d.get("institution", "")
            else:
                data = sb.table(table).select("*").or_(f"programme_name.ilike.%{q}%,university_name.ilike.%{q}%").limit(limit).execute().data
                for d in data:
                    d["region"] = reg
            results.extend(data)
        except:
            pass
    return results[:limit]


@app.get("/api/programmes/all/stats")
def all_programmes_stats():
    sb = get_sb()
    tables_map = {
        "hk": "programmes_hk",
        "hk_nonjupas": "programmes_hk_nonjupas",
        "hk_selffinanced": "programmes_hk_selffinanced",
        "uk": "programmes_uk",
        "us": "programmes_us",
        "au": "programmes_au",
        "sg": "programmes_sg",
        "eu": "programmes_eu",
        "jp": "programmes_jp",
        "kr": "programmes_kr",
        "ca": "programmes_ca",
        "mainland": "programmes_mainland",
        "macau": "programmes_macau",
        "nz": "programmes_nz",
        "asia_other": "programmes_asia_other",
    }
    stats = {}
    total = 0
    for region, table in tables_map.items():
        try:
            count = len(sb.table(table).select("id").execute().data)
            stats[region] = count
            total += count
        except:
            stats[region] = 0
    return {"total_programmes": total, "regions": len(stats), "by_region": stats}

@app.get("/api/programmes/{region}")
def list_programmes_by_region(region: str, search: str = None, limit: int = 100):
    tables_map = {
        "hk": "programmes_hk",
        "hk_nonjupas": "programmes_hk_nonjupas",
        "hk_selffinanced": "programmes_hk_selffinanced",
        "uk": "programmes_uk",
        "us": "programmes_us",
        "au": "programmes_au",
        "sg": "programmes_sg",
        "eu": "programmes_eu",
        "jp": "programmes_jp",
        "kr": "programmes_kr",
        "ca": "programmes_ca",
        "mainland": "programmes_mainland",
        "macau": "programmes_macau",
        "nz": "programmes_nz",
        "asia_other": "programmes_asia_other",
    }
    if region not in tables_map:
        raise HTTPException(404, f"Region '{region}' not found. Available: {list(tables_map.keys())}")
    sb = get_sb()
    table = tables_map[region]
    q = sb.table(table).select("*")
    if search:
        if table == "programmes_hk_selffinanced":
            q = q.or_(f"programme_name_en.ilike.%{search}%,institution.ilike.%{search}%")
        else:
            q = q.or_(f"programme_name.ilike.%{search}%,university_name.ilike.%{search}%")
    return q.limit(limit).execute().data
