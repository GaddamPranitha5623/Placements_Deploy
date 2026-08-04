import os
from typing import Dict, Any, List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

import pdfplumber
import requests
from collections import Counter
from datetime import datetime
import json

from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
#from google.colab import userdata # Import userdata

# Install missing libraries
#!pip install pdfplumber

# ---------- LLM (Gemini) setup ----------
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Original line

# Retrieve the API key from secrets, making this cell robust


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API:
    print("Error: GOOGLE_API_KEY not set. Please configure it in Render environment variables.")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",   # or gemini-1.5-pro if available
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY,
)

# ---------- Tools for placement agent ----------

@tool
def parse_resume(file_path: str) -> Dict[str, Any]:
    """
    Extract text from a resume PDF and return:
    raw_text, skills, experience, education.
    """
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    import re
    skills = []
    experience = []
    education = []

    lines = text.splitlines()
    for line in lines:
        l = line.lower()
        if "skills" in l and ":" in l:
            skills_text = line.split(":", 1)[1]
            skills = [s.strip() for s in re.split(r"[,\u2022•]", skills_text) if s.strip()]
        elif "experience" in l:
            experience.append(line.strip())
        elif "education" in l:
            education.append(line.strip())

    return {
        "raw_text": text,
        "skills": skills,
        "experience": experience,
        "education": education,
    }


@tool
def job_search(target_role: str) -> Dict[str, Any]:
    """
    Summarize typical responsibilities and required skills for a given role.
    Uses a template (no external job API), suitable for student projects.
    """
    role_lower = target_role.lower()
    responsibilities = [
        "Understand business requirements and translate them into technical solutions.",
        "Collaborate with cross-functional teams on development and deployment.",
        "Write clean, maintainable, and testable code.",
        "Participate in code reviews and debugging.",
    ]
    skills = [
        "Data structures and algorithms",
        "Version control (Git, GitHub)",
        "Database fundamentals (SQL/NoSQL)",
        "Problem-solving and system design basics",
    ]

    if "data" in role_lower:
        responsibilities.append("Clean, process, and analyze data for insights.")
        skills.extend(["Python", "Pandas", "NumPy", "Data visualization tools"])
    elif "ml" in role_lower or "machine learning" in role_lower:
        responsibilities.append("Train, validate, and deploy machine learning models.")
        skills.extend(["Python", "Scikit-learn", "TensorFlow/PyTorch", "Statistics"])
    elif "web" in role_lower or "frontend" in role_lower or "backend" in role_lower:
        responsibilities.append("Develop and maintain web applications.")
        skills.extend(["HTML/CSS/JS", "Frameworks (React/Django/etc.)", "REST APIs"])

    return {
        "role": target_role,
        "responsibilities": responsibilities,
        "required_skills": list(dict.fromkeys(skills)),
    }


@tool
def skill_gap_analyzer(resume_data: Dict[str, Any], job_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare resume skills vs job-required skills using Gemini and
    return missing_skills, partial_skills, strengths, summary.
    """
    prompt = f"""
You are a placement mentor.

Resume skills: {resume_data.get("skills")}
Resume experience: {resume_data.get("experience")}
Job role: {job_profile.get("role")}
Role required skills: {job_profile.get("required_skills")}
Role responsibilities: {job_profile.get("responsibilities")}

1. List skills clearly missing.
2. List skills partially covered or weak.
3. List clear strengths.
4. Give a short summary targeted at a final-year student.

Return JSON with keys:
missing_skills, partial_skills, strengths, summary.
"""
    response = llm.invoke(prompt)
    try:
        data = json.loads(response.content)
    except Exception:
        data = {
            "missing_skills": [],
            "partial_skills": [],
            "strengths": [],
            "summary": response.content,
        }
    return data


@tool
def project_ideas(skill_gap: Dict[str, Any], target_role: str) -> List[Dict[str, Any]]:
    """
    Recommend 3–5 practical project ideas to close skill gaps for the given role.
    Each idea: title, description, main_tech_stack, difficulty.
    """
    prompt = f"""
You are a mentor helping a student prepare for campus placements as a {target_role}.

Missing skills: {skill_gap.get("missing_skills")}
Partial skills: {skill_gap.get("partial_skills")}
Strengths: {skill_gap.get("strengths")}

Recommend 3-5 practical project ideas that:
- Can be built in 2-4 weeks,
- Are good for GitHub portfolio,
- Target the missing/weak skills.

Return JSON list, each item with:
title, description, main_tech_stack, difficulty.
"""
    response = llm.invoke(prompt)
    try:
        ideas = json.loads(response.content)
    except Exception:
        ideas = [{
            "title": "Portfolio Project",
            "description": response.content,
            "main_tech_stack": [],
            "difficulty": "medium",
        }]
    return ideas


@tool
def github_profile_analyzer(username: str) -> Dict[str, Any]:
    """
    Analyze a GitHub user's profile using public GitHub API:
    total_repos, top_languages, total_stars, active_repos_90d, most_starred_repo, followers.
    """
    repos_url = f"https://api.github.com/users/{username}/repos"
    resp = requests.get(repos_url, params={"per_page": 100, "sort": "updated"})
    repos = resp.json()

    if isinstance(repos, dict) and repos.get("message"):
        return {"error": repos.get("message")}

    languages = Counter()
    total_stars = 0
    active_repos_90d = 0
    original = [r for r in repos if not r.get("fork")]

    for repo in original:
        lang = repo.get("language")
        if lang:
            languages[lang] += 1
        total_stars += repo.get("stargazers_count", 0)

        updated_str = repo.get("updated_at")
        if updated_str:
            updated = datetime.strptime(updated_str, "%Y-%m-%dT%H:%M:%SZ")
            if (datetime.now() - updated).days < 90:
                active_repos_90d += 1

    most_starred = None
    if original:
        most_starred = max(
            original,
            key=lambda r: r.get("stargazers_count", 0)
        ).get("name")

    profile_url = f"https://api.github.com/users/{username}"
    profile_resp = requests.get(profile_url)
    profile = profile_resp.json()

    return {
        "username": username,
        "name": profile.get("name"),
        "public_repos": profile.get("public_repos"),
        "followers": profile.get("followers"),
        "top_languages": dict(languages.most_common(5)),
        "total_stars": total_stars,
        "active_repos_90d": active_repos_90d,
        "most_starred_repo": most_starred,
        "profile_url": profile.get("html_url"),
    }


def placement_agent(resume_pdf_path: str, target_role: str, github_username: str) -> Dict[str, Any]:
    """
    Implements your flow chart:
    1. Parse resume.
    2. Job search for target role.
    3. Skill gap analysis.
    4. Project recommendations.
    5. GitHub evaluation.
    6. Final synthesis (JSON).
    """
    resume_data = parse_resume.invoke({"file_path": resume_pdf_path})
    job_profile = job_search.invoke({"target_role": target_role})
    skill_gap = skill_gap_analyzer.invoke({"resume_data": resume_data, "job_profile": job_profile})
    projects = project_ideas.invoke({"skill_gap": skill_gap, "target_role": target_role})
    github_eval = github_profile_analyzer.invoke({"username": github_username})

    synthesis_prompt = f"""
Student target role: {target_role}
GitHub username: {github_username}

RESUME DATA:
{resume_data}

JOB PROFILE:
{job_profile}

SKILL GAP:
{skill_gap}

PROJECT IDEAS:
{projects}

GITHUB EVALUATION:
{github_eval}

Create a structured JSON response with keys:
- job_profile
- skill_gap
- recommended_projects
- github_evaluation
- overall_advice

Return ONLY JSON.
"""
    final_response = llm.invoke(synthesis_prompt)
    try:
        result = json.loads(final_response.content)
    except Exception:
        result = {
            "job_profile": job_profile,
            "skill_gap": skill_gap,
            "recommended_projects": projects,
            "github_evaluation": github_eval,
            "overall_advice": final_response.content,
        }
    return result

# ---------- FastAPI app ----------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "Placement-ready AI agent running"}


@app.post("/analyze")
async def analyze(
    resume: UploadFile = File(...),
    target_role: str = Form(...),
    github_username: str = Form(...),
):
    """
    POST /analyze with form-data:
      - resume: PDF file
      - target_role: e.g., "Data Analyst"
      - github_username: GitHub ID

    Returns placement analysis JSON:
    job_profile, skill_gap, recommended_projects, github_evaluation, overall_advice.
    """
    temp_path = f"/tmp/{resume.filename}"
    with open(temp_path, "wb") as f:
        f.write(await resume.read())

    result = placement_agent(
        resume_pdf_path=temp_path,
        target_role=target_role,
        github_username=github_username,
    )
    return result
