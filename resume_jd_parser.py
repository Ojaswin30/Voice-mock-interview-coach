import io
import json
from typing import Dict, Any, Optional
import pypdf
from groq import Groq
from memory_graph import CandidateMemoryGraph


def extract_text_from_upload(uploaded_file) -> str:
    """Extracts text content from an uploaded PDF or plain text file with stream seek safety."""
    if uploaded_file is None:
        return ""

    try:
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)

        filename = getattr(uploaded_file, "name", "").lower()
        if filename.endswith(".pdf"):
            reader = pypdf.PdfReader(uploaded_file)
            pages_text = []
            for page in reader.pages:
                text = page.extract_text()
                if text and text.strip():
                    pages_text.append(text.strip())
            extracted = "\n\n".join(pages_text).strip()
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
            return extracted
        else:
            raw = uploaded_file.read()
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
            if isinstance(raw, bytes):
                return raw.decode("utf-8", errors="ignore").strip()
            return str(raw).strip()
    except Exception as e:
        print(f"Error parsing uploaded file: {e}")
        return ""


def analyze_resume_and_jd(
    groq_client: Groq,
    resume_text: str,
    jd_text: str = "",
    model_name: str = "llama-3.1-8b-instant",
) -> Dict[str, Any]:
    """
    Analyzes the candidate's resume and target job description using a fast Groq LLM call.
    Extracts concrete claims, tech stack, key requirements, and interview probe areas.
    """
    if not resume_text or len(resume_text.strip()) < 20:
        return {}

    prompt = f"""You are an elite technical executive recruiter and hiring manager.
Analyze the provided candidate resume and optional target Job Description (JD).

CANDIDATE RESUME:
---
{resume_text[:6000]}
---

TARGET JOB DESCRIPTION (JD):
---
{jd_text[:4000] if jd_text.strip() else "None provided (use resume to infer target role and domain)"}
---

Extract structured information in JSON format with these exact keys:
1. "candidate_name": String. Full name of the candidate if mentioned in the resume (default "Candidate").
2. "target_role": String. The specific job title being targeted (e.g. "Senior Backend Engineer", "Staff ML Engineer").
3. "experience_summary": String. 2-3 sentences summarizing the candidate's core background, seniority, and primary tech stack.
4. "key_claims": Array of objects: [{{"claim": "specific project, architecture, metric, or system the candidate built/led", "topic": "domain/tech name"}}]. Extract 4 to 8 concrete claims made on the resume that an interviewer should probe to verify real depth.
5. "jd_requirements": Array of strings. 3 to 6 critical technical or behavioral requirements from the JD (or industry standard if no JD was provided).
6. "probe_areas": Array of strings. 3 to 5 high-priority questions or potential gaps between the resume and JD where the interviewer should challenge the candidate (e.g. tradeoffs, unmentioned failure handling, scaling bottlenecks).

Return ONLY valid JSON.
"""

    candidate_models = []
    if model_name and any(ok in model_name.lower() for ok in ["llama", "mixtral", "gemma", "qwen", "deepseek"]):
        candidate_models.append(model_name)
    candidate_models.extend(["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"])
    seen = set()
    models_to_try = [m for m in candidate_models if not (m in seen or seen.add(m))]

    for m in models_to_try:
        try:
            resp = groq_client.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system", "content": "You are a resume and job description analysis engine. Respond strictly with a JSON object."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=900,
            )
            raw = resp.choices[0].message.content.strip()
            return json.loads(raw)
        except Exception as e:
            err_str = str(e).lower()
            if "model_not_found" in err_str or "does not exist" in err_str:
                continue
            print(f"Resume/JD analysis error with model {m}: {e}")
            break

    return {}


def seed_graph_from_resume_and_jd(
    graph: CandidateMemoryGraph,
    candidate_id: str,
    analysis_result: Dict[str, Any],
):
    """
    Seeds the candidate's persistent Knowledge Graph with claims from their resume
    and probe areas from the Job Description.
    """
    if not analysis_result:
        return

    # 1. Update candidate role
    name = analysis_result.get("candidate_name") or "Candidate"
    role = analysis_result.get("target_role") or "Software Engineer"
    graph.register_candidate(candidate_id, name, role)

    # 2. Add resume claims as unprobed claims into the graph
    for item in analysis_result.get("key_claims", []):
        if isinstance(item, dict):
            claim_text = item.get("claim", "")
            topic = item.get("topic", "")
        elif isinstance(item, str):
            claim_text = item
            topic = ""
        else:
            continue

        if claim_text and len(claim_text.strip()) > 5:
            graph.add_claim(candidate_id, claim_text.strip(), topic)

    # 3. Add probe areas / potential gaps to weaknesses table as probe targets
    for probe in analysis_result.get("probe_areas", []):
        if isinstance(probe, str) and len(probe.strip()) > 5:
            graph.add_weakness(candidate_id, "Resume/JD Probe Area", probe.strip())
