"""
Interview Page View for Voice Mock-Interview Coach.
Handles Resume & JD grounding, live voice conversation, audio recording, 
Whisper STT inference, Web Speech TTS playback, filler analysis,
dynamic Knowledge Graph memory extraction, and Lead Committee evaluations.
"""
import os
import io
import time
import wave
import json
import re
import hashlib
import tempfile
from typing import List, Dict, Any, Optional

import numpy as np
import streamlit as st
from faster_whisper import WhisperModel
from groq import Groq

from memory_graph import CandidateMemoryGraph, extract_and_update_memory
from resume_jd_parser import (
    extract_text_from_upload,
    analyze_resume_and_jd,
    seed_graph_from_resume_and_jd,
)
from .speech import render_speech_player


# ---------------------------------------------------------
# Caching & Model Initialization
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_whisper_model(model_size: str = "base"):
    """
    Loads and caches the faster-whisper model locally.
    Defaults to CPU with int8 quantization for high speed & low RAM.
    """
    try:
        return WhisperModel(model_size, device="cpu", compute_type="int8")
    except Exception:
        return WhisperModel(model_size, device="cpu", compute_type="float32")


def decode_wav_to_numpy(audio_bytes: bytes, target_sr: int = 16000) -> Optional[np.ndarray]:
    """
    Decodes in-memory WAV audio bytes directly to a float32 NumPy array at 16000 Hz.
    Completely bypasses PyAV / FFmpeg to eliminate any C-extension binary incompatibility.
    """
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            nframes = wf.getnframes()
            raw_data = wf.readframes(nframes)

        if sampwidth == 2:  # 16-bit PCM (standard for st.audio_input)
            data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 1:  # 8-bit PCM
            data = (np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif sampwidth == 4:  # 32-bit float or int32
            try:
                data = np.frombuffer(raw_data, dtype=np.float32)
            except Exception:
                data = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            return None

        # Convert stereo to mono by averaging channels
        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1)

        # Resample to 16 kHz if necessary
        if framerate != target_sr and len(data) > 0:
            new_len = int(len(data) * target_sr / framerate)
            data = np.interp(
                np.linspace(0, len(data), new_len, endpoint=False),
                np.arange(len(data)),
                data,
            ).astype(np.float32)

        return data.astype(np.float32)
    except Exception:
        return None


def transcribe_audio_bytes(audio_bytes: bytes, whisper_model: WhisperModel) -> str:
    """
    Transcribes raw audio bytes using faster-whisper.
    1. Uses pure-Python/NumPy in-memory WAV decoding (zero PyAV dependency).
    2. Falls back to temporary file only if audio is not WAV.
    """
    numpy_audio = decode_wav_to_numpy(audio_bytes, target_sr=16000)
    if numpy_audio is not None and len(numpy_audio) > 0:
        segments, _ = whisper_model.transcribe(
            numpy_audio,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        text_segments = [seg.text.strip() for seg in segments if seg.text.strip()]
        return " ".join(text_segments).strip()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio_path = temp_audio.name

    try:
        segments, _ = whisper_model.transcribe(
            temp_audio_path,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        text_segments = [seg.text.strip() for seg in segments if seg.text.strip()]
        return " ".join(text_segments).strip()
    finally:
        if os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except OSError:
                pass


def analyze_filler_words(text: str) -> Dict[str, Any]:
    """
    Detects common filler words in transcribed speech.
    Returns counts and individual occurrences.
    """
    filler_patterns = {
        "um": r"\b(um|umm|ummm)\b",
        "uh": r"\b(uh|uhh|uhhh)\b",
        "like": r"\b(like)\b",
        "you know": r"\b(you know)\b",
        "actually": r"\b(actually)\b",
        "basically": r"\b(basically)\b",
        "literally": r"\b(literally)\b",
        "i mean": r"\b(i mean)\b",
        "kind of / sort of": r"\b(kind of|sort of|kinda|sorta)\b",
        "right?": r"\b(right\?|right)\b",
    }

    counts = {}
    total_fillers = 0
    words = re.findall(r"\b\w+\b", text.lower())
    total_words = len(words)

    for label, pattern in filler_patterns.items():
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        count = len(matches)
        if count > 0:
            counts[label] = count
            total_fillers += count

    density = (total_fillers / total_words * 100) if total_words > 0 else 0.0

    return {
        "total_fillers": total_fillers,
        "total_words": total_words,
        "filler_density_pct": round(density, 1),
        "breakdown": counts,
    }


def get_resume_and_jd_summaries(candidate_name: str, fallback_role: str):
    """Formats resume and JD context to supply into interviewer prompts."""
    resume_summary = ""
    jd_summary = ""
    if st.session_state.resume_analysis:
        analysis = st.session_state.resume_analysis
        claims = [c.get("claim", c) if isinstance(c, dict) else c for c in analysis.get("key_claims", [])]
        resume_summary = (
            f"Candidate: {analysis.get('candidate_name', candidate_name)}\n"
            f"Target Role: {analysis.get('target_role', fallback_role)}\n"
            f"Experience Summary: {analysis.get('experience_summary', '')}\n"
            f"Key Resume Claims: {json.dumps(claims)}"
        )
        jd_summary = json.dumps(analysis.get("jd_requirements", []))
    elif st.session_state.resume_text:
        resume_summary = st.session_state.resume_text[:2500]

    if st.session_state.jd_text and not jd_summary:
        jd_summary = st.session_state.jd_text[:2500]

    return resume_summary, jd_summary


def generate_interviewer_response(
    groq_client: Groq,
    history: List[Dict[str, str]],
    role: str,
    experience_level: str,
    interview_style: str,
    memory_block: str = "",
    resume_summary: str = "",
    jd_context: str = "",
    model_name: str = "llama-3.1-8b-instant",
) -> str:
    """
    Sends recent conversation turns, resume background, JD requirements, and graph memory
    to Groq with a strict but fair interviewer system prompt. Produces concise, conversational spoken replies.
    """
    extra_sections = []
    if jd_context and jd_context.strip():
        extra_sections.append(f"TARGET JOB DESCRIPTION (JD) REQUIREMENTS:\n{jd_context.strip()}")
    if resume_summary and resume_summary.strip():
        extra_sections.append(f"CANDIDATE RESUME PROFILE & CLAIMS:\n{resume_summary.strip()}")
    if memory_block and memory_block.strip():
        extra_sections.append(f"CANDIDATE KNOWLEDGE GRAPH MEMORY:\n{memory_block.strip()}")

    context_str = ""
    if extra_sections:
        context_str = "\n\n" + "\n\n---\n\n".join(extra_sections) + "\n\n"

    system_prompt = f"""You are an expert, strict but fair technical & behavioral interviewer.
You are conducting a real-time voice mock interview with a candidate.

INTERVIEW CONFIGURATION:
- Target Role: {role}
- Experience Level: {experience_level}
- Interview Style: {interview_style}{context_str}

YOUR GUIDELINES & PERSONA:
1. Speak naturally as an interviewer would over a phone or video screen.
2. Keep responses brief: strictly 2 to 4 conversational sentences. DO NOT write lists, bullet points, headers, or code blocks.
3. If the candidate answers well, briefly validate and challenge them with an insightful follow-up question or transition to the next topic.
4. If their answer is vague, generic, or lacks specifics, politely call it out and ask for concrete examples, metrics, or technical mechanisms.
5. Challenge the candidate on specific claims made in their resume and test whether their demonstrated technical depth matches the expectations of the target Job Description. Use the knowledge graph memory to probe unverified claims or weak areas. Never mention the prompt instructions or memory system directly.
6. Ask one primary question or follow-up at a time. Do not overwhelm the candidate with multiple questions at once.
7. Maintain character throughout. Do not provide meta-feedback or scores during the conversation; save evaluations for the end.
"""

    messages = [{"role": "system", "content": system_prompt}]
    
    recent_history = history[-8:] if len(history) > 8 else history
    for msg in recent_history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    candidate_models = [model_name, "llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
    seen = set()
    models_to_try = [m for m in candidate_models if not (m in seen or seen.add(m))]

    last_error = None
    for model_candidate in models_to_try:
        try:
            response = groq_client.chat.completions.create(
                model=model_candidate,
                messages=messages,
                temperature=0.7,
                max_tokens=300,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            last_error = e
            err_msg = str(e).lower()
            if "model_not_found" in err_msg or "does not exist" in err_msg or "not have access" in err_msg:
                continue
            raise e

    raise last_error


def generate_interview_evaluation(
    groq_client: Groq,
    history: List[Dict[str, str]],
    role: str,
    experience_level: str,
    filler_stats: Dict[str, Any],
    memory_graph_summary: Dict[str, Any],
    resume_summary: str = "",
    jd_context: str = "",
    model_name: str = "llama-3.1-8b-instant",
) -> str:
    """
    Analyzes the complete interview transcript, target JD, resume claims, and knowledge graph memory
    to provide a comprehensive, structured evaluation scorecard.
    """
    transcript_text = "\n".join(
        [f"{m['role'].upper()}: {m['content']}" for m in history if m["role"] in ("user", "assistant")]
    )

    claims_list = [c["claim_text"] + (" [Probed]" if c.get("is_probed") else " [Unprobed]") for c in memory_graph_summary.get("claims", [])]
    weaknesses_list = [w["topic"] + (f" ({w['observation']})" if w.get("observation") else "") for w in memory_graph_summary.get("weaknesses", [])]
    topics_list = [t["topic_name"] for t in memory_graph_summary.get("topics", [])]

    eval_prompt = f"""You are the lead hiring committee member reviewing the transcript of a completed mock interview.
Role: {role}
Seniority: {experience_level}

Target Job Description (JD):
---
{jd_context if jd_context else "Standard expectations for " + role}
---

Candidate Resume Profile & Claims:
---
{resume_summary if resume_summary else "Standard background"}
---

Full Transcript of the interview:
---
{transcript_text}
---

Candidate's Persistent Memory Graph:
- Technical Claims: {json.dumps(claims_list)}
- Topics Covered: {json.dumps(topics_list)}
- Weak Areas Observed: {json.dumps(weaknesses_list)}
- Filler Words Detected: {filler_stats.get('total_fillers', 0)} ({filler_stats.get('filler_density_pct', 0)}% density)

Please provide a comprehensive, rigorous scorecard and feedback in the following Markdown structure:

# 📊 Interview Scorecard & Feedback

### Overall Rating: [Score]/10 — [Verdict: e.g. Strong Hire / Hire / Lean Hire / Needs Improvement / Strong No Hire]

---

### 1. Dimension Scores & Breakdown
- **Job Description Alignment & Technical Fit: [Score]/10**
  *How strongly did the candidate's demonstrated competence match the must-have requirements in the JD?*
- **Resume Claim Verification: [Score]/10**
  *Did the candidate's answers substantiate the claims made on their resume with specific technical depth, metrics, and architecture?*
- **Clarity & Communication: [Score]/10**
  *Assessment of articulation, conciseness, and presence.*
- **Structure & Framework: [Score]/10**
  *Use of structured communication (e.g. STAR method, problem decomposition, clear narrative arc).*
- **Delivery & Filler Words: [Score]/10**
  *Assessment of filler words ({filler_stats.get('total_fillers', 0)} detected, {filler_stats.get('filler_density_pct', 0)}% density) and conversational confidence.*

---

### 2. Key Strengths 🌟
- [Bullet points of standout responses, strong reasoning, or solid technical claims verified]

---

### 3. Actionable Areas for Improvement 🎯
- [Specific examples where the candidate was vague, stumbled on tradeoffs, or failed to substantiate a resume claim]
- [Actionable advice for the real interview with this specific JD]

---

### 4. Exemplary Answers & Reframing 💡
*Pick 1 question where the candidate could have scored significantly higher and show how an ideal {experience_level} candidate would answer it concisely using the STAR framework.*

---

Be candid, constructive, and uncompromising in your high bar for excellence.
"""

    candidate_models = [model_name, "llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
    seen = set()
    models_to_try = [m for m in candidate_models if not (m in seen or seen.add(m))]

    last_error = None
    for model_candidate in models_to_try:
        try:
            response = groq_client.chat.completions.create(
                model=model_candidate,
                messages=[
                    {"role": "system", "content": "You are a world-class executive and engineering hiring coach."},
                    {"role": "user", "content": eval_prompt},
                ],
                temperature=0.4,
                max_tokens=1500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            last_error = e
            err_msg = str(e).lower()
            if "model_not_found" in err_msg or "does not exist" in err_msg or "not have access" in err_msg:
                continue
            raise e

    raise last_error


# ---------------------------------------------------------
# Core Interview View Function
# ---------------------------------------------------------
def render_interview_page(
    groq_client: Optional[Groq],
    memory_graph: CandidateMemoryGraph,
    user_manager,
    current_user: Optional[Dict[str, Any]] = None,
):
    """Renders the core voice mock interview room and scorecard view."""
    user = current_user or st.session_state.get("user") or {}
    candidate_id = user.get("id", "guest")
    candidate_name = user.get("name", "Candidate")
    target_role = st.session_state.get("target_role", "Software Engineer (Full-Stack)")
    experience_level = st.session_state.get("experience_level", "Senior (5-8 yrs)")
    interview_style = st.session_state.get("interview_style", "Strict & Rigorous (FAANG-Style)")
    whisper_model_size = st.session_state.get("whisper_model_size", "base")
    groq_model_name = st.session_state.get("groq_model_name", "llama-3.1-8b-instant")

    st.title("🎙️ Voice Mock-Interview Room")
    st.caption(
        f"Candidate: **{candidate_name}** | Track: **{target_role}** ({experience_level}) | Demeanor: **{interview_style}**"
    )

    # 1. Setup Step: Resume & JD Input (Before interview starts)
    if not st.session_state.interview_started:
        st.markdown(
            """
            Prepare for your interview by providing your **Resume** and the **Target Job Description**. 
            The interviewer will analyze your actual claims, probe for technical depth, and test your experience against the JD.
            """
        )

        with st.expander("📄 Step 1: Upload Resume & Job Description (JD)", expanded=True):
            col_res, col_jd = st.columns(2)
            with col_res:
                st.markdown("**Candidate Resume**")
                uploaded_resume = st.file_uploader(
                    "Upload Resume (PDF or TXT)",
                    type=["pdf", "txt"],
                    key="resume_uploader",
                    help="PDF or plain text resume.",
                )
                typed_resume = st.text_area(
                    "Or paste resume text:",
                    value=st.session_state.resume_text,
                    height=140,
                    placeholder="Paste work experience, past projects, technical stack...",
                    key="pasted_resume_area",
                )
                if uploaded_resume is not None:
                    extracted = extract_text_from_upload(uploaded_resume)
                    if extracted:
                        st.session_state.resume_text = extracted
                elif typed_resume.strip():
                    st.session_state.resume_text = typed_resume.strip()

            with col_jd:
                st.markdown("**Target Job Description (JD)**")
                typed_jd = st.text_area(
                    "Paste Job Description / Requirements:",
                    value=st.session_state.jd_text,
                    height=205,
                    placeholder="Paste role responsibilities, required qualifications, tech stack...",
                    key="pasted_jd_area",
                )
                if typed_jd.strip():
                    st.session_state.jd_text = typed_jd.strip()

            col_parse_btn, _ = st.columns([3, 4])
            with col_parse_btn:
                if st.button("⚡ Parse Resume & JD into Knowledge Graph", use_container_width=True):
                    raw_resume = ""
                    if uploaded_resume is not None:
                        raw_resume = extract_text_from_upload(uploaded_resume)
                    if not raw_resume and typed_resume.strip():
                        raw_resume = typed_resume.strip()

                    if raw_resume:
                        st.session_state.resume_text = raw_resume

                    if not st.session_state.resume_text.strip():
                        st.warning("Please upload a PDF/TXT resume or paste your resume text first!")
                    else:
                        with st.spinner("Analyzing resume against JD and seeding claims to Memory Graph..."):
                            analysis = analyze_resume_and_jd(
                                groq_client=groq_client,
                                resume_text=st.session_state.resume_text,
                                jd_text=st.session_state.jd_text,
                                model_name=groq_model_name,
                            )
                            if analysis and (
                                analysis.get("key_claims") or analysis.get("experience_summary")
                            ):
                                st.session_state.resume_analysis = analysis
                                seed_graph_from_resume_and_jd(memory_graph, candidate_id, analysis)
                                st.success("✅ Resume & JD analyzed! Claims and probe areas seeded to Knowledge Graph.")
                                st.rerun()
                            else:
                                st.error("Could not extract structured data from the resume. Please paste text directly.")

        # Analyzed Profile Preview
        if st.session_state.resume_analysis:
            analysis = st.session_state.resume_analysis
            with st.expander("🔍 Analyzed Profile & Claims (Ready to be Tested)", expanded=True):
                col_a1, col_a2 = st.columns(2)
                with col_a1:
                    st.markdown(f"**Target Role:** {analysis.get('target_role', target_role)}")
                    st.markdown(f"**Candidate:** {analysis.get('candidate_name', candidate_name)}")
                    st.markdown(f"**Summary:** *{analysis.get('experience_summary', '')}*")
                    st.markdown("**📌 Key Resume Claims (Interviewer will test):**")
                    for claim_item in analysis.get("key_claims", [])[:6]:
                        c_txt = (
                            claim_item.get("claim", claim_item)
                            if isinstance(claim_item, dict)
                            else claim_item
                        )
                        st.markdown(f"- {c_txt}")
                with col_a2:
                    st.markdown("**🎯 Core JD Requirements:**")
                    for req in analysis.get("jd_requirements", [])[:5]:
                        st.markdown(f"- {req}")
                    st.markdown("**⚠️ Identified Probe / Focus Areas:**")
                    for probe in analysis.get("probe_areas", [])[:4]:
                        st.markdown(f"- 🔍 *{probe}*")

        st.markdown("<br>", unsafe_allow_html=True)
        col_start, _ = st.columns([2.5, 5])
        with col_start:
            if st.button("🚀 Start Interview", type="primary", use_container_width=True):
                with st.spinner("Preparing your personalized interviewer..."):
                    filler_note = ""
                    mem_block = memory_graph.get_candidate_memory_block(
                        candidate_id, filler_notes=filler_note
                    )
                    resume_sum, jd_sum = get_resume_and_jd_summaries(candidate_name, target_role)

                    initial_prompt = (
                        f"Start the interview for a {experience_level} {target_role}. "
                        "Introduce yourself briefly as the interviewer in 1-2 friendly but professional sentences, "
                        "mention that you reviewed their resume for this specific position, and ask an opening question directly "
                        "challenging or probing one of the projects or claims from their background."
                    )

                    try:
                        opening_reply = generate_interviewer_response(
                            groq_client=groq_client,
                            history=[{"role": "user", "content": initial_prompt}],
                            role=target_role,
                            experience_level=experience_level,
                            interview_style=interview_style,
                            memory_block=mem_block,
                            resume_summary=resume_sum,
                            jd_context=jd_sum,
                            model_name=groq_model_name,
                        )
                        st.session_state.messages.append({"role": "assistant", "content": opening_reply})
                        st.session_state.interview_started = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to reach Groq API: {str(e)}")

    # 2. Live Active Interview Conversation
    if st.session_state.interview_started:
        for idx, msg in enumerate(st.session_state.messages):
            if msg["role"] == "assistant":
                with st.chat_message("assistant", avatar="🧑‍💼"):
                    st.markdown(f"**Interviewer:** {msg['content']}")
                    is_latest = idx == len(st.session_state.messages) - 1
                    render_speech_player(
                        text=msg["content"],
                        auto_play=is_latest,
                        key_id=f"tts_msg_{idx}",
                    )
            elif msg["role"] == "user":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(f"**You (Candidate):** {msg['content']}")
                    if "filler_info" in msg:
                        f_info = msg["filler_info"]
                        if f_info.get("total_fillers", 0) > 0:
                            st.caption(
                                f"💬 *Fillers detected: {f_info['total_fillers']} ({f_info['filler_density_pct']}% density)*"
                            )

        st.markdown("---")

        # Response Controls (Microphone & Audio Input)
        if not st.session_state.interview_completed:
            st.subheader("🎤 Your Turn to Answer")
            whisper_model = load_whisper_model(whisper_model_size)

            audio_data = st.audio_input(
                "Record your answer (Click to start, click again to stop and submit)"
            )

            with st.expander("⌨️ Or type your answer manually (fallback)", expanded=False):
                typed_answer = st.text_area("Type your response here:", height=80, key="typed_input")
                send_typed_btn = st.button("Submit Typed Answer")

            def process_candidate_answer(answer_text: str):
                filler_stats = analyze_filler_words(answer_text)
                c_fillers = st.session_state.cumulative_fillers
                c_fillers["total_fillers"] += filler_stats["total_fillers"]
                c_fillers["total_words"] += filler_stats["total_words"]
                if c_fillers["total_words"] > 0:
                    c_fillers["filler_density_pct"] = round(
                        (c_fillers["total_fillers"] / c_fillers["total_words"]) * 100, 1
                    )
                for f_word, f_cnt in filler_stats["breakdown"].items():
                    c_fillers["breakdown"][f_word] = c_fillers["breakdown"].get(f_word, 0) + f_cnt

                st.session_state.messages.append({
                    "role": "user",
                    "content": answer_text,
                    "filler_info": filler_stats,
                })

                last_question = ""
                for m in reversed(st.session_state.messages[:-1]):
                    if m["role"] == "assistant":
                        last_question = m["content"]
                        break

                with st.spinner("🧠 Updating candidate memory graph..."):
                    extract_and_update_memory(
                        groq_client=groq_client,
                        graph=memory_graph,
                        candidate_id=candidate_id,
                        last_question=last_question,
                        user_answer=answer_text,
                        model_name="llama-3.1-8b-instant",
                    )

                top_fillers = list(c_fillers["breakdown"].keys())[:2]
                f_note = f"frequent '{', '.join(top_fillers)}'" if top_fillers else ""
                mem_block = memory_graph.get_candidate_memory_block(
                    candidate_id, filler_notes=f_note
                )
                resume_sum, jd_sum = get_resume_and_jd_summaries(candidate_name, target_role)

                with st.spinner("🤔 Interviewer is thinking..."):
                    try:
                        reply = generate_interviewer_response(
                            groq_client=groq_client,
                            history=st.session_state.messages,
                            role=target_role,
                            experience_level=experience_level,
                            interview_style=interview_style,
                            memory_block=mem_block,
                            resume_summary=resume_sum,
                            jd_context=jd_sum,
                            model_name=groq_model_name,
                        )
                        st.session_state.messages.append({"role": "assistant", "content": reply})
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to generate response from Groq: {str(e)}")

            if audio_data is not None:
                audio_bytes = audio_data.getvalue()
                audio_hash = hashlib.md5(audio_bytes).hexdigest()
                if audio_hash != st.session_state.last_processed_audio_hash:
                    st.session_state.last_processed_audio_hash = audio_hash
                    with st.spinner("🎧 Transcribing your voice with faster-whisper..."):
                        transcript = transcribe_audio_bytes(audio_bytes, whisper_model)
                    if transcript.strip():
                        process_candidate_answer(transcript.strip())
                    else:
                        st.warning("Could not detect any speech in the audio recording.")

            if send_typed_btn and typed_answer.strip():
                process_candidate_answer(typed_answer.strip())

            st.markdown("<br>", unsafe_allow_html=True)
            col_fin, col_rst, _ = st.columns([2.5, 2, 3])
            with col_fin:
                if st.button("🏁 Finish Interview & Get Scorecard", type="primary", use_container_width=True):
                    user_replies_count = sum(1 for m in st.session_state.messages if m["role"] == "user")
                    if user_replies_count == 0:
                        st.warning("Please answer at least one question before completing the interview!")
                    else:
                        with st.spinner("Analyzing performance against JD and preparing comprehensive scorecard..."):
                            try:
                                graph_summary = memory_graph.get_full_graph_state(candidate_id)
                                resume_sum, jd_sum = get_resume_and_jd_summaries(candidate_name, target_role)
                                eval_report = generate_interview_evaluation(
                                    groq_client=groq_client,
                                    history=st.session_state.messages,
                                    role=target_role,
                                    experience_level=experience_level,
                                    filler_stats=st.session_state.cumulative_fillers,
                                    memory_graph_summary=graph_summary,
                                    resume_summary=resume_sum,
                                    jd_context=jd_sum,
                                    model_name=groq_model_name,
                                    )
                                st.session_state.evaluation_report = eval_report
                                st.session_state.interview_completed = True

                                # Automatically save session into database history
                                try:
                                    user_manager.save_interview_session(
                                        candidate_id=candidate_id,
                                        target_role=target_role,
                                        experience_level=experience_level,
                                        evaluation_report=eval_report,
                                        messages=st.session_state.messages,
                                        filler_stats=st.session_state.cumulative_fillers,
                                    )
                                except Exception:
                                    pass

                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Failed to generate evaluation report: {str(e)}")

            with col_rst:
                if st.button("🔄 Restart Interview", use_container_width=True):
                    st.session_state.messages = []
                    st.session_state.last_processed_audio_hash = None
                    st.session_state.interview_started = False
                    st.session_state.interview_completed = False
                    st.session_state.evaluation_report = None
                    st.session_state.cumulative_fillers = {
                        "total_fillers": 0,
                        "total_words": 0,
                        "filler_density_pct": 0.0,
                        "breakdown": {},
                    }
                    st.rerun()

        # Scorecard View (When finished)
        if st.session_state.interview_completed and st.session_state.evaluation_report:
            st.success("🎉 Mock Interview Finished! Here is your hiring assessment.")
            m1, m2, m3, m4 = st.columns(4)
            user_turns = sum(1 for m in st.session_state.messages if m["role"] == "user")
            graph_summary = memory_graph.get_full_graph_state(candidate_id)
            probed_count = sum(1 for c in graph_summary.get("claims", []) if c.get("is_probed"))

            m1.metric("Questions Answered", user_turns)
            m2.metric("Total Words Spoken", st.session_state.cumulative_fillers.get("total_words", 0))
            m3.metric(
                "Filler Word Density",
                f"{st.session_state.cumulative_fillers.get('filler_density_pct', 0.0)}%",
                help="Target: < 2.5%",
            )
            m4.metric("Claims Probed", f"{probed_count} / {len(graph_summary.get('claims', []))}")

            st.markdown("---")
            st.markdown(st.session_state.evaluation_report)

            st.markdown("---")
            col_again, col_hist_btn = st.columns([2.5, 2.5])
            with col_again:
                if st.button("🔄 Practice Another Interview", type="primary", use_container_width=True):
                    st.session_state.messages = []
                    st.session_state.last_processed_audio_hash = None
                    st.session_state.interview_started = False
                    st.session_state.interview_completed = False
                    st.session_state.evaluation_report = None
                    st.session_state.cumulative_fillers = {
                        "total_fillers": 0,
                        "total_words": 0,
                        "filler_density_pct": 0.0,
                        "breakdown": {},
                    }
                    st.rerun()
            with col_hist_btn:
                if st.button("📜 View All Past Evaluations", use_container_width=True):
                    st.session_state.current_page = "📜 History"
                    st.rerun()
