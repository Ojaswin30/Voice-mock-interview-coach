import os
import sys
import io
import wave
import uuid
import tempfile
import hashlib
import json
import re
from typing import List, Dict, Any, Optional

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from faster_whisper import WhisperModel
from groq import Groq
from dotenv import load_dotenv

from memory_graph import CandidateMemoryGraph, extract_and_update_memory
from resume_jd_parser import (
    extract_text_from_upload,
    analyze_resume_and_jd,
    seed_graph_from_resume_and_jd,
)
from auth import (
    UserManager,
    get_google_oauth_config,
    get_google_auth_url,
    exchange_code_for_user,
    verify_google_id_token,
    render_gis_signin_html,
)

load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Voice Mock-Interview Coach",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

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


def get_memory_graph() -> CandidateMemoryGraph:
    """Initializes the SQLite-backed candidate memory graph."""
    return CandidateMemoryGraph(db_path="candidate_memory.db")


user_manager = UserManager(db_path="candidate_memory.db")


def resolve_groq_api_key() -> str:
    """
    Retrieves the Groq API Key from Streamlit Secrets or environment variables.
    The key is never displayed or entered via the UI.
    """
    key = ""
    try:
        if "GROQ_API_KEY" in st.secrets:
            key = str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        pass

    if not key:
        key = os.environ.get("GROQ_API_KEY", "").strip()

    return key


def get_groq_client() -> Optional[Groq]:
    """Returns an authenticated Groq API client."""
    key = resolve_groq_api_key()
    if not key:
        return None
    return Groq(api_key=key)


# ---------------------------------------------------------
# Audio Transcription & Analysis
# ---------------------------------------------------------
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
    # Try pure in-memory decoding first
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

    # Fallback to temp file if not standard WAV
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


# ---------------------------------------------------------
# Text-to-Speech (Web Speech API Component)
# ---------------------------------------------------------
def render_speech_player(text: str, auto_play: bool = True, key_id: str = "tts_default"):
    """
    Renders an HTML/JavaScript component utilizing the browser's native Web Speech API
    (speechSynthesis). Includes autoplay handling, voice selection, rate slider, and replay controls.
    """
    clean_text = json.dumps(text)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        :root {{
          --primary-color: #2563eb;
          --primary-hover: #1d4ed8;
          --bg-color: #f8fafc;
          --card-bg: #ffffff;
          --text-color: #1e293b;
          --border-color: #e2e8f0;
          --accent-color: #10b981;
        }}
        @media (prefers-color-scheme: dark) {{
          :root {{
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --text-color: #f1f5f9;
            --border-color: #334155;
          }}
        }}
        body {{
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
          margin: 0;
          padding: 8px 12px;
          background: transparent;
          color: var(--text-color);
        }}
        .tts-card {{
          display: flex;
          align-items: center;
          justify-content: space-between;
          flex-wrap: wrap;
          gap: 12px;
          background-color: var(--card-bg);
          border: 1px solid var(--border-color);
          border-radius: 10px;
          padding: 10px 16px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .controls-left {{
          display: flex;
          align-items: center;
          gap: 10px;
        }}
        .btn {{
          display: inline-flex;
          align-items: center;
          gap: 6px;
          font-weight: 500;
          font-size: 13px;
          padding: 6px 14px;
          border-radius: 6px;
          border: none;
          cursor: pointer;
          transition: background-color 0.15s ease, transform 0.05s ease;
        }}
        .btn:active {{
          transform: scale(0.98);
        }}
        .btn-primary {{
          background-color: var(--primary-color);
          color: white;
        }}
        .btn-primary:hover {{
          background-color: var(--primary-hover);
        }}
        .btn-secondary {{
          background-color: transparent;
          border: 1px solid var(--border-color);
          color: var(--text-color);
        }}
        .btn-secondary:hover {{
          background-color: rgba(125, 125, 125, 0.1);
        }}
        .status-badge {{
          font-size: 12px;
          color: #64748b;
          display: flex;
          align-items: center;
          gap: 6px;
        }}
        .pulse {{
          display: inline-block;
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: var(--accent-color);
          animation: pulse-ring 1.5s infinite;
        }}
        @keyframes pulse-ring {{
          0% {{ opacity: 1; transform: scale(0.9); }}
          50% {{ opacity: 0.4; transform: scale(1.2); }}
          100% {{ opacity: 1; transform: scale(0.9); }}
        }}
        .rate-selector {{
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 12px;
          color: #64748b;
        }}
        select {{
          background: var(--card-bg);
          color: var(--text-color);
          border: 1px solid var(--border-color);
          border-radius: 4px;
          padding: 3px 6px;
          font-size: 12px;
        }}
      </style>
    </head>
    <body>
      <div class="tts-card">
        <div class="controls-left">
          <button id="playBtn" class="btn btn-primary" onclick="togglePlay()">
            <span id="playIcon">🔊</span> <span id="playText">Play Voice</span>
          </button>
          <button id="stopBtn" class="btn btn-secondary" onclick="stopSpeech()">
            ⏹ Stop
          </button>
          <div id="statusBadge" class="status-badge" style="display: none;">
            <span class="pulse"></span> <span>Speaking...</span>
          </div>
        </div>
        <div class="rate-selector">
          <label for="speedSelect">Speed:</label>
          <select id="speedSelect" onchange="updateRate()">
            <option value="0.85">0.85x</option>
            <option value="1.0" selected>1.0x</option>
            <option value="1.15">1.15x</option>
            <option value="1.3">1.3x</option>
          </select>
        </div>
      </div>

      <script>
        const messageText = {clean_text};
        let isSpeaking = false;
        let selectedRate = 1.0;
        let synth = window.speechSynthesis;
        let utterance = null;

        function updateRate() {{
          const sel = document.getElementById("speedSelect");
          selectedRate = parseFloat(sel.value);
          if (isSpeaking) {{
            synth.cancel();
            speak();
          }}
        }}

        function speak() {{
          if (!synth) {{
            alert("Speech synthesis is not supported in this browser.");
            return;
          }}
          synth.cancel();

          utterance = new SpeechSynthesisUtterance(messageText);
          utterance.rate = selectedRate;
          utterance.pitch = 1.0;

          // Pick natural English voice if available
          const voices = synth.getVoices();
          const preferredVoice = voices.find(v => 
            v.lang.startsWith("en") && (v.name.includes("Google") || v.name.includes("Natural") || v.name.includes("Jenny") || v.name.includes("David") || v.name.includes("Samantha"))
          ) || voices.find(v => v.lang.startsWith("en"));

          if (preferredVoice) {{
            utterance.voice = preferredVoice;
          }}

          utterance.onstart = function() {{
            isSpeaking = true;
            document.getElementById("playIcon").innerText = "⏸";
            document.getElementById("playText").innerText = "Pause";
            document.getElementById("statusBadge").style.display = "flex";
          }};

          utterance.onend = function() {{
            isSpeaking = false;
            document.getElementById("playIcon").innerText = "🔁";
            document.getElementById("playText").innerText = "Replay";
            document.getElementById("statusBadge").style.display = "none";
          }};

          utterance.onerror = function(event) {{
            console.warn("SpeechSynthesis error:", event);
            isSpeaking = false;
            document.getElementById("playIcon").innerText = "🔊";
            document.getElementById("playText").innerText = "Play Voice";
            document.getElementById("statusBadge").style.display = "none";
          }};

          synth.speak(utterance);
        }}

        function togglePlay() {{
          if (synth.speaking && !synth.paused) {{
            synth.pause();
            isSpeaking = false;
            document.getElementById("playIcon").innerText = "▶️";
            document.getElementById("playText").innerText = "Resume";
            document.getElementById("statusBadge").style.display = "none";
          }} else if (synth.paused) {{
            synth.resume();
            isSpeaking = true;
            document.getElementById("playIcon").innerText = "⏸";
            document.getElementById("playText").innerText = "Pause";
            document.getElementById("statusBadge").style.display = "flex";
          }} else {{
            speak();
          }}
        }}

        function stopSpeech() {{
          synth.cancel();
          isSpeaking = false;
          document.getElementById("playIcon").innerText = "🔊";
          document.getElementById("playText").innerText = "Play Voice";
          document.getElementById("statusBadge").style.display = "none";
        }}

        if (speechSynthesis.onvoiceschanged !== undefined) {{
          speechSynthesis.onvoiceschanged = () => {{}};
        }}

        if ({str(auto_play).lower()}) {{
          window.addEventListener("DOMContentLoaded", () => {{
            try {{
              speak();
            }} catch(e) {{
              console.log("Autoplay prevented:", e);
            }}
          }});
        }}
      </script>
    </body>
    </html>
    """
    components.html(html_code, height=65, scrolling=False)


# ---------------------------------------------------------
# LLM: Interviewer & Feedback Generation via Groq
# ---------------------------------------------------------
def fetch_available_groq_models(groq_client: Optional[Groq]) -> List[str]:
    """
    Fetches the actual list of available models from the Groq API for this account.
    Filters strictly to standard chat completion LLMs (Llama, Mixtral, Gemma, Qwen, DeepSeek)
    and eliminates any audio, TTS, vision, guard, or specialized endpoints.
    """
    default_fallback_list = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ]
    if not groq_client:
        return default_fallback_list

    try:
        models_data = groq_client.models.list().data
        excluded_keywords = [
            "whisper", "guard", "embed", "safeguard", "vision", "tool",
            "orpheus", "canopylabs", "audio", "tts", "voice", "speech"
        ]
        allowed_chat_keywords = ["llama", "mixtral", "mistral", "gemma", "qwen", "deepseek"]

        chat_models = [
            m.id for m in models_data
            if any(ok in m.id.lower() for ok in allowed_chat_keywords)
            and not any(ex in m.id.lower() for ex in excluded_keywords)
        ]
        if chat_models:
            priority = [
                "llama-3.1-8b-instant",
                "llama-3.3-70b-versatile",
                "llama-3.1-70b-versatile",
                "mixtral-8x7b-32768",
                "gemma2-9b-it",
            ]
            chat_models.sort(key=lambda m: priority.index(m) if m in priority else 100)
            return chat_models
    except Exception:
        pass

    return default_fallback_list


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
    
    # Keep recent turns verbatim (last 8 turns) so the model follows the thread
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
# Session State Initialization
# ---------------------------------------------------------
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_processed_audio_hash" not in st.session_state:
        st.session_state.last_processed_audio_hash = None
    if "interview_started" not in st.session_state:
        st.session_state.interview_started = False
    if "interview_completed" not in st.session_state:
        st.session_state.interview_completed = False
    if "evaluation_report" not in st.session_state:
        st.session_state.evaluation_report = None
    if "cumulative_fillers" not in st.session_state:
        st.session_state.cumulative_fillers = {
            "total_fillers": 0,
            "total_words": 0,
            "filler_density_pct": 0.0,
            "breakdown": {},
        }
    if "resume_text" not in st.session_state:
        st.session_state.resume_text = ""
    if "jd_text" not in st.session_state:
        st.session_state.jd_text = ""
    if "resume_analysis" not in st.session_state:
        st.session_state.resume_analysis = None
    if "user" not in st.session_state:
        st.session_state.user = None


init_session_state()
memory_graph = get_memory_graph()

# ---------------------------------------------------------
# Groq Authentication Check
# ---------------------------------------------------------
groq_client = get_groq_client()

if not groq_client:
    st.title("🎙️ Voice Mock-Interview Coach")
    st.error(
        "🔑 **GROQ_API_KEY not detected.**\n\n"
        "Please configure your `GROQ_API_KEY` in:\n"
        "- **Streamlit Cloud**: App Settings → Secrets (`GROQ_API_KEY = \"gsk_...\"`)\n"
        "- **Local Development**: `.streamlit/secrets.toml` or `.env` file."
    )
    st.info("You can obtain a free Groq API key at [console.groq.com](https://console.groq.com).")
    st.stop()


# ---------------------------------------------------------
# User Authentication & Google Sign-In
# ---------------------------------------------------------
oauth_cfg = get_google_oauth_config()

# Always clean up legacy or malicious cid/uid query params to prevent unauthorized URL-based session hijacking
if "cid" in st.query_params:
    try:
        del st.query_params["cid"]
    except Exception:
        pass
if "uid" in st.query_params:
    try:
        del st.query_params["uid"]
    except Exception:
        pass

# 1. Native Streamlit OpenID Connect (st.login / st.user) integration
if hasattr(st, "user") and getattr(st.user, "is_logged_in", False):
    user_email = getattr(st.user, "email", "") or ""
    user_name = getattr(st.user, "name", "") or (user_email.split("@")[0].title() if user_email else "Candidate")
    user_sub = getattr(st.user, "sub", "") or hashlib.md5(user_email.encode()).hexdigest()[:16]
    user_id = f"google_{user_sub}"
    db_user = user_manager.upsert_user(user_id, user_email, user_name, "")
    st.session_state.user = db_user
    st.session_state.candidate_id = user_id

# 2. Handle Google ID token (?google_token=...) callback
google_token = st.query_params.get("google_token")
if google_token and not st.session_state.get("user"):
    with st.spinner("Authenticating with Google..."):
        google_user = verify_google_id_token(google_token)
        if google_user:
            user_id = f"google_{google_user.get('sub', '')}"
            db_user = user_manager.upsert_user(
                user_id=user_id,
                email=google_user.get("email", ""),
                name=google_user.get("name", "Google User"),
                picture_url=google_user.get("picture", ""),
            )
            st.session_state.user = db_user
            st.session_state.candidate_id = user_id
            try:
                del st.query_params["google_token"]
            except Exception:
                pass
            st.rerun()

# 3. Handle Google OAuth code (?code=...) callback
auth_code = st.query_params.get("code")
if auth_code and oauth_cfg.get("client_id") and oauth_cfg.get("client_secret") and not st.session_state.get("user"):
    with st.spinner("Authenticating with Google..."):
        google_user = exchange_code_for_user(auth_code, oauth_cfg)
        if google_user:
            user_id = f"google_{google_user.get('sub', '')}"
            db_user = user_manager.upsert_user(
                user_id=user_id,
                email=google_user.get("email", ""),
                name=google_user.get("name", "Google User"),
                picture_url=google_user.get("picture", ""),
            )
            st.session_state.user = db_user
            st.session_state.candidate_id = user_id
            try:
                del st.query_params["code"]
            except Exception:
                pass
            st.rerun()

# 4. If not signed in, show Sign-In portal (NEVER auto-log in via unauthenticated URL parameters)
if not st.session_state.get("user"):
    st.title("🎙️ Voice Mock-Interview Coach")
    st.markdown("Sign in to sync your **Knowledge Graph memory**, **resume claims**, and **interview performance** permanently to your account.")

    col_login, _ = st.columns([4, 3])
    with col_login:
        st.markdown("### 🔐 Sign In")

        # Native Streamlit OIDC Login if configured in secrets [auth]
        has_native_auth = False
        try:
            if "auth" in st.secrets and hasattr(st, "login"):
                has_native_auth = True
        except Exception:
            pass

        has_google_secret = bool(oauth_cfg.get("client_secret"))
        has_google_client = bool(oauth_cfg.get("client_id"))

        # 1. Google Identity Services (GIS) Button from Campus Voting Platform
        if has_google_client:
            components.html(render_gis_signin_html(oauth_cfg["client_id"]), height=60)

        # 2. Native Streamlit OIDC or Direct OAuth fallback
        if has_native_auth:
            if st.button("Sign in with Google (OIDC)", type="primary", use_container_width=True):
                st.login()
        elif has_google_client and has_google_secret:
            google_auth_url = get_google_auth_url(oauth_cfg)
            st.markdown(
                f"""
                <a href="{google_auth_url}" target="_self" style="
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    gap: 12px;
                    width: 100%;
                    background-color: #ffffff;
                    color: #1f2937;
                    border: 1px solid #d1d5db;
                    border-radius: 8px;
                    padding: 10px 18px;
                    font-size: 14px;
                    font-weight: 500;
                    text-decoration: none;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    margin-top: 10px;
                ">
                    <svg width="18" height="18" viewBox="0 0 48 48">
                        <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
                        <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
                        <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>
                        <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
                    </svg>
                    Direct Google OAuth Sign-In
                </a>
                """,
                unsafe_allow_html=True,
            )

    st.stop()


# ---------------------------------------------------------
# Sidebar Configuration (Authenticated)
# ---------------------------------------------------------
with st.sidebar:
    st.title("🎙️ Interview Settings")

    current_user = st.session_state.user
    current_user = current_user or {}
    candidate_id = current_user.get("id", "guest")
    candidate_name = current_user.get("name", "Candidate")

    st.markdown(f"👤 **{candidate_name}**")
    st.caption(f"📧 `{current_user.get('email', '')}`")

    if st.button("🚪 Log Out", use_container_width=True):
        if hasattr(st, "logout") and getattr(getattr(st, "user", None), "is_logged_in", False):
            try:
                st.logout()
            except Exception:
                pass
        st.session_state.user = None
        st.session_state.candidate_id = None
        st.session_state.messages = []
        st.session_state.interview_started = False
        st.session_state.interview_completed = False
        st.session_state.resume_analysis = None
        st.rerun()

    st.markdown("---")

    # Target Role Selection
    role_presets = [
        "Software Engineer (Full-Stack)",
        "Backend / Systems Engineer",
        "Frontend Engineer",
        "Machine Learning / AI Engineer",
        "Data Scientist",
        "Engineering Manager / Tech Lead",
        "Product Manager",
        "Behavioral & Leadership",
        "Custom Role...",
    ]
    selected_role_preset = st.selectbox("Target Role", role_presets, index=0)
    if selected_role_preset == "Custom Role...":
        target_role = st.text_input("Enter Target Role / Description", value="DevOps Engineer")
    else:
        target_role = selected_role_preset

    # Register candidate in graph
    memory_graph.register_candidate(candidate_id, candidate_name, target_role)

    # Seniority Level
    experience_level = st.selectbox(
        "Experience Level",
        ["Entry-Level / Intern", "Mid-Level (2-4 yrs)", "Senior (5-8 yrs)", "Staff / Principal (8+ yrs)", "Director / Executive"],
        index=2,
    )

    # Interview Style
    interview_style = st.selectbox(
        "Interviewer Demeanor",
        [
            "Strict & Rigorous (FAANG-Style)",
            "Standard Professional Hiring Manager",
            "Rapid-Fire Technical Inquisitor",
            "Behavioral & Deep Probe (STAR Method)",
            "Supportive & Constructive Coach",
        ],
        index=0,
    )

    # Model & Whisper Settings
    st.markdown("---")
    st.subheader("Engine Settings")
    whisper_model_size = st.selectbox(
        "Whisper STT Model",
        ["base", "small"],
        index=0,
        help="'base' is lightning fast on CPU. 'small' offers slightly higher accuracy.",
    )

    available_models = fetch_available_groq_models(groq_client)
    default_model_idx = 0
    if "llama-3.1-8b-instant" in available_models:
        default_model_idx = available_models.index("llama-3.1-8b-instant")
    elif "llama-3.3-70b-versatile" in available_models:
        default_model_idx = available_models.index("llama-3.3-70b-versatile")

    groq_model_name = st.selectbox(
        "Groq LLM Model",
        available_models,
        index=default_model_idx,
        help="Fast models detected from your Groq account.",
    )

    # Candidate Knowledge Graph Inspector
    st.markdown("---")
    with st.expander("🧠 Candidate Knowledge Graph", expanded=False):
        graph_state = memory_graph.get_full_graph_state(candidate_id)
        claims = graph_state.get("claims", [])
        weaknesses = graph_state.get("weaknesses", [])
        topics = graph_state.get("topics", [])
        strengths = graph_state.get("strengths", [])

        st.caption(f"Persistent memory for **{candidate_name}**")

        st.markdown("**📌 Claims:**")
        if claims:
            for c in claims[:6]:
                status = "✅ Probed" if c["is_probed"] else "⏳ Unprobed"
                st.markdown(f"- *{c['claim_text']}* ({status})")
        else:
            st.caption("No claims extracted yet.")

        st.markdown("**⚠️ Weaknesses Flagged:**")
        if weaknesses:
            for w in weaknesses[:5]:
                obs = f": {w['observation']}" if w.get("observation") else ""
                st.markdown(f"- **{w['topic']}**{obs}")
        else:
            st.caption("No weak areas flagged yet.")

        st.markdown("**📚 Topics Covered:**")
        if topics:
            topics_str = ", ".join([t["topic_name"] for t in topics[:8]])
            st.markdown(f"*{topics_str}*")
        else:
            st.caption("No topics recorded yet.")

        if st.button("🗑️ Reset Candidate Memory", help="Wipes this candidate's memory graph entries", use_container_width=True):
            memory_graph.reset_candidate_memory(candidate_id)
            st.success("Memory cleared!")
            st.rerun()

        st.markdown("---")
        st.markdown("**💾 Backup & Restore Memory:**")
        backup_dict = memory_graph.export_candidate_memory(candidate_id)
        st.download_button(
            "📥 Download Memory (JSON)",
            data=json.dumps(backup_dict, indent=2),
            file_name=f"{candidate_id}_memory.json",
            mime="application/json",
            use_container_width=True,
            help="Download your memory graph as JSON to keep across Streamlit restarts",
        )

        uploaded_backup = st.file_uploader(
            "Restore Memory (.json)",
            type=["json"],
            key="restore_json_file",
            help="Upload a previously exported candidate memory file",
        )
        if uploaded_backup is not None:
            try:
                uploaded_backup.seek(0)
                loaded_data = json.loads(uploaded_backup.read().decode("utf-8"))
                memory_graph.import_candidate_memory(candidate_id, loaded_data)
                st.success("Memory restored!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to restore memory: {e}")

    # Resume & JD Status in Sidebar
    if st.session_state.resume_analysis:
        st.markdown("---")
        st.caption(f"📄 **Resume:** {st.session_state.resume_analysis.get('candidate_name', 'Loaded')} ({len(st.session_state.resume_analysis.get('key_claims', []))} claims)")
        if st.session_state.jd_text:
            st.caption("📋 **Job Description:** Linked")
    elif st.session_state.resume_text:
        st.markdown("---")
        st.caption("📄 **Resume:** Loaded")

    st.markdown("---")
    if st.button("🔄 Reset / Start New Interview", use_container_width=True):
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


# ---------------------------------------------------------
# Main App Header & Status
# ---------------------------------------------------------
st.title("🎙️ Voice Mock-Interview Coach")
st.caption(
    f"Candidate: **{candidate_name}** | Role: **{target_role}** ({experience_level}) | Demeanor: **{interview_style}**"
)


# ---------------------------------------------------------
# Helpers for Context Summaries
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Start Interview Trigger & Resume / JD Setup
# ---------------------------------------------------------
if not st.session_state.interview_started:
    st.markdown(
        """
        ### Welcome to your voice mock interview!
        Prepare for your interview by providing your **Resume** and the **Target Job Description**. 
        The interviewer will analyze your actual claims, probe for gaps, and test your depth against the JD.
        """
    )

    # Resume & JD Input Section
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
                # Ensure resume text is fresh and captured from upload or text
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
                        if analysis and (analysis.get("key_claims") or analysis.get("experience_summary")):
                            st.session_state.resume_analysis = analysis
                            seed_graph_from_resume_and_jd(memory_graph, candidate_id, analysis)
                            st.success("✅ Resume & JD analyzed! Claims and probe areas seeded to Knowledge Graph.")
                            st.rerun()
                        else:
                            st.error(
                                "Could not extract structured data from the resume. "
                                "If this is a scanned/image PDF, please copy and paste the text into the box above, "
                                "and make sure a chat LLM (such as `llama-3.1-8b-instant`) is selected in the sidebar."
                            )

    # Show Analyzed Profile Preview if Available
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
                    c_txt = claim_item.get("claim", claim_item) if isinstance(claim_item, dict) else claim_item
                    st.markdown(f"- {c_txt}")
            with col_a2:
                st.markdown("**🎯 Core JD Requirements:**")
                for req in analysis.get("jd_requirements", [])[:5]:
                    st.markdown(f"- {req}")
                st.markdown("**⚠️ Identified Probe / Focus Areas:**")
                for probe in analysis.get("probe_areas", [])[:4]:
                    st.markdown(f"- 🔍 *{probe}*")

    # Show Memory Preview if candidate has previous history
    existing_memory = memory_graph.get_candidate_memory_block(candidate_id)
    if existing_memory:
        with st.expander("🧠 Persistent Memory Graph for Profile", expanded=False):
            st.code(existing_memory, language="markdown")

    st.markdown("<br>", unsafe_allow_html=True)
    col_start, _ = st.columns([2, 5])
    with col_start:
        if st.button("🚀 Start Interview", type="primary", use_container_width=True):
            with st.spinner("Preparing your personalized interviewer..."):
                filler_note = ""
                mem_block = memory_graph.get_candidate_memory_block(candidate_id, filler_notes=filler_note)
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
                    st.info("Try selecting a different model such as `llama-3.1-8b-instant` in the sidebar or check your Groq API key.")
    st.stop()


# ---------------------------------------------------------
# Interview Active View: Chat History & Voice Player
# ---------------------------------------------------------

# Display Conversation History
for idx, msg in enumerate(st.session_state.messages):
    if msg["role"] == "assistant":
        with st.chat_message("assistant", avatar="🧑‍💼"):
            st.markdown(f"**Interviewer:** {msg['content']}")
            is_latest = (idx == len(st.session_state.messages) - 1)
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

# ---------------------------------------------------------
# Response Controls & Audio Recording
# ---------------------------------------------------------
if not st.session_state.interview_completed:
    st.subheader("🎤 Your Turn to Answer")

    whisper_model = load_whisper_model(whisper_model_size)

    # Audio input component
    audio_data = st.audio_input("Record your answer (Click to start, click again to stop and submit)")

    # Optional text fallback input
    with st.expander("⌨️ Or type your answer manually (fallback)", expanded=False):
        typed_answer = st.text_area("Type your response here:", height=80, key="typed_input")
        send_typed_btn = st.button("Submit Typed Answer")

    # Helper function to process an answer turn
    def process_candidate_answer(answer_text: str):
        # 1. Analyze filler words
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

        # 2. Add answer to session messages
        st.session_state.messages.append({
            "role": "user",
            "content": answer_text,
            "filler_info": filler_stats,
        })

        # 3. Find the last question asked by interviewer
        last_question = ""
        for m in reversed(st.session_state.messages[:-1]):
            if m["role"] == "assistant":
                last_question = m["content"]
                break

        # 4. Extract new facts and update the graph (separate cheap Groq JSON call)
        with st.spinner("🧠 Updating candidate memory graph..."):
            extract_and_update_memory(
                groq_client=groq_client,
                graph=memory_graph,
                candidate_id=candidate_id,
                last_question=last_question,
                user_answer=answer_text,
                model_name="llama-3.1-8b-instant",
            )

        # 5. Retrieve memory block from graph
        top_fillers = list(c_fillers["breakdown"].keys())[:2]
        f_note = f"frequent '{', '.join(top_fillers)}'" if top_fillers else ""
        mem_block = memory_graph.get_candidate_memory_block(candidate_id, filler_notes=f_note)

        # 6. Retrieve resume and JD context
        resume_sum, jd_sum = get_resume_and_jd_summaries(candidate_name, target_role)

        # 7. Groq generates interviewer response (with last 8 turns verbatim + memory block + resume/JD)
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

    # Process Audio Input
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
                st.warning("Could not detect any speech in the audio recording. Please try speaking closer to the microphone.")

    # Process Typed Fallback
    if send_typed_btn and typed_answer.strip():
        process_candidate_answer(typed_answer.strip())

    # Finish Interview Action Bar
    st.markdown("<br>", unsafe_allow_html=True)
    c_btn1, c_btn2, _ = st.columns([2, 2, 3])
    with c_btn1:
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
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to generate evaluation report: {str(e)}")

# ---------------------------------------------------------
# Interview Completed View: Scorecard & Metrics
# ---------------------------------------------------------
if st.session_state.interview_completed and st.session_state.evaluation_report:
    st.success("🎉 Mock Interview Finished! Here is your hiring assessment.")

    # Show metrics banner
    m1, m2, m3, m4 = st.columns(4)
    user_turns = sum(1 for m in st.session_state.messages if m["role"] == "user")
    graph_summary = memory_graph.get_full_graph_state(candidate_id)
    probed_count = sum(1 for c in graph_summary.get("claims", []) if c.get("is_probed"))

    m1.metric("Questions Answered", user_turns)
    m2.metric("Total Words Spoken", st.session_state.cumulative_fillers.get("total_words", 0))
    m3.metric(
        "Filler Word Density",
        f"{st.session_state.cumulative_fillers.get('filler_density_pct', 0.0)}%",
        help="Target: < 2.5% for executive & senior roles",
    )
    m4.metric("Claims Probed", f"{probed_count} / {len(graph_summary.get('claims', []))}")

    st.markdown("---")
    st.markdown(st.session_state.evaluation_report)

    st.markdown("---")
    col_again, _ = st.columns([2, 4])
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
