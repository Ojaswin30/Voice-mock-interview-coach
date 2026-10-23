# 🎙️ Voice Mock-Interview Coach

A real-time, interactive voice mock-interview coach powered by local speech-to-text (`faster-whisper`), low-latency LLMs on Groq, the browser's native Web Speech API (`speechSynthesis`), persistent Knowledge Graph memory, and Streamlit (`st.audio_input`).

---

## 🚀 Features

- **Resume & Job Description (JD) Intelligence**: Upload your resume (PDF or TXT) and paste the target Job Description. The app extracts your background claims and the JD requirements, seeds them into the Knowledge Graph, and tailors the interviewer to probe whether your real depth matches your resume and the role.
- **Direct Voice Recording**: Record your answers using Streamlit's native `st.audio_input` microphone recorder with zero-latency in-memory WAV decoding.
- **Local & Free Speech-to-Text**: Powered by `faster-whisper` running locally on CPU with int8 quantization. No API costs, no rate limits.
- **Persistent Candidate Memory Graph**: Tracks technical claims (probed vs. unprobed), conceptual weaknesses, covered topics, and strengths across turns and sessions. Memory is seamlessly injected into interviewer prompts so questions stay cohesive over long interviews.
- **Strict but Fair Interviewer**: Driven by Groq (`llama-3.1-8b-instant` for instant replies or `llama-3.3-70b-versatile` for deep technical rigor), roleplaying as a senior hiring manager. Replies are concise (2–4 spoken sentences) and challenge hand-wavy explanations.
- **Zero-Cost Spoken Replies**: Uses the browser's native Web Speech API (`speechSynthesis`) via an embedded HTML component. No cloud TTS subscription needed.
- **Speech Analytics & Filler Words**: Tracks filler words (*um, uh, like, you know, basically*) and measures filler word density percentage in real time.
- **In-Depth Scorecard**: Finish the interview anytime to receive a hiring decision, rating out of 10, dimension scores (Clarity, Structure, Technical Depth, Delivery), strengths, and actionable improvements with STAR reframing.

---

## 🧠 Memory Graph Architecture & Turn Flow

Each conversational turn follows this pipeline:
1. **User speaks** → `faster-whisper` transcribes the audio locally on CPU via in-memory decoding.
2. **Update the graph** → A fast, dedicated Groq call (`llama-3.1-8b-instant` in JSON mode) analyzes the answer against the interviewer's question to extract new claims, flag weak areas, record covered topics, and mark probed claims.
3. **Retrieve memory** → The candidate's persistent graph produces a targeted memory block (unprobed claims, weak topics, covered topics, and filler word tendencies).
4. **Build the prompt** → System prompt incorporates the interviewer role + resume claims + JD requirements + memory block, followed by the **last 8 turns verbatim** so the model tracks immediate conversational thread continuity.
5. **Groq replies** → The interviewer generates the next question or follow-up, which is spoken back via the Web Speech API (`speechSynthesis`).

---

## 🛠️ Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **User Interface** | [Streamlit](https://streamlit.io/) | With `st.audio_input` for microphone audio capture |
| **Speech-to-Text** | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Fast CTranslate2 Whisper inference running locally on CPU (in-memory decoding) |
| **LLM & Extraction** | [Groq Cloud](https://console.groq.com) | Low-latency inference with JSON mode extraction (`llama-3.1-8b-instant`, `llama-3.3-70b-versatile`) |
| **Document Parsing** | [pypdf](https://github.com/py-pdf/pypdf) | Pure-Python PDF extraction for candidate resumes |
| **Memory Graph** | SQLite | Persistent candidate knowledge graph tracking claims, weaknesses, and covered topics |
| **Text-to-Speech** | [Web Speech API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API) | Client-side native browser speech synthesis |

---

## 🤖 Recommended Groq LLM Models

The application dynamically detects models enabled on your Groq key and filters strictly to conversational chat LLMs:

- **`llama-3.1-8b-instant` (Default & Recommended)**: Near-instant generation speed (~800 tokens/sec), universally accessible on all free-tier accounts, and ideal for fast conversational turnarounds and JSON fact extraction.
- **`llama-3.3-70b-versatile`**: Meta's flagship open model for advanced reasoning and FAANG-level deep technical grilling.
- **`mixtral-8x7b-32768`**: Robust mixture-of-experts model for diverse technical domains.

---

## 🔒 Multi-User Privacy & Database Architecture

When deployed on Streamlit Cloud, the SQLite database (`candidate_memory.db`) resides on the server container. To guarantee complete privacy and isolation between different users:

- **Immutable Private IDs**: Every visitor is automatically provisioned a unique, non-guessable identifier (`cand_<16-character-uuid>`) stored in session state and synced to the URL query parameter (`?cid=cand_...`).
- **No ID Collisions**: Users can type their friendly display name ("Ojaswin") for the interviewer to address them politely, while all relational database foreign keys are strictly bound to the private immutable UUID.
- **Session Continuity**: Refreshing or bookmarking your URL allows you to return to the exact same profile and memory.
- **Backup & Restore**: In the sidebar under **🧠 Candidate Knowledge Graph**, click **📥 Download Memory (JSON)** to export your profile, and use **📤 Restore Memory** to load it back anytime—even across Streamlit Cloud redeployments.
- **Fresh Start**: Click **🆕 Start New Profile / Session** to generate a clean, isolated ID instantly.

---

## 📦 Setup & Deployment

### 1. Streamlit Cloud Deployment (Recommended)
1. Push your repository to GitHub.
2. Deploy the app on [Streamlit Community Cloud](https://share.streamlit.io/).
3. In **App Settings → Secrets**, add your Groq API key:
   ```toml
   GROQ_API_KEY = "gsk_your_groq_api_key_here"
   ```
   *(The key is never requested or displayed in the UI).*

### 2. Local Setup
```bash
# Clone repository
git clone https://github.com/Ojaswin30/Voice-mock-interview-coach.git
cd Voice-mock-interview-coach

# Create & activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # On Windows
# source .venv/bin/activate # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

#### Configure Secrets locally & on Streamlit Cloud
Create a `.streamlit/secrets.toml` file (or set in Streamlit Cloud App Settings → Secrets):
```toml
# .streamlit/secrets.toml

# Groq API Key (Required)
GROQ_API_KEY = "gsk_your_groq_api_key_here"

# Google OpenID Connect (OIDC) Authentication with Streamlit native st.login
[auth]
redirect_uri = "https://voice-mock-interview-coach.streamlit.app/oauth2callback" # Local: "http://localhost:8501/oauth2callback"
cookie_secret = "a_random_32_character_string_for_cookie_encryption"
client_id = "374708726653-rl5217itnoic8sdm0ql2952ggqpn7qtv.apps.googleusercontent.com"
client_secret = "GOCSPX-your-client-secret"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```
Or export via terminal:
```powershell
$env:GROQ_API_KEY="gsk_your_groq_api_key_here"
```

#### Run the App
```bash
streamlit run app.py
```

---

## 🔐 Authentication & Security

- **Strict Identity & Authorization**: Candidate identity and interview data are **never** stored in public URL query parameters (`?cid=` and `?uid=` are completely removed and ignored).
- **No URL-based session hijacking**: Pasting the app link into another browser, a phone, or an incognito window will **always** prompt the user to sign in. Nobody can access your interview history just by possessing a URL.
- **Google Sign-In**: Native OpenID Connect authentication powered by Streamlit (`st.login()` / `st.user`), securing browser sessions via encrypted, HTTP-only cookies.
- **Account Resume**: You can also sign in or create an account with your email address. All claims, weakness tracking, and scorecards are securely keyed to your account in SQLite.
- **Memory Backup & Restore**: Download and upload full JSON backups of your Knowledge Graph directly from the sidebar.

---

## 🎯 How to Use

1. **Sign In**: Log in with Google or your email. Your candidate identity and Knowledge Graph memory are tied directly to your account.
2. **Configure Role & Interviewer**: In the sidebar, select your target role, experience level, interviewer demeanor, and Groq model.
3. **Upload Resume & Paste JD**: Upload your resume (`.pdf` or `.txt`) and paste the job description. Click **⚡ Parse Resume & JD into Knowledge Graph** to extract claims and probe areas.
4. **Start Interview**: Click **🚀 Start Interview** to begin. The interviewer introduces themselves and asks a tailored opening question.
5. **Listen & Speak**: The interviewer speaks aloud automatically via browser TTS. Use the microphone recorder to speak your answer and press stop to submit.
6. **Iterate**: The interviewer challenges ambiguities, tests technical tradeoffs, asks follow-up questions, or transitions to new topics.
7. **Get Evaluated**: Click **🏁 Finish Interview & Get Scorecard** when you're done to review your rating, JD alignment, resume verification, dimension breakdown, filler word count, and model STAR answers.
