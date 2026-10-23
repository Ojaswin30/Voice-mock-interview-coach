"""
Voice Mock-Interview Coach - Main Application Entrypoint.
Orchestrates authentication, session initialization, sidebar navigation, 
and modular view dispatching.
"""
import streamlit as st
from dotenv import load_dotenv

# Load local environment variables immediately before any view imports
load_dotenv()

from memory_graph import CandidateMemoryGraph
from auth import UserManager
from views import (
    render_home_page,
    render_profile_page,
    render_interview_page,
    render_history_page,
    render_persona_page,
    render_model_settings_page,
    render_auth_portal,
    handle_oauth_callbacks,
    get_groq_client,
    fetch_available_groq_models,
)

# Set page configuration
st.set_page_config(
    page_title="Voice Mock-Interview Coach",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    """Initializes persistent application session variables."""
    defaults = {
        "user": None,
        "candidate_id": None,
        "messages": [],
        "last_processed_audio_hash": None,
        "interview_started": False,
        "interview_completed": False,
        "evaluation_report": None,
        "resume_text": "",
        "jd_text": "",
        "resume_analysis": None,
        "current_page": "🏠 Home",
        "cumulative_fillers": {
            "total_fillers": 0,
            "total_words": 0,
            "filler_density_pct": 0.0,
            "breakdown": {},
        },
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()

# ---------------------------------------------------------
# Core Services & Authentication
# ---------------------------------------------------------
user_manager = UserManager(db_path="candidate_memory.db")
memory_graph = CandidateMemoryGraph(db_path="candidate_memory.db")
groq_client = get_groq_client()

# Process any pending OAuth callbacks (Streamlit Native / GitHub)
handle_oauth_callbacks(user_manager)

# Groq API Key Verification
if not groq_client:
    st.title("🎙️ Voice Mock-Interview Coach")
    st.error(
        "🔑 **GROQ_API_KEY not detected.**\n\n"
        "Please configure your `GROQ_API_KEY` in:\n"
        "- **Streamlit Cloud**: App Settings → Secrets (`GROQ_API_KEY = \"...\"`)\n"
        "- **Local Development**: `.streamlit/secrets.toml` or `.env` file."
    )
    st.info("You can obtain a free Groq API key at [console.groq.com](https://console.groq.com).")
    st.stop()

# Auth Gate: If user not logged in, render the candidate portal
if not st.session_state.get("user"):
    render_auth_portal(user_manager)
    st.stop()

# ---------------------------------------------------------
# Authenticated App Context & Candidate Context
# ---------------------------------------------------------
current_user = st.session_state.user or {}
candidate_id = current_user.get("id", "guest")
candidate_name = current_user.get("name", "Candidate")
candidate_email = current_user.get("email", "")

if "target_role" not in st.session_state:
    st.session_state.target_role = (
        current_user.get("target_role_decrypted")
        or "Software Engineer (Full-Stack)"
    )
if "experience_level" not in st.session_state:
    st.session_state.experience_level = (
        current_user.get("experience_level_decrypted")
        or "Senior (5-8 yrs)"
    )
if "interview_style" not in st.session_state:
    st.session_state.interview_style = "Strict & Rigorous (FAANG-Style)"
if "whisper_model_size" not in st.session_state:
    st.session_state.whisper_model_size = "base"

available_models = fetch_available_groq_models(groq_client)
if "groq_model_name" not in st.session_state or st.session_state.groq_model_name not in available_models:
    default_groq = "llama-3.1-8b-instant" if "llama-3.1-8b-instant" in available_models else (available_models[0] if available_models else "llama-3.1-8b-instant")
    st.session_state.groq_model_name = default_groq

# Ensure candidate is registered in knowledge graph
memory_graph.register_candidate(candidate_id, candidate_name, st.session_state.target_role)

# ---------------------------------------------------------
# Sidebar Navigation Menu (Modular Multi-Page Setup)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎙️ Coach Portal")
    st.markdown(f"👤 **{candidate_name}**")
    if candidate_email:
        st.caption(f"📧 `{candidate_email}`")

    st.markdown("---")

    # Inject Rectangular Navigation Tiles CSS (Equal width, compact height, tight spacing, no bullets)
    st.markdown(
        """
        <style>
        /* 1. Outer radio wrapper stretches full width */
        [data-testid="stSidebar"] div[data-testid="stRadio"] {
            width: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
        }

        /* 2. Completely eliminate the phantom widget label above Home */
        [data-testid="stSidebar"] div[data-testid="stRadio"] > label,
        [data-testid="stSidebar"] div[data-testid="stRadio"] label[data-testid="stWidgetLabel"] {
            display: none !important;
            height: 0 !important;
            min-height: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            border: none !important;
            background: transparent !important;
        }

        /* 3. Radiogroup container: full 100% width, stretch items equally, reduced in-between gap */
        [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] {
            display: flex !important;
            flex-direction: column !important;
            align-items: stretch !important;
            gap: 4px !important;
            width: 100% !important;
        }

        /* 4. Ensure each option container stretches to 100% equal width */
        [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > div {
            width: 100% !important;
            align-self: stretch !important;
            display: flex !important;
        }

        /* 5. Sleek, uniform rectangular tiles: reduced height, equal width */
        [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label {
            width: 100% !important;
            align-self: stretch !important;
            background: rgba(30, 41, 59, 0.6) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 8px !important;
            padding: 7px 12px !important;
            min-height: 36px !important;
            margin: 0 !important;
            cursor: pointer !important;
            transition: all 0.15s ease !important;
            box-sizing: border-box !important;
            display: flex !important;
            align-items: center !important;
            justify-content: flex-start !important;
        }

        /* 6. Completely hide radio bullet circles */
        [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label div[data-testid="stRadioCircle"] {
            display: none !important;
        }
        [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label div:first-child:not([data-testid="stMarkdownContainer"]) {
            display: none !important;
        }

        /* 7. Tile text formatting */
        [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label [data-testid="stMarkdownContainer"] {
            width: 100% !important;
        }
        [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label p {
            font-size: 0.88rem !important;
            font-weight: 500 !important;
            margin: 0 !important;
            color: #cbd5e1 !important;
            line-height: 1.3 !important;
        }

        /* 8. Hover effect */
        [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
            background: rgba(59, 130, 246, 0.14) !important;
            border-color: rgba(59, 130, 246, 0.35) !important;
        }
        [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:hover p {
            color: #f8fafc !important;
        }

        /* 9. Active selected rectangular tile */
        [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.32), rgba(59, 130, 246, 0.18)) !important;
            border: 1px solid #3b82f6 !important;
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.22) !important;
        }
        [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p {
            color: #60a5fa !important;
            font-weight: 600 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    menu_pages = [
        "🏠 Home",
        "👤 Profile",
        "🎙️ Interview",
        "📜 History",
        "🧠 Persona",
        "⚙️ Model",
    ]
    if "current_page" not in st.session_state:
        st.session_state.current_page = "🏠 Home"

    selected_page = st.radio(
        "Navigation",
        menu_pages,
        index=menu_pages.index(st.session_state.current_page) if st.session_state.current_page in menu_pages else 0,
        label_visibility="collapsed",
        key="sidebar_navigation_radio",
    )
    st.session_state.current_page = selected_page

    st.markdown("---")

    # Status indicators in sidebar
    if st.session_state.interview_started and not st.session_state.interview_completed:
        st.warning("🔴 **Interview in progress**")
        if selected_page != "🎙️ Interview":
            if st.button("Return to Interview", use_container_width=True):
                st.session_state.current_page = "🎙️ Interview"
                st.rerun()

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
        st.session_state.current_page = "🏠 Home"
        st.rerun()

# ---------------------------------------------------------
# Modular Page Dispatcher (Calls Dedicated View Modules)
# ---------------------------------------------------------
if selected_page == "🏠 Home":
    render_home_page()

elif selected_page == "👤 Profile":
    render_profile_page(
        user_manager=user_manager,
        current_user=current_user,
    )

elif selected_page == "🎙️ Interview":
    render_interview_page(
        groq_client=groq_client,
        memory_graph=memory_graph,
        user_manager=user_manager,
        current_user=current_user,
    )

elif selected_page == "📜 History":
    render_history_page(
        user_manager=user_manager,
        candidate_id=candidate_id,
    )

elif selected_page == "🧠 Persona":
    render_persona_page(
        memory_graph=memory_graph,
        candidate_id=candidate_id,
        candidate_name=candidate_name,
    )

elif selected_page == "⚙️ Model":
    render_model_settings_page(
        groq_client=groq_client,
    )
