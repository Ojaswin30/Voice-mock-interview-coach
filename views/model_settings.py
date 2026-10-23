"""
Model Settings Page View for Voice Mock-Interview Coach.
Configures Groq LLM model, Interviewer Demeanor/Persona, 
faster-whisper STT parameters, and Browser Web Speech TTS audio.
"""
import os
from typing import List, Optional
import streamlit as st
from groq import Groq

from .speech import render_speech_player


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
    try:
        return Groq(api_key=key)
    except Exception:
        return None


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


def render_model_settings_page(groq_client: Optional[Groq] = None):
    """Renders the Engine & Model settings configuration page."""
    st.title("⚙️ Engine & Model Configuration")
    st.caption("Customize speech recognition, Groq LLM models, and interviewer personality.")

    available_models = fetch_available_groq_models(groq_client)
    groq_model_name = st.session_state.get("groq_model_name", "llama-3.1-8b-instant")
    interview_style = st.session_state.get("interview_style", "Strict & Rigorous (FAANG-Style)")
    whisper_model_size = st.session_state.get("whisper_model_size", "base")

    with st.container(border=True):
        st.subheader("🤖 Large Language Model (Groq)")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            groq_idx = (
                available_models.index(groq_model_name)
                if groq_model_name in available_models
                else 0
            )
            new_groq_model = st.selectbox(
                "Active Groq LLM Model",
                available_models,
                index=groq_idx,
                help="High-speed open models running on Groq LPUs.",
            )
            st.session_state.groq_model_name = new_groq_model

        with col_m2:
            demeanor_options = [
                "Strict & Rigorous (FAANG-Style)",
                "Standard Professional Hiring Manager",
                "Rapid-Fire Technical Inquisitor",
                "Behavioral & Deep Probe (STAR Method)",
                "Supportive & Constructive Coach",
            ]
            dem_idx = (
                demeanor_options.index(interview_style)
                if interview_style in demeanor_options
                else 0
            )
            new_demeanor = st.selectbox(
                "Interviewer Demeanor / Persona",
                demeanor_options,
                index=dem_idx,
            )
            st.session_state.interview_style = new_demeanor

    with st.container(border=True):
        st.subheader("🎙️ Speech Recognition (faster-whisper STT)")
        col_stt1, col_stt2 = st.columns(2)
        with col_stt1:
            whisper_options = ["base", "small"]
            whisper_idx = (
                whisper_options.index(whisper_model_size)
                if whisper_model_size in whisper_options
                else 0
            )
            new_whisper_size = st.selectbox(
                "Whisper STT Model Size",
                whisper_options,
                index=whisper_idx,
                help="'base' is optimized for CPU real-time inference. 'small' offers slightly higher accuracy.",
            )
            st.session_state.whisper_model_size = new_whisper_size
        with col_stt2:
            st.caption("STT Engine: **faster-whisper (CTranslate2)**")
            st.caption("Quantization: **INT8 CPU (optimized)**")
            st.caption("Latency: **~150ms - 350ms per audio chunk**")

    with st.container(border=True):
        st.subheader("🔊 Speech Synthesis (TTS)")
        st.markdown("Browser Web Speech API is used for zero-latency streaming interviewer audio.")
        render_speech_player(
            "Hello, this is a test of the speech synthesis engine. I am ready for your technical interview.",
            auto_play=False,
            key_id="tts_test_preview",
        )

    st.success("✅ Engine settings are automatically saved and applied to your interview sessions.")
