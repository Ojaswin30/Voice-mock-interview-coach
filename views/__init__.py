"""
Views package for Voice Mock-Interview Coach.
Modular components for Home, Profile, Interview, History, Persona, Model Settings,
Authentication Portal, and Speech Player.
"""
from .home import render_home_page
from .profile import render_profile_page
from .interview import render_interview_page
from .history import render_history_page
from .persona import render_persona_page
from .model_settings import (
    render_model_settings_page,
    resolve_groq_api_key,
    get_groq_client,
    fetch_available_groq_models,
)
from .auth_portal import render_auth_portal, handle_oauth_callbacks
from .speech import render_speech_player

__all__ = [
    "render_home_page",
    "render_profile_page",
    "render_interview_page",
    "render_history_page",
    "render_persona_page",
    "render_model_settings_page",
    "render_auth_portal",
    "handle_oauth_callbacks",
    "render_speech_player",
    "resolve_groq_api_key",
    "get_groq_client",
    "fetch_available_groq_models",
]
