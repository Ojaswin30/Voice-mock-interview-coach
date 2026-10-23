import os
import json
import urllib.parse
import sqlite3
import contextlib
from typing import Optional, Dict, Any
import requests
import streamlit as st


class UserManager:
    """Manages user accounts and persistent login sessions in SQLite."""

    def __init__(self, db_path: str = "candidate_memory.db"):
        self.db_path = db_path
        self._init_db()

    @contextlib.contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE,
                    name TEXT,
                    picture_url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def upsert_user(
        self, user_id: str, email: str, name: str, picture_url: str = ""
    ) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO users (id, email, name, picture_url, last_login)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET 
                    email = excluded.email,
                    name = excluded.name,
                    picture_url = excluded.picture_url,
                    last_login = CURRENT_TIMESTAMP
                """,
                (user_id, email, name, picture_url),
            )
            conn.commit()
        return self.get_user_by_id(user_id) or {
            "id": user_id,
            "email": email,
            "name": name,
            "picture_url": picture_url,
        }


def get_google_oauth_config() -> Dict[str, str]:
    """Retrieves Google OAuth client ID, client secret, and redirect URI from secrets/env."""
    client_id = ""
    client_secret = ""
    redirect_uri = ""

    try:
        if "GOOGLE_CLIENT_ID" in st.secrets:
            client_id = str(st.secrets["GOOGLE_CLIENT_ID"]).strip()
        if "GOOGLE_CLIENT_SECRET" in st.secrets:
            client_secret = str(st.secrets["GOOGLE_CLIENT_SECRET"]).strip()
        if "GOOGLE_REDIRECT_URI" in st.secrets:
            redirect_uri = str(st.secrets["GOOGLE_REDIRECT_URI"]).strip()
    except Exception:
        pass

    if not client_id:
        client_id = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
    if not client_secret:
        client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
    if not redirect_uri:
        redirect_uri = os.environ.get("GOOGLE_REDIRECT_URI", "").strip()

    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri or "http://localhost:8501/",
    }


def get_google_auth_url(config: Dict[str, str]) -> str:
    """Generates the Google OAuth2 authorization URL."""
    params = {
        "client_id": config["client_id"],
        "redirect_uri": config["redirect_uri"],
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "online",
        "prompt": "select_account",
    }
    return f"https://accounts.google.com/o/oauth2/v2/auth?{urllib.parse.urlencode(params)}"


def exchange_code_for_user(code: str, config: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Exchanges authorization code with Google for tokens and fetches user profile."""
    try:
        token_resp = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": config["client_id"],
                "client_secret": config["client_secret"],
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": config["redirect_uri"],
            },
            timeout=10,
        )
        if not token_resp.ok:
            print("Google token exchange failed:", token_resp.text)
            return None

        tokens = token_resp.json()
        access_token = tokens.get("access_token")
        if not access_token:
            return None

        # Fetch profile
        user_resp = requests.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        if not user_resp.ok:
            return None

        return user_resp.json()
    except Exception as e:
        print(f"OAuth exchange error: {e}")
        return None


def verify_google_id_token(id_token: str) -> Optional[Dict[str, Any]]:
    """
    Verifies a Google Identity Services ID token (JWT) using Google's public tokeninfo endpoint.
    Requires only client_id (no client_secret needed).
    """
    try:
        resp = requests.get(
            f"https://oauth2.googleapis.com/tokeninfo?id_token={id_token}",
            timeout=10,
        )
        if resp.ok:
            data = resp.json()
            return {
                "sub": data.get("sub"),
                "email": data.get("email"),
                "name": data.get("name") or (data.get("email", "").split("@")[0].title() if data.get("email") else "Google User"),
                "picture": data.get("picture", ""),
            }
        print("Google ID token validation failed:", resp.text)
        return None
    except Exception as e:
        print(f"Google ID token verification error: {e}")
        return None

