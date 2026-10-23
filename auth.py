import os
import json
import urllib.parse
import sqlite3
import contextlib
import hashlib
import hmac
from typing import Optional, Dict, Any
import requests
import streamlit as st


class UserManager:
    """Manages user accounts, secure authentication, and persistent login sessions in SQLite."""

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
                    password_hash TEXT,
                    password_salt TEXT,
                    auth_provider TEXT DEFAULT 'google',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

            # Ensure schema migration for existing databases
            cursor.execute("PRAGMA table_info(users)")
            existing_cols = {col["name"] for col in cursor.fetchall()}
            if "password_hash" not in existing_cols:
                cursor.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
            if "password_salt" not in existing_cols:
                cursor.execute("ALTER TABLE users ADD COLUMN password_salt TEXT")
            if "auth_provider" not in existing_cols:
                cursor.execute("ALTER TABLE users ADD COLUMN auth_provider TEXT DEFAULT 'google'")
            conn.commit()

    def _hash_password(self, password: str, salt: bytes) -> str:
        return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120000).hex()

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def upsert_user(
        self, user_id: str, email: str, name: str, picture_url: str = "", auth_provider: str = "google"
    ) -> Dict[str, Any]:
        """Upserts a verified OAuth user (e.g. Google Sign-In)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO users (id, email, name, picture_url, auth_provider, last_login)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET 
                    email = excluded.email,
                    name = excluded.name,
                    picture_url = excluded.picture_url,
                    auth_provider = excluded.auth_provider,
                    last_login = CURRENT_TIMESTAMP
                """,
                (user_id, email, name, picture_url, auth_provider),
            )
            conn.commit()
        return self.get_user_by_id(user_id) or {
            "id": user_id,
            "email": email,
            "name": name,
            "picture_url": picture_url,
            "auth_provider": auth_provider,
        }

    def register_email_user(self, email: str, password: str, name: str = "") -> Dict[str, Any]:
        """Registers a new user with verified email and salted hashed password."""
        email_clean = email.strip().lower()
        if not email_clean or "@" not in email_clean:
            raise ValueError("Please provide a valid email address.")
        if len(password) < 6:
            raise ValueError("Password must be at least 6 characters.")

        existing = self.get_user_by_email(email_clean)
        if existing:
            if existing.get("auth_provider") == "google":
                raise ValueError("This email is registered with Google Sign-In. Please sign in with Google.")
            raise ValueError("An account with this email already exists. Please log in.")

        salt = os.urandom(16)
        p_hash = self._hash_password(password, salt)
        user_id = f"usr_{hashlib.md5(email_clean.encode()).hexdigest()[:16]}"
        display_name = name.strip() or email_clean.split("@")[0].title()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO users (id, email, name, password_hash, password_salt, auth_provider, last_login)
                VALUES (?, ?, ?, ?, ?, 'email', CURRENT_TIMESTAMP)
                """,
                (user_id, email_clean, display_name, p_hash, salt.hex()),
            )
            conn.commit()
        return self.get_user_by_id(user_id)

    def authenticate_email_user(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """Verifies email and password. Returns user dict on success, None on failure."""
        email_clean = email.strip().lower()
        user = self.get_user_by_email(email_clean)
        if not user:
            return None

        # Check if password exists (e.g. Google-only accounts won't have a local password)
        if not user.get("password_hash") or not user.get("password_salt"):
            return None

        salt = bytes.fromhex(user["password_salt"])
        expected_hash = self._hash_password(password, salt)
        if hmac.compare_digest(expected_hash, user["password_hash"]):
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?", (user["id"],))
                conn.commit()
            return dict(user)
        return None


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

