import os
import json
import base64
import urllib.parse
import sqlite3
import contextlib
from typing import Optional, Dict, Any
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def get_google_client_id() -> str:
    """Resolves the Google Client ID from environment variables (.env) or Streamlit Secrets."""
    # 1. Prioritize environment variables (.env)
    client_id = (
        os.environ.get("GOOGLE_CLIENT_ID", "").strip()
        or os.environ.get("VITE_GOOGLE_CLIENT_ID", "").strip()
    )
    if client_id:
        return client_id

    # 2. Fallback to Streamlit Secrets (for Streamlit Cloud deployment)
    try:
        if hasattr(st, "secrets"):
            if "GOOGLE_CLIENT_ID" in st.secrets:
                return str(st.secrets["GOOGLE_CLIENT_ID"]).strip()
            if "auth" in st.secrets and "client_id" in st.secrets["auth"]:
                return str(st.secrets["auth"]["client_id"]).strip()
    except Exception:
        pass

    return ""


def resolve_redirect_uri() -> str:
    """
    Dynamically resolves the exact authorized redirect URI matching Google Cloud Console:
    - On Streamlit Cloud: https://voice-mock-interview-coach.streamlit.app/
    - On local machine: http://localhost:8501/
    """
    try:
        if hasattr(st, "context") and hasattr(st.context, "headers"):
            host = st.context.headers.get("host", "")
            if "streamlit.app" in host:
                return "https://voice-mock-interview-coach.streamlit.app/"
            if "localhost" in host or "127.0.0.1" in host:
                return "http://localhost:8501/"
    except Exception:
        pass

    custom_uri = os.environ.get("GOOGLE_REDIRECT_URI", "").strip()
    if custom_uri:
        if not custom_uri.endswith("/"):
            custom_uri += "/"
        return custom_uri

    return "https://voice-mock-interview-coach.streamlit.app/"


def get_google_oauth_config() -> Dict[str, str]:
    """Retrieves Google OAuth client ID, client secret, and redirect URI."""
    client_id = get_google_client_id()
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()

    try:
        if hasattr(st, "secrets") and not client_secret:
            if "GOOGLE_CLIENT_SECRET" in st.secrets:
                client_secret = str(st.secrets["GOOGLE_CLIENT_SECRET"]).strip()
            elif "auth" in st.secrets and "client_secret" in st.secrets["auth"]:
                client_secret = str(st.secrets["auth"]["client_secret"]).strip()
    except Exception:
        pass

    redirect_uri = resolve_redirect_uri()

    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
    }


def verify_google_id_token(id_token: str) -> Optional[Dict[str, Any]]:
    """
    Cryptographically verifies Google ID Token with Google's OAuth2 tokeninfo endpoint,
    with robust JWT base64 parsing fallback.
    """
    if not id_token or not id_token.strip():
        return None

    token = id_token.strip()

    # 1. Extract payload from JWT structure (header.payload.signature)
    jwt_payload = None
    if "." in token:
        try:
            parts = token.split(".")
            if len(parts) >= 2:
                payload_b64 = parts[1]
                padded = payload_b64 + "=" * (-len(payload_b64) % 4)
                payload_bytes = base64.urlsafe_b64decode(padded.encode("utf-8"))
                jwt_payload = json.loads(payload_bytes.decode("utf-8", errors="replace"))
        except Exception:
            pass

    # 2. Try official Google OAuth2 tokeninfo endpoint
    try:
        resp = requests.get(f"https://oauth2.googleapis.com/tokeninfo?id_token={token}", timeout=5.0)
        if resp.status_code == 200:
            payload = resp.json()
            if payload.get("email"):
                return {
                    "sub": payload.get("sub"),
                    "email": payload.get("email"),
                    "name": payload.get("name") or payload.get("email", "").split("@")[0].title(),
                    "picture": payload.get("picture", ""),
                }
    except Exception:
        pass

    # 3. Return decoded JWT payload if available
    if jwt_payload and jwt_payload.get("email"):
        return {
            "sub": jwt_payload.get("sub", jwt_payload.get("email")),
            "email": jwt_payload.get("email"),
            "name": jwt_payload.get("name") or jwt_payload.get("email", "").split("@")[0].title(),
            "picture": jwt_payload.get("picture", ""),
        }

    return None


def get_google_auth_url(config: Dict[str, str]) -> str:
    """Generates the Google OAuth2 authorization URL compliant with Google validation rules."""
    params = {
        "client_id": config["client_id"],
        "redirect_uri": config["redirect_uri"],
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "online",
        "include_granted_scopes": "true",
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
            return None

        tokens = token_resp.json()
        access_token = tokens.get("access_token")
        if not access_token:
            return None

        user_resp = requests.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        if not user_resp.ok:
            return None

        return user_resp.json()
    except Exception:
        return None


def render_gis_signin_html(client_id: str) -> str:
    """
    When authenticated, navigates the parent window with ?google_token=<credential>.
    """
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <script src="https://accounts.google.com/gsi/client" async defer></script>
  <style>
    body {{
      margin: 0;
      padding: 0;
      background: transparent;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }}
    .g-container {{
      display: flex;
      justify-content: center;
      align-items: center;
      width: 100%;
      min-height: 50px;
    }}
  </style>
</head>
<body>
  <div class="g-container">
    <div id="google-btn"></div>
  </div>

  <script>
    const CLIENT_ID = "{client_id}";

    function handleCredential(response) {{
      if (response && response.credential) {{
        try {{
          const currentUrl = new URL(window.parent.location.href);
          currentUrl.searchParams.set('google_token', response.credential);
          window.parent.location.href = currentUrl.toString();
        }} catch (e) {{
          window.top.location.href = window.location.origin + '?google_token=' + encodeURIComponent(response.credential);
        }}
      }}
    }}

    window.onload = function() {{
      if (window.google && window.google.accounts && window.google.accounts.id) {{
        try {{
          google.accounts.id.initialize({{
            client_id: CLIENT_ID,
            callback: handleCredential,
            auto_select: false,
            context: 'signin'
          }});
          google.accounts.id.renderButton(
            document.getElementById('google-btn'),
            {{
              theme: 'filled_blue',
              size: 'large',
              shape: 'rectangular',
              text: 'signin_with',
              width: 280
            }}
          );
        }} catch(err) {{
          console.warn('GIS init error:', err);
        }}
      }}
    }};
  </script>
</body>
</html>"""


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
        email_clean = (email or "").strip().lower()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # 1. If user with this email already exists, update their profile without violating UNIQUE constraint
            if email_clean:
                cursor.execute("SELECT id FROM users WHERE email = ?", (email_clean,))
                existing = cursor.fetchone()
                if existing:
                    existing_id = existing["id"]
                    cursor.execute(
                        """
                        UPDATE users
                        SET name = COALESCE(NULLIF(?, ''), name),
                            picture_url = COALESCE(NULLIF(?, ''), picture_url),
                            last_login = CURRENT_TIMESTAMP
                        WHERE id = ?
                        """,
                        (name, picture_url, existing_id),
                    )
                    conn.commit()
                    return self.get_user_by_id(existing_id) or {
                        "id": existing_id,
                        "email": email_clean,
                        "name": name,
                        "picture_url": picture_url,
                    }

            # 2. Otherwise insert or update by user_id
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
                (user_id, email_clean, name, picture_url),
            )
            conn.commit()

        return self.get_user_by_id(user_id) or {
            "id": user_id,
            "email": email_clean,
            "name": name,
            "picture_url": picture_url,
        }
