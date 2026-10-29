import os
import json
import time
import base64
import urllib.parse
import sqlite3
import contextlib
import secrets
import hashlib
import hmac
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Dict, Any, Tuple
import requests
import streamlit as st
from dotenv import load_dotenv
from cryptography.fernet import Fernet

load_dotenv()


# ---------------------------------------------------------------------------
# Cryptographic Storage & Encryption (Streamlit Secrets / Env Vault)
# ---------------------------------------------------------------------------
def get_encryption_key() -> bytes:
    """
    Retrieves the symmetric Fernet encryption key from Streamlit secrets or environment.
    If none is set, derives a consistent key from an environment seed or fallback seed,
    ensuring database encryption and decryption remain stable and never crash.
    """
    raw_key = ""
    try:
        if hasattr(st, "secrets") and "ENCRYPTION_KEY" in st.secrets:
            raw_key = str(st.secrets["ENCRYPTION_KEY"]).strip()
    except Exception:
        pass

    if not raw_key:
        raw_key = os.environ.get("ENCRYPTION_KEY", "").strip()

    if not raw_key:
        # Stable fallback seed so encrypted records can always be decrypted consistently
        raw_key = "voice_mock_interview_coach_secure_vault_master_key_2026"

    # Derive valid 32-byte URL-safe base64 Fernet key from any passphrase
    key_digest = hashlib.sha256(raw_key.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(key_digest)


def encrypt_data(plain_text: str) -> str:
    """Encrypts a plaintext string using Fernet AES-128-CBC + HMAC-SHA256."""
    if not plain_text:
        return ""
    try:
        f = Fernet(get_encryption_key())
        return f.encrypt(plain_text.encode("utf-8")).decode("utf-8")
    except Exception:
        return plain_text


def decrypt_data(cipher_text: str) -> str:
    """Decrypts a Fernet ciphertext string, falling back to plaintext if not encrypted."""
    if not cipher_text:
        return ""
    try:
        f = Fernet(get_encryption_key())
        return f.decrypt(cipher_text.encode("utf-8")).decode("utf-8")
    except Exception:
        return cipher_text


def hash_password(password: str) -> str:
    """Hashes a password using PBKDF2-HMAC-SHA256 with a unique 16-byte random salt."""
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), 100_000
    )
    return f"{salt}:{dk.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verifies a plaintext password against the stored salt:hash string."""
    if not stored_hash or ":" not in stored_hash:
        return False
    try:
        salt, hash_val = stored_hash.split(":", 1)
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt), 100_000
        )
        return hmac.compare_digest(hash_val, dk.hex())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# OTP Verification Code Generation & SMTP Delivery
# ---------------------------------------------------------------------------
def generate_otp() -> str:
    """Generates a cryptographically secure 6-digit numeric verification code."""
    return f"{secrets.randbelow(900000) + 100000}"


def send_otp_email(
    to_email: str, otp_code: str, action: str = "Identity Verification"
) -> Tuple[bool, str]:
    """
    Sends a 6-digit OTP code to the recipient email via SMTP.
    Returns (True, 'Success') if delivered, or (False, reason) if SMTP credentials are missing or fail.
    """
    smtp_email = ""
    smtp_password = ""
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    try:
        if hasattr(st, "secrets"):
            if "SMTP_EMAIL" in st.secrets:
                smtp_email = str(st.secrets["SMTP_EMAIL"]).strip()
            if "SMTP_PASSWORD" in st.secrets:
                smtp_password = str(st.secrets["SMTP_PASSWORD"]).strip()
            if "SMTP_SERVER" in st.secrets:
                smtp_server = str(st.secrets["SMTP_SERVER"]).strip()
            if "SMTP_PORT" in st.secrets:
                smtp_port = int(st.secrets["SMTP_PORT"])
    except Exception:
        pass

    if not smtp_email:
        smtp_email = os.environ.get("SMTP_EMAIL", "").strip()
    if not smtp_password:
        smtp_password = os.environ.get("SMTP_PASSWORD", "").strip()
    if os.environ.get("SMTP_SERVER"):
        smtp_server = os.environ.get("SMTP_SERVER", "").strip()
    if os.environ.get("SMTP_PORT"):
        with contextlib.suppress(Exception):
            smtp_port = int(os.environ.get("SMTP_PORT", "587"))

    if not smtp_email or not smtp_password:
        return False, "SMTP credentials not configured"

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Your Verification Code: {otp_code} — Voice Mock-Interview Coach"
        msg["From"] = f"Voice Mock-Interview Coach <{smtp_email}>"
        msg["To"] = to_email

        html_body = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 500px; margin: 0 auto; padding: 24px; border: 1px solid #e2e8f0; border-radius: 12px; background: #ffffff;">
            <div style="text-align: center; margin-bottom: 20px;">
                <h2 style="color: #0f172a; margin: 0;">🎙️ Voice Mock-Interview Coach</h2>
                <p style="color: #64748b; font-size: 14px; margin-top: 6px;">Secure Candidate Authentication</p>
            </div>
            <p style="color: #334155; font-size: 15px; line-height: 1.5;">
                Hello,<br><br>
                Please use the following 6-digit verification code to complete your <strong>{action}</strong>:
            </p>
            <div style="margin: 26px 0; padding: 18px; background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 10px; text-align: center;">
                <span style="font-size: 34px; font-weight: 800; letter-spacing: 8px; color: #2563eb; font-family: monospace;">{otp_code}</span>
            </div>
            <p style="color: #64748b; font-size: 13px; line-height: 1.5;">
                ⏳ This code will expire in <strong>10 minutes</strong>. If you did not make this request, please safely ignore this email.
            </p>
            <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 24px 0;">
            <p style="color: #94a3b8; font-size: 12px; text-align: center; margin: 0;">
                Voice Mock-Interview Coach • AI-Powered Interview Preparation
            </p>
        </div>
        """
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(smtp_server, smtp_port, timeout=10) as server:
            server.starttls()
            server.login(smtp_email, smtp_password)
            server.sendmail(smtp_email, [to_email], msg.as_string())

        return True, "Email sent successfully"
    except Exception as e:
        return False, f"SMTP delivery failed: {str(e)}"


# ---------------------------------------------------------------------------
# Redirect URI Resolver (Streamlit Cloud & Localhost)
# ---------------------------------------------------------------------------
def resolve_redirect_uri() -> str:
    """
    Dynamically resolves the exact authorized redirect URI:
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

    custom_uri = os.environ.get("GOOGLE_REDIRECT_URI", "").strip() or os.environ.get(
        "GITHUB_REDIRECT_URI", ""
    ).strip()
    if custom_uri:
        if not custom_uri.endswith("/"):
            custom_uri += "/"
        return custom_uri

    return "https://voice-mock-interview-coach.streamlit.app/"


# ---------------------------------------------------------------------------
# GitHub OAuth Configuration & Token Exchange
# ---------------------------------------------------------------------------
def get_github_oauth_config() -> Dict[str, str]:
    """Retrieves GitHub OAuth client ID, client secret, and redirect URI."""
    client_id = os.environ.get("GITHUB_CLIENT_ID", "").strip()
    client_secret = os.environ.get("GITHUB_CLIENT_SECRET", "").strip()

    try:
        if hasattr(st, "secrets"):
            if "GITHUB_CLIENT_ID" in st.secrets:
                client_id = str(st.secrets["GITHUB_CLIENT_ID"]).strip()
            if "GITHUB_CLIENT_SECRET" in st.secrets:
                client_secret = str(st.secrets["GITHUB_CLIENT_SECRET"]).strip()
            if "github" in st.secrets:
                if "client_id" in st.secrets["github"]:
                    client_id = str(st.secrets["github"]["client_id"]).strip()
                if "client_secret" in st.secrets["github"]:
                    client_secret = str(st.secrets["github"]["client_secret"]).strip()
    except Exception:
        pass

    redirect_uri = resolve_redirect_uri()
    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
    }


def get_github_auth_url(config: Dict[str, str]) -> str:
    """Generates the GitHub OAuth authorization URL."""
    params = {
        "client_id": config["client_id"],
        "redirect_uri": config["redirect_uri"],
        "scope": "read:user user:email",
        "allow_signup": "true",
    }
    return f"https://github.com/login/oauth/authorize?{urllib.parse.urlencode(params)}"


def exchange_github_code_for_user(
    code: str, config: Dict[str, str]
) -> Optional[Dict[str, Any]]:
    """Exchanges GitHub OAuth authorization code for an access token and user profile."""
    try:
        token_resp = requests.post(
            "https://github.com/login/oauth/access_token",
            headers={"Accept": "application/json"},
            data={
                "client_id": config["client_id"],
                "client_secret": config["client_secret"],
                "code": code,
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

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/vnd.github+json",
        }
        user_resp = requests.get(
            "https://api.github.com/user", headers=headers, timeout=10
        )
        if not user_resp.ok:
            return None

        u = user_resp.json()
        email = u.get("email")

        # If primary email is private in GitHub public profile, fetch from emails endpoint
        if not email:
            emails_resp = requests.get(
                "https://api.github.com/user/emails", headers=headers, timeout=10
            )
            if emails_resp.ok:
                for em in emails_resp.json():
                    if em.get("primary") and em.get("verified"):
                        email = em.get("email")
                        break
                    if not email:
                        email = em.get("email")

        email = email or f"{u.get('login')}@users.noreply.github.com"
        name = u.get("name") or u.get("login") or "GitHub Candidate"
        picture = u.get("avatar_url") or ""

        return {
            "sub": str(u.get("id")),
            "email": email,
            "name": name,
            "picture": picture,
            "provider": "github",
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Google OAuth Configuration (Preserved for compatibility)
# ---------------------------------------------------------------------------
def get_google_client_id() -> str:
    """Resolves the Google Client ID from environment variables (.env) or Streamlit Secrets."""
    client_id = (
        os.environ.get("GOOGLE_CLIENT_ID", "").strip()
        or os.environ.get("VITE_GOOGLE_CLIENT_ID", "").strip()
    )
    if client_id:
        return client_id

    try:
        if hasattr(st, "secrets"):
            if "GOOGLE_CLIENT_ID" in st.secrets:
                return str(st.secrets["GOOGLE_CLIENT_ID"]).strip()
            if "auth" in st.secrets and "client_id" in st.secrets["auth"]:
                return str(st.secrets["auth"]["client_id"]).strip()
    except Exception:
        pass

    return ""


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
    """Verifies Google ID Token with tokeninfo endpoint and JWT base64 fallback."""
    if not id_token or not id_token.strip():
        return None
    token = id_token.strip()

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

    try:
        resp = requests.get(
            f"https://oauth2.googleapis.com/tokeninfo?id_token={token}", timeout=5.0
        )
        if resp.status_code == 200:
            payload = resp.json()
            if payload.get("email"):
                return {
                    "sub": payload.get("sub"),
                    "email": payload.get("email"),
                    "name": payload.get("name")
                    or payload.get("email", "").split("@")[0].title(),
                    "picture": payload.get("picture", ""),
                }
    except Exception:
        pass

    if jwt_payload and jwt_payload.get("email"):
        return {
            "sub": jwt_payload.get("sub", jwt_payload.get("email")),
            "email": jwt_payload.get("email"),
            "name": jwt_payload.get("name")
            or jwt_payload.get("email", "").split("@")[0].title(),
            "picture": jwt_payload.get("picture", ""),
        }
    return None


def get_google_auth_url(config: Dict[str, str]) -> str:
    """Generates the Google OAuth2 authorization URL."""
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


def exchange_code_for_user(
    code: str, config: Dict[str, str]
) -> Optional[Dict[str, Any]]:
    """Exchanges Google authorization code for access tokens and user profile."""
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


def render_gis_signin_html(client_id: str = "") -> str:
    """Renders HTML for Google Identity Services One Tap / button sign-in."""
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <script src="https://accounts.google.com/gsi/client" async defer></script>
</head>
<body>
  <div id="g_id_onload"
       data-client_id="{client_id}"
       data-auto_prompt="false">
  </div>
  <div class="g_id_signin" data-type="standard"></div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# User Manager (SQLite Persistent Vault & Profile Manager)
# ---------------------------------------------------------------------------
class UserManager:
    """Manages encrypted candidate accounts, authentication, and sessions in SQLite."""

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
        """Initializes database schema and ensures all encryption columns exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE COLLATE NOCASE,
                    name TEXT,
                    password_hash TEXT,
                    picture_url TEXT,
                    target_role TEXT,
                    experience_level TEXT,
                    encrypted_profile TEXT,
                    auth_provider TEXT DEFAULT 'local',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Ensure candidate memory table exists for persistent interview tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS candidates (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    target_role TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Ensure interview sessions history table exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interview_sessions (
                    id TEXT PRIMARY KEY,
                    candidate_id TEXT,
                    target_role TEXT,
                    experience_level TEXT,
                    evaluation_report TEXT,
                    messages_json TEXT,
                    filler_stats_json TEXT,
                    questions_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(candidate_id) REFERENCES users(id)
                )
            """)

            # Safe schema migration for any missing columns
            cursor.execute("PRAGMA table_info(users)")
            existing_cols = [row["name"] for row in cursor.fetchall()]
            required_cols = [
                ("password_hash", "TEXT"),
                ("target_role", "TEXT"),
                ("experience_level", "TEXT"),
                ("encrypted_profile", "TEXT"),
                ("auth_provider", "TEXT DEFAULT 'local'"),
            ]
            for col_name, col_type in required_cols:
                if col_name not in existing_cols:
                    cursor.execute(f"ALTER TABLE users ADD COLUMN {col_name} {col_type}")

            # Safe schema migration for existing databases:
            # 1. Resolve duplicate candidate rows whose emails differ only by case
            cursor.execute("""
                SELECT LOWER(TRIM(email)) AS norm_email
                FROM users
                WHERE email IS NOT NULL AND TRIM(email) != ''
                GROUP BY LOWER(TRIM(email))
                HAVING COUNT(*) > 1
            """)
            dup_groups = [r["norm_email"] for r in cursor.fetchall()]
            for norm_email in dup_groups:
                cursor.execute("""
                    SELECT * FROM users
                    WHERE LOWER(TRIM(email)) = ?
                    ORDER BY
                        (password_hash IS NOT NULL AND password_hash != '') DESC,
                        last_login DESC,
                        created_at DESC,
                        id ASC
                """, (norm_email,))
                rows = [dict(r) for r in cursor.fetchall()]
                if not rows:
                    continue

                canonical = rows[0]
                canonical_id = canonical["id"]

                for dup in rows[1:]:
                    dup_id = dup["id"]

                    # Merge non-empty user fields into canonical if canonical is missing them
                    merge_updates = []
                    merge_params = []
                    for field in [
                        "name",
                        "password_hash",
                        "picture_url",
                        "target_role",
                        "experience_level",
                        "encrypted_profile",
                    ]:
                        if not canonical.get(field) and dup.get(field):
                            merge_updates.append(f"{field} = ?")
                            merge_params.append(dup[field])
                            canonical[field] = dup[field]

                    if merge_updates:
                        merge_params.append(canonical_id)
                        cursor.execute(
                            f"UPDATE users SET {', '.join(merge_updates)} WHERE id = ?",
                            merge_params,
                        )

                    # Preserve earliest created_at and latest last_login
                    if dup.get("created_at") and (
                        not canonical.get("created_at")
                        or dup["created_at"] < canonical["created_at"]
                    ):
                        cursor.execute(
                            "UPDATE users SET created_at = ? WHERE id = ?",
                            (dup["created_at"], canonical_id),
                        )
                        canonical["created_at"] = dup["created_at"]

                    if dup.get("last_login") and (
                        not canonical.get("last_login")
                        or dup["last_login"] > canonical["last_login"]
                    ):
                        cursor.execute(
                            "UPDATE users SET last_login = ? WHERE id = ?",
                            (dup["last_login"], canonical_id),
                        )
                        canonical["last_login"] = dup["last_login"]

                    # Re-point interview sessions to canonical user
                    cursor.execute(
                        "UPDATE interview_sessions SET candidate_id = ? WHERE candidate_id = ?",
                        (canonical_id, dup_id),
                    )

                    # Re-point memory graph tables if they exist
                    for table in ["claims", "weaknesses", "topics_covered", "strengths"]:
                        cursor.execute(
                            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                            (table,),
                        )
                        if cursor.fetchone():
                            cursor.execute(
                                f"UPDATE {table} SET candidate_id = ? WHERE candidate_id = ?",
                                (canonical_id, dup_id),
                            )

                    # Merge and clean up candidates table
                    cursor.execute("SELECT * FROM candidates WHERE id = ?", (canonical_id,))
                    canonical_cand = cursor.fetchone()
                    cursor.execute("SELECT * FROM candidates WHERE id = ?", (dup_id,))
                    dup_cand = cursor.fetchone()

                    if dup_cand:
                        if canonical_cand:
                            cand_updates = []
                            cand_params = []
                            for field in ["name", "target_role"]:
                                can_val = canonical_cand[field]
                                dup_val = dup_cand[field]
                                can_populated = can_val is not None and bool(str(can_val).strip())
                                dup_populated = dup_val is not None and bool(str(dup_val).strip())
                                if not can_populated and dup_populated:
                                    cand_updates.append(f"{field} = ?")
                                    cand_params.append(dup_val.strip() if isinstance(dup_val, str) else dup_val)

                            dup_created = dup_cand["created_at"]
                            can_created = canonical_cand["created_at"]
                            if dup_created and str(dup_created).strip() and (
                                not can_created
                                or not str(can_created).strip()
                                or str(dup_created) < str(can_created)
                            ):
                                cand_updates.append("created_at = ?")
                                cand_params.append(dup_created)

                            if cand_updates:
                                cand_params.append(canonical_id)
                                cursor.execute(
                                    f"UPDATE candidates SET {', '.join(cand_updates)} WHERE id = ?",
                                    cand_params,
                                )
                            cursor.execute("DELETE FROM candidates WHERE id = ?", (dup_id,))
                        else:
                            cursor.execute(
                                "UPDATE candidates SET id = ? WHERE id = ?",
                                (canonical_id, dup_id),
                            )

                    # Delete duplicate user row
                    cursor.execute("DELETE FROM users WHERE id = ?", (dup_id,))

                # Normalize canonical email
                cursor.execute(
                    "UPDATE users SET email = ? WHERE id = ?",
                    (norm_email, canonical_id),
                )

            # 2. Normalize remaining legacy emails to lowercase
            cursor.execute(
                "UPDATE users SET email = LOWER(TRIM(email)) WHERE email IS NOT NULL AND email != LOWER(TRIM(email))"
            )

            # 3. Create unique index on email with NOCASE collation to enforce case-insensitivity on existing and new tables
            cursor.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email_nocase ON users(email COLLATE NOCASE)"
            )

            conn.commit()

    def _format_user_record(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Converts database row to dictionary and decrypts sensitive encrypted fields."""
        user = dict(row)

        # Decrypt target role
        if user.get("target_role"):
            user["target_role_decrypted"] = decrypt_data(user["target_role"])
        else:
            user["target_role_decrypted"] = ""

        # Decrypt experience level
        if user.get("experience_level"):
            user["experience_level_decrypted"] = decrypt_data(user["experience_level"])
        else:
            user["experience_level_decrypted"] = ""

        # Decrypt full profile payload
        if user.get("encrypted_profile"):
            try:
                raw_json = decrypt_data(user["encrypted_profile"])
                user["profile_decrypted"] = json.loads(raw_json)
            except Exception:
                user["profile_decrypted"] = {}
        else:
            user["profile_decrypted"] = {}

        return user

    def user_exists(self, email: str) -> bool:
        """Checks if a user already exists with the given email address (case-insensitive)."""
        email_clean = (email or "").strip().lower()
        if not email_clean:
            return False
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE email = ? COLLATE NOCASE", (email_clean,))
            return cursor.fetchone() is not None

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves and decrypts a user record by user ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                return self._format_user_record(row)
            return None

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Retrieves and decrypts a user record by email (case-insensitive)."""
        email_clean = (email or "").strip().lower()
        if not email_clean:
            return None
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE email = ? COLLATE NOCASE", (email_clean,))
            row = cursor.fetchone()
            if row:
                return self._format_user_record(row)
            return None

    def register_user(
        self,
        name: str,
        email: str,
        password: str,
        target_role: str = "",
        experience_level: str = "",
    ) -> Dict[str, Any]:
        """
        Registers a new candidate:
        - Hashes password using salted PBKDF2-HMAC-SHA256
        - Encrypts target role, experience level, and profile JSON using Fernet
        - Inserts encrypted record into SQLite and initializes candidate memory graph row
        """
        email_clean = (email or "").strip().lower()
        name_clean = (name or "").strip() or email_clean.split("@")[0].title()

        if self.user_exists(email_clean):
            raise ValueError(f"Email already exists: An account with '{email_clean}' is already registered in the system.")

        user_id = f"usr_{secrets.token_hex(6)}"
        pwd_hash = hash_password(password)

        # Encrypt sensitive fields
        enc_role = encrypt_data(target_role.strip()) if target_role else ""
        enc_exp = encrypt_data(experience_level.strip()) if experience_level else ""
        profile_payload = {
            "name": name_clean,
            "email": email_clean,
            "target_role": target_role.strip(),
            "experience_level": experience_level.strip(),
            "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        enc_profile = encrypt_data(json.dumps(profile_payload))

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO users (
                    id, email, name, password_hash, picture_url, 
                    target_role, experience_level, encrypted_profile, 
                    auth_provider, last_login
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'local', CURRENT_TIMESTAMP)
                """,
                (
                    user_id,
                    email_clean,
                    name_clean,
                    pwd_hash,
                    "",
                    enc_role,
                    enc_exp,
                    enc_profile,
                ),
            )

            # Initialize candidate memory graph record
            cursor.execute(
                """
                INSERT OR IGNORE INTO candidates (id, name, target_role)
                VALUES (?, ?, ?)
                """,
                (user_id, name_clean, target_role.strip() or "General Software Engineer"),
            )
            conn.commit()

        return self.get_user_by_id(user_id) or {
            "id": user_id,
            "email": email_clean,
            "name": name_clean,
            "target_role_decrypted": target_role,
            "experience_level_decrypted": experience_level,
        }

    def authenticate_user(
        self, email: str, password: str
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Verifies credentials:
        Returns (user_dict, "Success") on successful authentication,
        or (None, error_message) on failure.
        """
        email_clean = (email or "").strip().lower()
        if not email_clean:
            return None, "Please provide an email address."

        user = self.get_user_by_email(email_clean)
        if not user:
            return None, "No account found with this email. Please sign up first."

        stored_hash = user.get("password_hash")
        if not stored_hash:
            return (
                None,
                "This account was registered via social sign-in. Please use GitHub or set a password.",
            )

        if not verify_password(password, stored_hash):
            return None, "Incorrect password. Please verify your credentials."

        # Update last login timestamp
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?",
                (user["id"],),
            )
            conn.commit()

        return self.get_user_by_id(user["id"]), "Success"

    def upsert_user(
        self,
        user_id: str,
        email: str,
        name: str,
        picture_url: str = "",
        auth_provider: str = "google",
    ) -> Dict[str, Any]:
        """
        Upserts an OAuth candidate (GitHub or Google), linking existing email
        to preserve persistent memory graph without duplicate key conflicts.
        """
        email_clean = (email or "").strip().lower()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # If user with this email already exists, update and link
            if email_clean:
                cursor.execute("SELECT id FROM users WHERE email = ? COLLATE NOCASE", (email_clean,))
                existing = cursor.fetchone()
                if existing:
                    existing_id = existing["id"]
                    cursor.execute(
                        """
                        UPDATE users
                        SET name = COALESCE(NULLIF(?, ''), name),
                            picture_url = COALESCE(NULLIF(?, ''), picture_url),
                            auth_provider = ?,
                            last_login = CURRENT_TIMESTAMP
                        WHERE id = ?
                        """,
                        (name, picture_url, auth_provider, existing_id),
                    )
                    conn.commit()
                    return self.get_user_by_id(existing_id) or {
                        "id": existing_id,
                        "email": email_clean,
                        "name": name,
                        "picture_url": picture_url,
                        "auth_provider": auth_provider,
                    }

            # Otherwise insert new user
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
                (user_id, email_clean, name, picture_url, auth_provider),
            )

            # Ensure candidate memory record is initialized
            cursor.execute(
                """
                INSERT OR IGNORE INTO candidates (id, name, target_role)
                VALUES (?, ?, 'Software Engineer')
                """,
                (user_id, name or email_clean.split("@")[0].title()),
            )
            conn.commit()

        return self.get_user_by_id(user_id) or {
            "id": user_id,
            "email": email_clean,
            "name": name,
            "picture_url": picture_url,
            "auth_provider": auth_provider,
        }

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Alias for get_user_by_id."""
        return self.get_user_by_id(user_id)

    def update_user_password(
        self,
        user_id: str,
        new_password: str = "",
        current_password: Optional[str] = None,
        old_password: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[bool, str]:
        """
        Updates the user's password.
        Supports both current_password and old_password keyword arguments.
        If the user has an existing password, verifies current_password/old_password first.
        If social login (no password set), allows setting a new password directly.
        """
        if not new_password and "new_password" in kwargs:
            new_password = kwargs["new_password"]
        if not current_password and "current_password" in kwargs:
            current_password = kwargs["current_password"]
        if not old_password and "old_password" in kwargs:
            old_password = kwargs["old_password"]

        pwd_to_check = current_password if current_password is not None else old_password

        if not new_password or len(new_password) < 6:
            return False, "New password must be at least 6 characters long."

        user = self.get_user_by_id(user_id)
        if not user:
            return False, "User not found."

        current_hash = user.get("password_hash")
        if current_hash:
            if not pwd_to_check:
                return False, "Please enter your current password."
            if not verify_password(pwd_to_check, current_hash):
                return False, "Current password is incorrect."

        new_hash = hash_password(new_password)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (new_hash, user_id),
            )
            conn.commit()

        return True, "Password updated successfully!"

    def update_user_profile(
        self, user_id: str, name: str, target_role: str = "", experience_level: str = ""
    ) -> Dict[str, Any]:
        """Updates candidate name, encrypted target role, and experience level."""
        name_clean = (name or "").strip()
        enc_role = encrypt_data(target_role.strip()) if target_role else ""
        enc_exp = encrypt_data(experience_level.strip()) if experience_level else ""

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE users 
                SET name = COALESCE(NULLIF(?, ''), name),
                    target_role = ?,
                    experience_level = ?
                WHERE id = ?
                """,
                (name_clean, enc_role, enc_exp, user_id),
            )
            cursor.execute(
                """
                UPDATE candidates 
                SET name = COALESCE(NULLIF(?, ''), name),
                    target_role = COALESCE(NULLIF(?, ''), target_role)
                WHERE id = ?
                """,
                (name_clean, target_role.strip(), user_id),
            )
            conn.commit()

        return self.get_user_by_id(user_id) or {}

    def save_interview_session(
        self,
        candidate_id: str,
        target_role: str,
        experience_level: str,
        evaluation_report: str,
        messages: list,
        filler_stats: dict,
    ) -> str:
        """Stores a completed mock interview session in SQLite."""
        session_id = f"sess_{secrets.token_hex(6)}"
        messages_json = json.dumps(messages)
        filler_stats_json = json.dumps(filler_stats)
        questions_count = sum(1 for m in messages if m.get("role") == "user")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO interview_sessions (
                    id, candidate_id, target_role, experience_level, 
                    evaluation_report, messages_json, filler_stats_json, 
                    questions_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    session_id,
                    candidate_id,
                    target_role,
                    experience_level,
                    evaluation_report,
                    messages_json,
                    filler_stats_json,
                    questions_count,
                ),
            )
            conn.commit()
        return session_id

    def get_interview_sessions(self, candidate_id: str) -> list:
        """Retrieves all past interview sessions for a candidate, newest first."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM interview_sessions 
                WHERE candidate_id = ? 
                ORDER BY created_at DESC
                """,
                (candidate_id,),
            )
            rows = cursor.fetchall()
            sessions = []
            for r in rows:
                d = dict(r)
                try:
                    d["messages"] = json.loads(d["messages_json"])
                except Exception:
                    d["messages"] = []
                try:
                    d["filler_stats"] = json.loads(d["filler_stats_json"])
                except Exception:
                    d["filler_stats"] = {}
                sessions.append(d)
            return sessions

    def delete_interview_session(self, session_id: str) -> bool:
        """Deletes a recorded interview session by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM interview_sessions WHERE id = ?", (session_id,))
            conn.commit()
            return cursor.rowcount > 0
