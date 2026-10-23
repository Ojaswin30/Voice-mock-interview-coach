"""
Authentication Portal View for Voice Mock-Interview Coach.
Handles Sign-In (Email + Password + OTP), GitHub OAuth, New Candidate Registration (Encrypted),
and Full-Height Centered Candidate Landing Page UI.
"""
import os
import time
import base64
import hashlib
import secrets
import smtplib
import contextlib
import urllib.parse
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Dict, Any, Tuple
import requests
import streamlit as st


# ---------------------------------------------------------------------------
# Independent OTP Generation & Delivery (Zero external module dependencies)
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

        return True, "Code delivered successfully"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Independent GitHub OAuth Functions
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
    redirect_uri = (
        os.environ.get("GITHUB_REDIRECT_URI", "").strip()
        or "https://voice-mock-interview-coach.streamlit.app/"
    )
    return {"client_id": client_id, "client_secret": client_secret, "redirect_uri": redirect_uri}


def get_github_auth_url(config: Dict[str, str]) -> str:
    """Generates the GitHub OAuth authorization URL."""
    params = {
        "client_id": config.get("client_id", ""),
        "redirect_uri": config.get("redirect_uri", ""),
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
        return {
            "sub": str(u.get("id")),
            "name": u.get("name") or u.get("login") or "GitHub User",
            "email": email or f"github_{u.get('id')}@users.noreply.github.com",
            "picture": u.get("avatar_url", ""),
        }
    except Exception:
        return None


def handle_oauth_callbacks(user_manager: Any):
    """
    Handles URL query parameters for OAuth authentication (Streamlit Native, GitHub, Google)
    and removes query parameters to prevent URL session leakage.
    """
    # Clean up legacy or malicious query params
    for param in ("cid", "uid"):
        if param in st.query_params:
            try:
                del st.query_params[param]
            except Exception:
                pass

    # 1. Native Streamlit OpenID Connect (st.login / st.user)
    if hasattr(st, "user") and getattr(st.user, "is_logged_in", False):
        user_email = getattr(st.user, "email", "") or ""
        user_name = getattr(st.user, "name", "") or (
            user_email.split("@")[0].title() if user_email else "Candidate"
        )
        user_sub = getattr(st.user, "sub", "") or hashlib.md5(user_email.encode()).hexdigest()[:16]
        user_id = f"google_{user_sub}"
        db_user = user_manager.upsert_user(user_id, user_email, user_name, "")
        st.session_state.user = db_user
        st.session_state.candidate_id = user_id

    # 2. GitHub OAuth Callback (?code=...)
    github_cfg = get_github_oauth_config()
    gh_code = st.query_params.get("code")
    if (
        gh_code
        and github_cfg.get("client_id")
        and github_cfg.get("client_secret")
        and not st.session_state.get("user")
    ):
        with st.spinner("Authenticating with GitHub..."):
            gh_user = exchange_github_code_for_user(gh_code, github_cfg)
            if gh_user:
                user_id = f"github_{gh_user.get('sub', '')}"
                db_user = user_manager.upsert_user(
                    user_id=user_id,
                    email=gh_user.get("email", ""),
                    name=gh_user.get("name", "GitHub User"),
                    picture_url=gh_user.get("picture", ""),
                    auth_provider="github",
                )
                st.session_state.user = db_user
                st.session_state.candidate_id = db_user["id"]
                try:
                    del st.query_params["code"]
                except Exception:
                    pass
                st.toast(f"Signed in as {db_user.get('name', 'Candidate')} via GitHub!")
                st.rerun()


def get_background_base64() -> str:
    """Loads and enhances background_image.jpeg if needed, returning base64 string."""
    enhanced_path = "background_enhanced.jpg"
    orig_path = "background_image.jpeg"

    if not os.path.exists(enhanced_path) and os.path.exists(orig_path):
        try:
            from PIL import Image, ImageEnhance, ImageFilter
            with Image.open(orig_path) as img:
                target_w = 1920
                target_h = int(img.height * (1920 / img.width))
                upscaled = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                sharpened = ImageEnhance.Sharpness(upscaled).enhance(1.8)
                contrasted = ImageEnhance.Contrast(sharpened).enhance(1.15)
                colored = ImageEnhance.Color(contrasted).enhance(1.1)
                final_img = colored.filter(ImageFilter.UnsharpMask(radius=1.8, percent=140, threshold=2))
                final_img.save(enhanced_path, "JPEG", quality=95, optimize=True)
        except Exception:
            pass

    target_file = enhanced_path if os.path.exists(enhanced_path) else (orig_path if os.path.exists(orig_path) else "")
    if target_file:
        try:
            with open(target_file, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            pass
    return ""


# -----------------------------------------------------------------------------
# Authentication Dialogs: Sign-In, GitHub Setup, and Encrypted Registration
# -----------------------------------------------------------------------------
@st.dialog("🔑 Sign In to Existing Account")
def signin_dialog(user_manager: Any):
    step = st.session_state.get("signin_step", 1)

    if step == 1:
        st.markdown(
            "<p style='color:#cbd5e1; font-size:0.9rem; margin-bottom:1.2rem;'>"
            "Enter your registered candidate email and password to receive an identity verification code (OTP)."
            "</p>",
            unsafe_allow_html=True,
        )
        email = st.text_input(
            "Candidate Email Address",
            placeholder="e.g. candidate@example.com",
            key="dlg_signin_email",
        )
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your account password",
            key="dlg_signin_pwd",
        )

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        if st.button("Send Verification Code →", type="primary", use_container_width=True):
            email_clean = (email or "").strip().lower()
            if not email_clean or not password:
                st.error("Please provide both email address and password.")
                return

            if not user_manager.user_exists(email_clean):
                st.error("No account found with this email. Please click 'Sign Up as New Candidate'.")
                return

            otp_code = generate_otp()
            st.session_state.pending_signin = {
                "email": email_clean,
                "password": password,
                "otp": otp_code,
                "timestamp": time.time(),
            }

            sent, msg = send_otp_email(
                email_clean, otp_code, action="Sign-In Verification"
            )
            st.session_state.signin_step = 2
            st.session_state.signin_otp_sent = sent
            st.session_state.signin_otp_msg = msg
            st.rerun()

    elif step == 2:
        pending = st.session_state.get("pending_signin", {})
        email_clean = pending.get("email", "")
        otp_expected = pending.get("otp", "")
        pwd = pending.get("password", "")

        st.markdown(
            f"<p style='color:#cbd5e1; font-size:0.92rem; margin-bottom:1rem;'>"
            f"A 6-digit identity verification code was issued for <strong>{email_clean}</strong>.<br>"
            f"<span style='color:#94a3b8; font-size:0.82rem;'>Enter the OTP below to confirm your identity.</span>"
            f"</p>",
            unsafe_allow_html=True,
        )

        if not st.session_state.get("signin_otp_sent", False):
            st.info(
                f"📧 **Demo Verification Code:** `{otp_expected}`\n\n"
                f"*(Configure `SMTP_EMAIL` and `SMTP_PASSWORD` in secrets to deliver via real email)*"
            )

        otp_input = st.text_input(
            "6-Digit Verification Code (OTP)",
            max_chars=6,
            placeholder="e.g. 123456",
            key="dlg_signin_otp",
        )

        col_verify, col_back = st.columns([1, 1])
        with col_verify:
            if st.button("Verify Identity & Sign In", type="primary", use_container_width=True):
                if not otp_input or len(otp_input.strip()) != 6:
                    st.error("Please enter the complete 6-digit verification code.")
                    return

                # 1. Verify OTP code
                if otp_input.strip() != otp_expected:
                    st.error("Invalid verification code. Please check and try again.")
                    return

                if time.time() - pending.get("timestamp", 0) > 600:
                    st.error("Verification code has expired. Please request a new code.")
                    st.session_state.signin_step = 1
                    st.rerun()
                    return

                # 2. Check password in DB once OTP is verified
                db_user, auth_msg = user_manager.authenticate_user(email_clean, pwd)
                if not db_user:
                    st.error(f"Password verification failed: {auth_msg}")
                    return

                # Authenticated successfully
                st.session_state.user = db_user
                st.session_state.candidate_id = db_user["id"]
                st.session_state.signin_step = 1
                if "pending_signin" in st.session_state:
                    del st.session_state.pending_signin
                st.toast(f"Welcome back, {db_user.get('name', 'Candidate')}!")
                st.rerun()

        with col_back:
            if st.button("← Back", use_container_width=True):
                st.session_state.signin_step = 1
                st.rerun()


@st.dialog("🐙 GitHub Authentication Setup")
def github_setup_dialog():
    st.markdown(
        """
        ### 🐙 Link GitHub OAuth
        GitHub Sign-In is completely coded and ready to activate!
        
        To allow candidates to sign in with GitHub without domain restrictions:
        1. Open [GitHub Developer Settings → OAuth Apps](https://github.com/settings/developers).
        2. Click **New OAuth App** with:
           - **Application name:** `Voice Mock-Interview Coach`
           - **Homepage URL:** `https://voice-mock-interview-coach.streamlit.app/`
           - **Authorization callback URL:**
             - Production: `https://voice-mock-interview-coach.streamlit.app/`
             - Localhost: `http://localhost:8501/`
        3. Copy your **Client ID** and **Client Secret** into your `.env` or Streamlit Secrets:
        ```toml
        GITHUB_CLIENT_ID = "your_github_client_id"
        GITHUB_CLIENT_SECRET = "your_github_client_secret"
        ```
        Once added, this button will immediately log candidates in with 1 click!
        """
    )
    if st.button("Close", use_container_width=True):
        st.rerun()


@st.dialog("✨ Sign Up as a New Candidate")
def signup_dialog(user_manager: Any):
    step = st.session_state.get("signup_step", 1)

    if step == 1:
        st.markdown(
            "<p style='color:#cbd5e1; font-size:0.9rem; margin-bottom:1rem;'>"
            "Fill in your candidate details below. Your target role, experience track, and performance history are encrypted and stored in our SQLite memory vault."
            "</p>",
            unsafe_allow_html=True,
        )

        name = st.text_input("Full Name", placeholder="e.g. Jane Doe", key="dlg_signup_name")
        email = st.text_input("Email Address", placeholder="e.g. jane@example.com", key="dlg_signup_email")

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            pwd = st.text_input(
                "Create Password",
                type="password",
                placeholder="At least 6 characters",
                key="dlg_signup_pwd",
            )
        with col_p2:
            pwd_confirm = st.text_input(
                "Confirm Password",
                type="password",
                placeholder="Repeat password",
                key="dlg_signup_pwd_conf",
            )

        role_options = [
            "Software Engineer (Full-Stack)",
            "Backend / Systems Engineer",
            "Frontend Engineer",
            "Machine Learning / AI Engineer",
            "Data Scientist",
            "Engineering Manager / Tech Lead",
            "Product Manager",
            "DevOps / Infrastructure Engineer",
        ]
        target_role = st.selectbox("Target Interview Track", role_options, key="dlg_signup_role")

        exp_options = [
            "Entry-Level / New Grad (0-2 years)",
            "Mid-Level Engineer (2-5 years)",
            "Senior Engineer (5-8 years)",
            "Staff / Principal / Lead (8+ years)",
        ]
        exp_level = st.selectbox("Experience Level", exp_options, key="dlg_signup_exp")

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        if st.button("Continue to OTP Verification →", type="primary", use_container_width=True):
            email_clean = (email or "").strip().lower()
            name_clean = (name or "").strip()

            if not name_clean:
                st.error("Please enter your name.")
                return
            if not email_clean or "@" not in email_clean or "." not in email_clean:
                st.error("Please enter a valid email address.")
                return
            if not pwd or len(pwd) < 6:
                st.error("Password must be at least 6 characters long.")
                return
            if pwd != pwd_confirm:
                st.error("Passwords do not match. Please re-enter.")
                return
            if user_manager.user_exists(email_clean):
                st.error("An account with this email already exists. Please use 'Sign In to Existing Account'.")
                return

            otp_code = generate_otp()
            st.session_state.pending_signup = {
                "name": name_clean,
                "email": email_clean,
                "password": pwd,
                "target_role": target_role,
                "experience_level": exp_level,
                "otp": otp_code,
                "timestamp": time.time(),
            }

            sent, msg = send_otp_email(
                email_clean, otp_code, action="New Candidate Registration"
            )
            st.session_state.signup_step = 2
            st.session_state.signup_otp_sent = sent
            st.session_state.signup_otp_msg = msg
            st.rerun()

    elif step == 2:
        pending = st.session_state.get("pending_signup", {})
        email_clean = pending.get("email", "")
        otp_expected = pending.get("otp", "")

        st.markdown(
            f"<p style='color:#cbd5e1; font-size:0.92rem; margin-bottom:1rem;'>"
            f"A 6-digit verification code was generated for <strong>{email_clean}</strong>.<br>"
            f"<span style='color:#94a3b8; font-size:0.82rem;'>Enter the OTP below to finalize your encrypted account.</span>"
            f"</p>",
            unsafe_allow_html=True,
        )

        if not st.session_state.get("signup_otp_sent", False):
            st.info(
                f"📧 **Demo Verification Code:** `{otp_expected}`\n\n"
                f"*(Configure `SMTP_EMAIL` and `SMTP_PASSWORD` in secrets to deliver via real email)*"
            )

        otp_input = st.text_input(
            "6-Digit Verification Code (OTP)",
            max_chars=6,
            placeholder="e.g. 123456",
            key="dlg_signup_otp",
        )

        col_reg, col_back = st.columns([1, 1])
        with col_reg:
            if st.button("Verify & Create Encrypted Profile", type="primary", use_container_width=True):
                if not otp_input or len(otp_input.strip()) != 6:
                    st.error("Please enter the complete 6-digit verification code.")
                    return

                if otp_input.strip() != otp_expected:
                    st.error("Invalid verification code. Please check and try again.")
                    return

                if time.time() - pending.get("timestamp", 0) > 600:
                    st.error("Verification code expired. Please fill the registration form again.")
                    st.session_state.signup_step = 1
                    st.rerun()
                    return

                try:
                    new_user = user_manager.register_user(
                        name=pending.get("name", ""),
                        email=email_clean,
                        password=pending.get("password", ""),
                        target_role=pending.get("target_role", ""),
                        experience_level=pending.get("experience_level", ""),
                    )
                    st.session_state.user = new_user
                    st.session_state.candidate_id = new_user["id"]
                    st.session_state.signup_step = 1
                    if "pending_signup" in st.session_state:
                        del st.session_state.pending_signup
                    st.toast(f"Account registered & encrypted! Welcome, {new_user.get('name', 'Candidate')}!")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Registration failed: {str(ex)}")

        with col_back:
            if st.button("← Back", use_container_width=True):
                st.session_state.signup_step = 1
                st.rerun()


def render_auth_portal(user_manager: Any):
    """
    Renders the Candidate Sign-In & Onboarding Portal with full-height centering,
    background image, and authentication options (Existing Account, GitHub, New Candidate).
    """
    bg_b64 = get_background_base64()
    bg_css = f"""
    .stApp {{
        background: linear-gradient(rgba(10, 15, 29, 0.85), rgba(10, 15, 29, 0.92)), url("data:image/jpeg;base64,{bg_b64}") !important;
        background-size: cover !important;
        background-position: center center !important;
        background-attachment: fixed !important;
        background-repeat: no-repeat !important;
    }}
    """ if bg_b64 else ""

    st.markdown(
        f"""
        <style>
        {bg_css}
        /* Hide Streamlit header, sidebar & footer on login to eliminate dead space */
        header[data-testid="stHeader"], [data-testid="stSidebar"], footer {{
            display: none !important;
        }}
        html, body, .stApp, [data-testid="stAppViewContainer"] {{
            height: 100vh !important;
            max-height: 100vh !important;
            overflow: hidden !important;
        }}
        .main {{
            height: 100vh !important;
            max-height: 100vh !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            align-items: center !important;
            padding: 0 !important;
            margin: 0 !important;
            overflow: hidden !important;
        }}
        .main .block-container {{
            max-width: 1220px !important;
            width: 92% !important;
            padding-top: 0 !important;
            padding-bottom: 0 !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            margin: auto !important;
        }}
        /* Stretch both columns to match full length */
        [data-testid="stHorizontalBlock"] {{
            align-items: stretch !important;
        }}
        [data-testid="column"] {{
            display: flex !important;
            flex-direction: column !important;
        }}
        .hero-badge {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 5px 13px;
            background: rgba(59, 130, 246, 0.14);
            border: 1px solid rgba(59, 130, 246, 0.35);
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            color: #60a5fa;
            margin-bottom: 0.75rem;
        }}
        .hero-title {{
            font-size: 2.25rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            line-height: 1.15;
            margin-bottom: 0.6rem;
            color: #f8fafc;
        }}
        .hero-summary {{
            font-size: 0.98rem;
            color: #cbd5e1;
            line-height: 1.5;
            margin-bottom: 1.25rem;
        }}
        .agenda-grid {{
            display: flex;
            flex-direction: column;
            gap: 0.65rem;
        }}
        .agenda-item {{
            display: flex;
            align-items: flex-start;
            gap: 12px;
            background: rgba(15, 23, 42, 0.65);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 10px 15px;
            backdrop-filter: blur(8px);
            transition: all 0.2s ease;
        }}
        .agenda-item:hover {{
            background: rgba(15, 23, 42, 0.85);
            border-color: rgba(59, 130, 246, 0.4);
            transform: translateX(4px);
        }}
        .agenda-icon {{
            font-size: 1.25rem;
            line-height: 1.2;
            padding-top: 1px;
        }}
        .agenda-title {{
            font-size: 0.91rem;
            font-weight: 700;
            color: #f1f5f9;
            margin-bottom: 2px;
        }}
        .agenda-desc {{
            font-size: 0.81rem;
            color: #94a3b8;
            line-height: 1.4;
        }}
        [data-testid="column"]:nth-child(2) {{
            background: rgba(15, 23, 42, 0.76) !important;
            border: 1px solid rgba(255, 255, 255, 0.14) !important;
            border-radius: 20px !important;
            padding: 2.2rem 2rem !important;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5) !important;
            backdrop-filter: blur(16px) !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            box-sizing: border-box !important;
        }}
        .signin-title {{
            font-size: 1.65rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
            color: #f8fafc;
        }}
        .signin-desc {{
            font-size: 0.88rem;
            color: #94a3b8;
            line-height: 1.5;
            margin-bottom: 1.25rem;
        }}
        .github-login-btn {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            width: 100%;
            background-color: #24292f;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            padding: 10px 16px;
            font-size: 14px;
            font-weight: 600;
            text-decoration: none !important;
            box-sizing: border-box;
            transition: all 0.2s ease;
            margin-bottom: 0.25rem;
        }}
        .github-login-btn:hover {{
            background-color: #1a1e22;
            border-color: rgba(255, 255, 255, 0.4);
            color: #ffffff !important;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }}
        .login-footer {{
            margin-top: 1.25rem;
            font-size: 0.78rem;
            color: #64748b;
            text-align: center;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    col_agenda, col_auth = st.columns([6, 4], gap="large")

    with col_agenda:
        st.markdown(
            """
            <div class="hero-badge">🎙️ AI-Powered Interview Preparation</div>
            <div class="hero-title">Voice Mock-Interview Coach</div>
            <div class="hero-summary">
                Practice realistic, high-stakes technical interviews with an AI that speaks, listens, and probes technical claims in real time.<br>
                Backed by a persistent Knowledge Graph that tracks your demonstrated depth, covered topics, and weak areas across sessions.
            </div>
            <div class="agenda-grid">
                <div class="agenda-item">
                    <div class="agenda-icon">🗣️</div>
                    <div>
                        <div class="agenda-title">1. Real-Time Voice Conversation</div>
                        <div class="agenda-desc">Speak answers naturally using faster-whisper speech recognition and listen to immediate interviewer audio via browser speech synthesis.</div>
                    </div>
                </div>
                <div class="agenda-item">
                    <div class="agenda-icon">📄</div>
                    <div>
                        <div class="agenda-title">2. Resume & Job Description Grounding</div>
                        <div class="agenda-desc">Extracts background claims and target role requirements to challenge specific project tradeoffs and architectural decisions.</div>
                    </div>
                </div>
                <div class="agenda-item">
                    <div class="agenda-icon">🧠</div>
                    <div>
                        <div class="agenda-title">3. Persistent Knowledge Graph Memory</div>
                        <div class="agenda-desc">Unprobed claims, weak explanations, and strengths evolve turn-by-turn and persist permanently to your account across sessions.</div>
                    </div>
                </div>
                <div class="agenda-item">
                    <div class="agenda-icon">📊</div>
                    <div>
                        <div class="agenda-title">4. Lead Hiring Committee Evaluation</div>
                        <div class="agenda-desc">Receive comprehensive dimension ratings, filler word delivery analytics, and STAR framework model answers upon completion.</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_auth:
        github_cfg = get_github_oauth_config()
        has_github_configured = bool(
            github_cfg.get("client_id") and github_cfg.get("client_secret")
        )

        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 1.25rem;">
                <div class="signin-title">Candidate Portal</div>
                <div class="signin-desc">
                    Choose your authentication method to access your personalized interview session and persistent memory graph.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # -------------------------------------------------------------
        # Option 1: Sign in to Existing Account (Email + Password + OTP)
        # -------------------------------------------------------------
        if st.button("🔑 Sign In to Existing Account", type="primary", use_container_width=True):
            st.session_state.signin_step = 1
            signin_dialog(user_manager)

        st.markdown(
            "<div style='text-align: center; color: #64748b; font-size: 0.74rem; font-weight: 700; letter-spacing: 0.05em; margin: 0.65rem 0;'>— OR CONTINUE WITH —</div>",
            unsafe_allow_html=True,
        )

        # -------------------------------------------------------------
        # Option 2: Sign-In with GitHub
        # -------------------------------------------------------------
        if has_github_configured:
            github_auth_url = get_github_auth_url(github_cfg)
            st.markdown(
                f"""
                <a href="{github_auth_url}" target="_top" class="github-login-btn">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                        <path fill-rule="evenodd" clip-rule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"/>
                    </svg>
                    Sign in with GitHub
                </a>
                """,
                unsafe_allow_html=True,
            )
        else:
            if st.button("🐙 Sign in with GitHub", use_container_width=True):
                github_setup_dialog()

        st.markdown(
            "<div style='text-align: center; color: #64748b; font-size: 0.74rem; font-weight: 700; letter-spacing: 0.05em; margin: 0.65rem 0;'>— NEW CANDIDATE? —</div>",
            unsafe_allow_html=True,
        )

        # -------------------------------------------------------------
        # Option 3: Sign Up as a New Candidate (Encrypted DB storage)
        # -------------------------------------------------------------
        if st.button("✨ Sign Up as New Candidate", use_container_width=True):
            st.session_state.signup_step = 1
            signup_dialog(user_manager)

        st.markdown(
            """
            <div class="login-footer">
                🔒 End-to-end encrypted session • Vault-secured memory graph
            </div>
            """,
            unsafe_allow_html=True,
        )
