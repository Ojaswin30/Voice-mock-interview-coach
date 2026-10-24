"""
Profile Page View for Voice Mock-Interview Coach.
Displays personal/job track information, read-only registered email, and password management.
"""
from typing import Dict, Any, Optional
import streamlit as st


def render_profile_page(
    user_manager,
    current_user: Optional[Dict[str, Any]] = None,
    candidate_id: Optional[str] = None,
    candidate_name: Optional[str] = None,
    candidate_email: Optional[str] = None,
    target_role: Optional[str] = None,
    experience_level: Optional[str] = None,
):
    """Renders Candidate Profile, verified email display, and password update form."""
    user = current_user or st.session_state.get("user") or {}
    candidate_id = candidate_id or user.get("id", "guest")
    candidate_name = candidate_name or user.get("name", "Candidate")
    candidate_email = candidate_email or user.get("email", "")
    target_role = target_role or st.session_state.get("target_role", "Software Engineer (Full-Stack)")
    experience_level = experience_level or st.session_state.get("experience_level", "Senior (5-8 yrs)")

    st.title("👤 Candidate Profile & Account Settings")
    st.caption("Manage your verified identity, target interview track, and password security.")

    # 1. Candidate Personal Info & Job Track
    st.subheader("📋 Personal & Track Information")
    with st.container(border=True):
        col_em1, col_em2 = st.columns([3, 1])
        with col_em1:
            st.text_input(
                "Registered Email Address (Permanent)",
                value=candidate_email or "Not registered",
                disabled=True,
                help="Your registered email address cannot be changed to preserve persistent memory graph consistency.",
            )
        with col_em2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            st.success("🔒 Verified Account", icon="✅")

        col_n1, col_n2 = st.columns(2)
        with col_n1:
            edit_name = st.text_input("Full Name", value=candidate_name)
        with col_n2:
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
            role_idx = (
                role_options.index(target_role) if target_role in role_options else 0
            )
            edit_role = st.selectbox("Target Interview Track", role_options, index=role_idx)

        exp_options = [
            "Entry-Level / Intern (0-2 yrs)",
            "Mid-Level (2-4 yrs)",
            "Senior (5-8 yrs)",
            "Staff / Principal (8+ yrs)",
            "Director / Executive",
        ]
        exp_idx = (
            exp_options.index(experience_level)
            if experience_level in exp_options
            else 2
        )
        edit_exp = st.selectbox("Experience Seniority", exp_options, index=exp_idx)

        if st.button("💾 Save Profile Changes", type="primary"):
            updated_user = user_manager.update_user_profile(
                candidate_id,
                name=edit_name,
                target_role=edit_role,
                experience_level=edit_exp,
            )
            st.session_state.user = updated_user
            st.session_state.target_role = edit_role
            st.session_state.experience_level = edit_exp
            st.toast("Profile updated and encrypted successfully!")
            st.rerun()

    # 2. Password Update Section
    st.subheader("🔑 Password & Security")
    with st.container(border=True):
        has_pwd = bool(user.get("password_hash"))
        if has_pwd:
            curr_pwd = st.text_input(
                "Current Password", type="password", key="profile_current_pwd"
            )
        else:
            curr_pwd = ""
            st.info(
                "ℹ️ Your account was created via social login. Setting a password enables email/password sign-in."
            )

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            new_pwd = st.text_input(
                "New Password",
                type="password",
                placeholder="At least 6 characters",
                key="profile_new_pwd",
            )
        with col_p2:
            conf_pwd = st.text_input(
                "Confirm New Password",
                type="password",
                placeholder="Re-type new password",
                key="profile_conf_pwd",
            )

        if st.button("🔄 Update Password"):
            if not new_pwd or len(new_pwd) < 6:
                st.error("New password must be at least 6 characters long.")
            elif new_pwd != conf_pwd:
                st.error("New passwords do not match.")
            else:
                success, msg = user_manager.update_user_password(
                    user_id=candidate_id,
                    new_password=new_pwd,
                    current_password=curr_pwd if has_pwd else None,
                )
                if success:
                    st.success("✅ Password updated successfully!")
                    # Refresh session user
                    st.session_state.user = user_manager.get_user_by_id(candidate_id)
                    st.rerun()
                else:
                    st.error(f"❌ Failed to update password: {msg}")
