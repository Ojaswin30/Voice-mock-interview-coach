"""
History Page View for Voice Mock-Interview Coach.
Lists historical completed interview sessions, Lead Hiring Committee scorecards, 
delivery analytics, and turn-by-turn conversation transcripts.
"""
import streamlit as st


def render_history_page(user_manager, candidate_id: str):
    """Renders the historical list of completed mock interviews."""
    st.title("📜 Interview History & Past Evaluations")
    st.caption("Review your historical interview performances, hiring committee reports, and delivery analytics.")

    sessions = user_manager.get_interview_sessions(candidate_id)

    if not sessions:
        st.info(
            "ℹ️ No past interview sessions found for your profile. Practice an interview on the **🎙️ Interview** page to see your history here!"
        )
        if st.button("Start Your First Mock Interview →", type="primary"):
            st.session_state.current_page = "🎙️ Interview"
            st.rerun()
    else:
        hm1, hm2 = st.columns(2)
        hm1.metric("Total Interviews Completed", len(sessions))
        hm2.metric("Latest Session", sessions[0].get("created_at", "")[:16])

        st.markdown("---")

        for idx, sess in enumerate(sessions):
            sess_id = sess["id"]
            sess_date = sess.get("created_at", "")[:16]
            sess_role = sess.get("target_role", "Software Engineer")
            sess_exp = sess.get("experience_level", "")
            q_count = sess.get("questions_count", 0)

            with st.expander(
                f"📅 **{sess_date}** — 🎯 **{sess_role}** ({sess_exp}) • {q_count} Questions",
                expanded=(idx == 0),
            ):
                col_h_actions = st.columns([5, 1])
                with col_h_actions[1]:
                    if st.button(
                        "🗑️ Delete",
                        key=f"del_sess_{sess_id}",
                        help="Delete this interview session record",
                    ):
                        user_manager.delete_interview_session(sess_id)
                        st.toast("Interview record removed.")
                        st.rerun()

                # Metrics banner
                f_stats = sess.get("filler_stats", {})
                sm1, sm2, sm3 = st.columns(3)
                sm1.metric("Questions Answered", q_count)
                sm2.metric("Words Spoken", f_stats.get("total_words", 0))
                sm3.metric("Filler Density", f"{f_stats.get('filler_density_pct', 0.0)}%")

                st.markdown("#### 📊 Lead Hiring Committee Report")
                st.markdown(sess.get("evaluation_report", "No report available."))

                # Conversation transcript
                msgs = sess.get("messages", [])
                if msgs:
                    with st.expander("💬 View Full Conversation Transcript", expanded=False):
                        for m in msgs:
                            if m.get("role") == "assistant":
                                st.markdown(f"**Interviewer:** {m.get('content', '')}")
                            elif m.get("role") == "user":
                                st.markdown(f"**You:** {m.get('content', '')}")
