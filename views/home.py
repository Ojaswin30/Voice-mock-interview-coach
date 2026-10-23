"""
Home Page View for Voice Mock-Interview Coach.
Provides platform overview, core capability highlights, and step-by-step instructions.
"""
import streamlit as st


def render_home_page():
    """Renders the comprehensive Home page."""
    st.title("🎙️ Voice Mock-Interview Coach")
    st.markdown(
        """
        Welcome to your personalized AI interview coach. Practice realistic, high-stakes technical interviews with 
        an adaptive AI interviewer that speaks, listens, and probes technical claims in real time. 
        Backed by an evolving **Persistent Knowledge Graph** that tracks your demonstrated depth, covered topics, 
        and weak areas across interview sessions.
        """
    )

    st.markdown("---")

    col_h1, col_h2 = st.columns(2, gap="large")
    with col_h1:
        st.subheader("🚀 Platform Features")
        st.markdown(
            """
            - 🗣️ **Real-Time Voice Conversation:** Speak your answers naturally using `faster-whisper` and listen to immediate interviewer speech synthesized in your browser.
            - 📄 **Resume & Job Description Grounding:** Analyzes your background claims and specific target job requirements to challenge real architectural tradeoffs.
            - 🧠 **Persistent Knowledge Graph:** Tracks unprobed claims, weak explanations, and proven strengths that permanently update across turns and sessions.
            - 📊 **Lead Hiring Committee Evaluation:** Get comprehensive dimension ratings, filler word delivery analytics, and STAR framework model answers upon completion.
            """
        )

    with col_h2:
        st.subheader("📋 Step-by-Step Instructions")
        st.markdown(
            """
            1. **Verify Your Profile:** Check your target role, seniority level, and password security under **👤 Profile**.
            2. **Choose Engine Settings:** Customize your Groq LLM model and interviewer demeanor under **⚙️ Model**.
            3. **Upload Context:** Head over to **🎙️ Interview** to upload your resume and paste your target Job Description. Click **Parse into Knowledge Graph**.
            4. **Conduct Interview:** Click **Start Interview**, speak your responses clearly into the microphone, and listen to the interviewer's live questions.
            5. **Inspect Persona:** Visit **🧠 Persona** to see what claims were probed, what strengths you proved, and what weaknesses were flagged.
            6. **Review Past Sessions:** Access all your completed assessments in **📜 History**.
            """
        )

    st.markdown("---")
    c_btn1, c_btn2, _ = st.columns([2.5, 2.5, 3])
    with c_btn1:
        if st.button("🚀 Enter Interview Room →", type="primary", use_container_width=True):
            st.session_state.current_page = "🎙️ Interview"
            st.rerun()
    with c_btn2:
        if st.button("🧠 View Candidate Knowledge Graph →", use_container_width=True):
            st.session_state.current_page = "🧠 Persona"
            st.rerun()
