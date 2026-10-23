"""
Persona Page View for Voice Mock-Interview Coach.
Visualizes the Candidate Knowledge Graph: claims, probed/unprobed status, 
flagged weaknesses, proven strengths, and covered topics.
"""
import json
import streamlit as st


def render_persona_page(memory_graph, candidate_id: str, candidate_name: str):
    """Renders the Candidate Knowledge Graph state, analytics, and backup options."""
    st.title("🧠 Candidate Knowledge Graph")
    st.caption(
        f"Persistent memory state for **{candidate_name}** — tracking demonstrated depth, unprobed claims, and weak explanations across all sessions."
    )

    graph_state = memory_graph.get_full_graph_state(candidate_id)
    claims = graph_state.get("claims", [])
    weaknesses = graph_state.get("weaknesses", [])
    topics = graph_state.get("topics", [])
    strengths = graph_state.get("strengths", [])

    probed_count = sum(1 for c in claims if c.get("is_probed"))

    pm1, pm2, pm3, pm4 = st.columns(4)
    pm1.metric("Total Claims", len(claims), f"{probed_count} probed")
    pm2.metric("Strengths Proven", len(strengths))
    pm3.metric("Weaknesses Flagged", len(weaknesses))
    pm4.metric("Topics Covered", len(topics))

    st.markdown("---")

    tab_claims, tab_weak, tab_str, tab_top = st.tabs([
        "📌 Claims & Probing Status",
        "⚠️ Identified Weaknesses",
        "💪 Strengths Demonstrated",
        "📚 Topics Covered",
    ])

    with tab_claims:
        if claims:
            for c in claims:
                col_c1, col_c2 = st.columns([4, 1])
                with col_c1:
                    status_badge = "✅ **PROBED**" if c.get("is_probed") else "⏳ **UNPROBED**"
                    st.markdown(f"{status_badge} — *{c.get('claim_text', '')}*")
                with col_c2:
                    if c.get("topic"):
                        st.caption(f"Topic: `{c['topic']}`")
        else:
            st.info("No claims extracted yet. Upload a resume on the Interview page to populate claims.")

    with tab_weak:
        if weaknesses:
            for w in weaknesses:
                obs = f": *{w['observation']}*" if w.get("observation") else ""
                st.markdown(f"- ⚠️ **{w.get('topic', 'Topic')}**{obs}")
        else:
            st.success("No weak explanations flagged yet! Outstanding performance.")

    with tab_str:
        if strengths:
            for s in strengths:
                st.markdown(f"- 💪 **{s.get('strength_text', '')}**")
        else:
            st.info("Complete an interview session to identify key strengths.")

    with tab_top:
        if topics:
            topics_list = [f"`{t.get('topic_name', '')}`" for t in topics]
            st.markdown(" ".join(topics_list))
        else:
            st.info("No topics covered yet.")

    st.markdown("---")
    st.subheader("💾 Backup, Restore & Reset Memory")
    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        backup_dict = memory_graph.export_candidate_memory(candidate_id)
        st.download_button(
            "📥 Download Memory (JSON)",
            data=json.dumps(backup_dict, indent=2),
            file_name=f"{candidate_id}_memory.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_b2:
        uploaded_backup = st.file_uploader(
            "Restore Memory (.json)",
            type=["json"],
            key="restore_json_file_persona",
            label_visibility="collapsed",
        )
        if uploaded_backup is not None:
            try:
                uploaded_backup.seek(0)
                loaded_data = json.loads(uploaded_backup.read().decode("utf-8"))
                memory_graph.import_candidate_memory(candidate_id, loaded_data)
                st.success("Memory restored successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to restore: {e}")
    with col_b3:
        if st.button("🗑️ Reset Candidate Memory", type="secondary", use_container_width=True):
            memory_graph.reset_candidate_memory(candidate_id)
            st.toast("Memory graph reset!")
            st.rerun()
