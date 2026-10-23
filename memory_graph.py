import sqlite3
import json
import os
import contextlib
import datetime
from typing import List, Dict, Any, Optional
from groq import Groq


class CandidateMemoryGraph:
    """
    Persistent SQLite-backed Knowledge Graph for Candidate Interview Memory.
    Tracks claims (with probed/unprobed status), weaknesses, covered topics, and strengths.
    """

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
                CREATE TABLE IF NOT EXISTS candidates (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    target_role TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS claims (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id TEXT,
                    claim_text TEXT,
                    topic TEXT,
                    is_probed INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(candidate_id) REFERENCES candidates(id)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS weaknesses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id TEXT,
                    topic TEXT,
                    observation TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(candidate_id) REFERENCES candidates(id)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS topics_covered (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id TEXT,
                    topic_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(candidate_id) REFERENCES candidates(id)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strengths (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id TEXT,
                    strength_text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(candidate_id) REFERENCES candidates(id)
                )
            """)
            conn.commit()

    def register_candidate(self, candidate_id: str, name: str, target_role: str):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO candidates (id, name, target_role)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET target_role=excluded.target_role
                """,
                (candidate_id, name, target_role),
            )
            conn.commit()

    def add_claim(self, candidate_id: str, claim_text: str, topic: str = ""):
        claim_clean = claim_text.strip()
        if not claim_clean:
            return
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Avoid duplicate claims
            cursor.execute(
                "SELECT id FROM claims WHERE candidate_id = ? AND LOWER(claim_text) = LOWER(?)",
                (candidate_id, claim_clean),
            )
            if not cursor.fetchone():
                cursor.execute(
                    "INSERT INTO claims (candidate_id, claim_text, topic, is_probed) VALUES (?, ?, ?, 0)",
                    (candidate_id, claim_clean, topic),
                )
                conn.commit()

    def mark_claim_probed(self, candidate_id: str, claim_substring: str):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE claims 
                SET is_probed = 1 
                WHERE candidate_id = ? AND is_probed = 0 AND LOWER(claim_text) LIKE LOWER(?)
                """,
                (candidate_id, f"%{claim_substring.strip()}%"),
            )
            conn.commit()

    def add_weakness(self, candidate_id: str, topic: str, observation: str):
        topic_clean = topic.strip()
        if not topic_clean:
            return
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM weaknesses WHERE candidate_id = ? AND LOWER(topic) = LOWER(?)",
                (candidate_id, topic_clean),
            )
            if not cursor.fetchone():
                cursor.execute(
                    "INSERT INTO weaknesses (candidate_id, topic, observation) VALUES (?, ?, ?)",
                    (candidate_id, topic_clean, observation.strip()),
                )
                conn.commit()

    def add_covered_topic(self, candidate_id: str, topic_name: str):
        topic_clean = topic_name.strip()
        if not topic_clean:
            return
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM topics_covered WHERE candidate_id = ? AND LOWER(topic_name) = LOWER(?)",
                (candidate_id, topic_clean),
            )
            if not cursor.fetchone():
                cursor.execute(
                    "INSERT INTO topics_covered (candidate_id, topic_name) VALUES (?, ?)",
                    (candidate_id, topic_clean),
                )
                conn.commit()

    def add_strength(self, candidate_id: str, strength_text: str):
        strength_clean = strength_text.strip()
        if not strength_clean:
            return
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM strengths WHERE candidate_id = ? AND LOWER(strength_text) = LOWER(?)",
                (candidate_id, strength_clean),
            )
            if not cursor.fetchone():
                cursor.execute(
                    "INSERT INTO strengths (candidate_id, strength_text) VALUES (?, ?)",
                    (candidate_id, strength_clean),
                )
                conn.commit()

    def get_candidate_memory_block(self, candidate_id: str, filler_notes: str = "") -> str:
        """
        Retrieves relevant graph history for this candidate and builds the structured memory block
        injected into the interviewer system prompt.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Unprobed claims
            cursor.execute(
                "SELECT claim_text FROM claims WHERE candidate_id = ? AND is_probed = 0 ORDER BY id DESC LIMIT 5",
                (candidate_id,),
            )
            unprobed_claims = [row["claim_text"] for row in cursor.fetchall()]

            # Probed claims
            cursor.execute(
                "SELECT claim_text FROM claims WHERE candidate_id = ? AND is_probed = 1 ORDER BY id DESC LIMIT 3",
                (candidate_id,),
            )
            probed_claims = [row["claim_text"] for row in cursor.fetchall()]

            # Weaknesses
            cursor.execute(
                "SELECT topic, observation FROM weaknesses WHERE candidate_id = ? ORDER BY id DESC LIMIT 4",
                (candidate_id,),
            )
            weaknesses = [f"{row['topic']} ({row['observation']})" if row['observation'] else row['topic'] for row in cursor.fetchall()]

            # Covered topics
            cursor.execute(
                "SELECT topic_name FROM topics_covered WHERE candidate_id = ? ORDER BY id DESC LIMIT 6",
                (candidate_id,),
            )
            covered = [row["topic_name"] for row in cursor.fetchall()]

            # Strengths
            cursor.execute(
                "SELECT strength_text FROM strengths WHERE candidate_id = ? ORDER BY id DESC LIMIT 3",
                (candidate_id,),
            )
            strengths = [row["strength_text"] for row in cursor.fetchall()]

        # Build formatted block
        lines = []
        if unprobed_claims:
            for c in unprobed_claims:
                lines.append(f"- Claimed: {c} (not yet probed)")
        if probed_claims:
            for c in probed_claims:
                lines.append(f"- Claimed: {c} (probed)")
        if weaknesses:
            lines.append(f"- Weak on: {', '.join(weaknesses)}")
        if covered:
            lines.append(f"- Covered: {', '.join(covered)}")
        if strengths:
            lines.append(f"- Strengths: {', '.join(strengths)}")
        if filler_notes:
            lines.append(f"- Filler words / Delivery: {filler_notes}")

        if not lines:
            return ""

        block = (
            "Candidate memory from previous sessions & earlier turns:\n"
            + "\n".join(lines)
            + "\n\nUse this context to ask targeted follow-ups on unprobed claims or probe known weak areas.\n"
            + "Do not mention this memory directly to the candidate."
        )
        return block

    def get_full_graph_state(self, candidate_id: str) -> Dict[str, Any]:
        """Returns structured memory graph state for UI visualization."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, claim_text, topic, is_probed, created_at FROM claims WHERE candidate_id = ? ORDER BY id DESC",
                (candidate_id,),
            )
            claims = [dict(row) for row in cursor.fetchall()]

            cursor.execute(
                "SELECT id, topic, observation, created_at FROM weaknesses WHERE candidate_id = ? ORDER BY id DESC",
                (candidate_id,),
            )
            weaknesses = [dict(row) for row in cursor.fetchall()]

            cursor.execute(
                "SELECT id, topic_name, created_at FROM topics_covered WHERE candidate_id = ? ORDER BY id DESC",
                (candidate_id,),
            )
            topics = [dict(row) for row in cursor.fetchall()]

            cursor.execute(
                "SELECT id, strength_text, created_at FROM strengths WHERE candidate_id = ? ORDER BY id DESC",
                (candidate_id,),
            )
            strengths = [dict(row) for row in cursor.fetchall()]

        return {
            "claims": claims,
            "weaknesses": weaknesses,
            "topics": topics,
            "strengths": strengths,
        }

    def reset_candidate_memory(self, candidate_id: str):
        """Clears memory entries for a candidate to restart fresh."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM claims WHERE candidate_id = ?", (candidate_id,))
            cursor.execute("DELETE FROM weaknesses WHERE candidate_id = ?", (candidate_id,))
            cursor.execute("DELETE FROM topics_covered WHERE candidate_id = ?", (candidate_id,))
            cursor.execute("DELETE FROM strengths WHERE candidate_id = ?", (candidate_id,))
            conn.commit()

    def export_candidate_memory(self, candidate_id: str) -> Dict[str, Any]:
        """Exports all graph memory entries for backup, download, and migration."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, target_role, created_at FROM candidates WHERE id = ?", (candidate_id,))
            cand = cursor.fetchone()
            full_state = self.get_full_graph_state(candidate_id)
            return {
                "candidate_id": candidate_id,
                "profile": dict(cand) if cand else {},
                "memory": full_state,
            }

    def import_candidate_memory(self, candidate_id: str, data: Dict[str, Any]):
        """Imports graph memory entries from a backup JSON payload."""
        if not data:
            return
        profile = data.get("profile", {})
        if profile:
            self.register_candidate(
                candidate_id,
                profile.get("name", "Candidate"),
                profile.get("target_role", "Software Engineer"),
            )

        mem = data.get("memory", {})
        for c in mem.get("claims", []):
            self.add_claim(candidate_id, c.get("claim_text", ""), c.get("topic", ""))
            if c.get("is_probed"):
                self.mark_claim_probed(candidate_id, c.get("claim_text", ""))
        for w in mem.get("weaknesses", []):
            self.add_weakness(candidate_id, w.get("topic", ""), w.get("observation", ""))
        for t in mem.get("topics", []):
            self.add_covered_topic(candidate_id, t.get("topic_name", ""))
        for s in mem.get("strengths", []):
            self.add_strength(candidate_id, s.get("strength_text", ""))


def extract_and_update_memory(
    groq_client: Groq,
    graph: CandidateMemoryGraph,
    candidate_id: str,
    last_question: str,
    user_answer: str,
    model_name: str = "llama-3.1-8b-instant",
) -> Dict[str, Any]:
    """
    Performs a lightweight, dedicated Groq call using JSON mode to extract facts, claims,
    topics, and weak areas from the candidate's latest answer, then commits them to the graph.
    """
    if not user_answer or len(user_answer.strip()) < 10:
        return {}

    extraction_prompt = f"""You are an automated interview knowledge-graph extraction engine.
Analyze the candidate's latest verbal answer to the interviewer's question.

Interviewer's Question:
\"\"\"{last_question}\"\"\"

Candidate's Answer:
\"\"\"{user_answer}\"\"\"

Extract structured memory facts in JSON format with the following keys:
1. "new_claims": Array of strings. Specific technical experiences, past projects, metrics, tools, or architectural accomplishments the candidate explicitly claimed they built, designed, or led (e.g. "built a Kafka ingestion pipeline at a previous job"). Keep empty if none.
2. "probed_claim_keywords": Array of strings. If the interviewer's question probed an existing technical topic or past project and the candidate answered it, list keywords of the claim that was probed.
3. "weak_topics": Array of objects: [{{"topic": "topic name", "observation": "brief note on why it was weak or superficial"}}]. Flag topics where the candidate struggled, had conceptual gaps, gave hand-wavy explanations, or failed to explain tradeoffs.
4. "covered_topics": Array of strings. Technical or behavioral subject areas touched upon in this answer (e.g. "Kafka", "Data Partitioning", "Conflict Resolution").
5. "strengths": Array of strings. Standout technical mastery or well-reasoned decisions demonstrated in this answer.

Return ONLY valid JSON.
"""

    candidate_models = [model_name, "llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
    seen = set()
    models_to_try = [m for m in candidate_models if not (m in seen or seen.add(m))]

    parsed = {}
    for m in models_to_try:
        try:
            resp = groq_client.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system", "content": "You are a JSON fact extractor. Respond strictly with a JSON object."},
                    {"role": "user", "content": extraction_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=600,
            )
            raw_content = resp.choices[0].message.content.strip()
            parsed = json.loads(raw_content)
            break
        except Exception as e:
            err_str = str(e).lower()
            if "model_not_found" in err_str or "does not exist" in err_str:
                continue
            # JSON decode or other error
            break

    if not parsed:
        return {}

    # Commit extracted facts into the SQLite graph
    try:
        # 1. New claims
        for claim in parsed.get("new_claims", []):
            if isinstance(claim, str) and len(claim.strip()) > 5:
                graph.add_claim(candidate_id, claim.strip())

        # 2. Mark probed claims
        for kw in parsed.get("probed_claim_keywords", []):
            if isinstance(kw, str) and kw.strip():
                graph.mark_claim_probed(candidate_id, kw.strip())

        # 3. Weak topics
        for w in parsed.get("weak_topics", []):
            if isinstance(w, dict):
                graph.add_weakness(candidate_id, w.get("topic", ""), w.get("observation", ""))
            elif isinstance(w, str) and len(w.strip()) > 3:
                graph.add_weakness(candidate_id, w.strip(), "Hand-wavy or incomplete explanation")

        # 4. Covered topics
        for topic in parsed.get("covered_topics", []):
            if isinstance(topic, str) and len(topic.strip()) > 2:
                graph.add_covered_topic(candidate_id, topic.strip())

        # 5. Strengths
        for s in parsed.get("strengths", []):
            if isinstance(s, str) and len(s.strip()) > 3:
                graph.add_strength(candidate_id, s.strip())

    except Exception as e:
        print(f"Error writing to memory graph: {e}")

    return parsed
