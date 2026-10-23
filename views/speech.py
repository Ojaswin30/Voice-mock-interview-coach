"""
Speech Synthesis (TTS) Component using native Browser Web Speech API.
Provides zero-latency, zero-cost, streaming natural voices with autoplay, replay, and speed controls.
"""
import json
import streamlit.components.v1 as components


def render_speech_player(text: str, auto_play: bool = True, key_id: str = "tts_default"):
    """
    Renders an HTML/JavaScript component utilizing the browser's native Web Speech API
    (speechSynthesis). Includes autoplay handling, voice selection, rate slider, and replay controls.
    """
    clean_text = json.dumps(text)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        :root {{
          --primary-color: #2563eb;
          --primary-hover: #1d4ed8;
          --bg-color: #f8fafc;
          --card-bg: #ffffff;
          --text-color: #1e293b;
          --border-color: #e2e8f0;
          --accent-color: #10b981;
        }}
        @media (prefers-color-scheme: dark) {{
          :root {{
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --text-color: #f1f5f9;
            --border-color: #334155;
          }}
        }}
        body {{
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
          margin: 0;
          padding: 8px 12px;
          background: transparent;
          color: var(--text-color);
        }}
        .tts-card {{
          display: flex;
          align-items: center;
          justify-content: space-between;
          flex-wrap: wrap;
          gap: 12px;
          background-color: var(--card-bg);
          border: 1px solid var(--border-color);
          border-radius: 10px;
          padding: 10px 16px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .controls-left {{
          display: flex;
          align-items: center;
          gap: 10px;
        }}
        .btn {{
          display: inline-flex;
          align-items: center;
          gap: 6px;
          font-weight: 500;
          font-size: 13px;
          padding: 6px 14px;
          border-radius: 6px;
          border: none;
          cursor: pointer;
          transition: background-color 0.15s ease, transform 0.05s ease;
        }}
        .btn:active {{
          transform: scale(0.98);
        }}
        .btn-primary {{
          background-color: var(--primary-color);
          color: white;
        }}
        .btn-primary:hover {{
          background-color: var(--primary-hover);
        }}
        .btn-secondary {{
          background-color: transparent;
          border: 1px solid var(--border-color);
          color: var(--text-color);
        }}
        .btn-secondary:hover {{
          background-color: rgba(125, 125, 125, 0.1);
        }}
        .status-badge {{
          font-size: 12px;
          color: #64748b;
          display: flex;
          align-items: center;
          gap: 6px;
        }}
        .pulse {{
          display: inline-block;
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: var(--accent-color);
          animation: pulse-ring 1.5s infinite;
        }}
        @keyframes pulse-ring {{
          0% {{ opacity: 1; transform: scale(0.9); }}
          50% {{ opacity: 0.4; transform: scale(1.2); }}
          100% {{ opacity: 1; transform: scale(0.9); }}
        }}
        .rate-selector {{
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 12px;
          color: #64748b;
        }}
        select {{
          background: var(--card-bg);
          color: var(--text-color);
          border: 1px solid var(--border-color);
          border-radius: 4px;
          padding: 3px 6px;
          font-size: 12px;
        }}
      </style>
    </head>
    <body>
      <div class="tts-card">
        <div class="controls-left">
          <button id="playBtn" class="btn btn-primary" onclick="togglePlay()">
            <span id="playIcon">🔊</span> <span id="playText">Play Voice</span>
          </button>
          <button id="stopBtn" class="btn btn-secondary" onclick="stopSpeech()">
            ⏹ Stop
          </button>
          <div id="statusBadge" class="status-badge" style="display: none;">
            <span class="pulse"></span> <span>Speaking...</span>
          </div>
        </div>
        <div class="rate-selector">
          <label for="speedSelect">Speed:</label>
          <select id="speedSelect" onchange="updateRate()">
            <option value="0.85">0.85x</option>
            <option value="1.0" selected>1.0x</option>
            <option value="1.15">1.15x</option>
            <option value="1.3">1.3x</option>
          </select>
        </div>
      </div>

      <script>
        const messageText = {clean_text};
        let isSpeaking = false;
        let selectedRate = 1.0;
        let synth = window.speechSynthesis;
        let utterance = null;

        function updateRate() {{
          const sel = document.getElementById("speedSelect");
          selectedRate = parseFloat(sel.value);
          if (isSpeaking) {{
            synth.cancel();
            speak();
          }}
        }}

        function speak() {{
          if (!synth) {{
            alert("Speech synthesis is not supported in this browser.");
            return;
          }}
          synth.cancel();

          utterance = new SpeechSynthesisUtterance(messageText);
          utterance.rate = selectedRate;
          utterance.pitch = 1.0;

          // Pick natural English voice if available
          const voices = synth.getVoices();
          const preferredVoice = voices.find(v => 
            v.lang.startsWith("en") && (v.name.includes("Google") || v.name.includes("Natural") || v.name.includes("Jenny") || v.name.includes("David") || v.name.includes("Samantha"))
          ) || voices.find(v => v.lang.startsWith("en"));

          if (preferredVoice) {{
            utterance.voice = preferredVoice;
          }}

          utterance.onstart = function() {{
            isSpeaking = true;
            document.getElementById("playIcon").innerText = "⏸";
            document.getElementById("playText").innerText = "Pause";
            document.getElementById("statusBadge").style.display = "flex";
          }};

          utterance.onend = function() {{
            isSpeaking = false;
            document.getElementById("playIcon").innerText = "🔁";
            document.getElementById("playText").innerText = "Replay";
            document.getElementById("statusBadge").style.display = "none";
          }};

          utterance.onerror = function(event) {{
            console.warn("SpeechSynthesis error:", event);
            isSpeaking = false;
            document.getElementById("playIcon").innerText = "🔊";
            document.getElementById("playText").innerText = "Play Voice";
            document.getElementById("statusBadge").style.display = "none";
          }};

          synth.speak(utterance);
        }}

        function togglePlay() {{
          if (synth.speaking && !synth.paused) {{
            synth.pause();
            isSpeaking = false;
            document.getElementById("playIcon").innerText = "▶️";
            document.getElementById("playText").innerText = "Resume";
            document.getElementById("statusBadge").style.display = "none";
          }} else if (synth.paused) {{
            synth.resume();
            isSpeaking = true;
            document.getElementById("playIcon").innerText = "⏸";
            document.getElementById("playText").innerText = "Pause";
            document.getElementById("statusBadge").style.display = "flex";
          }} else {{
            speak();
          }}
        }}

        function stopSpeech() {{
          synth.cancel();
          isSpeaking = false;
          document.getElementById("playIcon").innerText = "🔊";
          document.getElementById("playText").innerText = "Play Voice";
          document.getElementById("statusBadge").style.display = "none";
        }}

        if (speechSynthesis.onvoiceschanged !== undefined) {{
          speechSynthesis.onvoiceschanged = () => {{}};
        }}

        if ({str(auto_play).lower()}) {{
          window.addEventListener("DOMContentLoaded", () => {{
            try {{
              speak();
            }} catch(e) {{
              console.log("Autoplay prevented:", e);
            }}
          }});
        }}
      </script>
    </body>
    </html>
    """
    components.html(html_code, height=65, scrolling=False)
