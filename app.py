# app.py — EmoBERT Streamlit Frontend
# Run with: streamlit run app.py
# Make sure api.py is running first: uvicorn api:app --reload

import streamlit as st
import requests
import plotly.graph_objects as go
from datetime import datetime

# ── Config ─────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"

EMOTIONS = {
    "sadness":  {"emoji": "😢", "color": "#5B8FD4", "bg": "#EBF2FC"},
    "joy":      {"emoji": "😄", "color": "#E8A020", "bg": "#FEF6E4"},
    "love":     {"emoji": "❤️",  "color": "#D44F6E", "bg": "#FDEEF2"},
    "anger":    {"emoji": "😠", "color": "#D44F3B", "bg": "#FDECEA"},
    "fear":     {"emoji": "😨", "color": "#8A5FC4", "bg": "#F3EDFC"},
    "surprise": {"emoji": "😲", "color": "#2EAF8A", "bg": "#E6F7F3"},
}

EXAMPLES = [
    "Just got promoted at work! Best day of my life! 🎉",
    "I miss my grandmother so much, it still hurts every day.",
    "How could they do this to us?! This is absolutely outrageous!",
    "Walking alone at night, every single sound makes me jump.",
    "OMG I just won tickets to the concert!! I can't believe it!!",
    "You are my sunshine, my everything, I love you so much 🌸",
]

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EmoBERT",
    page_icon="🧠",
    layout="centered"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main { padding-top: 2rem; }

    .header-container {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }

    .header-title {
        font-size: 3rem;
        font-weight: 600;
        letter-spacing: -1px;
        margin: 0;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .header-subtitle {
        font-size: 1rem;
        color: #888;
        margin-top: 0.25rem;
        font-weight: 300;
        letter-spacing: 0.5px;
    }

    .api-badge {
        display: inline-block;
        background: #e8f5e9;
        color: #2e7d32;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-top: 0.5rem;
        font-family: 'DM Mono', monospace;
    }

    .api-badge-error {
        background: #ffebee;
        color: #c62828;
    }

    .emotion-card {
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(0,0,0,0.06);
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    }

    .emotion-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        opacity: 0.6;
        margin-bottom: 0.25rem;
    }

    .emotion-name {
        font-size: 2.2rem;
        font-weight: 600;
        letter-spacing: -0.5px;
        margin: 0;
    }

    .emotion-confidence {
        font-size: 0.9rem;
        opacity: 0.7;
        margin-top: 0.25rem;
        font-family: 'DM Mono', monospace;
    }

    .history-item {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin: 0.4rem 0;
        border-left: 3px solid #ddd;
        font-size: 0.9rem;
    }

    .history-text {
        color: #333;
        font-weight: 400;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 300px;
    }

    .history-meta {
        color: #999;
        font-size: 0.75rem;
        font-family: 'DM Mono', monospace;
        margin-top: 2px;
    }

    .char-count {
        text-align: right;
        font-size: 0.75rem;
        color: #aaa;
        font-family: 'DM Mono', monospace;
        margin-top: -10px;
        margin-bottom: 8px;
    }

    .stButton > button {
        border-radius: 10px;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
    }

    .stTextArea textarea {
        border-radius: 10px;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.95rem;
    }

    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Check API health ────────────────────────────────────────────────────────
def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200
    except:
        return False

# ── Call prediction API ─────────────────────────────────────────────────────
def get_prediction(text: str):
    try:
        r = requests.post(
            f"{API_URL}/predict",
            json={"text": text},
            timeout=10
        )
        if r.status_code == 200:
            return r.json(), None
        return None, r.json().get("detail", "Unknown error")
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API. Make sure api.py is running."
    except Exception as e:
        return None, str(e)

# ── Plotly chart ────────────────────────────────────────────────────────────
def make_chart(scores: dict):
    labels = list(scores.keys())
    values = list(scores.values())
    colors = [EMOTIONS[l]["color"] for l in labels]

    fig = go.Figure(go.Bar(
        x=values,
        y=[f'{EMOTIONS[l]["emoji"]} {l.capitalize()}' for l in labels],
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(width=0),
        ),
        text=[f"{v}%" for v in values],
        textposition="outside",
        textfont=dict(family="DM Mono", size=12),
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 110], showgrid=False, visible=False),
        yaxis=dict(autorange="reversed", tickfont=dict(size=13, family="DM Sans")),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=60, t=10, b=10),
        height=260,
        showlegend=False,
    )
    return fig

# ── Session state ───────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Header ──────────────────────────────────────────────────────────────────
api_ok = check_api()
badge_class = "api-badge" if api_ok else "api-badge api-badge-error"
badge_text  = "● API connected" if api_ok else "● API offline — run: uvicorn api:app --reload"

st.markdown(f"""
<div class="header-container">
    <h1 class="header-title">🧠 EmoBERT</h1>
    <p class="header-subtitle">Emotion Detection · Powered by BERT · REST API</p>
    <span class="{badge_class}">{badge_text}</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Main layout ─────────────────────────────────────────────────────────────
col_main, col_history = st.columns([3, 1.2])

with col_main:

    # Text input
    text_input = st.text_area(
        "Enter text to analyze:",
        placeholder="Type a tweet, sentence, or social media post...",
        height=130,
        max_chars=1000,
        label_visibility="collapsed",
    )

    # Character count
    char_count = len(text_input) if text_input else 0
    st.markdown(f'<div class="char-count">{char_count} / 1000</div>', unsafe_allow_html=True)

    # Analyze button
    analyze = st.button(
        "Analyze Emotion →",
        use_container_width=True,
        type="primary",
        disabled=not api_ok
    )

    # Examples
    with st.expander("✨ Try an example"):
        for ex in EXAMPLES:
            if st.button(ex, use_container_width=True, key=ex):
                text_input = ex
                analyze = True

    # Results
    if analyze and text_input.strip():
        with st.spinner("Analyzing..."):
            result, error = get_prediction(text_input)

        if error:
            st.error(f"Error: {error}")

        elif result:
            emotion    = result["emotion"]
            confidence = result["confidence"]
            scores     = result["scores"]
            emoji      = result["emoji"]
            meta       = EMOTIONS[emotion]

            # Emotion card
            st.markdown(f"""
            <div class="emotion-card" style="background:{meta['bg']}; border-color:{meta['color']}22">
                <div class="emotion-label" style="color:{meta['color']}">Detected Emotion</div>
                <div class="emotion-name" style="color:{meta['color']}">{emoji} {emotion.capitalize()}</div>
                <div class="emotion-confidence">{confidence}% confidence</div>
            </div>
            """, unsafe_allow_html=True)

            # Chart
            st.markdown("**All emotion scores**")
            st.plotly_chart(make_chart(scores), use_container_width=True)

            # Save to history
            st.session_state.history.insert(0, {
                "text":    text_input[:60] + ("..." if len(text_input) > 60 else ""),
                "emotion": emotion,
                "emoji":   emoji,
                "color":   meta["color"],
                "conf":    confidence,
                "time":    datetime.now().strftime("%H:%M"),
            })
            st.session_state.history = st.session_state.history[:5]

    elif analyze:
        st.warning("Please enter some text first.")

# ── History sidebar ─────────────────────────────────────────────────────────
with col_history:
    st.markdown("**Recent**")

    if not st.session_state.history:
        st.markdown('<div style="color:#bbb; font-size:0.85rem;">No analysis yet</div>', unsafe_allow_html=True)
    else:
        for item in st.session_state.history:
            st.markdown(f"""
            <div class="history-item" style="border-left-color:{item['color']}">
                <div style="font-size:1.1rem">{item['emoji']} {item['emotion'].capitalize()}</div>
                <div class="history-text">{item['text']}</div>
                <div class="history-meta">{item['conf']}% · {item['time']}</div>
            </div>
            """, unsafe_allow_html=True)

    if st.session_state.history:
        if st.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.rerun()

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#bbb; font-size:0.8rem'>"
    "EmoBERT · BERT fine-tuned on dair-ai/emotion · 93.5% accuracy · Built by Saanvi Chauhan"
    "</div>",
    unsafe_allow_html=True
)
