import streamlit as st
import joblib
import nltk
import PyPDF2
import docx
from detector.ambiguity_detector import detect_ambiguity, highlight_sentence

st.set_page_config(
    page_title="SRS Ambiguity Detector",
    page_icon="🔍",
    layout="wide"
)

nltk.download('punkt')

model = joblib.load("ambiguity_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ---------------- ML FUNCTION ----------------
def ml_predict(sentence):
    vec = vectorizer.transform([sentence])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    confidence = max(prob)
    return (
        "Ambiguous" if pred == 1 else "Clear",
        round(confidence * 100, 2)
    )

# ---------------- REPORT FUNCTION ----------------
def generate_report(results):
    report = "SRS Ambiguity Detection Report\n"
    report += "=" * 50 + "\n\n"
    for item in results:
        report += f"Sentence: {item['sentence']}\n"
        report += f"Final Decision: {item['final']}\n"
        report += f"ML Prediction: {item['ml']} ({item['confidence']}%)\n"
        if item["issues"]:
            report += "Ambiguities:\n"
            for term, reason, suggestion in item["issues"]:
                report += f" - {term}: {reason}\n"
                report += f"   Suggestion: {suggestion}\n"
        report += "\n" + "-" * 50 + "\n\n"
    return report

# ---------------- FILTER FUNCTION ----------------
def is_valid_requirement(sentence):
    sentence = sentence.strip().lower()

    # Too short
    if len(sentence) < 20:
        return False

    # Skip headings / numbering
    if sentence.startswith(("1.", "2.", "3.", "4.", "5.")):
        return False

    # Skip common non-requirement keywords
    skip_words = [
        "introduction",
        "purpose",
        "scope",
        "definitions",
        "overview",
        "software requirements specification",
        "version",
    ]

    if any(word in sentence for word in skip_words):
        return False

    # Skip lines without modal verbs (strong heuristic 🔥)
    if not any(word in sentence for word in ["shall", "must", "should", "may"]):
        return False

    return True

# ---------------- STYLING ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #080c14 !important;
    color: #c9d4e8;
    font-family: 'Syne', sans-serif;
}

[data-testid="block-container"] {
    padding: 2rem 3rem 4rem !important;
}

/* ── HEADER ── */
.hero-header {
    display: flex;
    align-items: flex-start;
    gap: 1.5rem;
    padding: 2.2rem 0 1.8rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 2.5rem;
}
.hero-badge {
    background: linear-gradient(135deg, #00c8ff22, #6328ff22);
    border: 1px solid rgba(0,200,255,0.25);
    border-radius: 14px;
    padding: 14px 16px;
    font-size: 2rem;
    line-height: 1;
    flex-shrink: 0;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    color: #f0f6ff;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin: 0 0 0.4rem;
}
.hero-title .grad {
    background: linear-gradient(90deg, #00c8ff, #6328ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 0.88rem;
    color: #4b5a72;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.04em;
    margin: 0;
}

/* ── SECTION LABELS ── */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #00c8ff;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-label::after {
    content: "";
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(0,200,255,0.25), transparent);
}

/* ── RADIO ── */
div[data-testid="stRadio"] > label { display: none; }
div[data-testid="stRadio"] > div {
    display: flex; gap: 10px; flex-direction: row !important;
}
div[data-testid="stRadio"] > div > label {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 9px 18px;
    cursor: pointer;
    color: #6b7a90;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    transition: all 0.15s;
}
div[data-testid="stRadio"] > div > label:has(input:checked) {
    background: rgba(0,200,255,0.08);
    border-color: rgba(0,200,255,0.4);
    color: #00c8ff;
}

/* ── TEXTAREA ── */
textarea {
    background: rgba(0,8,20,0.6) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 10px !important;
    color: #c9d4e8 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
}
textarea:focus {
    border-color: rgba(0,200,255,0.35) !important;
    box-shadow: none !important;
}

/* ── BUTTON ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(90deg, #00c8ff, #6328ff);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 11px 28px;
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    width: 100%;
    transition: opacity 0.15s;
}
div[data-testid="stButton"] > button:hover { opacity: 0.88; }

/* ── METRICS ── */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #4b5a72 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif;
    font-size: 2rem !important;
    font-weight: 800;
    color: #f0f6ff !important;
}

/* ── RESULT CARDS ── */
.result-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-left: 3px solid #ef4444;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.result-card.review { border-left-color: #f59e0b; }

.card-sentence {
    font-family: 'Space Mono', monospace;
    font-size: 0.84rem;
    color: #c9d4e8;
    line-height: 1.65;
    margin-bottom: 0.8rem;
}
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 5px;
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}
.badge-ambiguous { background: rgba(239,68,68,0.12); color: #f87171; border: 1px solid rgba(239,68,68,0.25); }
.badge-review    { background: rgba(245,158,11,0.12); color: #fbbf24; border: 1px solid rgba(245,158,11,0.25); }

.issue-row {
    padding: 0.6rem 0.8rem;
    background: rgba(0,0,0,0.2);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 7px;
    margin-top: 0.4rem;
    font-size: 0.82rem;
}
.issue-term       { color: #f87171; font-weight: 700; font-family: 'Space Mono', monospace; }
.issue-reason     { color: #5a7090; margin-top: 2px; }
.issue-suggestion { color: #60a5fa; margin-top: 2px; }

hr {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.07) !important;
    margin: 1.6rem 0 !important;
}

[data-testid="stDownloadButton"] > button {
    background: rgba(0,200,255,0.07) !important;
    border: 1px solid rgba(0,200,255,0.2) !important;
    color: #00c8ff !important;
    border-radius: 9px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    font-weight: 700;
}
[data-testid="stDownloadButton"] > button:hover {
    border-color: rgba(0,200,255,0.45) !important;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ──
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">🔍</div>
    <div>
        <div class="hero-title">SRS <span class="grad">Ambiguity</span> Detector</div>
        <p class="hero-sub">NLP + ML &nbsp;&middot;&nbsp; Requirement Analysis &nbsp;&middot;&nbsp; Precision Flagging</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── INPUT + HOW IT WORKS ──
left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    st.markdown('<div class="section-label">Input Source</div>', unsafe_allow_html=True)

    option = st.radio("Select Input Type:", ["✍️ Enter Text", "📂 Upload File"], horizontal=True)
    text = ""

    if option == "✍️ Enter Text":
        text = st.text_area(
            "Paste your SRS requirement text below:",
            height=180,
            placeholder="e.g. The system should handle large amounts of data efficiently and provide fast responses..."
        )
    elif option == "📂 Upload File":
        uploaded_file = st.file_uploader("Upload .txt, .pdf, or .docx", type=["txt", "pdf", "docx"])
        if uploaded_file is not None:
            file_type = uploaded_file.name.split(".")[-1]
            if file_type == "txt":
                text = uploaded_file.read().decode("utf-8")
            elif file_type == "pdf":
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    content = page.extract_text()
                    if content:
                        text += content + " "
            elif file_type == "docx":
                doc = docx.Document(uploaded_file)
                text = ""
                for para in doc.paragraphs:
                    text += para.text + " "
            text = text.replace("\n", " ")
            st.text_area("Extracted Content:", text, height=150)

    analyze = st.button("⚡ Run Analysis")

with right_col:
    st.markdown('<div class="section-label">How It Works</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);'
        'border-radius:14px;padding:1.4rem;display:flex;flex-direction:column;gap:1.1rem;">'

        '<div style="display:flex;gap:12px;align-items:flex-start;">'
        '<div style="background:rgba(0,200,255,0.08);border:1px solid rgba(0,200,255,0.2);'
        'border-radius:7px;padding:4px 10px;font-size:0.72rem;font-weight:700;color:#00c8ff;'
        'flex-shrink:0;font-family:monospace;">01</div>'
        '<div><div style="font-weight:700;color:#f0f6ff;font-size:0.88rem;">Rule-Based NLP</div>'
        '<div style="color:#4b5a72;font-size:0.79rem;font-family:\'Space Mono\',monospace;margin-top:3px;">'
        'Flags ambiguous terms, vague quantifiers, and weak verbs</div></div></div>'

        '<div style="display:flex;gap:12px;align-items:flex-start;">'
        '<div style="background:rgba(0,200,255,0.08);border:1px solid rgba(0,200,255,0.2);'
        'border-radius:7px;padding:4px 10px;font-size:0.72rem;font-weight:700;color:#00c8ff;'
        'flex-shrink:0;font-family:monospace;">02</div>'
        '<div><div style="font-weight:700;color:#f0f6ff;font-size:0.88rem;">ML Classifier</div>'
        '<div style="color:#4b5a72;font-size:0.79rem;font-family:\'Space Mono\',monospace;margin-top:3px;">'
        'TF-IDF vectorizer + trained model with confidence score</div></div></div>'

        '<div style="display:flex;gap:12px;align-items:flex-start;">'
        '<div style="background:rgba(0,200,255,0.08);border:1px solid rgba(0,200,255,0.2);'
        'border-radius:7px;padding:4px 10px;font-size:0.72rem;font-weight:700;color:#00c8ff;'
        'flex-shrink:0;font-family:monospace;">03</div>'
        '<div><div style="font-weight:700;color:#f0f6ff;font-size:0.88rem;">Hybrid Decision</div>'
        '<div style="color:#4b5a72;font-size:0.79rem;font-family:\'Space Mono\',monospace;margin-top:3px;">'
        'Combined verdict: Ambiguous / Clear / Needs Review</div></div></div>'

        '</div>',
        unsafe_allow_html=True
    )

# ── ANALYSIS ──
if analyze:
    if text.strip() == "":
        st.warning("⚠️ Please provide some requirement text before running analysis.")
    else:
        sentences = nltk.sent_tokenize(text)

        total_sentences = 0
        ambiguous_count = 0
        results = []

        for sent in sentences:
            if not is_valid_requirement(sent):
                continue
            total_sentences += 1
            if detect_ambiguity(sent):
                ambiguous_count += 1

        st.markdown("---")
        st.markdown('<div class="section-label">Document Summary</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Sentences", total_sentences)
        c2.metric("Ambiguous", ambiguous_count)
        c3.metric("Clear", total_sentences - ambiguous_count)
        c4.metric("Clarity Score", f"{round((1 - ambiguous_count / max(total_sentences, 1)) * 100)}%")

        st.markdown("---")
        st.markdown('<div class="section-label">Detailed Analysis</div>', unsafe_allow_html=True)

        for sent in sentences:
            if not is_valid_requirement(sent):
                continue

            ambiguous_words = detect_ambiguity(sent)
            ml_result, confidence = ml_predict(sent)

            if ambiguous_words:
                final = "Ambiguous"
            elif ml_result == "Ambiguous" and confidence >= 70:
                final = "Ambiguous"
            elif ml_result == "Clear" and confidence >= 70:
                final = "Clear"
            else:
                final = "Needs Review"

            results.append({
                "sentence": sent,
                "final": final,
                "ml": ml_result,
                "confidence": confidence,
                "issues": ambiguous_words
            })

            if final in ("Ambiguous", "Needs Review"):
                card_class = "result-card" if final == "Ambiguous" else "result-card review"
                badge_class = "badge-ambiguous" if final == "Ambiguous" else "badge-review"
                badge_icon = "🔴" if final == "Ambiguous" else "🟡"
                highlighted = highlight_sentence(sent, ambiguous_words)

                st.markdown(f"""
                <div class="{card_class}">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;">
                        <span class="badge {badge_class}">{badge_icon} {final}</span>
                        <span style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#4b5a72;">
                            ML: {ml_result} &middot; {confidence}% confidence
                        </span>
                    </div>
                    <div class="card-sentence">{highlighted}</div>
                """, unsafe_allow_html=True)

                if ambiguous_words:
                    st.markdown(
                        '<div style="font-family:\'Space Mono\',monospace;font-size:0.67rem;'
                        'letter-spacing:0.12em;text-transform:uppercase;color:#f87171;margin-bottom:0.4rem;">'
                        'Detected Issues</div>',
                        unsafe_allow_html=True
                    )
                    for term, reason, suggestion in ambiguous_words:
                        st.markdown(f"""
                        <div class="issue-row">
                            <span class="issue-term">{term}</span>
                            <div class="issue-reason">{reason}</div>
                            <div class="issue-suggestion">💡 {suggestion}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

        clear_sentences = [r for r in results if r["final"] == "Clear"]
        if clear_sentences:
            st.markdown(
                f'<div style="background:rgba(0,200,255,0.04);border:1px solid rgba(0,200,255,0.12);'
                f'border-radius:8px;padding:0.75rem 1rem;margin-top:0.4rem;'
                f'font-family:\'Space Mono\',monospace;font-size:0.8rem;color:#00c8ff;">'
                f'✓ {len(clear_sentences)} sentence{"s" if len(clear_sentences) != 1 else ""} passed without issues'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        report_text = generate_report(results)
        st.download_button(
            label="⬇ Download Full Report (.txt)",
            data=report_text,
            file_name="ambiguity_report.txt",
            mime="text/plain"
        )