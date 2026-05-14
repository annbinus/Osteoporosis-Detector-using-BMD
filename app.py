import streamlit as st
import boto3
import json
import time
import uuid
from datetime import datetime
from PIL import Image
import io

# ── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OsteoScan AI",
    page_icon="🦴",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&display=swap');

:root {
    --bg:        #0a0c0f;
    --surface:   #12161c;
    --border:    #1e2530;
    --accent:    #00e5a0;
    --warn:      #f5a623;
    --danger:    #ff4757;
    --text:      #e8edf5;
    --muted:     #5a6478;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
}

.stApp { background-color: var(--bg) !important; }

h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
    color: var(--text) !important;
    letter-spacing: -0.02em;
}

/* Hero */
.hero {
    text-align: center;
    padding: 3rem 0 2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2.5rem;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem;
    color: var(--text);
    letter-spacing: -0.03em;
    margin: 0;
    line-height: 1.1;
}
.hero-accent { color: var(--accent); }
.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.75rem;
}

/* Upload zone */
.upload-label {
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5rem;
}

/* Result cards */
.result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1.5rem;
    margin: 1rem 0;
}
.result-label {
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.4rem;
}
.result-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    line-height: 1;
}
.normal    { color: var(--accent); }
.osteopenia { color: var(--warn); }
.osteoporosis { color: var(--danger); }

/* Prob bar */
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 0.6rem 0;
    font-size: 0.75rem;
}
.prob-label { width: 110px; color: var(--muted); }
.prob-track {
    flex: 1;
    height: 3px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
}
.prob-fill-normal       { height: 100%; background: var(--accent); border-radius: 2px; }
.prob-fill-osteopenia   { height: 100%; background: var(--warn);   border-radius: 2px; }
.prob-fill-osteoporosis { height: 100%; background: var(--danger); border-radius: 2px; }
.prob-pct { width: 40px; text-align: right; color: var(--text); }

/* Parsed values table */
.parsed-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
    margin-top: 0.75rem;
}
.parsed-item {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 0.5rem 0.75rem;
}
.parsed-key {
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
}
.parsed-val {
    font-size: 0.9rem;
    color: var(--accent);
    margin-top: 0.1rem;
}

/* Status badge */
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 2px;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 500;
}
.badge-success { background: rgba(0,229,160,0.1); color: var(--accent); border: 1px solid rgba(0,229,160,0.3); }
.badge-error   { background: rgba(255,71,87,0.1);  color: var(--danger); border: 1px solid rgba(255,71,87,0.3); }

/* Divider */
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}

/* File uploader override */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 4px !important;
}

/* Button */
.stButton > button {
    background: var(--accent) !important;
    color: #0a0c0f !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 500 !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #00c988 !important;
}

/* Spinner */
.stSpinner > div { border-top-color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

# ── AWS Config ────────────────────────────────────────────────────────────
UPLOAD_BUCKET  = 'osteo-dxa-uploads'
RESULTS_BUCKET = 'osteo-s3-demo-bucket'
RESULTS_PREFIX = 'results'
REGION         = 'us-east-1'

s3 = boto3.client('s3', region_name=REGION)

# ── Hero ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-title">Osteo<span class="hero-accent">Scan</span> AI</p>
    <p class="hero-sub">DXA Report · Bone Density Classification · AWS Pipeline</p>
</div>
""", unsafe_allow_html=True)

# ── Upload Section ────────────────────────────────────────────────────────
st.markdown('<p class="upload-label">Upload DXA Report</p>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    label="",
    type=['png', 'jpg', 'jpeg', 'pdf'],
    label_visibility="collapsed"
)

if uploaded:
    # Show preview for images
    if uploaded.type.startswith('image'):
        img = Image.open(uploaded)
        st.image(img, caption='Uploaded report', use_column_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    if st.button("Analyse Report"):
        with st.spinner("Processing..."):

            # ── Step 1: Upload to S3 ──────────────────────────────────────
            ext       = uploaded.name.split('.')[-1]
            unique_id = str(uuid.uuid4())[:8]
            s3_key    = f"upload_{unique_id}.{ext}"

            uploaded.seek(0)
            s3.upload_fileobj(uploaded, UPLOAD_BUCKET, s3_key)
            st.toast("Uploaded to S3", icon="☁️")

            # ── Step 2: Wait for Lambda to process ────────────────────────
            base_name   = s3_key.rsplit('.', 1)[0]
            result_data = None
            max_wait    = 30
            waited      = 0

            while waited < max_wait:
                time.sleep(2)
                waited += 2

                # Check results bucket for matching file
                response = s3.list_objects_v2(
                    Bucket=RESULTS_BUCKET,
                    Prefix=f'{RESULTS_PREFIX}/{base_name}'
                )

                if response.get('Contents'):
                    result_key = response['Contents'][0]['Key']
                    obj        = s3.get_object(Bucket=RESULTS_BUCKET, Key=result_key)
                    result_data = json.loads(obj['Body'].read())
                    break

            if not result_data:
                st.error("Timeout — Lambda did not respond in 30s. Check CloudWatch logs.")
                st.stop()

        # ── Step 3: Display Results ───────────────────────────────────────
        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        if result_data.get('status') == 'error':
            st.markdown(f"""
            <span class="badge badge-error">Error</span>
            <div class="result-card" style="margin-top:0.75rem">
                <p class="result-label">Details</p>
                <p style="color:#ff4757;font-size:0.85rem">{result_data.get('error')}</p>
            </div>
            """, unsafe_allow_html=True)

        else:
            pred  = result_data['prediction']
            conf  = result_data['confidence']
            probs = result_data['probabilities']

            # Class color
            cls_map = {
                'Normal':       ('normal',       '✓'),
                'Osteopenia':   ('osteopenia',   '⚠'),
                'Osteoporosis': ('osteoporosis', '✕'),
            }
            cls_class, cls_icon = cls_map.get(pred, ('normal', '?'))

            # Main result
            st.markdown(f"""
            <span class="badge badge-success">Analysis complete</span>
            <div class="result-card" style="margin-top:0.75rem">
                <p class="result-label">Diagnosis</p>
                <p class="result-value {cls_class}">{cls_icon} {pred}</p>
                <p style="font-size:0.75rem;color:#5a6478;margin-top:0.5rem">
                    Confidence: {conf:.1%}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Probability bars
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<p class="result-label">Probability Distribution</p>', unsafe_allow_html=True)

            bar_classes = {
                'Normal':       'prob-fill-normal',
                'Osteopenia':   'prob-fill-osteopenia',
                'Osteoporosis': 'prob-fill-osteoporosis',
            }

            bars_html = ""
            for cls, prob in probs.items():
                fill_class = bar_classes[cls]
                bars_html += f"""
                <div class="prob-row">
                    <span class="prob-label">{cls}</span>
                    <div class="prob-track">
                        <div class="{fill_class}" style="width:{prob*100:.1f}%"></div>
                    </div>
                    <span class="prob-pct">{prob:.1%}</span>
                </div>"""

            st.markdown(bars_html + '</div>', unsafe_allow_html=True)

            # Parsed BMD values
            parsed = result_data.get('parsed_values', {})
            if parsed:
                st.markdown('<hr class="divider">', unsafe_allow_html=True)
                st.markdown('<p class="upload-label">Extracted BMD Values</p>', unsafe_allow_html=True)

                grid_html = '<div class="parsed-grid">'
                for col, val in parsed.items():
                    display_val = f"{val:.3f}" if isinstance(val, float) else str(val)
                    grid_html += f"""
                    <div class="parsed-item">
                        <p class="parsed-key">{col}</p>
                        <p class="parsed-val">{display_val}</p>
                    </div>"""
                grid_html += '</div>'
                st.markdown(grid_html, unsafe_allow_html=True)

            # Timestamp + source
            st.markdown(f"""
            <hr class="divider">
            <p style="font-size:0.65rem;color:#5a6478;letter-spacing:0.1em">
                PROCESSED · {result_data.get('timestamp', '')[:19].replace('T', ' ')} UTC
                &nbsp;·&nbsp;
                {result_data.get('source_file', '').split('/')[-1]}
            </p>
            """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("""
<hr class="divider">
<p style="font-size:0.62rem;color:#5a6478;letter-spacing:0.1em;text-align:center">
    NHANES 2015–2020 · XGBoost · 0.95 ROC AUC · AWS SageMaker
</p>
""", unsafe_allow_html=True)