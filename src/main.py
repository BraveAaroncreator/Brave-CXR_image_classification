from __future__ import annotations

import base64
import io
import json
import re
import sqlite3
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps

st.set_page_config(page_title='BRAVE AI', page_icon='B', layout='wide', initial_sidebar_state='collapsed')

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / 'hospital.db'
DEFAULT_IMAGE_PATH = BASE_DIR / 'temp_xray.png'
INFERENCE_WORKER_PATH = BASE_DIR / 'model_inference_worker.py'
BRAVE_ENV_PYTHON = Path(r'C:\Users\hp\Desktop\Brave_env\Scripts\python.exe')
CLASS_NAMES = ['Normal', 'Tuberculosis', 'Pneumonia', 'COVID-19']
NAV_ITEMS = ['Registry', 'AI Analysis', 'Dashboard', 'Worklist', 'Study Viewer']
RAW_MODEL_CLASSES = ['Covid', 'Normal', 'Pneumonia', 'Tuberculosis']
RAW_TO_DISPLAY = {
    'Covid': 'COVID-19',
    'Normal': 'Normal',
    'Pneumonia': 'Pneumonia',
    'Tuberculosis': 'Tuberculosis',
}
MODEL_PRIORITY = [
    'best_densenet121_model.keras',
    'best_resnet50_model.keras',
    'best_efficientnet_model.keras',
    'Brave.keras',
    'CNN_best_model.keras',
]
TARGET_LAYER_HINTS = {
    'densenet': ['conv5_block16_concat', 'relu'],
    'resnet50': ['conv5_block3_out'],
    'efficientnet': ['top_activation', 'top_conv'],
    'mobilenetv2': ['Conv_1', 'out_relu'],
    'vgg16': ['block5_conv3'],
}
BRANDING = {
    'platform_name': 'AI Radiology Assistant - Chest X-Ray Analysis System',
    'hospital_name': 'AI Radiology Assistant',
    'department': '',
    'tagline': 'See sooner. Decide with confidence.',
    'location': '',
    'contact': 'BraveHospital@Gmail.com | +251921329114',
}
DISCLAIMER_TEXT = 'Disclaimer: AI-generated result for decision support only. Final diagnosis must be confirmed by a qualified healthcare professional.'
ETHIOPIA_TZ = ZoneInfo('Africa/Addis_Ababa')
MRN_SEQUENCE_NAME = 'patient_mrn'
MRN_PREFIX = 'MRN-'
INTERNAL_ANALYSIS_TABLE = 'analysis_detail'
DISPLAY_RECORD_COLUMNS = [
    'Number',
    'Patient - MRN',
    'Patient Name',
    'Age',
    'Gender',
    'Predicted Class',
    'Confidence Score',
    'Prediction Date and Time',
    'Status',
    'priority',
    'modality',
    'Image Name',
    'Image Path',
    'Findings',
]


def utc_now_sql() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')


def now_gmt3() -> datetime:
    return datetime.now(ETHIOPIA_TZ)


def to_gmt3_timestamp(value: object) -> pd.Timestamp | None:
    if value is None or value == '':
        return None
    try:
        if isinstance(value, str):
            ts = pd.to_datetime(value, errors='coerce', utc=True)
            if ts is None or pd.isna(ts):
                return None
            return ts.tz_convert(ETHIOPIA_TZ)
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return None
        if ts.tzinfo is None:
            return ts.tz_localize(ETHIOPIA_TZ)
        return ts.tz_convert(ETHIOPIA_TZ)
    except Exception:
        return None


def format_gmt3(value: object, fmt: str = '%d %b %Y %H:%M Ethiopia Time') -> str:
    ts = to_gmt3_timestamp(value)
    if ts is None:
        return str(value or '')
    return ts.strftime(fmt)


def rerun() -> None:
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


def brave_logo_svg() -> str:
    return """
    <svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="BRAVE AI logo">
      <defs>
        <linearGradient id="braveGlow" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#1fd0bb" />
          <stop offset="100%" stop-color="#0c6e61" />
        </linearGradient>
        <linearGradient id="braveWarm" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#f0b35d" />
          <stop offset="100%" stop-color="#c07a1a" />
        </linearGradient>
      </defs>
      <rect x="8" y="8" width="104" height="104" rx="28" fill="rgba(255,255,255,0.06)" stroke="rgba(255,255,255,0.18)" />
      <circle cx="60" cy="60" r="32" fill="none" stroke="url(#braveGlow)" stroke-width="10" />
      <path d="M44 34h18c13 0 22 6 22 17 0 8-5 13-12 15 10 2 16 8 16 18 0 13-10 21-26 21H44z" fill="white" opacity="0.96"/>
      <path d="M54 44v17h10c7 0 11-3 11-8 0-6-4-9-11-9zm0 26v18h12c8 0 13-3 13-9 0-6-5-9-13-9z" fill="url(#braveGlow)"/>
      <path d="M82 31l8 8-18 18-8-8z" fill="url(#braveWarm)" opacity="0.95"/>
      <circle cx="89" cy="32" r="6" fill="#ffd7a1" />
    </svg>
    """


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Outfit:wght@500;600;700&display=swap');
        :root {
            --bg: #eaf0ee;
            --shell: linear-gradient(135deg, #f7fbfa 0%, #e8f2f0 60%, #f5faf9 100%);
            --surface: rgba(255,255,255,0.97);
            --ink: #132328;
            --muted: #2f474d;
            --accent: #0f8f7f;
            --accent-soft: rgba(15,143,127,0.12);
            --warn: #bf8a39;
            --danger: #a64742;
            --success: #0b775e;
            --border: rgba(16,34,40,0.10);
            --shadow: 0 22px 50px rgba(10, 22, 26, 0.10);
        }
        html, body, [class*="css"] {font-family: 'IBM Plex Sans', sans-serif; color: #111111;} 
        h1, h2, h3 {font-family: 'Outfit', sans-serif; letter-spacing: -0.02em;}
        .stApp {background: radial-gradient(circle at top left, rgba(15,143,127,0.12), transparent 28%), linear-gradient(180deg, #edf3f1 0%, #e4ece9 100%);} 
        [data-testid="stHeader"], [data-testid="collapsedControl"], [data-testid="stSidebar"] {display:none;} 
        .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1450px;}
        .shell {background: var(--shell); color: #111111; border-radius: 30px; padding: 1.3rem 1.4rem; box-shadow: 0 22px 45px rgba(8, 14, 16, 0.10); margin-bottom: 1rem; position: relative; overflow: hidden; border:1px solid rgba(16,34,40,0.10);}
        .shell::before {content:''; position:absolute; inset:0; background: radial-gradient(circle at 20% 20%, rgba(15,143,127,0.12), transparent 28%), radial-gradient(circle at 90% 18%, rgba(191,138,57,0.10), transparent 24%);} 
        .shell > div {position: relative; z-index: 1;} 
        .eyebrow {font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.18em; color: #111111;} 
        .brand {font-size: 2.05rem; font-weight: 700; margin: 0.18rem 0 0.18rem; color:#111111; line-height:1.1;} 
        .sub {color: #111111; max-width: 760px; margin: 0; font-size:0.96rem; font-weight:600; letter-spacing:0.01em;} 
        .shell-pills {display:flex; gap:0.6rem; flex-wrap:wrap; justify-content:flex-end;} 
        .shell-pill {border-radius:999px; border:1px solid rgba(16,34,40,0.12); background:rgba(255,255,255,0.70); padding:0.5rem 0.8rem; font-size:0.82rem; color:#111111;} 
        .brand-lockup {display:flex; gap:1rem; align-items:center; flex-wrap:wrap;}
        .brand-mark {width:84px; height:84px; border-radius:24px; background:rgba(255,255,255,0.85); box-shadow: inset 0 1px 0 rgba(255,255,255,0.12); padding:0.55rem; flex-shrink:0;}
        .brand-stack {display:flex; flex-direction:column; gap:0.35rem;}
        .brand-meta {display:flex; gap:0.5rem; flex-wrap:wrap; margin-top:0.15rem;}
        .brand-meta span {font-size:0.78rem; color:#111111; border:1px solid rgba(16,34,40,0.10); padding:0.28rem 0.55rem; border-radius:999px; background:rgba(255,255,255,0.55);}
        .section {background: var(--surface); border-radius: 22px; border: 1px solid rgba(255,255,255,0.72); box-shadow: var(--shadow); padding: 1rem 1.1rem; margin-bottom: 1rem;} 
        .kicker {color: #111111; font-size: 0.76rem; letter-spacing: 0.16em; text-transform: uppercase; margin-bottom: 0.25rem; font-weight:700;} 
        .section-title {font-size: 1.34rem; margin: 0; color:#10262c; font-weight:700;} 
        .section-copy {color: #111111; margin-top: 0.35rem; margin-bottom: 0; font-weight:500;} 
        .metric-card {background: linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(247,250,249,0.96) 100%); border-radius: 20px; border: 1px solid rgba(255,255,255,0.78); box-shadow: var(--shadow); padding: 1.05rem 1.1rem; min-height: 145px; margin-bottom: 1rem;} 
        .metric-label {color: #111111; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 0.65rem; font-weight:700;} 
        .metric-value {font-family:'Outfit', sans-serif; font-size: 2.2rem; line-height: 1; margin-bottom: 0.55rem;} 
        .metric-copy {color: #111111; font-size: 0.92rem; font-weight:500;} 
        .metric-badge {display:inline-block; margin-top:0.8rem; padding:0.34rem 0.65rem; background: var(--accent-soft); color:#111111; border-radius:999px; font-size:0.8rem; font-weight:600;} 
        .study-card, .queue-card {background: rgba(255,255,255,0.99); border-radius: 18px; border:1px solid var(--border); box-shadow: 0 18px 38px rgba(12,24,28,0.08); padding: 0.9rem 1rem; margin-bottom: 0.7rem;} 
        .study-title, .queue-title {font-weight:700; color: #10262c; font-size:1.02rem;} 
        .study-meta, .queue-sub, .soft {color: #111111; font-size: 0.9rem; font-weight:600;} 
        .study-body {margin-top:0.42rem; color:#18353a; font-weight:600; line-height:1.55;}
        .focus-mini {color:#111111; text-transform:uppercase; letter-spacing:0.14em; font-size:0.72rem; font-weight:700;}
        .focus-meta {color:#111111; font-weight:600; line-height:1.5;}
        .pill {display:inline-block; padding:0.28rem 0.62rem; border-radius:999px; font-size:0.78rem; font-weight:600; margin-right:0.35rem; margin-top:0.4rem;} 
        .neutral {background: rgba(64,84,89,0.10); color:#495d62;} 
        .success {background: rgba(11,119,94,0.14); color:#095f4b;} 
        .warning {background: rgba(191,138,57,0.16); color:#885f21;} 
        .danger {background: rgba(166,71,66,0.14); color:#8c3733;} 
        .focus {background: linear-gradient(180deg, rgba(255,255,255,0.99) 0%, rgba(244,249,248,0.97) 100%); color:#111111; border-radius:24px; border:1px solid rgba(16,34,40,0.10); box-shadow: var(--shadow); padding:1rem;} 
        .toolbar {display:flex; gap:0.45rem; flex-wrap:wrap; margin:0.8rem 0 0.2rem;} 
        .tool {border-radius:999px; border:1px solid rgba(16,34,40,0.10); background:rgba(255,255,255,0.92); padding:0.34rem 0.66rem; font-size:0.8rem; color:#111111; font-weight:600;} 
        .mini {text-transform:uppercase; letter-spacing:0.14em; font-size:0.72rem; color: #111111; font-weight:700;} 
        .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {border-radius:14px; border:1px solid rgba(12,31,36,0.10); background: linear-gradient(180deg, #bfe9e2 0%, #8fd4c7 100%); color:#111111 !important; font-weight:700; min-height:2.85rem; box-shadow:0 10px 22px rgba(15,143,127,0.12);} 
        .stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {color:#111111 !important; border-color:rgba(12,31,36,0.10); background: linear-gradient(180deg, #ccefe9 0%, #a7ddd2 100%);} 
        div[data-testid="stForm"] {background: rgba(255,255,255,0.94); border-radius: 20px; border:1px solid var(--border); box-shadow: var(--shadow); padding: 1rem 1rem 0.8rem;} 
        div[data-testid="stMetric"], div[data-testid="stDataFrame"] {border-radius: 18px; overflow:hidden; border:1px solid var(--border); box-shadow: var(--shadow);} 
        .section, .metric-card, .study-card, .queue-card, .focus, .shell-pill, div[data-testid="stMetric"] {transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;}
        .section:hover, .metric-card:hover, .study-card:hover, .queue-card:hover, .focus:hover, .shell-pill:hover, div[data-testid="stMetric"]:hover {transform: translateY(-3px); box-shadow: 0 28px 56px rgba(11, 23, 27, 0.14); border-color: rgba(15,143,127,0.24);}
        .analysis-panel {background: rgba(255,255,255,0.96); border-radius: 18px; border:1px solid var(--border); box-shadow: var(--shadow); padding: 1rem 1.05rem; margin: 0.95rem 0 1rem;}
        .analysis-grid {display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:0.7rem; margin-top:0.8rem;}
        .analysis-stat {background: linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(245,249,248,0.96) 100%); border:1px solid rgba(16,34,40,0.10); border-radius:16px; padding:0.78rem 0.86rem;}
        .analysis-key {font-size:0.72rem; text-transform:uppercase; letter-spacing:0.12em; color:#111111; font-weight:700;}
        .analysis-value {margin-top:0.28rem; color:#132328; font-weight:700; line-height:1.35;}
        .upload-panel {background: linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(247,250,249,0.96) 100%); border:1px solid rgba(16,34,40,0.10); border-radius:22px; box-shadow: var(--shadow); padding:1.05rem 1.1rem; margin-bottom:1rem;}
        .upload-title {font-size:1.24rem; font-weight:700; color:#10262c; margin:0 0 0.2rem;}
        .upload-copy {font-size:0.94rem; font-weight:600; color:#111111; margin:0;}
        .busy-button {display:flex; align-items:center; justify-content:center; gap:0.62rem; width:100%; min-height:2.85rem; border-radius:14px; border:1px solid rgba(12,31,36,0.10); background: linear-gradient(180deg, #bfe9e2 0%, #8fd4c7 100%); color:#111111; font-weight:700; box-shadow:0 10px 22px rgba(15,143,127,0.12);}
        .busy-spinner {width:1rem; height:1rem; border:2px solid rgba(17,17,17,0.30); border-top-color:#111111; border-radius:50%; animation:spin 0.85s linear infinite;}
        .gradcam-loader {display:flex; align-items:center; justify-content:center; gap:0.65rem; width:100%; min-height:3rem; border-radius:16px; border:1px solid rgba(16,34,40,0.10); background:rgba(255,255,255,0.96); box-shadow: var(--shadow); color:#111111; font-weight:700; margin:0.55rem 0 0.85rem;}
        @keyframes spin {to {transform: rotate(360deg);}}
        .result-panel {background: linear-gradient(180deg, rgba(255,255,255,0.99) 0%, rgba(245,249,248,0.97) 100%); border:1px solid rgba(16,34,40,0.10); border-radius:20px; box-shadow: var(--shadow); padding:1rem 1.05rem; margin:1rem 0;}
        .result-head {display:flex; justify-content:space-between; gap:1rem; align-items:flex-start; flex-wrap:wrap;}
        .result-title {font-size:1.4rem; font-weight:700; color:#10262c; margin:0;}
        .result-sub {font-size:0.93rem; color:#284046; font-weight:600; margin-top:0.22rem;}
        label[data-testid="stWidgetLabel"], label[data-testid="stWidgetLabel"] p, .stTextInput label, .stTextArea label, .stSelectbox label, .stFileUploader label, .stNumberInput label {color:#17343a !important; font-weight:700 !important;}
        .stCaption, .stAlert p, .stInfo p, .stWarning p, .stSuccess p {color:#111111 !important;}
        .stMarkdown p {color:#111111;}
        .stTextInput input, .stTextArea textarea, .stNumberInput input, div[data-baseweb="select"] > div {background: rgba(255,255,255,0.98) !important; color: #132328 !important; border-color: rgba(16,34,40,0.10) !important;}
        .stTextInput input::placeholder, .stTextArea textarea::placeholder {color:#6f8389 !important;}
        div[data-baseweb="select"] *, div[data-baseweb="base-input"] * {color:#132328 !important;}
        .stFileUploader {width:100% !important;}
        .stFileUploader > div {background: linear-gradient(180deg, rgba(250,252,251,0.98) 0%, rgba(239,246,244,0.96) 100%) !important; border:2px dashed rgba(15,143,127,0.30) !important; border-radius:24px !important; width:100% !important; box-shadow: var(--shadow);}
        [data-testid="stFileUploaderDropzone"] {padding:1.25rem 1rem !important; background:transparent !important;}
        .stFileUploader section, .stFileUploader small, .stFileUploader span, .stFileUploader label, [data-testid="stFileUploaderDropzoneInstructions"] {color:#18353a !important; font-weight:600 !important;}
        .stFileUploader small {text-transform: uppercase;}
        .stSpinner > div, .stSpinner label, .stSpinner span {color:#111111 !important; font-weight:700 !important;}
        .stDataFrame [role="gridcell"], .stDataFrame [role="columnheader"], .stDataFrame div {color:#132328 !important;}
        [data-baseweb="tab-list"] {gap:0.45rem; background:rgba(255,255,255,0.55); padding:0.34rem; border-radius:16px; border:1px solid var(--border); margin-bottom:0.85rem;}
        button[data-baseweb="tab"] {background:rgba(255,255,255,0.84); border-radius:12px; color:#17343a; font-weight:700; border:1px solid transparent;}
        button[data-baseweb="tab"][aria-selected="true"] {background:linear-gradient(180deg, #bfe9e2 0%, #8fd4c7 100%); color:#111111; border-color:rgba(12,31,36,0.08);}
        button[data-baseweb="tab"]:hover {background:rgba(15,143,127,0.10); color:#111111;}
        .stApp p, .stApp span, .stApp div, .stApp label, .stApp small, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 {color:#111111;}
        .pill, .neutral, .success, .warning, .danger {color:#111111 !important;}
        @media (max-width: 980px) {
            .analysis-grid {grid-template-columns:1fr;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    existing = {row['name'] for row in conn.execute(f'PRAGMA table_info({table})')}
    if column not in existing:
        conn.execute(f'ALTER TABLE {table} ADD COLUMN {column} {definition}')


def detect_last_registered_mrn(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        '''
        SELECT COUNT(*) AS last_value
        FROM patients
        '''
    ).fetchone()
    return int(row['last_value'] or 0) if row is not None else 0


def format_mrn(value: int) -> str:
    return f'{MRN_PREFIX}{int(value):04d}'


def ensure_patient_mrn_sequence(conn: sqlite3.Connection) -> None:
    conn.execute(
        '''
        CREATE TABLE IF NOT EXISTS mrn_sequence (
            sequence_name TEXT PRIMARY KEY,
            last_value INTEGER NOT NULL
        )
        '''
    )
    detected_last = detect_last_registered_mrn(conn)
    row = conn.execute(
        'SELECT last_value FROM mrn_sequence WHERE sequence_name = ?',
        (MRN_SEQUENCE_NAME,),
    ).fetchone()
    if row is None:
        conn.execute(
            'INSERT INTO mrn_sequence (sequence_name, last_value) VALUES (?, ?)',
            (MRN_SEQUENCE_NAME, detected_last),
        )
    else:
        current_last = int(row['last_value'] or 0)
        if current_last >= 1_000_000 and detected_last < 100_000:
            conn.execute(
                'UPDATE mrn_sequence SET last_value = ? WHERE sequence_name = ?',
                (detected_last, MRN_SEQUENCE_NAME),
            )
        elif current_last < detected_last:
            conn.execute(
                'UPDATE mrn_sequence SET last_value = ? WHERE sequence_name = ?',
                (detected_last, MRN_SEQUENCE_NAME),
            )


def allocate_next_mrn(conn: sqlite3.Connection) -> str:
    ensure_patient_mrn_sequence(conn)
    row = conn.execute(
        'SELECT last_value FROM mrn_sequence WHERE sequence_name = ?',
        (MRN_SEQUENCE_NAME,),
    ).fetchone()
    next_value = int(row['last_value'] or 0) + 1 if row is not None else 1
    conn.execute(
        'UPDATE mrn_sequence SET last_value = ? WHERE sequence_name = ?',
        (next_value, MRN_SEQUENCE_NAME),
    )
    return format_mrn(next_value)


def sync_patient_identity_references(
    conn: sqlite3.Connection,
    patient_id: int,
    mrn: str,
    name: str,
    age: int,
    gender: str,
    previous_mrn: str = '',
) -> None:
    conn.execute(
        f'''
        UPDATE {INTERNAL_ANALYSIS_TABLE}
        SET patient_mrn = ?, patient_name = ?, patient_age = ?, patient_gender = ?
        WHERE patient_id = ?
        ''',
        (mrn, name, int(age), gender, int(patient_id)),
    )
    if previous_mrn.strip():
        conn.execute(
            '''
            UPDATE hospital_records
            SET "Patient - MRN" = ?, "Patient Name" = ?, "Age" = ?, "Gender" = ?
            WHERE "Patient - MRN" = ?
            ''',
            (mrn, name, int(age), gender, previous_mrn.strip()),
        )
    conn.execute(
        '''
        UPDATE hospital_records
        SET "Patient - MRN" = ?, "Patient Name" = ?, "Age" = ?, "Gender" = ?
        WHERE "Patient Name" = ? AND COALESCE("Predicted Class", '') = ''
        ''',
        (mrn, name, int(age), gender, name),
    )


def repair_patient_mrns(conn: sqlite3.Connection) -> None:
    ensure_patient_mrn_sequence(conn)
    seen: set[str] = set()
    rows = conn.execute(
        'SELECT id, COALESCE(name, ?) AS name, COALESCE(age, 0) AS age, COALESCE(gender, ?) AS gender, COALESCE(mrn, "") AS mrn FROM patients ORDER BY id',
        ('Unassigned patient', 'Unknown'),
    ).fetchall()
    for row in rows:
        current_mrn = str(row['mrn'] or '').strip()
        normalized_key = current_mrn.upper()
        suspicious_auto_mrn = normalized_key.startswith(MRN_PREFIX) and len(re.sub(r'[^0-9]', '', current_mrn)) >= 7
        if current_mrn and normalized_key not in seen and not suspicious_auto_mrn:
            seen.add(normalized_key)
            continue
        new_mrn = allocate_next_mrn(conn)
        conn.execute('UPDATE patients SET mrn = ? WHERE id = ?', (new_mrn, int(row['id'])))
        sync_patient_identity_references(
            conn,
            int(row['id']),
            new_mrn,
            str(row['name'] or 'Unassigned patient'),
            int(row['age'] or 0),
            normalize_gender_label(str(row['gender'] or 'Unknown')),
            current_mrn,
        )
        seen.add(new_mrn.upper())


def ensure_patient_mrn_uniqueness(conn: sqlite3.Connection) -> None:
    conn.execute(
        '''
        CREATE UNIQUE INDEX IF NOT EXISTS idx_patients_mrn_unique
        ON patients(mrn)
        WHERE mrn IS NOT NULL AND TRIM(mrn) <> ''
        '''
    )


def create_display_records_view(conn: sqlite3.Connection, view_name: str) -> None:
    columns_sql = ',\n            '.join(f'"{column}"' for column in DISPLAY_RECORD_COLUMNS)
    conn.execute(f'DROP VIEW IF EXISTS "{view_name}"')
    conn.execute(
        f'''
        CREATE VIEW "{view_name}" AS
        SELECT
            {columns_sql}
        FROM hospital_records
        ORDER BY "Number" DESC
        '''
    )


def create_prediction_records_view(conn: sqlite3.Connection) -> None:
    create_display_records_view(conn, 'prediction_records')


def ensure_analysis_detail_table(conn: sqlite3.Connection) -> None:
    analysis_object = conn.execute(
        "SELECT type FROM sqlite_master WHERE name = 'analysis'"
    ).fetchone()
    detail_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (INTERNAL_ANALYSIS_TABLE,),
    ).fetchone()

    if analysis_object is not None and analysis_object['type'] == 'table':
        analysis_columns = [row['name'] for row in conn.execute('PRAGMA table_info(analysis)')]
        uses_internal_schema = any(
            column in analysis_columns
            for column in ['patient_id', 'result', 'report_summary', 'image_blob', 'prediction_id']
        )
        if uses_internal_schema:
            if not detail_exists:
                conn.execute(f'ALTER TABLE analysis RENAME TO {INTERNAL_ANALYSIS_TABLE}')
            else:
                conn.execute('ALTER TABLE analysis RENAME TO analysis_legacy_backup')

    conn.execute(
        f'''
        CREATE TABLE IF NOT EXISTS {INTERNAL_ANALYSIS_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            result TEXT,
            confidence REAL,
            date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT,
            priority TEXT,
            modality TEXT,
            source_site TEXT,
            report_summary TEXT,
            image_blob BLOB,
            image_name TEXT,
            image_kind TEXT,
            ai_engine TEXT,
            prediction_id TEXT,
            prediction_timestamp TEXT,
            image_id TEXT,
            image_path TEXT,
            upload_timestamp TEXT,
            patient_mrn TEXT,
            patient_name TEXT,
            patient_age INTEGER,
            patient_gender TEXT
        )
        '''
    )
    create_display_records_view(conn, 'analysis')


def create_hospital_records_table(conn: sqlite3.Connection) -> None:
    expected_columns = DISPLAY_RECORD_COLUMNS

    def create_table(name: str) -> None:
        conn.execute(
            f'''
            CREATE TABLE IF NOT EXISTS "{name}" (
                "Number" INTEGER PRIMARY KEY AUTOINCREMENT,
                "Patient - MRN" TEXT,
                "Patient Name" TEXT,
                "Age" INTEGER,
                "Gender" TEXT,
                "Predicted Class" TEXT,
                "Confidence Score" REAL,
                "Prediction Date and Time" TEXT,
                "Status" TEXT,
                "priority" TEXT,
                "modality" TEXT,
                "Image Name" TEXT,
                "Image Path" TEXT,
                "Findings" TEXT
            )
            '''
        )

    existing_rows = list(conn.execute('PRAGMA table_info(hospital_records)'))
    if not existing_rows:
        create_table('hospital_records')
        return

    existing_columns = [row['name'] for row in existing_rows]
    if existing_columns == expected_columns:
        return

    conn.execute('ALTER TABLE hospital_records RENAME TO hospital_records_legacy')
    create_table('hospital_records')
    legacy_columns = {row['name'] for row in conn.execute('PRAGMA table_info(hospital_records_legacy)')}

    def legacy_value(*candidates: str, default: str = "''") -> str:
        for candidate in candidates:
            if candidate in legacy_columns:
                return f'"{candidate}"'
        return default

    conn.execute(
        f'''
        INSERT INTO hospital_records (
            "Number", "Patient - MRN", "Patient Name", "Age", "Gender", "Predicted Class", "Confidence Score",
            "Prediction Date and Time", "Status", "priority", "modality", "Image Name", "Image Path", "Findings"
        )
        SELECT
            {legacy_value('Number', default='NULL')},
            {legacy_value('Patient - MRN', 'MRN', default="''")},
            {legacy_value('Patient Name', 'Name', default="''")},
            {legacy_value('Age', default='0')},
            {legacy_value('Gender', default="''")},
            {legacy_value('Predicted Class', default="''")},
            {legacy_value('Confidence Score', default='NULL')},
            {legacy_value('Prediction Date and Time', 'Upload Date and Time', default="''")},
            {legacy_value('Status', default="''")},
            {legacy_value('priority', default="''")},
            {legacy_value('modality', default="''")},
            {legacy_value('Image Name', 'Image - Path', 'Image Path', default="''")},
            {legacy_value('Image Path', 'Image - Path', default="''")},
            {legacy_value('Findings', default="''")}
        FROM hospital_records_legacy
        ORDER BY {legacy_value('Number', default='rowid')}
        '''
    )
    conn.execute('DROP TABLE hospital_records_legacy')


def sync_existing_hospital_records(conn: sqlite3.Connection) -> None:
    count_row = conn.execute('SELECT COUNT(*) FROM hospital_records').fetchone()
    existing = int(count_row[0] if not isinstance(count_row, sqlite3.Row) else count_row[0])
    if existing > 0:
        return
    rows = conn.execute(
        '''
        SELECT
            a.id AS number,
            COALESCE(p.mrn, '') AS mrn,
            COALESCE(p.name, 'Unassigned patient') AS name,
            COALESCE(p.age, 0) AS age,
            COALESCE(p.gender, 'Unknown') AS gender,
            COALESCE(a.result, '') AS predicted_class,
            COALESCE(a.confidence, NULL) AS confidence_score,
            COALESCE(a.prediction_timestamp, datetime(COALESCE(a.date, CURRENT_TIMESTAMP), '+3 hours') || ' Ethiopia Time') AS prediction_timestamp,
            COALESCE(a.status, '') AS status,
            COALESCE(a.priority, '') AS priority,
            COALESCE(a.modality, '') AS modality,
            COALESCE(a.image_name, '') AS image_name,
            COALESCE(a.image_path, a.image_name, '') AS image_path,
            COALESCE(a.report_summary, '') AS findings
        FROM analysis_detail a
        LEFT JOIN patients p ON p.id = a.patient_id
        ORDER BY a.id
        '''
    ).fetchall()
    for row in rows:
        conn.execute(
            '''
            INSERT INTO hospital_records (
                "Number", "Patient - MRN", "Patient Name", "Age", "Gender", "Predicted Class", "Confidence Score",
                "Prediction Date and Time", "Status", "priority", "modality", "Image Name", "Image Path", "Findings"
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                row['number'],
                row['mrn'],
                row['name'],
                row['age'],
                row['gender'],
                row['predicted_class'],
                row['confidence_score'],
                row['prediction_timestamp'],
                row['status'],
                row['priority'],
                row['modality'],
                row['image_name'],
                row['image_path'],
                row['findings'],
            ),
        )


def insert_hospital_record_registration(conn: sqlite3.Connection, mrn: str, name: str, age: int, gender: str) -> int:
    cursor = conn.execute(
        '''
        INSERT INTO hospital_records (
            "Patient - MRN", "Patient Name", "Age", "Gender", "Predicted Class", "Confidence Score",
            "Prediction Date and Time", "Status", "priority", "modality", "Image Name", "Image Path", "Findings"
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        (
            mrn,
            name,
            age,
            gender,
            '',
            None,
            '',
            '',
            '',
            '',
            '',
            '',
            '',
        ),
    )
    return int(cursor.lastrowid)


def update_hospital_record_prediction(
    conn: sqlite3.Connection,
    patient_id: int,
    result: str,
    confidence: float,
    status: str,
    priority: str,
    modality: str,
    image_name: str,
    image_path: str,
    prediction_time_label: str,
    findings: str,
) -> None:
    patient = conn.execute('SELECT COALESCE(mrn, \'\') AS mrn, COALESCE(name, \'Unassigned patient\') AS name, COALESCE(age, 0) AS age, COALESCE(gender, \'Unknown\') AS gender FROM patients WHERE id = ?', (int(patient_id),)).fetchone()
    if patient is None:
        return
    pending_row = conn.execute(
        '''
        SELECT "Number" FROM hospital_records
        WHERE "Patient - MRN" = ? AND COALESCE("Predicted Class", '') = ''
        ORDER BY "Number" DESC
        LIMIT 1
        ''',
        (patient['mrn'],),
    ).fetchone()
    values = (
        patient['mrn'],
        patient['name'],
        int(patient['age']),
        patient['gender'],
        result,
        round(float(confidence), 4),
        prediction_time_label,
        status,
        priority,
        modality,
        image_name,
        image_path,
        findings,
    )
    if pending_row is not None:
        conn.execute(
            '''
            UPDATE hospital_records
            SET "Patient - MRN" = ?, "Patient Name" = ?, "Age" = ?, "Gender" = ?, "Predicted Class" = ?,
                "Confidence Score" = ?, "Prediction Date and Time" = ?, "Status" = ?,
                "priority" = ?, "modality" = ?, "Image Name" = ?, "Image Path" = ?, "Findings" = ?
            WHERE "Number" = ?
            ''',
            values + (int(pending_row['Number']),),
        )
    else:
        conn.execute(
            '''
            INSERT INTO hospital_records (
                "Patient - MRN", "Patient Name", "Age", "Gender", "Predicted Class", "Confidence Score",
                "Prediction Date and Time", "Status", "priority", "modality", "Image Name", "Image Path", "Findings"
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            values,
        )


def sync_analysis_patient_fields(conn: sqlite3.Connection) -> None:
    conn.execute(
        '''
        UPDATE analysis_detail
        SET
            patient_mrn = COALESCE((SELECT p.mrn FROM patients p WHERE p.id = analysis_detail.patient_id), COALESCE(patient_mrn, '')),
            patient_name = COALESCE((SELECT p.name FROM patients p WHERE p.id = analysis_detail.patient_id), COALESCE(patient_name, '')),
            patient_age = COALESCE((SELECT p.age FROM patients p WHERE p.id = analysis_detail.patient_id), COALESCE(patient_age, 0)),
            patient_gender = COALESCE((SELECT p.gender FROM patients p WHERE p.id = analysis_detail.patient_id), COALESCE(patient_gender, ''))
        '''
    )


def refresh_hospital_records_from_analysis(conn: sqlite3.Connection) -> None:
    conn.execute(
        '''
        UPDATE hospital_records
        SET
            "Patient - MRN" = COALESCE((SELECT COALESCE(a.patient_mrn, p.mrn, '') FROM analysis_detail a LEFT JOIN patients p ON p.id = a.patient_id WHERE a.id = hospital_records."Number"), COALESCE("Patient - MRN", '')),
            "Patient Name" = COALESCE((SELECT COALESCE(a.patient_name, p.name, '') FROM analysis_detail a LEFT JOIN patients p ON p.id = a.patient_id WHERE a.id = hospital_records."Number"), COALESCE("Patient Name", '')),
            "Age" = COALESCE((SELECT COALESCE(a.patient_age, p.age, 0) FROM analysis_detail a LEFT JOIN patients p ON p.id = a.patient_id WHERE a.id = hospital_records."Number"), COALESCE("Age", 0)),
            "Gender" = COALESCE((SELECT COALESCE(a.patient_gender, p.gender, '') FROM analysis_detail a LEFT JOIN patients p ON p.id = a.patient_id WHERE a.id = hospital_records."Number"), COALESCE("Gender", '')),
            "Predicted Class" = COALESCE((SELECT COALESCE(a.result, '') FROM analysis_detail a WHERE a.id = hospital_records."Number"), COALESCE("Predicted Class", '')),
            "Confidence Score" = COALESCE((SELECT COALESCE(a.confidence, NULL) FROM analysis_detail a WHERE a.id = hospital_records."Number"), "Confidence Score"),
            "Prediction Date and Time" = COALESCE((SELECT COALESCE(a.prediction_timestamp, '') FROM analysis_detail a WHERE a.id = hospital_records."Number"), COALESCE("Prediction Date and Time", '')),
            "Status" = COALESCE((SELECT COALESCE(a.status, '') FROM analysis_detail a WHERE a.id = hospital_records."Number"), COALESCE("Status", '')),
            "priority" = COALESCE((SELECT COALESCE(a.priority, '') FROM analysis_detail a WHERE a.id = hospital_records."Number"), COALESCE("priority", '')),
            "modality" = COALESCE((SELECT COALESCE(a.modality, '') FROM analysis_detail a WHERE a.id = hospital_records."Number"), COALESCE("modality", '')),
            "Image Name" = COALESCE((SELECT COALESCE(a.image_name, '') FROM analysis_detail a WHERE a.id = hospital_records."Number"), COALESCE("Image Name", '')),
            "Image Path" = COALESCE((SELECT COALESCE(a.image_path, a.image_name, '') FROM analysis_detail a WHERE a.id = hospital_records."Number"), COALESCE("Image Path", '')),
            "Findings" = COALESCE((SELECT COALESCE(a.report_summary, '') FROM analysis_detail a WHERE a.id = hospital_records."Number"), COALESCE("Findings", ''))
        '''
    )

def migrate_schema(conn: sqlite3.Connection) -> None:
    # Older hospital.db files were created before MRN and workflow metadata existed.
    # We add missing columns in place so existing user data keeps working.
    for table, columns in {
        'patients': {
            'name': 'TEXT',
            'age': 'INTEGER',
            'gender': 'TEXT',
            'mrn': 'TEXT',
        },
        INTERNAL_ANALYSIS_TABLE: {
            'patient_id': 'INTEGER',
            'result': 'TEXT',
            'confidence': 'REAL',
            'date': 'TIMESTAMP',
            'status': 'TEXT',
            'priority': 'TEXT',
            'modality': 'TEXT',
            'source_site': 'TEXT',
            'report_summary': 'TEXT',
            'image_blob': 'BLOB',
            'image_name': 'TEXT',
            'image_kind': 'TEXT',
            'ai_engine': 'TEXT',
            'prediction_id': 'TEXT',
            'prediction_timestamp': 'TEXT',
            'image_id': 'TEXT',
            'image_path': 'TEXT',
            'upload_timestamp': 'TEXT',
            'patient_mrn': 'TEXT',
            'patient_name': 'TEXT',
            'patient_age': 'INTEGER',
            'patient_gender': 'TEXT',
        },
    }.items():
        for column, definition in columns.items():
            ensure_column(conn, table, column, definition)
    ensure_patient_mrn_sequence(conn)
    repair_patient_mrns(conn)
    ensure_patient_mrn_uniqueness(conn)
    sync_analysis_patient_fields(conn)
    create_hospital_records_table(conn)
    sync_existing_hospital_records(conn)
    refresh_hospital_records_from_analysis(conn)
    create_display_records_view(conn, 'analysis')
    create_prediction_records_view(conn)
    conn.commit()


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute('CREATE TABLE IF NOT EXISTS patients (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, age INTEGER, gender TEXT, mrn TEXT)')
    ensure_analysis_detail_table(conn)
    create_hospital_records_table(conn)
    migrate_schema(conn)
    conn.commit()
    return conn


def init_state() -> None:
    for key, value in {
        'page': 'Registry',
        'selected_study_id': None,
        'active_patient_id': None,
        'viewer_image_bytes': None,
        'viewer_upload_name': None,
        'viewer_upload_meta': None,
        'analysis_uploader_token': 0,
        'analysis_result': None,
        'analysis_busy': False,
        'analysis_scroll_to_result': False,
        'report_draft': '',
        'selected_model_name': None,
        'flash_message': None,
        'viewer_gradcam_cache': {},
    }.items():
        st.session_state.setdefault(key, value)


def fetch_patients(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query('SELECT id, name, age, gender, COALESCE(mrn, "") AS mrn FROM patients ORDER BY id DESC', conn)


def active_patient(conn: sqlite3.Connection) -> pd.Series | None:
    patients = fetch_patients(conn)
    if patients.empty:
        return None
    active_id = st.session_state.get('active_patient_id')
    if active_id is not None:
        match = patients[patients['id'].astype(int) == int(active_id)]
        if not match.empty:
            return match.iloc[0]
    st.session_state.active_patient_id = int(patients.iloc[0]['id'])
    return patients.iloc[0]


def fetch_studies(conn: sqlite3.Connection) -> pd.DataFrame:
    query = '''
        SELECT a.id AS study_id, a.patient_id, COALESCE(p.name, 'Unassigned patient') AS patient_name,
               COALESCE(p.age, 0) AS age, COALESCE(p.gender, 'Unknown') AS gender, COALESCE(p.mrn, '') AS mrn,
               COALESCE(a.result, 'Pending review') AS result, COALESCE(a.confidence, 0.0) AS confidence,
               COALESCE(a.status, 'Completed') AS status, COALESCE(a.priority, 'Routine') AS priority,
               COALESCE(a.modality, 'CR') AS modality, COALESCE(a.source_site, 'Radiology Hub') AS source_site,
               COALESCE(a.report_summary, 'Awaiting radiologist sign-off.') AS report_summary,
               COALESCE(a.image_name, '') AS image_name, COALESCE(a.image_kind, '') AS image_kind,
               COALESCE(a.ai_engine, '') AS ai_engine,
               COALESCE(a.prediction_id, 'PRED-' || printf('%06d', a.id)) AS prediction_id,
               COALESCE(a.prediction_timestamp, datetime(COALESCE(a.date, CURRENT_TIMESTAMP), '+3 hours') || ' Ethiopia Time') AS prediction_timestamp,
               COALESCE(a.image_id, 'IMG-' || printf('%06d', a.id)) AS image_id,
               COALESCE(a.image_path, a.image_name, '') AS image_path,
               COALESCE(a.upload_timestamp, datetime(COALESCE(a.date, CURRENT_TIMESTAMP), '+3 hours') || ' Ethiopia Time') AS upload_timestamp,
               COALESCE(a.date, datetime('now')) AS study_date
        FROM analysis_detail a LEFT JOIN patients p ON p.id = a.patient_id
        ORDER BY datetime(a.date) DESC
    '''
    frame = pd.read_sql_query(query, conn)
    if not frame.empty:
        frame['study_date'] = pd.to_datetime(frame['study_date'], errors='coerce', utc=True).dt.tz_convert(ETHIOPIA_TZ).dt.tz_localize(None)
    return frame


def demo_patients() -> pd.DataFrame:
    return pd.DataFrame([
        {'id': 101, 'name': 'SEQLAM NGUS', 'age': 23, 'gender': 'F', 'mrn': 'TTG-013-00953'},
        {'id': 102, 'name': 'EDEBIRHAN ASGEDOM', 'age': 40, 'gender': 'F', 'mrn': 'TTG-013-00952'},
        {'id': 103, 'name': 'BLETU GIRUM', 'age': 17, 'gender': 'F', 'mrn': 'TTG-013-00951'},
        {'id': 104, 'name': 'G/MICHEAL G/HIWOT', 'age': 42, 'gender': 'M', 'mrn': 'TTG-013-00949'},
        {'id': 105, 'name': 'KESHI TESFAY', 'age': 72, 'gender': 'M', 'mrn': 'TTG-013-00948'},
        {'id': 106, 'name': 'HELEN TESFAY', 'age': 31, 'gender': 'F', 'mrn': 'TTG-013-00947'},
        {'id': 107, 'name': 'MEDHANYE G/MEDHIN', 'age': 33, 'gender': 'M', 'mrn': 'TTG-013-00946'},
        {'id': 108, 'name': 'MULU G/FSEFW', 'age': 55, 'gender': 'F', 'mrn': 'TTG-013-00945'},
    ])


def demo_studies() -> pd.DataFrame:
    now = now_gmt3().replace(minute=0, second=0, microsecond=0, tzinfo=None)
    rows = [
        (-1, 101, 'SEQLAM NGUS', 23, 'F', 'TTG-013-00953', 'Normal', 0.98, 'Completed', 'Routine', 'CR', 'BRAVE AI | Addis Screening Hub', 'No acute cardiopulmonary abnormality identified. Continue routine follow-up.', 'demo_seq.png', 'png', 'Demo heuristic', now - timedelta(hours=2)),
        (-2, 102, 'EDEBIRHAN ASGEDOM', 40, 'F', 'TTG-013-00952', 'Tuberculosis', 0.82, 'Completed', 'High', 'CR', 'BRAVE AI | Addis Screening Hub', 'Upper zone fibrotic change with suspicious cavitary focus. Correlate with GeneXpert testing.', 'demo_tb_1.png', 'png', 'Demo heuristic', now - timedelta(hours=3, minutes=24)),
        (-3, 103, 'BLETU GIRUM', 17, 'F', 'TTG-013-00951', 'Pneumonia', 0.74, 'Completed', 'High', 'CR', 'BRAVE AI | Northern Referral Site', 'Patchy bilateral lower lobe air-space opacities suggest infectious consolidation.', 'demo_pna_1.png', 'png', 'Demo heuristic', now - timedelta(hours=4, minutes=8)),
        (-4, 104, 'G/MICHEAL G/HIWOT', 42, 'M', 'TTG-013-00949', 'Normal', 0.93, 'Completed', 'Routine', 'CR', 'BRAVE AI | Northern Referral Site', 'Lung fields are clear. Cardiomediastinal silhouette is within normal limits.', 'demo_normal_1.png', 'png', 'Demo heuristic', now - timedelta(hours=4, minutes=17)),
        (-5, 105, 'KESHI TESFAY', 72, 'M', 'TTG-013-00948', 'Tuberculosis', 0.64, 'Completed', 'Watchlist', 'CR', 'BRAVE AI | Coast Mobile Unit', 'Biapical pleural thickening and chronic scarring. Recommend targeted microbiology workup.', 'demo_tb_2.png', 'png', 'Demo heuristic', now - timedelta(hours=5, minutes=2)),
        (-6, 106, 'HELEN TESFAY', 31, 'F', 'TTG-013-00947', 'COVID-19', 0.58, 'Pending sign-off', 'Watchlist', 'CR', 'BRAVE AI | Coast Mobile Unit', 'Peripheral mixed interstitial and air-space change. Clinical and laboratory correlation advised.', 'demo_covid_1.png', 'png', 'Demo heuristic', now - timedelta(hours=6, minutes=11)),
        (-7, 107, 'MEDHANYE G/MEDHIN', 33, 'M', 'TTG-013-00946', 'Normal', 0.96, 'Completed', 'Routine', 'CR', 'BRAVE AI | Rift Valley Center', 'No active focal opacity. Pleural spaces are preserved.', 'demo_normal_2.png', 'png', 'Demo heuristic', now - timedelta(hours=6, minutes=22)),
        (-8, 108, 'MULU G/FSEFW', 55, 'F', 'TTG-013-00945', 'Tuberculosis', 0.91, 'Completed', 'Critical', 'CR', 'BRAVE AI | Rift Valley Center', 'Fibrosis in the left upper and mid zones with probable cavitation. Escalate for urgent evaluation.', 'demo_tb_3.png', 'png', 'Demo heuristic', now - timedelta(hours=7, minutes=5)),
    ]
    cols = ['study_id', 'patient_id', 'patient_name', 'age', 'gender', 'mrn', 'result', 'confidence', 'status', 'priority', 'modality', 'source_site', 'report_summary', 'image_name', 'image_kind', 'ai_engine', 'study_date']
    return pd.DataFrame(rows, columns=cols)


def operational_data(conn: sqlite3.Connection) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    patients = fetch_patients(conn)
    studies = fetch_studies(conn)
    if studies.empty:
        return (patients if not patients.empty else demo_patients(), demo_studies(), True)
    return patients, studies, False


def discover_model_files() -> list[Path]:
    discovered: list[Path] = []
    for preferred in MODEL_PRIORITY:
        candidate = BASE_DIR / preferred
        if candidate.exists():
            discovered.append(candidate)
    for candidate in sorted(BASE_DIR.glob('*.keras')):
        if candidate not in discovered:
            discovered.append(candidate)
    return discovered


def ensure_selected_model(model_files: list[Path]) -> None:
    names = [path.name for path in model_files]
    if names:
        # The UI no longer exposes a model picker, so always bind to the
        # highest-priority available model for deterministic inference.
        st.session_state.selected_model_name = names[0]
    else:
        st.session_state.selected_model_name = None


def selected_model_path(model_files: list[Path]) -> Path | None:
    ensure_selected_model(model_files)
    for path in model_files:
        if path.name == st.session_state.get('selected_model_name'):
            return path
    return None


def infer_model_kind(model_path: Path) -> str:
    name = model_path.name.lower()
    if 'resnet' in name:
        return 'resnet50'
    if 'efficientnet' in name:
        return 'efficientnet'
    if 'densenet' in name:
        return 'densenet'
    if 'mobilenet' in name:
        return 'mobilenetv2'
    if 'vgg16' in name:
        return 'vgg16'
    return 'custom'


@st.cache_resource(show_spinner=False)
def load_keras_model(model_path_str: str):
    def build_compat_model(model_path: Path):
        import numpy as np
        import tempfile
        from zipfile import ZipFile
        import tensorflow as tf

        model_kind = infer_model_kind(model_path)
        builders = {
            'densenet': lambda: tf.keras.applications.DenseNet121(weights=None, include_top=False, input_shape=(224, 224, 3), name='densenet121'),
            'resnet50': lambda: tf.keras.applications.ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3), name='resnet50'),
            'efficientnet': lambda: tf.keras.applications.EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3), name='efficientnetb0'),
            'mobilenetv2': lambda: tf.keras.applications.MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3), name='mobilenetv2'),
            'vgg16': lambda: tf.keras.applications.VGG16(weights=None, include_top=False, input_shape=(224, 224, 3), name='vgg16'),
        }
        if model_kind not in builders:
            raise ValueError(f'No compatibility builder is defined for {model_path.name}.')

        with tempfile.TemporaryDirectory() as td:
            with ZipFile(model_path, 'r') as archive:
                if 'model.weights.h5' not in archive.namelist():
                    raise ValueError(f'{model_path.name} does not contain model.weights.h5.')
                archive.extract('model.weights.h5', path=td)
            weights_path = Path(td) / 'model.weights.h5'
            base_model = builders[model_kind]()
            base_model.trainable = False
            model = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(shape=(224, 224, 3), name='input_layer_1'),
                    base_model,
                    tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling2d'),
                    tf.keras.layers.BatchNormalization(name='batch_normalization'),
                    tf.keras.layers.Dense(256, activation='relu', name='dense'),
                    tf.keras.layers.Dropout(0.5, name='dropout'),
                    tf.keras.layers.Dense(4, activation='softmax', name='dense_1'),
                ],
                name='sequential',
            )
            model(np.zeros((1, 224, 224, 3), dtype=np.float32), training=False)
            model.load_weights(weights_path)
            setattr(model, '_brave_loader_mode', 'compat')
            return model

    try:
        import tensorflow as tf

        model = tf.keras.models.load_model(model_path_str, compile=False)
        setattr(model, '_brave_loader_mode', 'standard')
        return model, None
    except Exception as exc:
        try:
            compat_model = build_compat_model(Path(model_path_str))
            return compat_model, None
        except Exception as compat_exc:
            return None, f'Primary load failed: {exc} | Compatibility load failed: {compat_exc}'


def predict_scores_external(image: Image.Image, model_path: Path) -> tuple[np.ndarray | None, dict[str, str] | None, str | None]:
    if not BRAVE_ENV_PYTHON.exists():
        return None, None, f'External inference python was not found at {BRAVE_ENV_PYTHON}.'
    if not INFERENCE_WORKER_PATH.exists():
        return None, None, f'External inference worker was not found at {INFERENCE_WORKER_PATH}.'

    try:
        with tempfile.TemporaryDirectory() as td:
            image_path = Path(td) / 'input.png'
            image.convert('RGB').save(image_path)
            completed = subprocess.run(
                [str(BRAVE_ENV_PYTHON), str(INFERENCE_WORKER_PATH), '--model', str(model_path), '--image', str(image_path)],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if completed.returncode != 0:
                message = completed.stderr.strip() or completed.stdout.strip() or f'External inference exited with code {completed.returncode}.'
                return None, None, message
            payload = json.loads(completed.stdout.strip())
            probs = np.array(payload['probs'], dtype=np.float32)
            backend = {
                'mode': str(payload.get('mode', 'real-external')),
                'engine': str(payload.get('engine', model_path.name)),
                'detail': str(payload.get('detail', 'external inference')),
            }
            return probs, backend, None
    except Exception as exc:
        return None, None, str(exc)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def preprocess_for_model(image: Image.Image, model_kind: str, target_size: tuple[int, int]) -> np.ndarray:
    rgb_image = image.convert('RGB').resize(target_size)
    batch = np.expand_dims(np.asarray(rgb_image, dtype=np.float32), axis=0)
    if model_kind == 'resnet50':
        from tensorflow.keras.applications.resnet50 import preprocess_input

        return preprocess_input(batch)
    if model_kind == 'efficientnet':
        from tensorflow.keras.applications.efficientnet import preprocess_input

        return preprocess_input(batch)
    if model_kind == 'densenet':
        from tensorflow.keras.applications.densenet import preprocess_input

        return preprocess_input(batch)
    if model_kind == 'mobilenetv2':
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        return preprocess_input(batch)
    if model_kind == 'vgg16':
        from tensorflow.keras.applications.vgg16 import preprocess_input

        return preprocess_input(batch)
    return batch / 255.0


def align_probabilities(raw_probs: np.ndarray) -> np.ndarray:
    mapped = {label: 0.0 for label in CLASS_NAMES}
    for raw_label, score in zip(RAW_MODEL_CLASSES, raw_probs.tolist()):
        display_label = RAW_TO_DISPLAY.get(raw_label)
        if display_label in mapped:
            mapped[display_label] = float(score)
    aligned = np.array([mapped[label] for label in CLASS_NAMES], dtype=np.float32)
    if aligned.sum() <= 0:
        return np.full(len(CLASS_NAMES), 1.0 / len(CLASS_NAMES), dtype=np.float32)
    return aligned / aligned.sum()


def pill(text: str, tone: str) -> str:
    return f"<span class='pill {tone}'>{text}</span>"


def result_tone(result: str) -> str:
    return {'Normal': 'success', 'Tuberculosis': 'danger', 'Pneumonia': 'warning', 'COVID-19': 'warning'}.get(result, 'neutral')


def priority_tone(priority: str) -> str:
    return {'Critical': 'danger', 'High': 'warning', 'Watchlist': 'warning', 'Routine': 'success'}.get(priority, 'neutral')


def section(title: str, copy: str, kicker: str) -> None:
    kicker_html = f"<div class='kicker'>{kicker}</div>" if kicker.strip() else ""
    copy_html = f"<p class='section-copy'>{copy}</p>" if copy.strip() else ""
    st.markdown(f"<div class='section'>{kicker_html}<p class='section-title'>{title}</p>{copy_html}</div>", unsafe_allow_html=True)


def flash_banner() -> None:
    message = st.session_state.get('flash_message')
    if message:
        st.success(message)
        st.session_state.flash_message = None


def shell(page: str, demo_mode: bool, model_files: list[Path]) -> None:
    st.markdown(
        f"<div class='shell'><div class='brand-lockup'><div class='brand-mark'>{brave_logo_svg()}</div><div class='brand-stack'><p class='brand'>{BRANDING['platform_name']}</p><p class='sub'>{BRANDING['tagline']}</p><p class='soft' style='margin:0.45rem 0 0; max-width:900px;'>{DISCLAIMER_TEXT}</p></div></div></div>",
        unsafe_allow_html=True,
    )
    cols = st.columns([1.0] * len(NAV_ITEMS))
    for i, item in enumerate(NAV_ITEMS):
        with cols[i]:
            if st.button(item, key=f'nav_{item}', use_container_width=True):
                st.session_state.page = item
                rerun()

def metrics(demo_mode: bool, studies: pd.DataFrame) -> list[dict[str, str]]:
    normal_count = int(studies['result'].eq('Normal').sum()) if not studies.empty else 0
    tb_count = int(studies['result'].eq('Tuberculosis').sum()) if not studies.empty else 0
    pneumonia_count = int(studies['result'].eq('Pneumonia').sum()) if not studies.empty else 0
    covid_count = int(studies['result'].eq('COVID-19').sum()) if not studies.empty else 0
    if demo_mode:
        return [
            {'label': 'Patients Registered', 'value': '879', 'copy': 'Across 4 active screening sites', 'badge': '24 newly triaged this week'},
            {'label': 'X-Rays Received', 'value': '412', 'copy': 'Studies routed into the digital worklist', 'badge': 'Median TAT 9.4 min'},
            {'label': 'X-Rays With AI Findings', 'value': '156', 'copy': 'Cases escalated for closer radiology attention', 'badge': '38% abnormality rate'},
            {'label': 'TB Presumptive Cases', 'value': '23', 'copy': 'Flagged for confirmatory lab review', 'badge': '7 in high-priority queue'},
            {'label': 'Normal Findings', 'value': str(normal_count), 'copy': 'Demo cases cleared without acute concern', 'badge': 'Reference screening output'},
            {'label': 'Tuberculosis Flags', 'value': str(tb_count), 'copy': 'Demo cases marked for TB-focused follow-up', 'badge': 'Escalation pathway active'},
            {'label': 'Pneumonia Flags', 'value': str(pneumonia_count), 'copy': 'Demo cases with pneumonia-pattern opacity', 'badge': 'Infection review suggested'},
            {'label': 'COVID-19 Flags', 'value': str(covid_count), 'copy': 'Demo cases with viral-pattern chest change', 'badge': 'Clinical review suggested'},
            {'label': 'Reports Within SLA', 'value': '96%', 'copy': 'Signed within the target turnaround window', 'badge': 'Operational target met'},
            {'label': 'Lab Confirmed TB Cases', 'value': '0', 'copy': 'Awaiting microbiology synchronization', 'badge': 'No synced lab file today'},
        ]
    abnormal = int(studies['result'].ne('Normal').sum())
    return [
        {'label': 'Studies Reviewed', 'value': f'{len(studies):,}', 'copy': 'Completed and queued imaging studies', 'badge': f'{abnormal} abnormal flags'},
        {'label': 'Normal Findings', 'value': str(normal_count), 'copy': 'Studies cleared without acute concern', 'badge': 'Auto-summarized for rapid reporting'},
        {'label': 'Tuberculosis Flags', 'value': str(tb_count), 'copy': 'Cases marked for infectious disease follow-up', 'badge': 'Escalation pathway active'},
        {'label': 'Pneumonia Flags', 'value': str(pneumonia_count), 'copy': 'Cases marked for pneumonia-focused review', 'badge': 'Clinical review suggested'},
        {'label': 'COVID-19 Flags', 'value': str(covid_count), 'copy': 'Cases marked for viral-pattern review', 'badge': 'Isolation workflow aware'},
        {'label': 'Average Confidence', 'value': f"{studies['confidence'].mean() * 100:.1f}%", 'copy': 'Mean model confidence across recorded studies', 'badge': 'Interpret with radiologist oversight'},
    ]


def render_metrics(data: list[dict[str, str]]) -> None:
    cols = st.columns(3)
    for i, item in enumerate(data):
        with cols[i % 3]:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>{item['label']}</div><div class='metric-value'>{item['value']}</div><div class='metric-copy'>{item['copy']}</div><div class='metric-badge'>{item['badge']}</div></div>", unsafe_allow_html=True)


def weekly(studies: pd.DataFrame, demo_mode: bool) -> pd.DataFrame:
    if demo_mode:
        dates = pd.date_range(end=now_gmt3().replace(tzinfo=None), periods=7, freq='W')
        return pd.DataFrame({'Week': dates, 'Patients Registered': [84, 96, 102, 108, 115, 118, 119], 'Scans Received': [72, 88, 95, 90, 104, 97, 103], 'Abnormal Scans': [18, 24, 21, 16, 23, 19, 17], 'TB Presumptive': [7, 11, 9, 6, 8, 10, 7]})
    frame = studies.copy()
    frame['Week'] = frame['study_date'].dt.to_period('W').apply(lambda r: r.start_time)
    grouped = frame.groupby('Week').agg(**{'Scans Received': ('study_id', 'count'), 'Abnormal Scans': ('result', lambda s: s.ne('Normal').sum()), 'TB Presumptive': ('result', lambda s: s.eq('Tuberculosis').sum())}).reset_index().sort_values('Week')
    grouped['Patients Registered'] = grouped['Scans Received']
    return grouped.tail(8)


def bucket_age(age: int) -> str:
    if age < 15:
        return '<15'
    if age < 35:
        return '15-34'
    if age < 45:
        return '35-44'
    if age < 60:
        return '45-59'
    return '60+'


def donut(series: pd.Series, title: str, palette: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    values = series.values if len(series) else np.array([1])
    labels = series.index.tolist() if len(series) else ['No data']
    ax.pie(values, labels=None, colors=palette[:len(values)], startangle=90, counterclock=False, wedgeprops={'width': 0.34, 'edgecolor': '#f9fbfa'})
    total = int(np.sum(values))
    ax.text(0, 0.08, f'{total}', ha='center', va='center', fontsize=26, fontweight='bold')
    ax.text(0, -0.12, 'TOTAL', ha='center', va='center', fontsize=10, color='#65757a')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    legend = [f'{label}   {value / max(total, 1) * 100:.1f}%' for label, value in zip(labels, values)]
    ax.legend(legend, loc='center left', bbox_to_anchor=(-0.24, 0.5), frameon=False, fontsize=9)
    ax.set(aspect='equal')
    fig.patch.set_facecolor('#ffffff')
    return fig


def line_chart(trend: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = {'Patients Registered': '#457b9d', 'Scans Received': '#0f8f7f', 'Abnormal Scans': '#c77d36', 'TB Presumptive': '#a33a35'}
    for col in ['Patients Registered', 'Scans Received', 'Abnormal Scans', 'TB Presumptive']:
        ax.plot(trend['Week'], trend[col], linewidth=2.3, marker='o', markersize=4.2, color=colors[col], label=col)
    ax.grid(alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc='upper left')
    ax.set_ylabel('Count')
    ax.set_xlabel('Week')
    fig.autofmt_xdate(rotation=0)
    fig.patch.set_facecolor('#ffffff')
    return fig


def site_chart(studies: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    if studies.empty:
        ax.text(0.5, 0.5, 'No site data available', ha='center', va='center', fontsize=12, color='#66757a')
        ax.axis('off')
        fig.patch.set_facecolor('#ffffff')
        return fig

    grouped = (
        studies.groupby('source_site')
        .agg(
            Studies=('study_id', 'count'),
            Abnormal=('result', lambda s: s.ne('Normal').sum()),
            Tuberculosis=('result', lambda s: s.eq('Tuberculosis').sum()),
        )
        .reset_index()
        .sort_values('Studies', ascending=False)
        .head(6)
    )

    labels = [str(site)[:18] + ('...' if len(str(site)) > 18 else '') for site in grouped['source_site']]
    x = np.arange(len(labels))
    width = 0.24
    ax.bar(x - width, grouped['Studies'], width=width, color='#0f8f7f', label='Studies')
    ax.bar(x, grouped['Abnormal'], width=width, color='#c77d36', label='Abnormal')
    ax.bar(x + width, grouped['Tuberculosis'], width=width, color='#a33a35', label='TB Flags')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.15)
    ax.set_ylabel('Count')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha='right')
    ax.legend(frameon=False, ncol=3, loc='upper left')
    ax.set_title('Site Wise Performance', fontsize=13, fontweight='bold', loc='left', pad=12)
    fig.patch.set_facecolor('#ffffff')
    return fig


def cascade_chart(demo_mode: bool, studies: pd.DataFrame) -> plt.Figure:
    if demo_mode:
        items = [('Patients Registered', 119), ('X-Rays Received', 103), ('X-Rays With AI Findings', 52), ('TB Presumptive', 23), ('Lab Tests Initiated', 0), ('Lab Confirmed', 0)]
    else:
        items = [('Studies Reviewed', len(studies)), ('Abnormal Findings', int(studies['result'].ne('Normal').sum())), ('TB Flags', int(studies['result'].eq('Tuberculosis').sum())), ('Pending Sign-Off', int(studies['status'].eq('Pending sign-off').sum())), ('High Priority', int(studies['priority'].isin(['High', 'Critical']).sum())), ('Closed Reports', int(studies['status'].eq('Completed').sum()))]
    labels = [item[0] for item in items]
    values = np.array([item[1] for item in items], dtype=float)
    widths = values / max(values.max(), 1)
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    y = np.arange(len(labels))
    ax.barh(y, widths, color='#2a9d8f', alpha=0.86, height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xticks([])
    for i, value in enumerate(values):
        ax.text(min(widths[i] + 0.03, 0.98), i, f'{int(value)}', va='center', fontsize=9)
    ax.spines[:].set_visible(False)
    ax.set_title('TB Care Cascade', fontsize=13, fontweight='bold', loc='left', pad=12)
    fig.patch.set_facecolor('#ffffff')
    return fig


def default_image() -> Image.Image:
    if DEFAULT_IMAGE_PATH.exists():
        return Image.open(DEFAULT_IMAGE_PATH).convert('RGB')
    canvas = np.zeros((620, 480, 3), dtype=np.uint8)
    canvas[:, :] = (28, 38, 42)
    cv2.putText(canvas, 'No study image available', (36, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (220, 226, 228), 2, cv2.LINE_AA)
    return Image.fromarray(canvas)


def viewer_image() -> Image.Image:
    raw = st.session_state.get('viewer_image_bytes')
    if raw:
        try:
            return Image.open(io.BytesIO(raw)).convert('RGB')
        except Exception:
            pass
    return default_image()


def filename_to_patient_name(filename: str) -> str:
    stem = Path(filename).stem.replace('_', ' ').replace('-', ' ')
    stem = re.sub(r'\s+', ' ', stem).strip()
    return stem.title() if stem else 'Unassigned Patient'


def parse_dicom_age(value: object) -> int:
    match = re.search(r'(\d+)', str(value or ''))
    return int(match.group(1)) if match else 0


def normalize_gender_label(value: object) -> str:
    text = str(value or '').strip().lower()
    if text in {'f', 'female'}:
        return 'Female'
    if text in {'m', 'male'}:
        return 'Male'
    return 'Other'


def decode_uploaded_image(filename: str, raw_bytes: bytes) -> tuple[Image.Image, dict[str, object]]:
    lower = filename.lower()
    metadata: dict[str, object] = {
        'patient_name': filename_to_patient_name(filename),
        'mrn': '',
        'age': 0,
        'gender': 'Other',
        'modality': 'CR',
        'kind': Path(filename).suffix.lower().lstrip('.') or 'png',
    }
    if lower.endswith('.dcm') or lower.endswith('.dicom'):
        try:
            import pydicom

            ds = pydicom.dcmread(io.BytesIO(raw_bytes), force=True)
            pixels = np.asarray(ds.pixel_array, dtype=np.float32)
            pixels = np.squeeze(pixels)
            if pixels.ndim == 3:
                pixels = pixels[0] if pixels.shape[0] not in (3, 4) else np.moveaxis(pixels, 0, -1)
            pixels = np.nan_to_num(pixels)
            pixels -= pixels.min()
            if float(pixels.max()) > 0:
                pixels /= pixels.max()
            pixels = (pixels * 255).clip(0, 255).astype(np.uint8)
            if str(getattr(ds, 'PhotometricInterpretation', '')).upper() == 'MONOCHROME1':
                pixels = 255 - pixels
            image = Image.fromarray(pixels).convert('RGB')
            metadata.update({
                'patient_name': str(getattr(ds, 'PatientName', '')).replace('^', ' ').strip() or metadata['patient_name'],
                'mrn': str(getattr(ds, 'PatientID', '')).strip(),
                'age': parse_dicom_age(getattr(ds, 'PatientAge', 0)),
                'gender': normalize_gender_label(getattr(ds, 'PatientSex', '')),
                'modality': str(getattr(ds, 'Modality', 'DX')).strip() or 'DX',
                'kind': 'dcm',
            })
            return image, metadata
        except Exception as exc:
            raise ValueError(f'DICOM decode failed for {filename}: {exc}') from exc

    image = Image.open(io.BytesIO(raw_bytes)).convert('RGB')
    return image, metadata


def load_study_image_blob(conn: sqlite3.Connection, study_id: int) -> bytes | None:
    row = conn.execute(f'SELECT image_blob FROM {INTERNAL_ANALYSIS_TABLE} WHERE id = ?', (study_id,)).fetchone()
    if row is None:
        return None
    return row['image_blob'] if isinstance(row, sqlite3.Row) else row[0]


def analysis_image_context() -> tuple[Image.Image, str, dict[str, object]]:
    upload_bytes = st.session_state.get('viewer_image_bytes')
    upload_name = st.session_state.get('viewer_upload_name')
    upload_meta = st.session_state.get('viewer_upload_meta')
    if upload_bytes and upload_name:
        try:
            image, metadata = decode_uploaded_image(upload_name, upload_bytes)
            return image, upload_name, dict(upload_meta or metadata)
        except Exception as exc:
            return default_image(), upload_name, {'error': str(exc)}

    fallback_name = DEFAULT_IMAGE_PATH.name if DEFAULT_IMAGE_PATH.exists() else 'Viewer placeholder'
    return default_image(), fallback_name, {}


def saved_study_image_context(conn: sqlite3.Connection, study: pd.Series) -> tuple[Image.Image, str, dict[str, object]]:
    study_id = int(study.get('study_id', 0) or 0)
    if study_id > 0:
        raw = load_study_image_blob(conn, study_id)
        image_name = str(study.get('image_name', '') or f'study_{study_id}')
        if raw:
            try:
                image, metadata = decode_uploaded_image(image_name or f'study_{study_id}', raw)
                return image, image_name, metadata
            except Exception:
                pass

    fallback_name = str(study.get('image_name') or (DEFAULT_IMAGE_PATH.name if DEFAULT_IMAGE_PATH.exists() else 'Viewer placeholder'))
    return default_image(), fallback_name, {}


def safe_slug(value: str) -> str:
    cleaned = re.sub(r'[^A-Za-z0-9._-]+', '_', value.strip())
    return cleaned.strip('._') or 'study'


def heuristic_scores(image: Image.Image) -> np.ndarray:
    arr = np.asarray(ImageOps.grayscale(image.resize((224, 224))), dtype=np.float32) / 255.0
    edge = cv2.Laplacian(arr, cv2.CV_32F).var()
    mean = arr.mean()
    std = arr.std()
    raw = np.array([1.2 + (0.52 - std) + mean * 0.35, 0.9 + std * 1.4 + max(0.0, 0.54 - mean) + edge * 0.02, 0.7 + (1.0 - mean) * 0.9 + std * 0.7, 0.55 + abs(0.5 - mean) * 1.1 + edge * 0.015], dtype=np.float32)
    raw = np.clip(raw, 0.05, None)
    return raw / raw.sum()


def heatmap(image: Image.Image) -> np.ndarray:
    gray = np.asarray(ImageOps.grayscale(image.resize((720, 720))), dtype=np.uint8)
    blur = cv2.GaussianBlur(gray, (0, 0), 9)
    detail = cv2.absdiff(gray, blur)
    heat = cv2.normalize(detail, None, 0, 255, cv2.NORM_MINMAX)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(base, 0.72, heat, 0.28, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def find_last_conv_layer(model):
    import tensorflow as tf

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            nested = find_last_conv_layer(layer)
            if nested is not None:
                return nested
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
        try:
            shape = layer.output_shape
        except Exception:
            shape = None
        if isinstance(shape, list):
            shape = shape[0] if shape else None
        if shape is not None and hasattr(shape, '__len__') and len(shape) == 4 and 'conv' in layer.name.lower():
            return layer
    return None


def resolve_gradcam_target_layer(model, model_kind: str):
    for layer_name in TARGET_LAYER_HINTS.get(model_kind, []):
        try:
            return model.get_layer(layer_name)
        except Exception:
            pass
        for layer in model.layers:
            if hasattr(layer, 'get_layer'):
                try:
                    return layer.get_layer(layer_name)
                except Exception:
                    pass
    return find_last_conv_layer(model)


def run_gradcam_forward(model, target_layer, batch):
    import tensorflow as tf

    for layer in model.layers:
        if not isinstance(layer, tf.keras.Model):
            continue
        try:
            nested_target = layer.get_layer(target_layer.name)
        except Exception:
            continue
        feature_model = tf.keras.models.Model(layer.inputs, [nested_target.output, layer.outputs[0]])
        with tf.GradientTape() as tape:
            conv_output, features = feature_model(batch, training=False)
            after_backbone = False
            x = features
            for outer_layer in model.layers:
                if outer_layer is layer:
                    after_backbone = True
                    continue
                if not after_backbone:
                    continue
                x = outer_layer(x, training=False)
            predictions = x
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        grads = tape.gradient(class_channel, conv_output)
        return conv_output, predictions, grads, pred_index

    grad_model = tf.keras.models.Model(model.inputs, [target_layer.output, model.outputs[0]])
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(batch, training=False)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_output)
    return conv_output, predictions, grads, pred_index


def blend_attention_map(image: Image.Image, normalized_heatmap: np.ndarray, alpha: float = 0.34) -> np.ndarray:
    base = np.asarray(image.convert('RGB'))
    heat_uint8 = np.uint8(np.clip(normalized_heatmap, 0, 1) * 255)
    colored = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(base, 1 - alpha, colored, alpha, 0)


def render_gradcam_overlay(image: Image.Image, normalized_heatmap: np.ndarray, display_label: str) -> np.ndarray:
    def thoracic_focus_prior(height: int, width: int) -> np.ndarray:
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
        x_mid = width / 2.0
        y_mid = height * 0.56
        thorax = np.exp(-((((xs - x_mid) / (width * 0.33)) ** 2) + (((ys - y_mid) / (height * 0.36)) ** 2)) * 1.6)

        left_shoulder = np.exp(-((((xs - width * 0.16) / (width * 0.18)) ** 2) + (((ys - height * 0.16) / (height * 0.14)) ** 2)) * 2.2)
        right_shoulder = np.exp(-((((xs - width * 0.84) / (width * 0.18)) ** 2) + (((ys - height * 0.16) / (height * 0.14)) ** 2)) * 2.2)
        left_axilla = np.exp(-((((xs - width * 0.18) / (width * 0.15)) ** 2) + (((ys - height * 0.31) / (height * 0.12)) ** 2)) * 2.0)
        right_axilla = np.exp(-((((xs - width * 0.82) / (width * 0.15)) ** 2) + (((ys - height * 0.31) / (height * 0.12)) ** 2)) * 2.0)

        prior = thorax * (1.0 - 0.72 * np.clip(left_shoulder + right_shoulder, 0, 1))
        prior *= (1.0 - 0.58 * np.clip(left_axilla + right_axilla, 0, 1))
        prior = cv2.GaussianBlur(prior.astype(np.float32), (0, 0), 17)
        return np.clip(prior, 0.12, 1.0)

    gray = np.asarray(image.convert('L'))
    base_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    smoothed = cv2.GaussianBlur(np.clip(normalized_heatmap, 0, 1).astype(np.float32), (0, 0), 11)
    center_prior = thoracic_focus_prior(smoothed.shape[0], smoothed.shape[1])
    focused_map = smoothed * (0.35 + 0.65 * center_prior)
    heat_uint8 = np.uint8(np.clip(focused_map, 0, 1) * 255)
    heat_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(base_rgb, 0.56, heat_rgb, 0.44, 0)

    threshold = max(0.46, float(np.percentile(focused_map, 89)))
    mask = np.uint8(focused_map >= threshold) * 255
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        dark_mask = gray < np.percentile(gray, 58)
        ranked: list[tuple[float, np.ndarray, tuple[int, int, int, int]]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area <= 120:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            contour_mask = np.zeros_like(mask, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)
            region = contour_mask.astype(bool)
            if not np.any(region):
                continue
            heat_score = float(focused_map[region].mean())
            dark_overlap = float(dark_mask[region].mean())
            center_score = float(center_prior[region].mean())
            score = heat_score * (0.45 + dark_overlap) * (0.5 + center_score) * np.sqrt(area)
            ranked.append((score, contour, (x, y, w, h)))

        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = ranked[:2]
        for _, contour, _ in selected:
            cv2.drawContours(overlay, [contour], -1, (255, 255, 255), 2)

        if selected:
            _, contour, (x, y, w, h) = selected[0]
            focus_x = x + w // 2
            focus_y = y + h // 2
            label = f'{display_label} focus'
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = max(0.55, min(0.82, overlay.shape[1] / 900.0))
            thickness = 2
            text_w, text_h = cv2.getTextSize(label, font, scale, thickness)[0]
            box_x = int(np.clip(x, 12, max(12, overlay.shape[1] - text_w - 24)))
            box_y = max(text_h + 18, y - 18)
            top_left = (box_x - 8, box_y - text_h - 10)
            bottom_right = (box_x + text_w + 8, box_y + 8)
            cv2.rectangle(overlay, top_left, bottom_right, (255, 255, 255), -1)
            cv2.putText(overlay, label, (box_x, box_y), font, scale, (38, 38, 38), thickness, cv2.LINE_AA)
            cv2.arrowedLine(overlay, (box_x + text_w // 2, bottom_right[1]), (focus_x, focus_y), (255, 255, 255), 2, tipLength=0.12)
            cv2.circle(overlay, (focus_x, focus_y), 6, (255, 255, 255), 2)
    return overlay


def gradcam_overlay_external(image: Image.Image, model_path: Path) -> tuple[np.ndarray | None, str | None, str | None]:
    if not BRAVE_ENV_PYTHON.exists():
        return None, None, f'External inference python was not found at {BRAVE_ENV_PYTHON}.'
    if not INFERENCE_WORKER_PATH.exists():
        return None, None, f'External inference worker was not found at {INFERENCE_WORKER_PATH}.'

    try:
        with tempfile.TemporaryDirectory() as td:
            image_path = Path(td) / 'input.png'
            overlay_path = Path(td) / 'gradcam_overlay.png'
            image.convert('RGB').save(image_path)
            completed = subprocess.run(
                [str(BRAVE_ENV_PYTHON), str(INFERENCE_WORKER_PATH), '--model', str(model_path), '--image', str(image_path), '--gradcam-output', str(overlay_path)],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if completed.returncode != 0:
                message = completed.stderr.strip() or completed.stdout.strip() or f'External Grad-CAM exited with code {completed.returncode}.'
                return None, None, message
            payload = json.loads(completed.stdout.strip())
            if not overlay_path.exists():
                return None, None, 'Grad-CAM overlay image was not created by the external worker.'
            overlay = np.asarray(Image.open(overlay_path).convert('RGB'))
            detail = str(payload.get('detail', f'Grad-CAM from {model_path.name}'))
            return overlay, detail, None
    except Exception as exc:
        return None, None, str(exc)


def gradcam_overlay(image: Image.Image, model_files: list[Path]) -> tuple[np.ndarray, str]:
    model_path = selected_model_path(model_files)
    if model_path is None:
        return heatmap(image), 'Fallback attention overlay shown because no .keras model is available.'

    external_overlay, external_detail, external_error = gradcam_overlay_external(image, model_path)
    if external_overlay is not None:
        return external_overlay, external_detail or f'Real Grad-CAM generated using {model_path.name}.'

    model, error = load_keras_model(str(model_path))
    if model is None:
        detail = f'Fallback attention overlay shown because model loading failed: {error}'
        if external_error:
            detail += f' | External Grad-CAM failed: {external_error}'
        return heatmap(image), detail

    try:
        import tensorflow as tf

        model_kind = infer_model_kind(model_path)
        target_layer = resolve_gradcam_target_layer(model, model_kind)
        if target_layer is None:
            return heatmap(image), f'Fallback attention overlay shown because no convolution layer was found in {model_path.name}.'

        input_shape = getattr(model, 'input_shape', (None, 224, 224, 3))
        height = int(input_shape[1] or 224)
        width = int(input_shape[2] or 224)
        batch = preprocess_for_model(image, model_kind, (width, height))
        model(batch, training=False)

        conv_output, predictions, grads, pred_index = run_gradcam_forward(model, target_layer, batch)
        if grads is None:
            return heatmap(image), f'Fallback attention overlay shown because Grad-CAM gradients were unavailable for {model_path.name}.'

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_output = conv_output[0]
        cam = tf.reduce_sum(conv_output * pooled_grads[tf.newaxis, tf.newaxis, :], axis=-1)
        cam = tf.maximum(cam, 0)
        cam_max = tf.reduce_max(cam)
        if float(cam_max) <= 0:
            return heatmap(image), f'Fallback attention overlay shown because Grad-CAM produced an empty map for {model_path.name}.'

        cam = cam / cam_max
        cam = tf.image.resize(cam[..., tf.newaxis], (image.size[1], image.size[0])).numpy().squeeze()
        display_label = RAW_TO_DISPLAY.get(RAW_MODEL_CLASSES[int(pred_index.numpy())], CLASS_NAMES[0])
        return render_gradcam_overlay(image, cam, display_label), f'Real Grad-CAM generated from layer "{target_layer.name}" using {model_path.name}.'
    except Exception as exc:
        detail = f'Fallback attention overlay shown because Grad-CAM failed: {exc}'
        if external_error:
            detail += f' | External Grad-CAM failed: {external_error}'
        return heatmap(image), detail


def summary(result: str, confidence: float) -> str:
    pct = confidence * 100
    if result == 'Normal':
        return f'No acute focal cardiopulmonary abnormality is highlighted on this chest X-ray. Cardiomediastinal silhouette appears within expected limits and no obvious pleural collection is emphasized by the model. AI confidence is {pct:.1f}%. Clinical correlation is recommended.'
    if result == 'Tuberculosis':
        return f'Upper-zone tuberculosis-suspicious radiographic changes are highlighted, with attention to possible fibronodular opacity or cavitary-pattern involvement. AI confidence is {pct:.1f}%. Recommend confirmatory laboratory testing and targeted clinical review. Clinical correlation is recommended.'
    if result == 'Pneumonia':
        return f'Patchy air-space opacity pattern consistent with pneumonia is highlighted, with likely focal or multifocal inflammatory consolidation. AI confidence is {pct:.1f}%. Consider infection-focused assessment and treatment planning. Clinical correlation is recommended.'
    return f'Bilateral viral-pattern chest opacity is highlighted, with peripheral or diffuse interstitial-airspace change that may fit COVID-19 patterning. AI confidence is {pct:.1f}%. Correlate with symptoms, exposure history, and laboratory findings. Clinical correlation is recommended.'


def predict_scores(image: Image.Image, model_files: list[Path]) -> tuple[np.ndarray, dict[str, str]]:
    model_path = selected_model_path(model_files)
    if model_path is None:
        return heuristic_scores(image), {'mode': 'demo', 'engine': 'Feature heuristic', 'detail': 'No .keras model found'}

    model, error = load_keras_model(str(model_path))
    if model is None:
        external_probs, external_backend, external_error = predict_scores_external(image, model_path)
        if external_probs is not None and external_backend is not None:
            return external_probs, external_backend
        detail = f'Load failed: {error}'
        if external_error:
            detail += f' | External inference failed: {external_error}'
        return heuristic_scores(image), {'mode': 'fallback', 'engine': model_path.name, 'detail': detail}

    try:
        input_shape = getattr(model, 'input_shape', (None, 224, 224, 3))
        height = int(input_shape[1] or 224)
        width = int(input_shape[2] or 224)
        model_kind = infer_model_kind(model_path)
        batch = preprocess_for_model(image, model_kind, (width, height))
        raw_output = np.asarray(model.predict(batch, verbose=0)[0], dtype=np.float32).flatten()
        if raw_output.size != 4:
            return heuristic_scores(image), {'mode': 'fallback', 'engine': model_path.name, 'detail': f'Unsupported output size: {raw_output.size}'}
        if not np.isfinite(raw_output).all():
            return heuristic_scores(image), {'mode': 'fallback', 'engine': model_path.name, 'detail': 'Model output contained invalid values'}
        probs = raw_output if np.isclose(raw_output.sum(), 1.0, atol=0.05) else softmax(raw_output)
        loader_mode = getattr(model, '_brave_loader_mode', 'standard')
        backend_mode = 'real-compat' if loader_mode == 'compat' else 'real'
        backend_detail = f'{model_kind} compatibility inference' if loader_mode == 'compat' else f'{model_kind} inference'
        return align_probabilities(probs), {'mode': backend_mode, 'engine': model_path.name, 'detail': backend_detail}
    except Exception as exc:
        return heuristic_scores(image), {'mode': 'fallback', 'engine': model_path.name, 'detail': f'Inference failed: {exc}'}


def report_bytes(
    study: pd.Series,
    result: str,
    confidence: float,
    impression: str,
    note: str,
    probs: np.ndarray,
    image: Image.Image,
    backend: dict[str, str],
    image_source: str,
) -> tuple[bytes, str, str]:
    patient_name = str(study.get('patient_name', 'Unknown patient')).strip() or 'Unknown patient'
    mrn = str(study.get('mrn', '')).strip() or 'NO-MRN'
    report_stamp = now_gmt3().strftime('%Y%m%d_%H%M%S')
    report_name = f"{safe_slug(mrn)}_{safe_slug(patient_name)}_{report_stamp}_AI_Radiology_Assistant_report.pdf"
    study_time_label = format_gmt3(study.get('study_date'), '%d %b %Y')
    finalized_label = format_gmt3(utc_now_sql())
    note_text = str(note or '').strip() or 'No additional note provided.'
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Image as PdfImage
        from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

        buffer = io.BytesIO()
        document = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=10 * mm, leftMargin=10 * mm, topMargin=16 * mm, bottomMargin=12 * mm)
        content_width = A4[0] - document.leftMargin - document.rightMargin
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='BrandTitle', fontName='Helvetica-Bold', fontSize=22, textColor=colors.HexColor('#15333a'), leading=26))
        styles.add(ParagraphStyle(name='BrandSub', fontName='Helvetica', fontSize=10.5, textColor=colors.HexColor('#5f7076'), leading=13))
        styles.add(ParagraphStyle(name='BodyCopy', fontName='Helvetica', fontSize=10.8, textColor=colors.HexColor('#1f2d31'), leading=16))
        styles.add(ParagraphStyle(name='DiagnosisTitle', fontName='Helvetica-Bold', fontSize=12, textColor=colors.white, leading=15, alignment=1))
        styles.add(ParagraphStyle(name='DiagnosisValue', fontName='Helvetica-Bold', fontSize=18, textColor=colors.HexColor('#15333a'), leading=22, alignment=1))
        styles.add(ParagraphStyle(name='SectionHeading', fontName='Helvetica-Bold', fontSize=12, textColor=colors.HexColor('#15333a'), leading=15))
        styles.add(ParagraphStyle(name='ImagePageTitle', fontName='Helvetica-Bold', fontSize=16, textColor=colors.HexColor('#15333a'), leading=20, alignment=1))

        def draw_pdf_chrome(canvas, doc) -> None:
            page_width, page_height = A4
            canvas.saveState()
            canvas.setStrokeColor(colors.HexColor('#d6e3e0'))
            logo_x = doc.leftMargin
            logo_y = page_height - 16 * mm
            logo_size = 9 * mm
            canvas.setFillColor(colors.HexColor('#12373d'))
            canvas.roundRect(logo_x, logo_y, logo_size, logo_size, 2.4 * mm, stroke=0, fill=1)
            canvas.setFillColor(colors.white)
            canvas.setFont('Helvetica-Bold', 11)
            canvas.drawCentredString(logo_x + (logo_size / 2), logo_y + 2.5 * mm, 'B')
            canvas.setFillColor(colors.HexColor('#d89b3f'))
            canvas.circle(logo_x + logo_size - 1.6 * mm, logo_y + logo_size - 1.6 * mm, 0.95 * mm, stroke=0, fill=1)
            text_x = logo_x + logo_size + 3 * mm
            canvas.setFillColor(colors.HexColor('#15333a'))
            canvas.setFont('Helvetica-Bold', 10)
            canvas.drawString(text_x, page_height - 12 * mm, BRANDING['hospital_name'])
            canvas.line(doc.leftMargin, page_height - 14 * mm, page_width - doc.rightMargin, page_height - 14 * mm)
            canvas.line(doc.leftMargin, 14 * mm, page_width - doc.rightMargin, 14 * mm)
            canvas.setFont('Helvetica', 6.2)
            canvas.drawString(doc.leftMargin, 9 * mm, f"{BRANDING['hospital_name']} | {DISCLAIMER_TEXT}")
            canvas.drawRightString(page_width - doc.rightMargin, 9 * mm, f'Page {canvas.getPageNumber()}')
            canvas.restoreState()

        story = [
            Paragraph(BRANDING['hospital_name'], styles['BrandTitle']),
            Paragraph(BRANDING['contact'], styles['BrandSub']),
            Spacer(1, 10),
        ]
        table_data = [
            ['Patient', patient_name],
            ['MRN', mrn],
            ['Age / Gender', f"{study['age']} / {study['gender']}"],
            ['Examination', f"Chest X-ray ({study['modality']}) | {study_time_label}"],
            ['Image Source', image_source],
        ]
        details = Table(table_data, colWidths=[42 * mm, content_width - (42 * mm)], rowHeights=[13 * mm] * len(table_data))
        details.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f4f8f7')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1f2d31')),
            ('GRID', (0, 0), (-1, -1), 0.35, colors.HexColor('#d7e4e1')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('PADDING', (0, 0), (-1, -1), 7),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        diagnosis_block = Table(
            [
                [Paragraph('IMPRESSION', styles['DiagnosisTitle'])],
                [Paragraph(f"{result} ({confidence * 100:.1f}% confidence)", styles['DiagnosisValue'])],
            ],
            colWidths=[content_width],
            rowHeights=[13 * mm, 29 * mm],
        )
        diagnosis_block.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#12373d')),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#eef6f4')),
            ('BOX', (0, 0), (-1, -1), 0.7, colors.HexColor('#cddfdc')),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.extend([details, Spacer(1, 12), diagnosis_block, Spacer(1, 10)])

        note_block = Table(
            [
                [Paragraph('Findings', styles['SectionHeading'])],
                [Paragraph(note_text.replace('\n', '<br/>'), styles['BodyCopy'])],
            ],
            colWidths=[content_width],
            rowHeights=[12 * mm, 42 * mm],
        )
        note_block.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f4f8f7')),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('BOX', (0, 0), (-1, -1), 0.55, colors.HexColor('#d7e4e1')),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        story.extend([note_block, Spacer(1, 12)])

        signature_table = Table(
            [
                ['Radiologist Signature', 'Report Finalized'],
                ['______________________________', finalized_label],
            ],
            colWidths=[content_width / 2, content_width / 2],
            rowHeights=[13 * mm, 22 * mm],
        )
        signature_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f4f8f7')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1f2d31')),
            ('GRID', (0, 0), (-1, -1), 0.35, colors.HexColor('#d7e4e1')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('PADDING', (0, 0), (-1, -1), 7),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))

        story.extend([
            Paragraph('Signature and Verification', styles['SectionHeading']),
            Spacer(1, 8),
            signature_table,
            Spacer(1, 8),
        ])

        image_buffer = io.BytesIO()
        rgb_image = image.convert('RGB')
        rgb_image.save(image_buffer, format='PNG')
        image_buffer.seek(0)
        max_width = A4[0] - document.leftMargin - document.rightMargin
        max_height = A4[1] - document.topMargin - document.bottomMargin - 1 * mm
        img_width, img_height = rgb_image.size
        width_ratio = max_width / float(img_width)
        height_ratio = max_height / float(img_height)
        scale = min(width_ratio, height_ratio)
        scaled_width = img_width * scale
        scaled_height = img_height * scale
        vertical_pad = max(0, (max_height - scaled_height) / 2)
        horizontal_pad = max(0, (max_width - scaled_width) / 2)

        story.extend([
            PageBreak(),
            Spacer(1, vertical_pad),
            Table(
                [[PdfImage(image_buffer, width=scaled_width, height=scaled_height)]],
                colWidths=[max_width],
            ),
        ])
        story[-1].setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), horizontal_pad),
            ('RIGHTPADDING', (0, 0), (-1, -1), horizontal_pad),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ]))
        document.build(story, onFirstPage=draw_pdf_chrome, onLaterPages=draw_pdf_chrome)
        return buffer.getvalue(), report_name, 'application/pdf'
    except Exception:
        html_image = io.BytesIO()
        image.convert('RGB').save(html_image, format='PNG')
        image_base64 = base64.b64encode(html_image.getvalue()).decode('ascii')
        note_html = note_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br/>')
        html = f"<html><body style='font-family: Arial, sans-serif; padding: 24px; color: #1f2d31;'><h1 style='margin-bottom:8px;'>{BRANDING['hospital_name']}</h1><p style='margin-bottom:16px;'>{BRANDING['contact']}</p><table style='width:100%; border-collapse:collapse; margin-top:8px; margin-bottom:18px;'><tr><td style='border:1px solid #d7e4e1; padding:12px; font-weight:bold; background:#f4f8f7;'>Patient</td><td style='border:1px solid #d7e4e1; padding:12px;'>{patient_name}</td></tr><tr><td style='border:1px solid #d7e4e1; padding:12px; font-weight:bold; background:#f4f8f7;'>MRN</td><td style='border:1px solid #d7e4e1; padding:12px;'>{mrn}</td></tr><tr><td style='border:1px solid #d7e4e1; padding:12px; font-weight:bold; background:#f4f8f7;'>Age / Gender</td><td style='border:1px solid #d7e4e1; padding:12px;'>{study['age']} / {study['gender']}</td></tr><tr><td style='border:1px solid #d7e4e1; padding:12px; font-weight:bold; background:#f4f8f7;'>Examination</td><td style='border:1px solid #d7e4e1; padding:12px;'>Chest X-ray ({study['modality']}) | {study_time_label}</td></tr><tr><td style='border:1px solid #d7e4e1; padding:12px; font-weight:bold; background:#f4f8f7;'>Image Source</td><td style='border:1px solid #d7e4e1; padding:12px;'>{image_source}</td></tr></table><h3 style='margin:20px 0 10px;'>IMPRESSION</h3><div style='border:1px solid #cddfdc; background:#eef6f4; padding:22px; font-weight:bold; text-align:center; font-size:20px; margin-bottom:12px;'>{result} ({confidence * 100:.1f}% confidence)</div><h3 style='margin:16px 0 10px;'>Findings</h3><div style='border:1px solid #d7e4e1; background:#ffffff; padding:18px; line-height:1.7; min-height:120px; margin-bottom:14px;'>{note_html}</div><h3 style='margin:18px 0 10px;'>Signature and Verification</h3><table style='width:100%; border-collapse:collapse; margin-bottom:10px;'><tr><td style='border:1px solid #d7e4e1; padding:12px; font-weight:bold; background:#f4f8f7;'>Radiologist Signature</td><td style='border:1px solid #d7e4e1; padding:12px; font-weight:bold; background:#f4f8f7;'>Report Finalized</td></tr><tr><td style='border:1px solid #d7e4e1; padding:18px;'>______________________________</td><td style='border:1px solid #d7e4e1; padding:18px;'>{finalized_label}</td></tr></table><div style='position:fixed; left:24px; right:24px; bottom:12px; font-size:10px; color:#44585d;'><span>{BRANDING['hospital_name']} | {DISCLAIMER_TEXT}</span></div><div style='page-break-before: always;'></div><div style='text-align:center;'><img src='data:image/png;base64,{image_base64}' alt='Study image' style='width:100%; height:95vh; object-fit:contain;'/></div></body></html>"
        return html.encode('utf-8'), report_name.replace('.pdf', '.html'), 'text/html'


def save_patient(conn: sqlite3.Connection, name: str, age: int, gender: str) -> tuple[int, str]:
    normalized_name = name.strip() or 'Unassigned Patient'
    normalized_gender = normalize_gender_label(gender)
    conn.execute('BEGIN IMMEDIATE')
    try:
        ensure_patient_mrn_sequence(conn)
        cursor = None
        generated_mrn = ''
        for _ in range(20):
            generated_mrn = allocate_next_mrn(conn)
            try:
                cursor = conn.execute(
                    'INSERT INTO patients (name, age, gender, mrn) VALUES (?, ?, ?, ?)',
                    (normalized_name, int(age), normalized_gender, generated_mrn),
                )
                break
            except sqlite3.IntegrityError as exc:
                if 'patients.mrn' not in str(exc) and 'idx_patients_mrn_unique' not in str(exc):
                    raise
        if cursor is None:
            raise RuntimeError('Automatic MRN generation could not allocate a unique MRN.')
        insert_hospital_record_registration(conn, generated_mrn, normalized_name, int(age), normalized_gender)
        conn.commit()
        return int(cursor.lastrowid), generated_mrn
    except Exception:
        conn.rollback()
        raise


def find_or_create_patient(conn: sqlite3.Connection, name: str, age: int, gender: str, mrn: str) -> int:
    normalized_name = name.strip() or 'Unassigned Patient'
    normalized_mrn = mrn.strip()
    normalized_gender = normalize_gender_label(gender)
    if normalized_mrn:
        row = conn.execute('SELECT id FROM patients WHERE mrn = ? ORDER BY id DESC LIMIT 1', (normalized_mrn,)).fetchone()
        if row:
            conn.execute('UPDATE patients SET name = ?, age = ?, gender = ? WHERE id = ?', (normalized_name, int(age), normalized_gender, int(row['id'])))
            sync_patient_identity_references(conn, int(row['id']), normalized_mrn, normalized_name, int(age), normalized_gender, normalized_mrn)
            conn.commit()
            return int(row['id'])
    row = conn.execute('SELECT id FROM patients WHERE lower(name) = lower(?) ORDER BY id DESC LIMIT 1', (normalized_name,)).fetchone()
    if row:
        existing = conn.execute('SELECT COALESCE(mrn, "") AS mrn FROM patients WHERE id = ?', (int(row['id']),)).fetchone()
        resolved_mrn = normalized_mrn or str(existing['mrn'] or '').strip()
        if not resolved_mrn:
            conn.execute('BEGIN IMMEDIATE')
            try:
                ensure_patient_mrn_sequence(conn)
                for _ in range(20):
                    resolved_mrn = allocate_next_mrn(conn)
                    try:
                        conn.execute('UPDATE patients SET name = ?, age = ?, gender = ?, mrn = ? WHERE id = ?', (normalized_name, int(age), normalized_gender, resolved_mrn, int(row['id'])))
                        break
                    except sqlite3.IntegrityError as exc:
                        if 'patients.mrn' not in str(exc) and 'idx_patients_mrn_unique' not in str(exc):
                            raise
                if not resolved_mrn:
                    raise RuntimeError('Automatic MRN generation could not allocate a unique MRN.')
                sync_patient_identity_references(conn, int(row['id']), resolved_mrn, normalized_name, int(age), normalized_gender, str(existing['mrn'] or ''))
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            return int(row['id'])
        conn.execute('UPDATE patients SET name = ?, age = ?, gender = ?, mrn = ? WHERE id = ?', (normalized_name, int(age), normalized_gender, resolved_mrn, int(row['id'])))
        sync_patient_identity_references(conn, int(row['id']), resolved_mrn, normalized_name, int(age), normalized_gender, str(existing['mrn'] or ''))
        conn.commit()
        return int(row['id'])
    if normalized_mrn:
        cursor = conn.execute('INSERT INTO patients (name, age, gender, mrn) VALUES (?, ?, ?, ?)', (normalized_name, int(age), normalized_gender, normalized_mrn))
        insert_hospital_record_registration(conn, normalized_mrn, normalized_name, int(age), normalized_gender)
        conn.commit()
        return int(cursor.lastrowid)
    patient_id, _ = save_patient(conn, normalized_name, int(age), normalized_gender)
    return patient_id


def save_analysis(
    conn: sqlite3.Connection,
    patient_id: int,
    result: str,
    confidence: float,
    note: str,
    priority: str,
    modality: str = 'CR',
    source_site: str = '',
    image_blob: bytes | None = None,
    image_name: str = '',
    image_kind: str = '',
    ai_engine: str = '',
    status: str = 'Completed',
) -> int:
    created_utc = utc_now_sql()
    prediction_time_label = format_gmt3(created_utc)
    patient = conn.execute('SELECT COALESCE(mrn, \'\') AS mrn, COALESCE(name, \'Unassigned patient\') AS name, COALESCE(age, 0) AS age, COALESCE(gender, \'Unknown\') AS gender FROM patients WHERE id = ?', (int(patient_id),)).fetchone()
    cursor = conn.execute(
        f'INSERT INTO {INTERNAL_ANALYSIS_TABLE} (patient_id, result, confidence, date, status, priority, modality, source_site, report_summary, image_blob, image_name, image_kind, ai_engine, patient_mrn, patient_name, patient_age, patient_gender) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
        (
            patient_id,
            result,
            confidence,
            created_utc,
            status,
            priority,
            modality,
            source_site or BRANDING['hospital_name'],
            note,
            image_blob,
            image_name,
            image_kind,
            ai_engine,
            '' if patient is None else str(patient['mrn'] or ''),
            'Unassigned patient' if patient is None else str(patient['name'] or 'Unassigned patient'),
            0 if patient is None else int(patient['age'] or 0),
            'Unknown' if patient is None else str(patient['gender'] or 'Unknown'),
        ),
    )
    study_id = int(cursor.lastrowid)
    conn.execute(
        f'UPDATE {INTERNAL_ANALYSIS_TABLE} SET prediction_id = ?, prediction_timestamp = ?, image_id = ?, image_path = ?, upload_timestamp = ? WHERE id = ?',
        (
            f'PRED-{study_id:06d}',
            prediction_time_label,
            f'IMG-{study_id:06d}',
            image_name or f'study_{study_id}',
            prediction_time_label,
            study_id,
        ),
    )
    update_hospital_record_prediction(
        conn,
        patient_id,
        result,
        confidence,
        status,
        priority,
        modality,
        image_name or f'study_{study_id}',
        image_name or f'study_{study_id}',
        prediction_time_label,
        note,
    )
    conn.commit()
    return study_id


def update_analysis_record(
    conn: sqlite3.Connection,
    study_id: int,
    result: str,
    confidence: float,
    note: str,
    priority: str,
    ai_engine: str,
) -> None:
    conn.execute(
        f'UPDATE {INTERNAL_ANALYSIS_TABLE} SET result = ?, confidence = ?, report_summary = ?, priority = ?, ai_engine = ?, status = ? WHERE id = ?',
        (result, confidence, note, priority, ai_engine, 'Completed', int(study_id)),
    )
    row = conn.execute(f'SELECT patient_id, COALESCE(image_path, image_name, ?) AS image_path, COALESCE(prediction_timestamp, ?) AS prediction_timestamp FROM {INTERNAL_ANALYSIS_TABLE} WHERE id = ?', (f'study_{int(study_id)}', format_gmt3(utc_now_sql()), int(study_id))).fetchone()
    if row is not None:
        analysis_row = conn.execute(f'SELECT status, priority, modality, COALESCE(image_name, ?) AS image_name, COALESCE(image_path, image_name, ?) AS image_path FROM {INTERNAL_ANALYSIS_TABLE} WHERE id = ?', (f'study_{int(study_id)}', f'study_{int(study_id)}', int(study_id))).fetchone()
        update_hospital_record_prediction(
            conn,
            int(row['patient_id']),
            result,
            confidence,
            '' if analysis_row is None else str(analysis_row['status'] or ''),
            priority,
            '' if analysis_row is None else str(analysis_row['modality'] or ''),
            '' if analysis_row is None else str(analysis_row['image_name'] or f'study_{int(study_id)}'),
            '' if analysis_row is None else str(analysis_row['image_path'] or f'study_{int(study_id)}'),
            str(row['prediction_timestamp'] or format_gmt3(utc_now_sql())),
            note,
        )
    conn.commit()


def clear_viewer_upload() -> None:
    st.session_state.viewer_image_bytes = None
    st.session_state.viewer_upload_name = None
    st.session_state.viewer_upload_meta = None
    st.session_state.analysis_result = None
    st.session_state.analysis_busy = False
    st.session_state.analysis_scroll_to_result = False
    st.session_state.report_draft = ''
    st.session_state.viewer_gradcam_cache = {}
    st.session_state.analysis_uploader_token = int(st.session_state.get('analysis_uploader_token', 0)) + 1


def batch_intake_frame(uploaded_files: list) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for upload in uploaded_files:
        try:
            _, metadata = decode_uploaded_image(upload.name, upload.getvalue())
        except Exception:
            metadata = {
                'patient_name': filename_to_patient_name(upload.name),
                'mrn': '',
                'age': 0,
                'gender': 'Other',
                'modality': 'CR',
            }
        rows.append({
            'File': upload.name,
            'Patient Name': str(metadata.get('patient_name') or filename_to_patient_name(upload.name)),
            'MRN': str(metadata.get('mrn') or ''),
            'Age': int(metadata.get('age') or 0),
            'Gender': str(metadata.get('gender') or 'Other'),
            'Modality': str(metadata.get('modality') or 'CR'),
        })
    return pd.DataFrame(rows)


def select_study(studies: pd.DataFrame) -> pd.Series:
    if st.session_state.selected_study_id is None:
        st.session_state.selected_study_id = int(studies.iloc[0]['study_id'])
        return studies.iloc[0]
    match = studies[studies['study_id'].astype(int) == int(st.session_state.selected_study_id)]
    return match.iloc[0] if not match.empty else studies.iloc[0]


def viewer_gradcam_cache_key(study_id: int, image_source: str, model_files: list[Path]) -> str:
    model_path = selected_model_path(model_files)
    model_name = model_path.name if model_path is not None else 'no-model'
    return f"{int(study_id)}|{image_source}|{model_name}"


def stored_probabilities(predicted: str, confidence: float) -> np.ndarray:
    probs = np.full(len(CLASS_NAMES), max(0.0, (1.0 - float(confidence))) / max(len(CLASS_NAMES) - 1, 1), dtype=np.float32)
    if predicted in CLASS_NAMES:
        probs[CLASS_NAMES.index(predicted)] = float(confidence)
    return probs / probs.sum() if probs.sum() > 0 else np.full(len(CLASS_NAMES), 1.0 / len(CLASS_NAMES), dtype=np.float32)

def dashboard(patients: pd.DataFrame, studies: pd.DataFrame, demo_mode: bool) -> None:
    section('Dashboard', '', '')
    filt = st.columns([2.2, 1.3, 1.3, 1.2])
    with filt[0]:
        st.text_input('Search patient, MRN, or site', placeholder='Search live worklist')
    with filt[1]:
        st.selectbox('Date window', ['Last 30 days', 'Last 7 days', 'Today'], index=0)
    with filt[2]:
        st.selectbox('Facility', sorted(studies['source_site'].astype(str).unique()), index=0)
    with filt[3]:
        st.selectbox('Modality', sorted(studies['modality'].astype(str).unique()), index=0)
    render_metrics(metrics(demo_mode, studies))
    trend = weekly(studies, demo_mode)
    age = patients['age'].fillna(0).astype(int).map(bucket_age).value_counts().sort_index()
    gender = patients['gender'].fillna('Unknown').value_counts()
    result_mix = studies['result'].value_counts().reindex(CLASS_NAMES, fill_value=0)
    top = st.columns([1.55, 1.05])
    with top[0]:
        section('Weekly Trends', '', '')
        st.pyplot(line_chart(trend), use_container_width=True)
    with top[1]:
        section('Site Performance', '', '')
        st.pyplot(site_chart(studies), use_container_width=True)
    donuts = st.columns(3)
    with donuts[0]:
        st.pyplot(donut(age, 'Age Distribution', ['#78c6c0', '#273d78', '#e2b39f', '#a8a4c3', '#d49739']), use_container_width=True)
    with donuts[1]:
        st.pyplot(donut(gender, 'Gender Distribution', ['#3397d1', '#85bf42', '#d6d6d6']), use_container_width=True)
    with donuts[2]:
        st.pyplot(donut(result_mix, 'Result Mix', ['#1b998b', '#d58b3b', '#b44b45', '#6c8aa6', '#cad7d4']), use_container_width=True)


def worklist(studies: pd.DataFrame, demo_mode: bool) -> None:
    section('Worklist', '', '')
    filt = st.columns([2.2, 1.2, 1.2, 1.2])
    with filt[0]:
        search = st.text_input('Find patient or MRN', placeholder='Search the queue')
    with filt[1]:
        status_f = st.selectbox('Status', ['All'] + sorted(studies['status'].astype(str).unique().tolist()))
    with filt[2]:
        result_f = st.selectbox('AI result', ['All'] + sorted(studies['result'].astype(str).unique().tolist()))
    with filt[3]:
        priority_f = st.selectbox('Priority', ['All'] + sorted(studies['priority'].astype(str).unique().tolist()))
    filtered = studies.copy()
    if search:
        filtered = filtered[filtered['patient_name'].str.contains(search, case=False) | filtered['mrn'].str.contains(search, case=False)]
    if status_f != 'All':
        filtered = filtered[filtered['status'] == status_f]
    if result_f != 'All':
        filtered = filtered[filtered['result'] == result_f]
    if priority_f != 'All':
        filtered = filtered[filtered['priority'] == priority_f]
    section('Study Queue', '', '')
    if filtered.empty:
        st.info('No studies match the current filters.')
        return
    for _, study in filtered.sort_values('study_date', ascending=False).iterrows():
        cols = st.columns([3.4, 1.35, 1.55, 1.0])
        with cols[0]:
            st.markdown(f"<div class='study-card'><div class='study-title'>{study['patient_name']}</div><div class='study-meta'>{format_gmt3(study['study_date'], '%d %b %Y %I:%M %p Ethiopia Time')} | {study['gender']} | {int(study['age'])}Y | {study['mrn']} | {study['source_site']}</div><div class='study-body'>{study['report_summary']}</div></div>", unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f"<div class='study-card'><div class='mini'>Workflow</div>{pill(study['status'], 'neutral')}{pill(study['priority'], priority_tone(study['priority']))}</div>", unsafe_allow_html=True)
        with cols[2]:
            st.markdown(f"<div class='study-card'><div class='mini'>AI Summary</div>{pill(study['result'], result_tone(study['result']))}<div class='study-body'>Confidence: <strong>{study['confidence'] * 100:.1f}%</strong><br>Modality: <strong>{study['modality']}</strong></div></div>", unsafe_allow_html=True)
        with cols[3]:
            if st.button('Open study', key=f"open_{study['study_id']}", use_container_width=True):
                st.session_state.selected_study_id = int(study['study_id'])
                clear_viewer_upload()
                st.session_state.page = 'Study Viewer'
                rerun()

def ai_analysis(conn: sqlite3.Connection, studies: pd.DataFrame, demo_mode: bool, model_files: list[Path]) -> None:
    section('AI Analysis', '', '')
    patient = active_patient(conn)
    _, center, _ = st.columns([0.25, 3.5, 0.25])
    with center:
        if patient is None:
            st.warning('Register a patient first in Registry before running AI Analysis.')
            if st.button('Open Registry', use_container_width=True):
                st.session_state.page = 'Registry'
                rerun()
            return

        st.markdown(
            "<div class='upload-panel'><p class='upload-title'>Register patient before analysis</p><p class='upload-copy'>Upload Chest X-ray image</p></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='analysis-panel'><div class='study-title'>{patient['name']}</div><div class='study-meta'>MRN {patient['mrn'] or 'MRN-UNASSIGNED'} | {int(patient['age'])} years | {patient['gender']}</div></div>",
            unsafe_allow_html=True,
        )
        upload = st.file_uploader(
            'Browse files',
            type=['png', 'jpg', 'jpeg', 'dcm', 'dicom'],
            label_visibility='collapsed',
            key=f"analysis_upload_{st.session_state.get('analysis_uploader_token', 0)}",
        )
        if upload is not None:
            raw_bytes = upload.getvalue()
            changed_upload = upload.name != st.session_state.get('viewer_upload_name') or raw_bytes != st.session_state.get('viewer_image_bytes')
            st.session_state.viewer_image_bytes = raw_bytes
            st.session_state.viewer_upload_name = upload.name
            if changed_upload:
                st.session_state.analysis_result = None
                st.session_state.analysis_busy = False
                st.session_state.report_draft = ''
            try:
                _, upload_metadata = decode_uploaded_image(upload.name, raw_bytes)
                st.session_state.viewer_upload_meta = upload_metadata
            except Exception as exc:
                st.session_state.viewer_upload_meta = {'error': str(exc)}
        image, image_source, image_meta = analysis_image_context()
        if st.session_state.get('viewer_image_bytes') is None:
            st.session_state.analysis_busy = False
            return

        image_kind = str(image_meta.get('kind') or Path(image_source).suffix.lower().lstrip('.') or 'image').upper()
        st.markdown(
            f"<div class='analysis-panel'><div class='study-title'>{Path(image_source).name}</div><div class='study-meta'>{image_kind}</div></div>",
            unsafe_allow_html=True,
        )
        st.image(image, use_container_width=True)
        if image_meta.get('error'):
            st.error(str(image_meta['error']))
        controls = st.columns(2)
        with controls[0]:
            if st.button('Clear file', use_container_width=True, disabled=st.session_state.get('viewer_image_bytes') is None):
                clear_viewer_upload()
                st.session_state.analysis_busy = False
                rerun()
        with controls[1]:
            analyze_slot = st.empty()
            analysis_ready = st.session_state.get('viewer_image_bytes') is not None and not bool(image_meta.get('error'))
            analyze_clicked = False
            if st.session_state.get('analysis_busy') and analysis_ready:
                analyze_slot.markdown(
                    "<div class='busy-button'><span class='busy-spinner'></span><span>Analyzing...</span></div>",
                    unsafe_allow_html=True,
                )
            else:
                analyze_clicked = analyze_slot.button('Analyze', use_container_width=True, disabled=not analysis_ready)
        if analyze_clicked:
            st.session_state.analysis_busy = True
            rerun()
        if st.session_state.get('analysis_busy') and st.session_state.get('viewer_image_bytes') is not None and not bool(image_meta.get('error')):
            probs, backend = predict_scores(image, model_files)
            top = int(np.argmax(probs))
            predicted = CLASS_NAMES[top]
            confidence = float(probs[top])
            st.session_state.analysis_result = {
                'predicted': predicted,
                'confidence': confidence,
                'probs': probs.tolist(),
                'backend': backend,
                'image_source': image_source,
            }
            st.session_state.report_draft = summary(predicted, confidence)
            st.session_state.analysis_scroll_to_result = True
            st.session_state.analysis_busy = False
            rerun()
        analysis_state = st.session_state.get('analysis_result')
        if analysis_state and analysis_state.get('image_source') != image_source:
            analysis_state = None
            st.session_state.analysis_result = None
        if analysis_state:
            predicted = str(analysis_state['predicted'])
            confidence = float(analysis_state['confidence'])
            probs = np.array(analysis_state['probs'], dtype=np.float32)
            backend = dict(analysis_state['backend'])
            if backend.get('mode') in {'fallback', 'demo'}:
                st.error(
                    'Real model inference is not active. The app is using fallback scoring. '
                    'Run the app with C:/Users/hp/Desktop/Brave_env/Scripts/python.exe -m streamlit run src/main.py'
                )

            st.markdown("<div id='analysis-result-anchor'></div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='result-panel'><div class='result-head'><div><p class='result-title'>{predicted}</p><div class='result-sub'>Confidence {confidence * 100:.1f}%</div></div>{pill(predicted, result_tone(predicted))}</div><div class='study-body'>Image: <strong>{Path(image_source).name}</strong></div></div>",
                unsafe_allow_html=True,
            )
            if st.session_state.get('analysis_scroll_to_result'):
                components.html(
                    """
                    <script>
                    const target = window.parent.document.getElementById('analysis-result-anchor');
                    if (target) {
                      target.scrollIntoView({behavior: 'smooth', block: 'start'});
                    }
                    </script>
                    """,
                    height=0,
                )
                st.session_state.analysis_scroll_to_result = False
            note = st.text_area('Findings', value=st.session_state.get('report_draft') or summary(predicted, confidence), height=170)
            export_study = pd.Series(
                {
                    'patient_name': str(patient['name']).strip() or 'Unassigned patient',
                    'mrn': str(patient['mrn']).strip() or 'MRN-UNASSIGNED',
                    'age': int(patient['age']),
                    'gender': normalize_gender_label(patient['gender']),
                    'modality': str((st.session_state.get('viewer_upload_meta') or {}).get('modality') or 'CR'),
                    'source_site': BRANDING['hospital_name'],
                    'study_date': now_gmt3(),
                }
            )
            actions = st.columns(2)
            with actions[0]:
                if st.button('Save to worklist', use_container_width=True, disabled=st.session_state.get('viewer_image_bytes') is None):
                    st.session_state.report_draft = note
                    upload_bytes = st.session_state.get('viewer_image_bytes')
                    upload_name = st.session_state.get('viewer_upload_name') or image_source
                    upload_kind = str((st.session_state.get('viewer_upload_meta') or {}).get('kind') or Path(upload_name).suffix.lower().lstrip('.') or 'png')
                    patient_id = int(patient['id'])
                    new_study_id = save_analysis(conn, patient_id, predicted, confidence, note, 'High' if predicted != 'Normal' else 'Routine', modality=export_study['modality'], source_site=export_study['source_site'], image_blob=upload_bytes, image_name=upload_name, image_kind=upload_kind, ai_engine=backend['engine'], status='Completed')
                    clear_viewer_upload()
                    st.session_state.selected_study_id = new_study_id
                    st.session_state.page = 'Worklist'
                    st.session_state.flash_message = f"Saved analyzed study for {export_study['patient_name']} ({export_study['mrn']})."
                    rerun()
            with actions[1]:
                export_bytes, export_name, export_mime = report_bytes(export_study, predicted, confidence, note, note, probs, image, backend, image_source)
                st.download_button('Download report', data=export_bytes, file_name=export_name, mime=export_mime, use_container_width=True)

def viewer(conn: sqlite3.Connection, studies: pd.DataFrame, demo_mode: bool, model_files: list[Path]) -> None:
    section('Study Viewer', '', '')
    study = select_study(studies)
    image, image_source, _ = saved_study_image_context(conn, study)
    _, center, _ = st.columns([0.25, 3.5, 0.25])
    with center:
        backend = {'engine': str(study.get('ai_engine') or 'AI Radiology Assistant'), 'mode': 'stored', 'detail': 'Saved study review'}
        predicted = str(study.get('result') or '').strip()
        confidence = float(study.get('confidence') or 0.0)
        if not predicted:
            probs, backend = predict_scores(image, model_files)
            if backend.get('mode') in {'fallback', 'demo'}:
                st.error(
                    'Real model inference is not active. The app is using fallback scoring. '
                    'Run the app with C:/Users/hp/Desktop/Brave_env/Scripts/python.exe -m streamlit run src/main.py'
                )
            top = int(np.argmax(probs))
            predicted = CLASS_NAMES[top]
            confidence = float(probs[top])
        else:
            probs = stored_probabilities(predicted, confidence if confidence > 0 else 0.95)
            if confidence <= 0:
                confidence = float(probs[CLASS_NAMES.index(predicted)]) if predicted in CLASS_NAMES else 0.95
        engine_label = str(study.get('ai_engine') or backend['engine'])
        st.markdown(
            f"<div class='focus'><h2 style='margin:0; color:#111111;'>{study['patient_name']}</h2>"
            f"<div class='focus-meta'>{study['mrn']} | {study['source_site']} | {study['modality']} | {format_gmt3(study['study_date'])}</div></div>",
            unsafe_allow_html=True,
        )
        viewer_mode = st.radio('View', ['Image', 'Grad-CAM'], horizontal=True, key=f"viewer_mode_{int(study.get('study_id', 0) or 0)}")
        export_view_image = image
        export_view_source = image_source
        study_id = int(study.get('study_id', 0) or 0)
        gradcam_cache = st.session_state.setdefault('viewer_gradcam_cache', {})
        gradcam_key = viewer_gradcam_cache_key(study_id, image_source, model_files)
        if viewer_mode == 'Image':
            st.image(image, use_container_width=True)
        else:
            cached_gradcam = gradcam_cache.get(gradcam_key)
            if cached_gradcam is None:
                loader_slot = st.empty()
                loader_slot.markdown(
                    "<div class='gradcam-loader'><span class='busy-spinner'></span><span>Generating Grad-CAM focus map...</span></div>",
                    unsafe_allow_html=True,
                )
                cam_overlay, cam_detail = gradcam_overlay(image, model_files)
                loader_slot.empty()
                gradcam_cache[gradcam_key] = {
                    'overlay': np.asarray(cam_overlay, dtype=np.uint8),
                    'detail': cam_detail,
                    'export_source': f"{image_source} | Grad-CAM",
                }
            cached_gradcam = gradcam_cache.get(gradcam_key, {})
            cam_overlay = np.asarray(cached_gradcam.get('overlay'), dtype=np.uint8)
            st.image(cam_overlay, use_container_width=True)
            export_view_image = Image.fromarray(cam_overlay.astype(np.uint8))
            export_view_source = str(cached_gradcam.get('export_source') or f"{image_source} | Grad-CAM")
        st.markdown(f"<div class='study-card'><div class='study-title'>{predicted}</div><div class='study-body'>Confidence: <strong>{confidence * 100:.1f}%</strong></div></div>", unsafe_allow_html=True)
        note = st.text_area('Findings', value=study['report_summary'] or summary(predicted, confidence), height=180, key=f"viewer_note_{int(study.get('study_id', 0) or 0)}")
        actions = st.columns(2)
        with actions[0]:
            if st.button('Save note', use_container_width=True, disabled=int(study.get('study_id', 0) or 0) <= 0):
                st.session_state.report_draft = note
                update_analysis_record(conn, int(study['study_id']), predicted, confidence, note, 'High' if predicted != 'Normal' else 'Routine', engine_label)
                st.session_state.flash_message = f"Updated the selected study note for {study['patient_name']}."
                rerun()
        with actions[1]:
            export_bytes, export_name, export_mime = report_bytes(study.copy(), predicted, confidence, note, note, probs, export_view_image, {'engine': engine_label, 'mode': 'stored', 'detail': 'Saved study review'}, export_view_source)
            st.download_button('Download report', data=export_bytes, file_name=export_name, mime=export_mime, use_container_width=True)

def registry(conn: sqlite3.Connection, patients: pd.DataFrame, studies: pd.DataFrame, demo_mode: bool) -> None:
    section('Registry', '', '')
    _, center, _ = st.columns([0.45, 2.5, 0.45])
    with center:
        with st.form('patient_intake'):
            st.markdown("<div class='mini'>Patient registration</div>", unsafe_allow_html=True)
            name = st.text_input('Full name')
            age_value = st.number_input('Age', min_value=0, max_value=120, value=None, step=1, placeholder='Enter age')
            gender = st.selectbox('Gender', ['', 'Female', 'Male'], format_func=lambda value: 'Select gender' if value == '' else value)
            st.text_input('Medical record number', value='Generated automatically after registration', disabled=True)
            submit = st.form_submit_button('Register and continue', use_container_width=True)
            if submit:
                if not name.strip():
                    st.error('A patient name is required.')
                elif age_value is None:
                    st.error('Age is required.')
                elif not gender:
                    st.error('Gender is required.')
                else:
                    patient_id, generated_mrn = save_patient(conn, name.strip(), int(age_value), gender)
                    st.session_state.active_patient_id = patient_id
                    st.session_state.page = 'AI Analysis'
                    st.session_state.flash_message = f"Patient {name.strip()} registered with {generated_mrn}. Continue with AI Analysis."
                    rerun()


def main() -> None:
    inject_styles()
    init_state()
    conn = db()
    patients, studies, demo_mode = operational_data(conn)
    model_files = discover_model_files()
    shell(st.session_state.page, demo_mode, model_files)
    flash_banner()
    if st.session_state.page == 'Dashboard':
        dashboard(patients, studies, demo_mode)
    elif st.session_state.page == 'Worklist':
        worklist(studies, demo_mode)
    elif st.session_state.page == 'AI Analysis':
        ai_analysis(conn, studies, demo_mode, model_files)
    elif st.session_state.page == 'Study Viewer':
        viewer(conn, studies, demo_mode, model_files)
    else:
        registry(conn, patients, studies, demo_mode)


if __name__ == '__main__':
    main()

