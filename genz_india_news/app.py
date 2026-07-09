"""Streamlit frontend for the genz_india_news pipeline.

One button triggers stages 1-3 (fetch -> score -> export) for every
configured sector, then the resulting Excel workbooks are available to
download directly from the page. Meant to be run manually once a day,
either locally (`streamlit run app.py`) or deployed on Streamlit Community
Cloud.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from config_utils import load_sectors, load_settings, setup_logging
from fetch_and_score import run_pipeline

st.set_page_config(page_title="GenZ India News Scoring", layout="wide")

settings = load_settings()
setup_logging(settings.get("log_file"))

OUTPUT_DIR = Path(settings.get("output_dir", "output"))
FULL_WORKBOOK_PATH = OUTPUT_DIR / "scored_headlines.xlsx"
SURVEY_CLEAN_PATH = OUTPUT_DIR / "survey_clean.xlsx"

st.title("India GenZ/Alpha News Scoring")
st.caption(
    "Fetches India news per sector from Google News, scores it for India + "
    "GenZ/Alpha relevance (no LLM, no external API), and exports "
    "survey-ready Excel workbooks -- no scheduler needed, just run this "
    "once a day."
)

sectors = load_sectors()
with st.expander(f"{len(sectors)} configured sectors"):
    for name, keywords in sectors.items():
        st.markdown(f"**{name}** -- {', '.join(keywords)}")

if "sector_results" not in st.session_state:
    st.session_state.sector_results = None
if "last_run_at" not in st.session_state:
    st.session_state.last_run_at = None

run_clicked = st.button("Run pipeline now", type="primary")

if run_clicked:
    with st.spinner("Fetching and scoring headlines across all sectors -- this takes a minute or two..."):
        try:
            st.session_state.sector_results = run_pipeline()
            st.session_state.last_run_at = datetime.now()
        except Exception as exc:
            st.error(f"Pipeline run failed: {exc}")
            st.session_state.sector_results = None

if st.session_state.last_run_at:
    st.caption(f"Last run: {st.session_state.last_run_at.strftime('%Y-%m-%d %H:%M:%S')}")

sector_results = st.session_state.sector_results

if sector_results:
    total = sum(len(v) for v in sector_results.values())
    st.success(f"Scored {total} headlines across {len(sector_results)} sectors.")

    cols = st.columns(len(sector_results) if len(sector_results) <= 7 else 7)
    for i, (sector, articles) in enumerate(sector_results.items()):
        cols[i % len(cols)].metric(sector, len(articles))

if FULL_WORKBOOK_PATH.exists() and SURVEY_CLEAN_PATH.exists():
    st.subheader("Download")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download scored_headlines.xlsx (full scores)",
            data=FULL_WORKBOOK_PATH.read_bytes(),
            file_name="scored_headlines.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with col2:
        st.download_button(
            "Download survey_clean.xlsx (respondent-facing)",
            data=SURVEY_CLEAN_PATH.read_bytes(),
            file_name="survey_clean.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.subheader("Preview")
    sheet_names = pd.ExcelFile(FULL_WORKBOOK_PATH).sheet_names
    selected_sheet = st.selectbox("Sector", sheet_names)
    preview_df = pd.read_excel(FULL_WORKBOOK_PATH, sheet_name=selected_sheet)
    st.dataframe(preview_df, use_container_width=True, hide_index=True)
else:
    st.info("No output yet -- click 'Run pipeline now' above to generate today's headlines.")
