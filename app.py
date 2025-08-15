import streamlit as st
import json, os
import streamlit as st
import google.generativeai as genai
import io
import pandas as pd
import time

st.set_page_config(page_title="Complaint Themes Extractor", layout="centered")

st.title("Complaint Themes Extractor")
st.caption("Step 1: Skeleton app — we’ll add upload & extraction next.")

st.success("If you can see this message after deployment, the app is wired up.")

st.write("Key loaded:", "GOOGLE_API_KEY" in st.secrets)
st.write("Model:", st.secrets.get("MODEL_NAME"))


st.header("Gemini test (single row)")

# 1) Check secret
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("No GOOGLE_API_KEY in Secrets. Add it in the app’s Settings → Secrets.")
    st.stop()

# 2) Configure SDK
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
model_name = st.secrets.get("MODEL_NAME", "gemini-2.5-flash-lite")

# 3) Inputs
system_prompt = st.text_area("System instruction (paste yours here)", height=200)
sample_text = st.text_area("One complaint_summary to test", height=100)
go = st.button("Test Gemini")

def call_gemini(system_instruction: str, complaint_text: str):
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_instruction
    )
    # Ask for strict JSON with only the fields we need
    generation_config = {
        "response_mime_type": "application/json",
        "response_schema": {
            "type": "object",
            "properties": {
                "all_case_themes": {"type": "array", "items": {"type": "string"}},
                "case_theme": {"type": "string"}
            },
            "required": ["all_case_themes"]
        }
    }
    resp = model.generate_content(
        [{"role": "user", "parts": [f'complaint_summary: """{complaint_text}"""']}],
        generation_config=generation_config,
        safety_settings=None,
    )
    # Prefer resp.text; fall back to parts if needed
    text = getattr(resp, "text", None)
    if not text and resp.candidates:
        parts = resp.candidates[0].content.parts
        text = "".join(getattr(p, "text", "") for p in parts)
    return json.loads(text)

if go:
    try:
        out = call_gemini(system_prompt.strip(), sample_text.strip())
        st.success("Got JSON:")
        st.json(out)
    except Exception as e:
        st.error(f"Failed to parse JSON: {e}")
        st.write("Raw response for debugging:")
        try:
            st.code(resp.text)  # may not exist if exception happened earlier
        except:
            pass

# ---------- Batch setup: upload & select column ----------
st.header("Batch extraction — upload & select column")

uploaded = st.file_uploader("Upload CSV with a complaint text column", type=["csv"])
if uploaded is not None:
    # Read CSV
    df = pd.read_csv(uploaded)
    st.write(f"Rows: {len(df):,} • Columns: {list(df.columns)}")

    # Choose the text column (default to 'complaint_summary' if present)
    default_col = "complaint_summary" if "complaint_summary" in df.columns else None
    text_col = st.selectbox(
        "Which column contains the complaint text?",
        options=list(df.columns),
        index=(list(df.columns).index(default_col) if default_col in df.columns else 0)
    )

    # Basic cleaning / checks
    work = df.copy()
    work["row_id"] = range(len(work))  # preserve original order
    missing = work[text_col].isna().sum()
    st.info(f"Selected text column: **{text_col}** • Missing values: **{missing}**")

    # Show a preview of what will be processed
    st.subheader("Preview (first 10 rows)")
    st.dataframe(work[["row_id", text_col]].head(10), use_container_width=True)

    st.caption("Looks good? In the next step we’ll add the Run button, progress bar, and CSV export.")
else:
    st.caption("Upload a CSV to continue.")



# ---------- Batch extraction: run + export ----------
st.header("Batch extraction — run & export")

# Reuse the uploaded df and text_col from the section above (guard if missing)
if "uploaded" not in locals() or uploaded is None:
    st.caption("Upload a CSV above to enable batch extraction.")
else:
    # Re-read so we have a fresh df in this scope
    df = pd.read_csv(uploaded)
    default_col = "complaint_summary" if "complaint_summary" in df.columns else df.columns[0]
    text_col = default_col if "text_col" not in locals() else text_col  # keep prior selection if set

    # Controls
    col1, col2, col3 = st.columns([1,1,2])
    with col1:
        skip_no_complaint = st.checkbox("Skip 'no complaint'", value=True)
    with col2:
        max_rows = st.number_input("Max rows (0 = all)", min_value=0, value=0, step=50)

    go_batch = st.button("Run extraction on uploaded CSV")

    if go_batch:
        if "GOOGLE_API_KEY" not in st.secrets:
            st.error("No GOOGLE_API_KEY in Secrets.")
            st.stop()
        if not system_prompt.strip():
            st.error("Paste your System instruction in the box above (Gemini test section).")
            st.stop()

        work = df.copy()
        work["row_id"] = range(len(work))
        # validity mask
        col_series = work[text_col].astype(str)
        valid = col_series.str.strip().ne("")
        if skip_no_complaint:
            valid &= col_series.str.strip().str.lower().ne("no complaint")

        idx = work.index[valid]
        if max_rows and max_rows > 0:
            idx = idx[:max_rows]

        st.write(f"Processing {len(idx)} of {len(work)} rows.")
        prog = st.progress(0.0)
        status = st.empty()
        results = []

        for k, i in enumerate(idx, start=1):
            txt = str(work.at[i, text_col])
            # retry/backoff
            attempts = 0
            while True:
                try:
                    out = call_gemini(system_prompt.strip(), txt.strip())
                    themes = out.get("all_case_themes", [])
                    break
                except Exception as e:
                    attempts += 1
                    if attempts >= 5:
                        st.warning(f"Row {work.at[i,'row_id']} failed after retries. Inserting empty list.")
                        themes = []
                        break
                    time.sleep(min(2**attempts, 10))

            results.append({
                "row_id": work.at[i, "row_id"],
                "all_case_themes": themes
            })
            prog.progress(k / len(idx))
            status.write(f"Processed {k}/{len(idx)}")

        # assemble ordered output
        out_df = pd.DataFrame(results).sort_values("row_id")
        st.subheader("Sample of results")
        st.dataframe(out_df.head(20), use_container_width=True)

        # export CSV (exact two columns)
        csv_bytes = out_df[["row_id", "all_case_themes"]].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download themes CSV",
            data=csv_bytes,
            file_name="all_case_themes.csv",
            mime="text/csv"
        )

        # simple guardrail
        st.info(f"Export ready. Source rows: {len(work)} • Labeled rows in file: {len(out_df)} (ordered by row_id)")

