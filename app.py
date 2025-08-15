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


# ---------- Gemini test (single row) ----------
import json
import streamlit as st
import google.generativeai as genai

st.header("Gemini test (single row)")

# 1) Check secret
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("No GOOGLE_API_KEY in Secrets. Add it in the app’s Settings → Secrets.")
    st.stop()

# 2) Configure SDK
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
MODEL_NAME = st.secrets.get("MODEL_NAME", "gemini-2.5-flash-lite")

# 3) Inputs
system_prompt = st.text_area("System instruction (paste yours here)", height=220, value=st.session_state.get("system_prompt", ""))
sample_text  = st.text_area("One complaint_summary to test", height=120, value=st.session_state.get("sample_text", ""))
go = st.button("Test Gemini")

def call_gemini(system_instruction: str, complaint_text: str):
    """
    Expects STRICT JSON with:
      {
        "all_case_themes": [string, ...],
        "subcategory_themes": [string, ...]
      }
    """
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=system_instruction
    )
    generation_config = {
        "response_mime_type": "application/json",
        "response_schema": {
            "type": "object",
            "properties": {
                "all_case_themes":    {"type": "array", "items": {"type": "string"}},
                "subcategory_themes": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["all_case_themes", "subcategory_themes"]
            # NOTE: no 'additionalProperties' — Gemini doesn't support that field
        }
    }

    resp = model.generate_content(
        [{"role": "user", "parts": [f'complaint_summary: """{complaint_text}"""']}],
        generation_config=generation_config,
    )

    # Prefer resp.text; fall back to parts if needed
    raw_text = getattr(resp, "text", None)
    if not raw_text and getattr(resp, "candidates", None):
        parts = resp.candidates[0].content.parts
        raw_text = "".join(getattr(p, "text", "") for p in parts)

    data = json.loads(raw_text or "{}")

    # Keep only the keys we care about (ignore any extras)
    data = {
        "all_case_themes": data.get("all_case_themes", []),
        "subcategory_themes": data.get("subcategory_themes", [])
    }
    return data, raw_text

if go:
    # remember inputs between reruns
    st.session_state["system_prompt"] = system_prompt
    st.session_state["sample_text"]  = sample_text

    try:
        out, raw = call_gemini(system_prompt.strip(), sample_text.strip())
        st.success("Got JSON:")
        st.json(out)
    except Exception as e:
        st.error(f"Failed to parse JSON: {e}")
        if 'raw' in locals() and raw:
            st.code(raw)


# ---------- Batch setup: upload & select column (robust) ----------
import pandas as pd
import streamlit as st

st.header("Batch extraction — upload & select column")

uploaded = st.file_uploader("Upload CSV (or Excel) with a complaint text column",
                            type=["csv", "xlsx"])

def load_table(file):
    import pandas as pd
    # Empty file guard
    try:
        nbytes = getattr(file, "size", None) or file.getbuffer().nbytes
        if nbytes == 0:
            raise pd.errors.EmptyDataError("empty file")
    except Exception:
        pass

    file.seek(0)
    if file.name.lower().endswith(".xlsx"):
        return pd.read_excel(file)
    # CSV: auto-detect delimiter, tolerate weird encodings
    try:
        file.seek(0)
        return pd.read_csv(file, sep=None, engine="python",
                           encoding="utf-8", on_bad_lines="skip")
    except pd.errors.EmptyDataError:
        st.error("The file looks empty or not a valid CSV. Make sure it has a header row and at least one data row.")
        st.stop()
    except UnicodeDecodeError:
        file.seek(0)
        return pd.read_csv(file, sep=None, engine="python",
                           encoding_errors="ignore", on_bad_lines="skip")

if uploaded is not None:
    df = load_table(uploaded)
    st.session_state["df"] = df  # keep it for the next section
    st.write(f"Rows: {len(df):,} • Columns: {list(df.columns)}")

    # Let the user choose the text column
    default_col = "complaint_summary" if "complaint_summary" in df.columns else df.columns[0]
    text_col = st.selectbox("Which column contains the complaint text?",
                            options=list(df.columns),
                            index=list(df.columns).index(default_col))
    st.session_state["text_col"] = text_col

    # Preview
    work = df.copy()
    work["row_id"] = range(len(work))
    missing = work[text_col].isna().sum()
    st.info(f"Selected text column: **{text_col}** • Missing values: **{missing}**")
    st.subheader("Preview (first 10 rows)")
    st.dataframe(work[["row_id", text_col]].head(10), use_container_width=True)
else:
    st.caption("Upload a CSV or Excel file to continue.")

# ---------- Batch extraction: run + export ----------
import time
import pandas as pd
import streamlit as st

st.header("Batch extraction — run & export")

df = st.session_state.get("df")
text_col = st.session_state.get("text_col")

if df is None or text_col is None:
    st.caption("Upload a file and select the text column above to enable batch extraction.")
else:
    col1, col2 = st.columns([1,1])
    with col1:
        skip_no_complaint = st.checkbox("Skip rows where text equals 'no complaint'", value=True)
    with col2:
        max_rows = st.number_input("Max rows (0 = all)", min_value=0, value=0, step=50)

    go_batch = st.button("Run extraction on uploaded file")

    if go_batch:
        if "GOOGLE_API_KEY" not in st.secrets:
            st.error("No GOOGLE_API_KEY in Secrets.")
            st.stop()
        if not system_prompt.strip():
            st.error("Paste your System instruction in the Gemini test box above.")
            st.stop()

        work = df.copy()
        work["row_id"] = range(len(work))
        col_series = work[text_col].astype(str)
        valid = col_series.str.strip().ne("")
        if skip_no_complaint:
            valid &= col_series.str.strip().str.lower().ne("no complaint")

        idx = work.index[valid]
        if max_rows:
            idx = idx[:max_rows]

        st.write(f"Processing {len(idx)} of {len(work)} rows.")
        prog = st.progress(0.0)
        status = st.empty()
        results = []

        for k, i in enumerate(idx, start=1):
            txt = str(work.at[i, text_col]).strip()
            # retry/backoff
            attempts = 0
            while True:
                try:
                    out = call_gemini(system_prompt.strip(), txt)
                    themes = out.get("all_case_themes", [])
                    break
                except Exception:
                    attempts += 1
                    if attempts >= 5:
                        themes = []
                        st.warning(f"Row {work.at[i,'row_id']} failed after retries. Saved empty list.")
                        break
                    time.sleep(min(2**attempts, 10))

            results.append({"row_id": work.at[i, "row_id"], "all_case_themes": themes})
            prog.progress(k / len(idx))
            status.write(f"Processed {k}/{len(idx)}")

        out_df = pd.DataFrame(results).sort_values("row_id")
        st.subheader("Sample of results")
        st.dataframe(out_df.head(20), use_container_width=True)

        csv_bytes = out_df[["row_id", "all_case_themes"]].to_csv(index=False).encode("utf-8")
        st.download_button("Download themes CSV", data=csv_bytes,
                           file_name="all_case_themes.csv", mime="text/csv")
        st.info(f"Export ready. Source rows: {len(work)} • Labeled rows: {len(out_df)} (ordered by row_id).")



