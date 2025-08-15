import streamlit as st
import json, os
import json
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


# ---------- Gemini test (single row) with summarize-on-error fallback ----------
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
system_prompt = st.text_area(
    "System instruction (paste yours here)",
    height=220,
    value=st.session_state.get("system_prompt", "")
)
sample_text = st.text_area(
    "One complaint_summary to test",
    height=120,
    value=st.session_state.get("sample_text", "")
)
go = st.button("Test Gemini")

def _parse_resp(resp):
    raw_text = getattr(resp, "text", None)
    if not raw_text and getattr(resp, "candidates", None):
        parts = resp.candidates[0].content.parts
        raw_text = "".join(getattr(p, "text", "") for p in parts)
    return raw_text or ""

def call_gemini_extract(system_instruction: str, complaint_text: str):
    """Primary extractor: expects three fields."""
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
                "subcategory_themes": {"type": "array", "items": {"type": "string"}},
                "evidence_spans": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"quote": {"type": "string"}},
                        "required": ["quote"]
                    }
                }
            },
            "required": ["all_case_themes", "subcategory_themes", "evidence_spans"]
        },
        "max_output_tokens": 256,
        "temperature": 0.2,
    }
    resp = model.generate_content(
        [{"role": "user", "parts": [f'complaint_summary: """{complaint_text}"""']}],
        generation_config=generation_config,
    )
    raw = _parse_resp(resp)
    data = json.loads(raw or "{}")

    # normalize
    spans = data.get("evidence_spans", []) or []
    norm_spans = []
    for s in spans:
        if isinstance(s, dict) and "quote" in s:
            norm_spans.append({"quote": str(s["quote"])})
        elif isinstance(s, str):
            norm_spans.append({"quote": s})
    out = {
        "all_case_themes": data.get("all_case_themes", []) or [],
        "subcategory_themes": data.get("subcategory_themes", []) or [],
        "evidence_spans": norm_spans,
    }
    return out, raw

def call_gemini_summarize(complaint_text: str) -> str:
    """Fallback: compress to core issues only, then we re-run extraction on the summary."""
    model = genai.GenerativeModel(model_name=MODEL_NAME)
    generation_config = {
        "response_mime_type": "application/json",
        "response_schema": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"]
        },
        "max_output_tokens": 200,
        "temperature": 0.2,
    }
    # Minimal, deterministic summary prompt
    summary_instruction = (
        "Summarize this complaint_summary into 3–5 short bullets (one paragraph OK) "
        "capturing ONLY substantive reasons/behaviors that could cause dissatisfaction or replacement. "
        "Remove admin/process details (calls, links, scheduling, follow-ups), names, and dates. "
        "Be concise and neutral."
    )
    resp = model.generate_content(
        [
            {"role": "user", "parts": [
                summary_instruction + f'\n\ncomplaint_summary: """{complaint_text}"""'
            ]}
        ],
        generation_config=generation_config,
    )
    raw = _parse_resp(resp)
    data = json.loads(raw or "{}")
    return data.get("summary", "").strip()

if go:
    # remember inputs between reruns
    st.session_state["system_prompt"] = system_prompt
    st.session_state["sample_text"]  = sample_text

    try:
        out, raw = call_gemini_extract(system_prompt.strip(), sample_text.strip())
        st.success("Got JSON:")
        st.json(out)
    except Exception as e1:
        st.warning(f"Primary extraction failed ({e1}). Trying summarize-then-extract fallback…")
        try:
            summary = call_gemini_summarize(sample_text.strip())
            if not summary:
                raise RuntimeError("Summarizer returned empty text")
            out2, raw2 = call_gemini_extract(system_prompt.strip(), summary)
            st.success("Fallback succeeded on summarized text.")
            with st.expander("Summary used for fallback"):
                st.write(summary)
            st.json(out2)
        except Exception as e2:
            st.error(f"Fallback also failed: {e2}")


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

# ---------- Batch extraction — run & export (with summarize-on-error fallback) ----------
import time, json
import pandas as pd
import streamlit as st

st.header("Batch extraction — run & export")

df = st.session_state.get("df")
text_col = st.session_state.get("text_col")

if df is None or text_col is None:
    st.caption("Upload a file and select the text column above to enable batch extraction.")
else:
    col1, col2 = st.columns([1, 1])
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

        # Valid rows: non-empty; optionally exclude literal "no complaint"
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

            # Try primary extractor first; on error, summarize then extract
            try:
                out, _raw = call_gemini_extract(system_prompt.strip(), txt)
            except Exception:
                try:
                    summary = call_gemini_summarize(txt)
                    out, _raw2 = call_gemini_extract(system_prompt.strip(), summary or txt)
                except Exception:
                    # Final fallback: empty payload
                    out = {
                        "all_case_themes": [],
                        "subcategory_themes": [],
                        "evidence_spans": []
                    }

            results.append({
                "row_id": int(work.at[i, "row_id"]),
                "all_case_themes": out.get("all_case_themes", []),
                "subcategory_themes": out.get("subcategory_themes", []),
                "evidence_spans": out.get("evidence_spans", []),
            })

            prog.progress(k / len(idx))
            status.write(f"Processed {k}/{len(idx)}")

        # Assemble ordered output
        out_df = pd.DataFrame(results).sort_values("row_id")

        st.subheader("Sample of results")
        st.dataframe(out_df.head(20), use_container_width=True)

        # Convert list / list-of-dicts to JSON strings for a clean CSV
        csv_df = out_df.copy()
        for col in ["all_case_themes", "subcategory_themes", "evidence_spans"]:
            csv_df[col] = csv_df[col].apply(lambda x: json.dumps(x, ensure_ascii=False))

        csv_bytes = csv_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download themes CSV",
            data=csv_bytes,
            file_name="themes_with_subcats_and_evidence.csv",
            mime="text/csv"
        )

        st.info(
            f"Export ready. Source rows: {len(work)} • Labeled rows: {len(out_df)} "
            "(ordered by row_id)."
        )
