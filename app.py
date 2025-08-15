import streamlit as st
import json, os
import streamlit as st
import google.generativeai as genai

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
