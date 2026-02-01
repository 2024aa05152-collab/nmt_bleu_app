import streamlit as st
import pandas as pd
import math

from nmt_backend import translate_and_evaluate

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="NMT with BLEU Evaluation",
    layout="wide"
)

# =========================
# SIDEBAR – DECODING SETTINGS & INFO
# =========================
st.sidebar.title("⚙️ Decoding Settings")

num_beams = st.sidebar.slider(
    "Number of Beams",
    min_value=1,
    max_value=10,
    value=5,
    help="Higher beams = better quality but slower"
)

num_candidates = st.sidebar.slider(
    "Candidate Outputs",
    min_value=1,
    max_value=num_beams,
    value=min(3, num_beams)
)

st.sidebar.caption("Candidate outputs ≤ number of beams")

# Quality Improvement Strategy (Task B)
st.sidebar.markdown("---")
st.sidebar.markdown("## 📈 Quality Improvement Strategies")

with st.sidebar.expander("How to Improve BLEU Score (Task B)"):
    st.markdown("""
    ### 1. **Domain-Specific Fine-tuning**
    - **Technical Dataset**: Train on scientific/technical parallel corpora (arXiv, patents)
    - **Adaptive Pretraining**: Continue pretraining on domain text before fine-tuning
    - **Transfer Learning**: Start from general NMT, fine-tune on technical domain

    ### 2. **Better Tokenization Strategies**
    - **SentencePiece/Byte-Pair Encoding**: Handle rare technical terms better
    - **Subword Regularization**: Improves robustness to spelling variations
    - **Domain Vocabulary**: Add domain-specific terms to vocabulary

    ### 3. **Ensemble Methods**
    - **Model Averaging**: Combine outputs from multiple NMT models
    - **Architecture Diversity**: Use Transformer, RNN, and CNN-based models
    - **Weighted Voting**: Give more weight to domain-adapted models

    ### 4. **Beam Search Optimization**
    - **Beam Width Tuning**: 5-10 beams for optimal quality-speed tradeoff
    - **Length Normalization**: Penalize very short/long translations
    - **N-gram Blocking**: Prevent repetition of n-grams
    - **Diverse Beam Search**: Ensure diversity among candidates

    ### 5. **Post-processing Techniques**
    - **Capitalization Correction**: Important for technical terms
    - **Named Entity Preservation**: Keep technical names unchanged
    - **Grammar Checking**: Ensure grammatical correctness
    - **Term Consistency**: Maintain consistent terminology
    """)

# Technical Examples
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧪 Technical Examples")

example = st.sidebar.selectbox(
    "Load technical example:",
    ["Select", "Quantum Physics", "Medical", "Computer Science", "Engineering", "Chemistry"]
)

if example != "Select":
    examples = {
        "Quantum Physics": "The wave function collapses upon measurement, yielding eigenvalues corresponding to observables.",
        "Medical": "The patient exhibited symptoms of tachycardia and dyspnea, with elevated troponin levels.",
        "Computer Science": "The algorithm employs dynamic programming with memoization to solve the optimization problem efficiently.",
        "Engineering": "The tensile strength of the composite material exceeds 500 MPa at room temperature.",
        "Chemistry": "The reaction proceeds via nucleophilic substitution with inversion of stereochemistry."
    }
    # Use session state to preserve the text
    if 'source_text' not in st.session_state:
        st.session_state.source_text = ""
    st.session_state.source_text = examples[example]

# =========================
# MAIN HEADER
# =========================
st.markdown(
    """
    <div style="display:flex; align-items:center; justify-content:center; gap:16px;">
        <div style="
            width:54px;
            height:46px;
            border-radius:10px;
            background-color:#1f77b4;
            color:white;
            font-size:22px;
            font-weight:700;
            display:flex;
            align-items:center;
            justify-content:center;">
            A | अ
        </div>
        <div>
            <h1 style="margin:0; text-align:center;">
                Neural Machine Translation with BLEU Evaluation
            </h1>
            <p style="margin:0; text-align:center; font-size:20px; font-weight:500;">
                Technical/Scientific Domain (English → Hindi)
            </p>
        </div>
    </div>
    <hr>
    """,
    unsafe_allow_html=True
)

# Domain Description
st.markdown("""
<div style="background-color:#f0f2f6; padding:15px; border-radius:10px; margin-bottom:20px;">
<h4 style="margin-top:0;">📋 Assignment: Neural Machine Translation with BLEU Evaluation</h4>
<p><strong>Domain:</strong> Technical/Scientific Translation</p>
<p><strong>Features:</strong> Multiple candidate translations, BLEU scoring with brevity penalty, n-gram precision analysis, quality improvement strategies</p>
</div>
""", unsafe_allow_html=True)

# =========================
# INPUT SECTION
# =========================
st.markdown("### Source Text (English)")
if 'source_text' in st.session_state:
    source_text = st.text_area(
        "Enter English text to translate",
        height=120,
        value=st.session_state.source_text
    )
else:
    source_text = st.text_area(
        "Enter English text to translate",
        height=120,
        placeholder="Enter technical/scientific text for translation..."
    )

st.markdown("---")

st.markdown("### Upload Reference Translation (Hindi)")
st.markdown("Upload one or more reference translations in .txt format")
uploaded_files = st.file_uploader(
    "Choose files",
    type=["txt"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

# Reference example
with st.expander("💡 Reference Translation Format"):
    st.markdown("""
    **Example reference.txt content:**
    ```
    वेव फंक्शन मापन पर पतन हो जाता है।
    ```

    **Tips:**
    - Use UTF-8 encoding
    - One sentence per line
    - Multiple references improve BLEU accuracy
    - Include domain-specific terminology
    """)

# =========================
# ACTION BUTTON
# =========================
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    run_btn = st.button("🚀 Translate & Evaluate", type="primary", use_container_width=True)

# =========================
# OUTPUT SECTION
# =========================
if run_btn:
    if not source_text:
        st.warning("⚠️ Please enter source text to translate.")
    elif not uploaded_files:
        st.warning("⚠️ Please upload at least one reference translation file.")
    else:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("## 🔍 Translation Results")

        # Display settings used
        st.info(f"**Settings:** {num_beams} beams, {num_candidates} candidates")

        try:
            # Read reference files
            reference_texts = []
            for file in uploaded_files:
                content = file.read().decode("utf-8").strip()
                reference_texts.append(content)

            # Call backend
            with st.spinner("Translating and evaluating..."):
                results, best_index, brevity_penalties = translate_and_evaluate(
                    source_text=source_text,
                    reference_texts=reference_texts,
                    num_beams=num_beams,
                    num_candidates=num_candidates
                )

            # Display results
            for idx, item in enumerate(results, start=1):
                is_best = (idx - 1) == best_index

                # Create expandable section for each candidate
                with st.expander(f"Candidate {idx} {'⭐ BEST' if is_best else ''} - BLEU: {item['bleu']:.4f}",
                                 expanded=is_best):

                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown("**Translation:**")
                        st.markdown(
                            f'<div style="background-color:#f8f9fa; padding:10px; border-radius:5px; border-left:4px solid {"#28a745" if is_best else "#6c757d"}">{item["translation"]}</div>',
                            unsafe_allow_html=True)

                    with col2:
                        st.markdown("**Scores:**")
                        st.metric("BLEU Score", f"{item['bleu']:.4f}")
                        if is_best:
                            st.success("Best candidate")

                    # Brevity Penalty
                    st.markdown(f"**Brevity Penalty:** `{brevity_penalties[idx - 1]:.4f}`")

                    # N-gram precision table
                    st.markdown("**N-gram Precision Scores:**")
                    df = pd.DataFrame({
                        "N-gram": list(item["ngrams"].keys()),
                        "Precision": [f"{val:.4f}" for val in list(item["ngrams"].values())]
                    })
                    st.table(df)

            # Summary statistics
            st.markdown("---")
            st.markdown("### 📊 Summary Statistics")

            col1, col2, col3, col4 = st.columns(4)
            bleu_scores = [r["bleu"] for r in results]

            with col1:
                st.metric("Highest BLEU", f"{max(bleu_scores):.4f}")
            with col2:
                st.metric("Average BLEU", f"{sum(bleu_scores) / len(bleu_scores):.4f}")
            with col3:
                st.metric("Number of Candidates", len(results))
            with col4:
                st.metric("References Used", len(reference_texts))

            # Download button
            st.markdown("---")
            results_data = []
            for idx, item in enumerate(results, start=1):
                results_data.append({
                    "Candidate": idx,
                    "Translation": item["translation"],
                    "BLEU_Score": item["bleu"],
                    "Brevity_Penalty": brevity_penalties[idx - 1],
                    "1-gram": item["ngrams"]["1-gram"],
                    "2-gram": item["ngrams"]["2-gram"],
                    "3-gram": item["ngrams"]["3-gram"],
                    "4-gram": item["ngrams"]["4-gram"],
                    "Best": "Yes" if (idx - 1) == best_index else "No"
                })

            results_df = pd.DataFrame(results_data)
            csv = results_df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="📥 Download All Results as CSV",
                data=csv,
                file_name=f"translation_results_{len(results)}_candidates.csv",
                mime="text/csv",
                help="Download complete evaluation results"
            )

        except Exception as e:
            st.error(f"❌ Error during translation: {str(e)}")
            st.error("Please check your input and try again.")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#666; font-size:0.9em;">
<p><strong>ASSIGNMENT – 2: Neural Machine Translation with BLEU Evaluation</strong></p>
<p>Task A: NMT System with Automatic Evaluation | Task B: Quality Improvement Strategy</p>
</div>
""", unsafe_allow_html=True)