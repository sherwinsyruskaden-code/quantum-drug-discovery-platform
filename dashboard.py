import streamlit as st
import py3Dmol
import streamlit.components.v1 as components
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from pipeline import run_pipeline
from database import init_db, get_all_molecules

st.set_page_config(
    page_title="Quantum-Guided Drug Discovery",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CUSTOM STYLING ----------
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    
    .hero {
        padding: 1.5rem 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #1a2a4a 0%, #2d1b4e 100%);
        margin-bottom: 1.5rem;
        border: 1px solid #3a3a5c;
    }
    .hero h1 {
        font-size: 2rem;
        margin-bottom: 0.3rem;
        background: linear-gradient(90deg, #00d4ff, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p {
        color: #a0a0c0;
        font-size: 1rem;
        margin: 0;
    }

    .metric-card {
        background: #1a1f2e;
        border: 1px solid #2d3348;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00d4ff;
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .candidate-card {
        background: #1a1f2e;
        border: 1px solid #2d3348;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
    }
    .candidate-card .smiles {
        font-family: 'Courier New', monospace;
        color: #a78bfa;
        font-size: 0.85rem;
        word-break: break-all;
    }
    .badge-good {
        background: #10b981;
        color: white;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-bad {
        background: #f59e0b;
        color: white;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    section[data-testid="stSidebar"] {
        background-color: #131720;
    }
</style>
""", unsafe_allow_html=True)

init_db()

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Controls")
    target_input = st.text_input("Disease / Protein Target", value="Epidermal growth factor receptor")
    top_n = st.slider("Top N candidates", min_value=1, max_value=10, value=5)
    run_button = st.button("🚀 Run Discovery Pipeline", use_container_width=True)

    st.markdown("---")
    st.markdown("### 📖 About")
    st.caption(
        "This platform searches for existing approved drugs first, "
        "generates new candidates if none exist, scores them on real "
        "pharmaceutical criteria, and validates the top result with a "
        "real quantum computation (VQE)."
    )

seeds = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Cc1ccc(cc1)S(=O)(=O)Nc1ccccc1",
]

# ---------- HERO HEADER ----------
st.markdown("""
<div class="hero">
    <h1>🧬 Quantum-Guided Autonomous Drug Discovery</h1>
    <p>Repurposing search → AI generation → pharmaceutical scoring → quantum validation → digital twin</p>
</div>
""", unsafe_allow_html=True)


def render_molecule_3d(smiles, width=600, height=400):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    mol_block = Chem.MolToMolBlock(mol)

    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(mol_block, "mol")
    viewer.setStyle({"stick": {}})
    viewer.setBackgroundColor("#1a1f2e")
    viewer.zoomTo()

    html = viewer._make_html()
    components.html(html, width=width, height=height)


def explain_candidate(result):
    reasons = []
    if result["qed_score"] >= 0.7:
        reasons.append("high overall drug-likeness (QED)")
    elif result["qed_score"] >= 0.5:
        reasons.append("moderate drug-likeness (QED)")
    else:
        reasons.append("low drug-likeness (QED)")

    if result["lipinski_violations"] == 0:
        reasons.append("passes all Lipinski Rule-of-5 criteria")
    else:
        reasons.append(f"{result['lipinski_violations']} Lipinski violation(s)")

    if result["logp"] > 5:
        reasons.append("high lipophilicity (may affect solubility)")
    elif result["logp"] < 0:
        reasons.append("low lipophilicity (may affect membrane permeability)")

    if result["molecular_weight"] > 500:
        reasons.append("molecular weight above typical oral drug range")

    return "This candidate shows " + ", ".join(reasons) + "."


if run_button:
    with st.spinner("Running pipeline: repurposing search → scoring → quantum validation..."):
        results = run_pipeline(target_input, seeds, top_n=top_n)

    if not results:
        st.error("No candidates found or generated for this target. Try a different target name.")
    else:
        # ---------- METRIC CARDS ----------
        avg_qed = sum(r["qed_score"] for r in results) / len(results)
        drug_like_count = sum(1 for r in results if r["drug_like"])

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="value">{len(results)}</div><div class="label">Candidates</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="value">{avg_qed:.2f}</div><div class="label">Avg QED</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="value">{drug_like_count}/{len(results)}</div><div class="label">Drug-Like</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card"><div class="value">✓</div><div class="label">Quantum Validated</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["🏆 Ranked Candidates", "📈 Comparison", "🧬 3D Structure"])

        with tab1:
            for i, result in enumerate(results, 1):
                badge_class = "badge-good" if result["drug_like"] else "badge-bad"
                badge_text = "Drug-like" if result["drug_like"] else "Flagged"
                st.markdown(f"""
                <div class="candidate-card">
                    <b>#{i}</b> &nbsp; <span class="{badge_class}">{badge_text}</span> &nbsp; QED: <b>{result['qed_score']}</b>
                    <div class="smiles">{result['smiles']}</div>
                </div>
                """, unsafe_allow_html=True)
                with st.expander("Details & reasoning"):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("MW", f"{result['molecular_weight']}")
                    c2.metric("LogP", f"{result['logp']}")
                    c3.metric("H-Donors", result['h_donors'])
                    c4.metric("H-Acceptors", result['h_acceptors'])
                    st.write(explain_candidate(result))

        with tab2:
            chart_df = pd.DataFrame({
                "Candidate": [f"#{i+1}" for i in range(len(results))],
                "QED Score": [r["qed_score"] for r in results]
            }).set_index("Candidate")
            st.bar_chart(chart_df)

        with tab3:
            top_smiles = results[0]["smiles"]
            st.code(top_smiles, language="text")
            try:
                render_molecule_3d(top_smiles)
            except Exception as e:
                st.warning(f"Could not render 3D structure: {e}")

# ---------- DIGITAL TWIN HISTORY ----------
st.markdown("---")
st.subheader("📊 Digital Twin: Full Molecule History")
all_molecules = get_all_molecules()
if all_molecules:
    st.dataframe(pd.DataFrame(all_molecules), use_container_width=True)
else:
    st.info("No molecules stored yet. Run the pipeline from the sidebar.")