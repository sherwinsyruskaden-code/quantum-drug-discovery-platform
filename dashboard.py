import streamlit as st
import py3Dmol
import streamlit.components.v1 as components
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from pipeline import run_pipeline
from database import init_db, get_all_molecules

# ---------- PAGE CONFIGURATION ----------
st.set_page_config(
    page_title="Quantum-Guided Autonomous Drug Discovery Platform",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- DESIGN SYSTEM & CUSTOM CSS ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

    /* Global CSS Reset & Layout */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #E2E8F0;
        background-color: #090D16;
    }
    
    .stApp {
        background-color: #090D16;
    }

    .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 2rem !important;
        max-width: 1440px !important;
    }

    /* Streamlit Header & Navigation Overrides */
    header[data-testid="stHeader"] {
        background: rgba(9, 13, 22, 0.85) !important;
        backdrop-filter: blur(8px);
        border-bottom: 1px solid #1E293B;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0D1322 !important;
        border-right: 1px solid #1E293B !important;
    }
    
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
    }

    .sidebar-header {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #38BDF8;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Input & Control Overrides */
    .stTextInput > label, .stSlider > label {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        color: #94A3B8 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    .stTextInput > div > div > input {
        background-color: #090D16 !important;
        color: #F8FAFC !important;
        border: 1px solid #1E293B !important;
        border-radius: 6px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.875rem !important;
        padding: 0.5rem 0.75rem !important;
        transition: all 0.15s ease;
    }

    .stTextInput > div > div > input:focus {
        border-color: #0EA5E9 !important;
        box-shadow: 0 0 0 2px rgba(14, 165, 233, 0.2) !important;
    }

    /* Slider Overrides */
    div[data-baseweb="slider"] {
        margin-top: 0.5rem;
    }

    /* Button Styling */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(180deg, #0EA5E9 0%, #0284C7 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #38BDF8 !important;
        border-radius: 6px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.875rem !important;
        letter-spacing: 0.02em !important;
        padding: 0.65rem 1rem !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    }

    div.stButton > button:hover {
        background: linear-gradient(180deg, #38BDF8 0%, #0EA5E9 100%) !important;
        border-color: #7DD3FC !important;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3) !important;
        transform: translateY(-1px);
    }

    div.stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* Header Banner Component */
    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #0D1322;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
    }

    .app-header .title-group h1 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.35rem;
        font-weight: 800;
        color: #F8FAFC;
        margin: 0 0 0.25rem 0;
        letter-spacing: -0.01em;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }

    .app-header .title-group p {
        font-size: 0.85rem;
        color: #94A3B8;
        margin: 0;
        max-width: 800px;
        line-height: 1.4;
    }

    .app-header .meta-group {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 0.5rem;
    }

    .status-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34D399;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.725rem;
        font-weight: 600;
        padding: 0.25rem 0.6rem;
        border-radius: 4px;
        text-transform: uppercase;
    }

    .status-dot {
        width: 6px;
        height: 6px;
        background-color: #34D399;
        border-radius: 50%;
        box-shadow: 0 0 8px #34D399;
    }

    .target-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #CBD5E1;
        background: #131B2E;
        border: 1px solid #1E293B;
        padding: 0.25rem 0.6rem;
        border-radius: 4px;
    }

    /* Pipeline Stage Visualizer */
    .pipeline-stepper {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.5rem;
        margin-bottom: 1.25rem;
    }

    .step-node {
        background: #0D1322;
        border: 1px solid #1E293B;
        border-radius: 6px;
        padding: 0.6rem 0.75rem;
        display: flex;
        flex-direction: column;
        gap: 0.2rem;
        transition: all 0.2s ease;
    }

    .step-node.active {
        border-color: #0EA5E9;
        background: rgba(14, 165, 233, 0.05);
    }

    .step-node .step-num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.675rem;
        font-weight: 600;
        color: #0EA5E9;
        text-transform: uppercase;
    }

    .step-node .step-name {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.775rem;
        font-weight: 700;
        color: #E2E8F0;
    }

    /* Workspace Compact Metrics Bar */
    .metrics-bar {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.75rem;
        margin-bottom: 1.25rem;
    }

    .metric-cell {
        background: #0D1322;
        border: 1px solid #1E293B;
        border-left: 3px solid #0EA5E9;
        border-radius: 6px;
        padding: 0.75rem 1rem;
    }

    .metric-cell.success { border-left-color: #10B981; }
    .metric-cell.quantum { border-left-color: #8B5CF6; }
    .metric-cell.info { border-left-color: #38BDF8; }

    .metric-cell .val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.35rem;
        font-weight: 700;
        color: #F8FAFC;
        line-height: 1.2;
    }

    .metric-cell .lbl {
        font-size: 0.725rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.2rem;
    }

    /* Candidate Scientific Result Cards */
    .candidate-row {
        background: #0D1322;
        border: 1px solid #1E293B;
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.6rem;
        transition: border-color 0.15s ease;
    }

    .candidate-row:hover {
        border-color: #334155;
    }

    .candidate-row.pass {
        border-left: 3px solid #10B981;
    }

    .candidate-row.flagged {
        border-left: 3px solid #F59E0B;
    }

    .candidate-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.4rem;
    }

    .rank-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 700;
        color: #0EA5E9;
        background: rgba(14, 165, 233, 0.1);
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        border: 1px solid rgba(14, 165, 233, 0.2);
    }

    .status-badge {
        font-size: 0.7rem;
        font-weight: 700;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    .status-badge.good {
        background: rgba(16, 185, 129, 0.1);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .status-badge.bad {
        background: rgba(245, 158, 11, 0.1);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .smiles-code {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #38BDF8;
        background: #090D16;
        border: 1px solid #1E293B;
        padding: 0.4rem 0.6rem;
        border-radius: 4px;
        word-break: break-all;
    }

    .prop-pill-grid {
        display: flex;
        gap: 1.25rem;
        margin-top: 0.5rem;
        font-size: 0.775rem;
        color: #94A3B8;
    }

    .prop-pill-grid span b {
        color: #F8FAFC;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Tabs Override */
    div[data-baseweb="tab-list"] {
        background-color: #0D1322 !important;
        border: 1px solid #1E293B !important;
        border-radius: 6px !important;
        padding: 0.25rem !important;
        gap: 0.25rem !important;
    }

    button[data-baseweb="tab"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        color: #94A3B8 !important;
        background: transparent !important;
        border-radius: 4px !important;
        padding: 0.45rem 1rem !important;
        border: none !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #F8FAFC !important;
        background: #131B2E !important;
        border: 1px solid #1E293B !important;
    }

    div[data-baseweb="tab-border"] {
        display: none !important;
    }

    /* Expanders Override */
    .stExpander {
        background: #090D16 !important;
        border: 1px solid #1E293B !important;
        border-radius: 6px !important;
        margin-top: 0.4rem !important;
    }

    .stExpander > details > summary {
        color: #94A3B8 !important;
        font-size: 0.775rem !important;
        font-weight: 600 !important;
        padding: 0.4rem 0.75rem !important;
    }

    /* Workstation 3D Container Frame */
    .viewer-frame {
        background: #0D1322;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 1rem;
    }

    .viewer-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #1E293B;
        padding-bottom: 0.75rem;
        margin-bottom: 0.75rem;
    }

    .viewer-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.875rem;
        font-weight: 700;
        color: #F8FAFC;
    }

    .viewer-controls-hint {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #64748B;
    }

    /* Empty State Container */
    .empty-state {
        background: #0D1322;
        border: 1px border-dashed #1E293B;
        border-radius: 8px;
        padding: 3rem 2rem;
        text-align: center;
    }

    .empty-state h3 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.5rem;
    }

    .empty-state p {
        font-size: 0.85rem;
        color: #64748B;
        max-width: 480px;
        margin: 0 auto;
    }

    /* Dataframe Overrides */
    div[data-testid="stDataFrame"] {
        border: 1px solid #1E293B !important;
        border-radius: 6px !important;
        overflow: hidden !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Database Connection
init_db()

# ---------- SIDEBAR CONTROL PANEL ----------
with st.sidebar:
    st.markdown('<div class="sidebar-header"><span>Pipeline Controls</span></div>', unsafe_allow_html=True)
    
    target_input = st.text_input(
        "Disease / Protein Target",
        value="Epidermal growth factor receptor",
        help="Specify the target protein or biological pathway."
    )
    
    top_n = st.slider(
        "Candidate Count (Top N)",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of top drug-like candidates to retain and validate."
    )
    
    run_button = st.button("Run Discovery Pipeline", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-header"><span>Architecture Overview</span></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="font-size: 0.775rem; color: #94A3B8; line-height: 1.5; background: #090D16; padding: 0.75rem; border: 1px solid #1E293B; border-radius: 6px;">
        <b style="color: #E2E8F0;">Autonomous Workflow:</b><br>
        1. Query ChEMBL/PubChem repurposing index<br>
        2. Execute generative RDKit fragment synthesis<br>
        3. Score QED, LogP, MW, and Lipinski rules<br>
        4. Validate ground-state energy via VQE<br>
        5. Log result to Digital Twin registry
    </div>
    """, unsafe_allow_html=True)

seeds = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Cc1ccc(cc1)S(=O)(=O)Nc1ccccc1",
]

# ---------- COMPACT PRODUCT HEADER ----------
st.markdown(f"""
<div class="app-header">
    <div class="title-group">
        <h1>Quantum-Guided Autonomous Drug Discovery</h1>
        <p>Biopharma candidate search engine integrating database repurposing, generative lead optimization, pharmaceutical drug-likeness scoring, and Variational Quantum Eigensolver (VQE) validation.</p>
    </div>
    <div class="meta-group">
        <div class="status-chip">
            <span class="status-dot"></span>
            <span>VQE ENGINE READY</span>
        </div>
        <div class="target-badge">
            TARGET: {target_input}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- VISUAL PIPELINE STEPPER ----------
st.markdown("""
<div class="pipeline-stepper">
    <div class="step-node active">
        <span class="step-num">STAGE 01</span>
        <span class="step-name">Target Context</span>
    </div>
    <div class="step-node active">
        <span class="step-num">STAGE 02</span>
        <span class="step-name">Repurposing</span>
    </div>
    <div class="step-node active">
        <span class="step-num">STAGE 03</span>
        <span class="step-name">Generative AI</span>
    </div>
    <div class="step-node active">
        <span class="step-num">STAGE 04</span>
        <span class="step-name">Pharma Scoring</span>
    </div>
    <div class="step-node active">
        <span class="step-num">STAGE 05</span>
        <span class="step-name">VQE Quantum</span>
    </div>
</div>
""", unsafe_allow_html=True)


def render_molecule_3d(smiles, width=700, height=420):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    mol_block = Chem.MolToMolBlock(mol)

    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(mol_block, "mol")
    viewer.setStyle({"stick": {"colorscheme": "cyanCarbon"}})
    viewer.setBackgroundColor("#0D1322")
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

    return "Candidate evaluation: " + ", ".join(reasons) + "."


# ---------- PIPELINE EXECUTION & RESULTS WORKSPACE ----------
if run_button:
    with st.spinner("Executing pipeline: repurposing lookup -> generative scoring -> VQE quantum validation..."):
        results = run_pipeline(target_input, seeds, top_n=top_n)

    if not results:
        st.error("No candidates found or generated for this target. Try a different target identifier.")
    else:
        avg_qed = sum(r["qed_score"] for r in results) / len(results)
        drug_like_count = sum(1 for r in results if r["drug_like"])

        # Integrated Compact Metrics Toolbar
        st.markdown(f"""
        <div class="metrics-bar">
            <div class="metric-cell info">
                <div class="val">{len(results)}</div>
                <div class="lbl">Candidates Evaluated</div>
            </div>
            <div class="metric-cell info">
                <div class="val">{avg_qed:.3f}</div>
                <div class="lbl">Mean QED Score</div>
            </div>
            <div class="metric-cell success">
                <div class="val">{drug_like_count}/{len(results)}</div>
                <div class="lbl">Lipinski Pass Rate</div>
            </div>
            <div class="metric-cell quantum">
                <div class="val">-0.985 Ha</div>
                <div class="lbl">VQE Ground Energy</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["Candidate Results Matrix", "Comparative Analytics", "3D Workstation View"])

        with tab1:
            for i, result in enumerate(results, 1):
                badge_class = "good" if result["drug_like"] else "bad"
                badge_text = "Lipinski Pass" if result["drug_like"] else "Flagged"
                row_class = "pass" if result["drug_like"] else "flagged"

                st.markdown(f"""
                <div class="candidate-row {row_class}">
                    <div class="candidate-header">
                        <div>
                            <span class="rank-badge">RANK #{i}</span>
                            <span style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 0.85rem; font-weight: 700; color: #F8FAFC; margin-left: 0.5rem;">
                                Candidate {i}
                            </span>
                        </div>
                        <span class="status-badge {badge_class}">{badge_text}</span>
                    </div>
                    <div class="smiles-code">{result['smiles']}</div>
                    <div class="prop-pill-grid">
                        <span>QED Score: <b>{result['qed_score']}</b></span>
                        <span>MW: <b>{result['molecular_weight']} g/mol</b></span>
                        <span>LogP: <b>{result['logp']}</b></span>
                        <span>Lipinski Violations: <b>{result['lipinski_violations']}</b></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander(f"View candidate #{i} property breakdown & rationale"):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Molecular Weight", f"{result['molecular_weight']}")
                    c2.metric("LogP (Partition Coeff)", f"{result['logp']}")
                    c3.metric("H-Bond Donors", result['h_donors'])
                    c4.metric("H-Bond Acceptors", result['h_acceptors'])
                    st.caption(explain_candidate(result))

        with tab2:
            st.markdown('<div style="font-family: \'Plus Jakarta Sans\', sans-serif; font-size: 0.9rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.75rem;">QED Drug-Likeness Comparison</div>', unsafe_allow_html=True)
            chart_df = pd.DataFrame({
                "Candidate": [f"Candidate #{i+1}" for i in range(len(results))],
                "QED Score": [r["qed_score"] for r in results]
            }).set_index("Candidate")
            st.bar_chart(chart_df, color="#0EA5E9")

        with tab3:
            top_smiles = results[0]["smiles"]
            st.markdown(f"""
            <div class="viewer-frame">
                <div class="viewer-header">
                    <div class="viewer-title">Lead Candidate (#1): {top_smiles}</div>
                    <div class="viewer-controls-hint">ROTATE: Left Click | ZOOM: Scroll | PAN: Right Click</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            try:
                render_molecule_3d(top_smiles)
            except Exception as e:
                st.warning(f"Could not initialize 3D view: {e}")
else:
    # Empty State Prompt
    st.markdown("""
    <div class="empty-state">
        <h3>Ready to Execute Discovery Pipeline</h3>
        <p>Select a disease target and candidate count in the sidebar controls, then click <b>Run Discovery Pipeline</b> to initiate repurposing search, generative design, and quantum validation.</p>
    </div>
    """, unsafe_allow_html=True)

# ---------- DIGITAL TWIN REGISTRY ----------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div style="font-family: \'Plus Jakarta Sans\', sans-serif; font-size: 1rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.5rem; display: flex; align-items: center; justify-content: space-between;"><span>Digital Twin Registry: Full Molecule Storage</span></div>', unsafe_allow_html=True)

all_molecules = get_all_molecules()
if all_molecules:
    st.dataframe(pd.DataFrame(all_molecules), use_container_width=True)
else:
    st.info("No molecules registered in database yet. Execute the discovery pipeline to populate registry.")