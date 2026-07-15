from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Crippen

def score_molecule(smiles):
    """
    Computes drug-likeness properties for a single molecule.
    Returns a dict, or None if the SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    h_donors = Descriptors.NumHDonors(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)
    qed_score = QED.qed(mol)  # 0 to 1, higher = more drug-like

    # Lipinski's Rule of 5 checks
    lipinski_violations = 0
    if mw > 500: lipinski_violations += 1
    if logp > 5: lipinski_violations += 1
    if h_donors > 5: lipinski_violations += 1
    if h_acceptors > 10: lipinski_violations += 1

    return {
        "smiles": smiles,
        "molecular_weight": round(mw, 2),
        "logp": round(logp, 2),
        "h_donors": h_donors,
        "h_acceptors": h_acceptors,
        "qed_score": round(qed_score, 3),
        "lipinski_violations": lipinski_violations,
        "drug_like": lipinski_violations <= 1  # standard cutoff
    }


def score_candidates(smiles_list):
    """
    Scores a list of molecules, returns list of dicts sorted by QED (best first).
    """
    scored = []
    for smi in smiles_list:
        result = score_molecule(smi)
        if result:
            scored.append(result)

    scored.sort(key=lambda x: x["qed_score"], reverse=True)
    return scored


if __name__ == "__main__":
    test_molecules = [
        "CC(C)Cc1ccccc1",
        "CC(C)Cc1ccccc1C(=O)O",
        "CC(C)Cc1ccccc1-c1ccccc1",
        "CC(C)Cc1ccccc1-c1ccccc1-c1ccccc1",
        "CC(C)Cc1ccccc1Nc1ccccc1",
    ]

    results = score_candidates(test_molecules)

    print(f"Scored {len(results)} molecules, ranked by drug-likeness:\n")
    for r in results:
        print(r)