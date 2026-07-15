from rdkit import Chem
from rdkit.Chem import BRICS

def generate_candidates(seed_smiles_list, num_candidates=10):
    all_fragments = set()
    for smi in seed_smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        frags = BRICS.BRICSDecompose(mol)
        all_fragments.update(frags)

    all_fragments = list(all_fragments)
    print(f"Collected {len(all_fragments)} unique fragments from {len(seed_smiles_list)} seed molecules.")

    if len(all_fragments) < 2:
        print("Not enough fragments to generate new molecules.")
        return []

    generator = BRICS.BRICSBuild([Chem.MolFromSmiles(f) for f in all_fragments])

    new_molecules = []
    seen = set()
    for mol in generator:
        try:
            Chem.SanitizeMol(mol)
            smi = Chem.MolToSmiles(mol)
            if smi not in seen:
                seen.add(smi)
                new_molecules.append(smi)
        except Exception:
            continue
        if len(new_molecules) >= num_candidates:
            break

    return new_molecules


if __name__ == "__main__":
    seed_molecules = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "Cc1ccc(cc1)S(=O)(=O)Nc1ccccc1",
    ]

    candidates = generate_candidates(seed_molecules, num_candidates=10)

    print(f"\nGenerated {len(candidates)} new candidate molecules:\n")
    for i, smi in enumerate(candidates, 1):
        print(f"{i}. {smi}")