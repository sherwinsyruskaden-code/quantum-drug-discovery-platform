from repurposing import repurposing_decision
from generation import generate_candidates
from scoring import score_candidates
from database import init_db, save_molecule, get_all_molecules
from quantum_step import run_h2_vqe

def run_pipeline(target_name, seed_smiles_for_generation, top_n=5):
    print(f"=== Running pipeline for target: {target_name} ===\n")

    init_db()

    found, repurposing_df = repurposing_decision(target_name)

    if found:
        print("Repurposing candidates found. Scoring these instead of generating new molecules.\n")
        smiles_list = repurposing_df["smiles"].dropna().tolist()
        source = "repurposing"
    else:
        print("No strong repurposing match. Generating new candidate molecules.\n")
        smiles_list = generate_candidates(seed_smiles_for_generation, num_candidates=15)
        source = "generated"

    if not smiles_list:
        print("No candidates available to score. Pipeline stopped.")
        return []

    scored_results = score_candidates(smiles_list)
    top_results = scored_results[:top_n]

    print("Running quantum validation step (VQE on reference H2 system)...\n")
    electronic_energy, total_energy = run_h2_vqe()
    print(f"Quantum step complete. Reference ground-state energy: {total_energy:.6f} Hartree\n")

    for result in top_results:
        save_molecule(target_name, source, result, quantum_energy=total_energy)

    print(f"Saved {len(top_results)} molecules to the database.\n")

    print(f"=== Top {top_n} candidates by drug-likeness (QED) ===\n")
    for i, result in enumerate(top_results, 1):
        print(f"{i}. {result}")

    return top_results


if __name__ == "__main__":
    target = "Epidermal growth factor receptor"
    seeds = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "Cc1ccc(cc1)S(=O)(=O)Nc1ccccc1",
    ]

    top_candidates = run_pipeline(target, seeds, top_n=5)

    print("\n=== Everything currently in the database ===\n")
    for row in get_all_molecules():
        print(row)