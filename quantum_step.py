import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA

def run_h2_vqe():
    """
    Runs VQE on the H2 molecule using its known qubit Hamiltonian
    (standard reference values at equilibrium bond length 0.735 Angstrom).
    Returns the computed ground-state energy in Hartree.
    """
    # Published H2 Hamiltonian, reduced to 2 qubits (well-known Qiskit reference example)
    hamiltonian = SparsePauliOp.from_list([
        ("II", -1.052373245772859),
        ("IZ", 0.39793742484318045),
        ("ZI", -0.39793742484318045),
        ("ZZ", -0.01128010425623538),
        ("XX", 0.18093119978423156),
    ])

    nuclear_repulsion_energy = 0.7055696146

    ansatz = TwoLocal(2, "ry", "cz", reps=2)
    optimizer = COBYLA(maxiter=200)
    estimator = StatevectorEstimator()

    vqe = VQE(estimator, ansatz, optimizer)
    result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)

    electronic_energy = result.eigenvalue.real
    total_energy = electronic_energy + nuclear_repulsion_energy

    return electronic_energy, total_energy


if __name__ == "__main__":
    print("Running VQE on H2 molecule (this may take a few seconds)...\n")
    electronic_energy, total_energy = run_h2_vqe()

    print(f"Electronic ground-state energy: {electronic_energy:.6f} Hartree")
    print(f"Total energy (with nuclear repulsion): {total_energy:.6f} Hartree")
    print(f"\nReference (known exact value): -1.137270 Hartree")
    print("If your total energy is close to this, VQE converged correctly.")