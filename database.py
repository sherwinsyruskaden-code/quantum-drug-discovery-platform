import sqlite3
import json
from datetime import datetime

DB_NAME = "drug_discovery.db"

def init_db():
    """Creates the database and table if they don't exist yet."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS molecules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            target_name TEXT,
            smiles TEXT,
            source TEXT,
            molecular_weight REAL,
            logp REAL,
            h_donors INTEGER,
            h_acceptors INTEGER,
            qed_score REAL,
            lipinski_violations INTEGER,
            drug_like INTEGER,
            quantum_energy REAL,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_molecule(target_name, source, scored_result, quantum_energy=None):
    """Saves one scored molecule to the database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO molecules (
            target_name, smiles, source, molecular_weight, logp,
            h_donors, h_acceptors, qed_score, lipinski_violations,
            drug_like, quantum_energy, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        target_name,
        scored_result["smiles"],
        source,
        scored_result["molecular_weight"],
        scored_result["logp"],
        scored_result["h_donors"],
        scored_result["h_acceptors"],
        scored_result["qed_score"],
        scored_result["lipinski_violations"],
        int(scored_result["drug_like"]),
        quantum_energy,
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()


def get_all_molecules(target_name=None):
    """Retrieves all stored molecules, optionally filtered by target."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    if target_name:
        cursor.execute("SELECT * FROM molecules WHERE target_name = ? ORDER BY qed_score DESC", (target_name,))
    else:
        cursor.execute("SELECT * FROM molecules ORDER BY qed_score DESC")
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    conn.close()
    return [dict(zip(columns, row)) for row in rows]


if __name__ == "__main__":
    init_db()
    print("Database initialized. Table 'molecules' is ready.")

    # quick test: fetch whatever's in there (should be empty first run)
    existing = get_all_molecules()
    print(f"Currently {len(existing)} molecules stored.")