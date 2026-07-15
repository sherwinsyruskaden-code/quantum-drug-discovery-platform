import requests
import pandas as pd

def get_target_id(target_name):
    url = "https://www.ebi.ac.uk/chembl/api/data/target.json"
    params = {
        "pref_name__icontains": target_name,
        "target_type": "SINGLE PROTEIN",
        "organism": "Homo sapiens",
        "limit": 5
    }
    try:
        response = requests.get(url, params=params, timeout=15)
    except requests.exceptions.RequestException as e:
        print("Connection failed:", e)
        return None

    if response.status_code != 200:
        return None

    targets = response.json().get("targets", [])
    if not targets:
        return None

    print("Candidate targets found:")
    for t in targets:
        print(f"  - {t['pref_name']}  ({t['target_chembl_id']})")

    best = targets[0]
    return best["target_chembl_id"], best["pref_name"]


def search_existing_drugs(target_name, max_results=10):
    target_info = get_target_id(target_name)
    if not target_info:
        print(f"No target found matching '{target_name}'")
        return pd.DataFrame()

    target_id, target_full_name = target_info
    print(f"\nUsing target: {target_full_name} ({target_id})\n")

    url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    params = {"target_chembl_id": target_id, "limit": max_results}
    try:
        response = requests.get(url, params=params, timeout=15)
    except requests.exceptions.RequestException as e:
        print("Connection failed:", e)
        return pd.DataFrame()

    if response.status_code != 200:
        print("API request failed:", response.status_code)
        return pd.DataFrame()

    activities = response.json().get("activities", [])

    results = []
    for act in activities:
        results.append({
            "molecule_chembl_id": act.get("molecule_chembl_id"),
            "molecule_name": act.get("molecule_pref_name"),
            "standard_type": act.get("standard_type"),
            "standard_value": act.get("standard_value"),
            "standard_units": act.get("standard_units"),
            "smiles": act.get("canonical_smiles"),
        })

    return pd.DataFrame(results)


def get_repurposing_candidates(target_name, max_results=10):
    df = search_existing_drugs(target_name, max_results=50)

    if df.empty:
        return df

    df = df.drop_duplicates(subset="molecule_chembl_id").reset_index(drop=True)

    return df.head(max_results)


def repurposing_decision(target_name):
    """
    Returns (found_candidates: bool, dataframe)
    """
    df = get_repurposing_candidates(target_name)

    if df.empty or len(df) < 3:
        print(f"\n[DECISION] Weak or no repurposing candidates for '{target_name}'.")
        print("[DECISION] Proceeding to molecule generation stage.\n")
        return False, df
    else:
        print(f"\n[DECISION] Found {len(df)} repurposing candidates for '{target_name}'.")
        print("[DECISION] Repurposing sufficient — skipping generation.\n")
        return True, df


if __name__ == "__main__":
    query = "Epidermal growth factor receptor"
    found, df = repurposing_decision(query)
    print(df)