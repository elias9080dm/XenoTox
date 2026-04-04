from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
import pandas as pd


def curation(smiles_col, df):

    RDLogger.DisableLog("rdApp.*")

    smiles_col = "SMILES"

    # =============================================================================
    # INITIALIZATION
    # =============================================================================
    df_qsar = df.copy()
    total_initial = len(df_qsar)
    df_qsar["SMILES_raw"] = df_qsar[smiles_col]

    curation_report = []
    std_report = []

    def add_report(step, desc, before, after):
        rem = before - after
        pct = round(rem / before * 100, 2) if before else 0
        curation_report.append({
            "Step": step,
            "Description": desc,
            "Entries_before": before,
            "Entries_after": after,
            "Removed": rem,
            "Removed_%": pct
        })

    # =============================================================================
    # STEP 1 – SEMANTIC FILTERING
    # =============================================================================

    before = len(df_qsar)

    mask = (
        df_qsar["Agonist_Activity"].str.lower().isin(["active", "inactive"])
    )

    df_qsar = df_qsar[mask].reset_index(drop=True)

    print("Actives:", (df_qsar["Agonist_Activity"] == "active").sum())
    print("Inactives:", (df_qsar["Agonist_Activity"] == "inactive").sum())

    add_report(1, "Activity filtering", before, len(df_qsar))

    # =============================================================================
    # STEP 2 – SMILES PARSING
    # =============================================================================
    before = len(df_qsar)

    # Drop rows where SMILES is NaN
    df_qsar.dropna(subset=[smiles_col], inplace=True)
    df_qsar = df_qsar.reset_index(drop=True)  # Reset index after dropping rows
    # Map SMILES to RDKit molecules.
    df_qsar["mol"] = df_qsar[smiles_col].map(Chem.MolFromSmiles)
    # Filter valid rows
    df_qsar = df_qsar[df_qsar["mol"].notna()].reset_index(drop=True)

    add_report(2, "SMILES parsing", before, len(df_qsar))

    # =============================================================================
    # STEP 3 – ORGANIC FILTER
    # =============================================================================
    before = len(df_qsar)

    def is_organic(mol):
        return any(a.GetAtomicNum() == 6 for a in mol.GetAtoms())

    df_qsar = df_qsar[df_qsar["mol"].map(is_organic)].reset_index(drop=True)

    add_report(3, "Filtering non-organic molecules", before, len(df_qsar))

    # =============================================================================
    # STEP 4 – STANDARDIZATION
    # =============================================================================
    mols = df_qsar["mol"]

    def smiles(m):
        return Chem.MolToSmiles(m, canonical=True)

    def report_changes(tag, before, after):
        chg = (before.map(smiles) != after.map(smiles))
        std_report.append({
            "Substep": tag,
            "Molecules": len(before),
            "Changed": chg.sum(),
            "Changed_%": round(chg.mean() * 100, 2)
        })

    # 4.1 Normalize
    m_norm = mols.map(rdMolStandardize.Normalize)
    report_changes("Normalize", mols, m_norm)

    # 4.2 Fragment parent (salt stripping)
    m_frag = m_norm.map(rdMolStandardize.FragmentParent)
    report_changes("FragmentParent", m_norm, m_frag)

    # 4.3 Uncharge
    uncharger = rdMolStandardize.Uncharger()
    m_unch = m_frag.map(uncharger.uncharge)
    report_changes("Uncharger", m_frag, m_unch)

    # 4.4 Canonical tautomer
    te = rdMolStandardize.TautomerEnumerator()
    m_std = m_unch.map(te.Canonicalize)
    report_changes("TautomerCanonical", m_unch, m_std)

    df_qsar["mol_std"] = m_std

    add_report(
        4,
        "Structural standardization",
        len(df_qsar),
        len(df_qsar)
    )

    # =============================================================================
    # STEP 5 – CANONICAL SMILES (FROM STANDARDIZED MOL)
    # =============================================================================
    before = len(df_qsar)

    df_qsar[smiles_col] = df_qsar["mol_std"].map(
        lambda m: Chem.MolToSmiles(m, canonical=True)
    )

    df_qsar = df_qsar[~df_qsar[smiles_col].str.contains(r"\*", na=False)]
    add_report(5, "Canonical SMILES generation", before, len(df_qsar))

    # =============================================================================
    # STEP 6 – DEDUPLICATION (FINAL)
    # =============================================================================
    before = len(df_qsar)

    df_qsar = df_qsar.drop_duplicates(
        subset=[smiles_col]).reset_index(drop=True)

    add_report(6, "Deduplication by standardized SMILES", before, len(df_qsar))

    # =============================================================================
    # FINALIZATION AND SUMMARY
    # =============================================================================
    df_qsar.drop(columns=["mol", "mol_std"], inplace=True)

    add_report(7, "Summary", total_initial, len(df_qsar))

    curation_report_df = pd.DataFrame(curation_report)
    std_report_df = pd.DataFrame(std_report)

    return df_qsar, curation_report_df, std_report_df
