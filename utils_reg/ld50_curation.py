import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger

import pandas as pd
import os
import glob


def combine_csv(folder_path, columns_to_keep):
    """
    Combines multiple CSV files into a single pandas DataFrame,
    keeping only the specified columns.

    Parameters:
    - folder_path : path to the directory containing CSV files.
    - columns_to_keep : list of column names to retain.
    """
    dataframes = []

    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    if not csv_files:
        raise ValueError("No CSV files found in the provided path.")

    for file in csv_files:
        try:
            df = pd.read_csv(file)

            # Keep only columns that exist in the current file
            available_columns = [
                col for col in columns_to_keep if col in df.columns
            ]
            df_filtered = df[available_columns]

            dataframes.append(df_filtered)

        except Exception as e:
            print(f"Error reading {file}: {e}")

    # Concatenate all DataFrames
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Add missing columns as NaN
    for col in columns_to_keep:
        if col not in combined_df.columns:
            combined_df[col] = pd.NA

    # Reorder columns
    combined_df = combined_df[columns_to_keep]

    return combined_df


def curate_data(df, smiles_col="SMILES", target_col="LD50"):

    RDLogger.DisableLog("rdApp.*")

    # =============================================================================
    # INITIALIZATION
    # =============================================================================
    df_qsar = df.copy()
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
        df_qsar['Risk assessment class'].str.contains('acute', case=False, na=False) &
        df_qsar['Species'].str.contains('rat', case=False, na=False) &
        df_qsar['Exposure route'].str.contains('oral', case=False, na=False) &
        df_qsar['Standard value'].str.startswith('=', na=False)
    )

    df_qsar = df_qsar[mask].reset_index(drop=True)

    add_report(1, "Semantic filtering (acute, rat, oral, exact value)",
               before, len(df_qsar))

    # =============================================================================
    # STEP 2 – TARGET CLEANING
    # =============================================================================
    before = len(df_qsar)

    df_qsar[target_col] = pd.to_numeric(df_qsar[target_col], errors='coerce')
    df_qsar = df_qsar.dropna(subset=[target_col]).reset_index(drop=True)

    add_report(2, "Target cleaning (numeric)", before, len(df_qsar))

    # =============================================================================
    # STEP 3 – SMILES PARSING
    # =============================================================================
    before = len(df_qsar)

    df_qsar = df_qsar.dropna(subset=[smiles_col]).reset_index(drop=True)

    df_qsar["mol"] = df_qsar[smiles_col].map(
        lambda x: Chem.MolFromSmiles(x) if isinstance(x, str) else None
    )

    df_qsar = df_qsar[df_qsar["mol"].notna()].reset_index(drop=True)

    add_report(3, "SMILES parsing", before, len(df_qsar))

    # =============================================================================
    # STEP 3.5 – SANITIZATION FILTER
    # =============================================================================
    def is_valid_mol(mol):
        try:
            Chem.SanitizeMol(mol)
            return True
        except:
            return False

    before = len(df_qsar)
    df_qsar = df_qsar[df_qsar["mol"].map(is_valid_mol)].reset_index(drop=True)

    add_report(3.5, "Sanitization filter", before, len(df_qsar))

    # =============================================================================
    # STEP 4 – ORGANIC FILTER
    # =============================================================================
    before = len(df_qsar)

    def is_organic(mol):
        return any(a.GetAtomicNum() == 6 for a in mol.GetAtoms())

    df_qsar = df_qsar[df_qsar["mol"].map(is_organic)].reset_index(drop=True)

    add_report(4, "Filtering non-organic molecules", before, len(df_qsar))

    # =============================================================================
    # STEP 5 – STANDARDIZATION (SAFE)
    # =============================================================================
    mols = df_qsar["mol"]

    def safe_apply(func, mol):
        try:
            return func(mol)
        except:
            return None

    def smiles_safe(m):
        try:
            return Chem.MolToSmiles(m, canonical=True) if m is not None else None
        except:
            return None

    def report_changes(tag, before, after):
        before_s = before.map(smiles_safe)
        after_s = after.map(smiles_safe)
        chg = (before_s != after_s)
        std_report.append({
            "Substep": tag,
            "Molecules": len(before),
            "Changed": chg.sum(),
            "Changed_%": round(chg.mean() * 100, 2)
        })

    # Normalize
    m_norm = mols.map(lambda m: safe_apply(rdMolStandardize.Normalize, m))
    report_changes("Normalize", mols, m_norm)

    # Fragment parent
    m_frag = m_norm.map(lambda m: safe_apply(
        rdMolStandardize.FragmentParent, m) if m is not None else None)
    report_changes("FragmentParent", m_norm, m_frag)

    # Uncharge
    uncharger = rdMolStandardize.Uncharger()
    m_unch = m_frag.map(lambda m: safe_apply(
        uncharger.uncharge, m) if m is not None else None)
    report_changes("Uncharger", m_frag, m_unch)

    # Tautomer canonicalization
    te = rdMolStandardize.TautomerEnumerator()
    m_std = m_unch.map(lambda m: safe_apply(
        te.Canonicalize, m) if m is not None else None)
    report_changes("TautomerCanonical", m_unch, m_std)

    # Remove failed molecules
    before = len(df_qsar)

    df_qsar["mol_std"] = m_std

    # Optional: keep track of failed SMILES
    failed_smiles = df_qsar[df_qsar["mol_std"].isna()]["SMILES_raw"]

    df_qsar = df_qsar[df_qsar["mol_std"].notna()].reset_index(drop=True)

    add_report(5, "Structural standardization (failed removed)",
               before, len(df_qsar))

    # =============================================================================
    # STEP 6 – CANONICAL SMILES
    # =============================================================================
    before = len(df_qsar)

    df_qsar[smiles_col] = df_qsar["mol_std"].map(
        lambda m: Chem.MolToSmiles(m, canonical=True)
    )

    df_qsar = df_qsar[~df_qsar[smiles_col].str.contains(r"\*", na=False)]

    add_report(6, "Canonical SMILES generation", before, len(df_qsar))

    # =============================================================================
    # STEP 7 – DEDUPLICATION (PRIORITIZE LOWEST LD50)
    # =============================================================================
    before = len(df_qsar)

    df_qsar = df_qsar.sort_values(target_col, ascending=True)
    df_qsar = df_qsar.drop_duplicates(
        subset=[smiles_col], keep='first').reset_index(drop=True)

    add_report(7, "Deduplication (lowest LD50 kept)", before, len(df_qsar))

    # =============================================================================
    # FINALIZATION
    # =============================================================================
    df_qsar.drop(columns=["mol", "mol_std"], inplace=True)

    return df_qsar, pd.DataFrame(curation_report), pd.DataFrame(std_report), failed_smiles
