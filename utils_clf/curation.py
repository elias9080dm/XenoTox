from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize


def curate_data(
    df_raw,
    smiles_col,
    activity_col,
):
    RDLogger.DisableLog("rdApp.*")

    initial_count = len(df_raw)

    # STEP 1 – BASIC SANITY (NaNs)
    df = df_raw.dropna(subset=[smiles_col, activity_col]).copy()

    # STEP 2 – SMILES PARSING
    df["mol"] = df[smiles_col].map(Chem.MolFromSmiles)
    df = df[df["mol"].notna()].copy()

    # STEP 3 – ORGANIC FILTER
    def is_organic(mol):
        return any(a.GetAtomicNum() == 6 for a in mol.GetAtoms())

    df = df[df["mol"].map(is_organic)].copy()

    # STEP 4 – STANDARDIZATION
    normalizer = rdMolStandardize.Normalize
    uncharger = rdMolStandardize.Uncharger()
    tautomer_enum = rdMolStandardize.TautomerEnumerator()

    def standardize(mol):
        try:
            # 4.1 Normalize (functional group normalization)
            mol = normalizer(mol)

            # 4.2 Fragment parent (salt stripping)
            mol = rdMolStandardize.FragmentParent(mol)

            # 4.3 Uncharge
            mol = uncharger.uncharge(mol)

            # 4.4 Canonical tautomer
            mol = tautomer_enum.Canonicalize(mol)

            return mol
        except Exception:
            return None

    df["mol_std"] = df["mol"].map(standardize)
    df = df[df["mol_std"].notna()].copy()

    # STEP 5 – CANONICAL SMILES FROM STANDARDIZED MOL
    df["SMILES_std"] = df["mol_std"].map(
        lambda m: Chem.MolToSmiles(m, canonical=True)
    )

    # delete wildcards
    df = df[~df["SMILES_std"].str.contains(r"\*", na=False)].copy()

    # STEP 6 – DEDUPLICATION WITH ACTIVITY PRIORITIZATION
    activity_mapping = {'active': 1, 'inactive': 0}
    df['activity_rank'] = df[activity_col].map(activity_mapping)

    # Ordenar para que 'active' aparezca primero en caso de SMILES_std duplicados
    df = df.sort_values(by=['SMILES_std', 'activity_rank'],
                        ascending=[True, False]).copy()

    # Eliminar duplicados, manteniendo la primera (ahora priorizada 'active')
    df = df.drop_duplicates(subset="SMILES_std", keep="first").copy()

    # Eliminar la columna temporal 'activity_rank'
    df = df.drop(columns=['activity_rank']).copy()

    # FINALIZATION
    df_final = (
        df[[activity_col, "SMILES_std"]]
        .rename(columns={"SMILES_std": "SMILES"})
        .reset_index(drop=True)
    )

    final_count = len(df_final)
    print(
        f"Curation completed: "
        f"{final_count} valid molecules "
        f"(out of {initial_count} initial)."
    )

    return df_final
