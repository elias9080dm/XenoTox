
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from joblib import Parallel, delayed
from tqdm import tqdm
from psutil import cpu_count


def descriptor_matrix(df_curated, smiles_col, activity_col):

    def compute_descriptores(smiles):
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return [np.nan] * len(Descriptors._descList)

        results = []

        for _, func in Descriptors._descList:
            try:
                d = func(mol)
                if d is None or isinstance(d, complex):
                    d = np.nan
            except Exception:
                d = np.nan

            results.append(d)

        return results

    # 2. Thread configuration
    n_cores = cpu_count()
    print(f"Initializing calculation with {n_cores} cores...")

    # 3. Parallel execution
    smiles_list = df_curated[smiles_col].tolist()

    resultados = Parallel(n_jobs=n_cores)(
        delayed(compute_descriptores)(s)
        for s in tqdm(smiles_list, desc="RDKit progress")
    )

    # 4. Create full DataFrame of descriptors
    feature_names = [name for name, _ in Descriptors._descList]

    X_full = pd.DataFrame(
        resultados,
        index=df_curated.index,
        columns=feature_names
    )

    # 5. Delete molecules with any NaN in descriptors
    mask_valid = ~X_full.isna().any(axis=1)

    X = X_full.loc[mask_valid].copy()
    df_curated = df_curated.loc[mask_valid].copy()

    # 6. Reset indexes consistently
    X.reset_index(drop=True, inplace=True)
    df_curated.reset_index(drop=True, inplace=True)

    y = df_curated[activity_col]

    return X, y
