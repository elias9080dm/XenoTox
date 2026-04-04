from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np


def filter_features(X_train, X_test, var_threshold=0.01, corr_threshold=0.9):

    # VARIANCE FILTER
    print(f"Variance Filter (threshold {var_threshold})")

    var_filter = VarianceThreshold(threshold=var_threshold)
    X_train_var_array = var_filter.fit_transform(X_train)

    # Retrieve retained feature names
    var_mask = var_filter.get_support()
    var_features = X_train.columns[var_mask]

    # Reconstruct Dataframe with origninal indices
    X_train_var = pd.DataFrame(
        X_train_var_array,
        columns=var_features,
        index=X_train.index
    )

    # Apply same transformation to test set
    X_test_var = X_test[var_features]
    print("Deleted features by variance:", len(
        X_train.columns) - len(var_features))
    print("Features after variance:", len(var_features))

    # CORRELATION FILTER
    print(f"\nCorrelation Filter (threshold {corr_threshold})")

    corr_matrix = pd.DataFrame(X_train_var, columns=var_features).corr().abs()

    # Upper triangular matrix
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Columns to drop
    to_drop_corr = [
        column for column in upper_triangle.columns
        if any(upper_triangle[column] > corr_threshold)
    ]

    # Final features
    filtered_features = [f for f in var_features if f not in to_drop_corr]

    # Apply to train and test
    X_train_filtered = pd.DataFrame(X_train_var, columns=var_features)[
        filtered_features]
    X_test_filtered = X_test_var[filtered_features]

    print("Deleted features by correlation:", len(to_drop_corr))
    print("Final features post-filtering:", len(filtered_features))

    return X_train_filtered, X_test_filtered
