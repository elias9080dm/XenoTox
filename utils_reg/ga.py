"""
DEPRECATED: Use utils_reg.robust_ga instead.
This module is kept only for backward compatibility.
"""
import warnings
from utils_reg.robust_ga import ga_feature_selection as _robust_ga


def ga_feature_selection(X, y, descriptor_names, **kwargs):
    warnings.warn(
        "utils_reg.ga is deprecated. Use utils_reg.robust_ga instead.",
        DeprecationWarning,
        stacklevel=2
    )
    selected_features, *_ = _robust_ga(X, y, descriptor_names, **kwargs)
    return selected_features
