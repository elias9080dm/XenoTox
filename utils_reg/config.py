# ============================================================
# QSAR Regression Pipeline - Central Configuration
# ============================================================
import subprocess as _subprocess


# Reproducibility
RANDOM_STATE = 42

# Data splitting
TEST_SIZE = 0.3

# Feature filtering
VAR_THRESHOLD = 0.01
CORR_THRESHOLD = 0.9

# Cross-validation (model optimization)
CV_FOLDS = 5

# Genetic Algorithm
GA_N_GEN = 40
GA_POP_SIZE = 60
GA_CV_FOLDS = 3        # Fewer folds in GA fitness for speed (19k samples)
GA_MAX_FEATURES = 40   # Regression benefits from more features than classification
GA_LAMBDA_PENALTY = 0.0005
GA_MAX_CACHE = 10_000
GA_CXPB = 0.6
GA_MUTPB = 0.4

# Optuna hyperparameter optimization
N_TRIALS = 30


# GPU detection (used by XGBoost)
def _has_gpu():
    try:
        _subprocess.check_output(["nvidia-smi"], stderr=_subprocess.DEVNULL)
        return True
    except Exception:
        return False


DEVICE = "cuda" if _has_gpu() else "cpu"

if DEVICE == "cuda":
    print("[config] GPU detected — XGBoost will use CUDA.")
else:
    print("[config] No GPU detected — using CPU.")
