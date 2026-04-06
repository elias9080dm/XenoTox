# ============================================================
# QSAR Classification Pipeline - Central Configuration
# ============================================================

# Reproducibility
RANDOM_STATE = 42

# Data splitting
TEST_SIZE = 0.3

# Feature filtering
VAR_THRESHOLD = 0.01
CORR_THRESHOLD = 0.9

# Cross-validation
CV_FOLDS = 5

# Genetic Algorithm
GA_N_GEN = 40
GA_POP_SIZE = 60
GA_MAX_FEATURES = 30
GA_LAMBDA_PENALTY = 0.005
GA_MAX_CACHE = 10_000
GA_CXPB = 0.7
GA_MUTPB = 0.3

# Optuna hyperparameter optimization
N_TRIALS = 25

# Classification decision threshold
DECISION_THRESHOLD = 0.5
