from deap import base, creator, tools, algorithms
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
import random

from utils_reg.config import (
    RANDOM_STATE,
    GA_N_GEN, GA_POP_SIZE, GA_CV_FOLDS,
    GA_MAX_FEATURES, GA_LAMBDA_PENALTY, GA_MAX_CACHE,
    GA_CXPB, GA_MUTPB
)

# ==========================================================
# GA Feature Selection (Regresión - Versión robusta)
# ==========================================================


def ga_feature_selection(
    X,
    y,
    descriptor_names,
    n_gen=GA_N_GEN,
    pop_size=GA_POP_SIZE,
    random_state=RANDOM_STATE
):
    """
    Genetic Algorithm feature selection for regression.

    Uses HistGradientBoostingRegressor as fitness model (fast, handles large datasets).
    Data passed in X must already be preprocessed (imputed + scaled).

    Parameters
    ----------
    X               : numpy array or DataFrame, shape (n_samples, n_features)
    y               : array-like, continuous target
    descriptor_names: list of feature names (len == X.shape[1])
    n_gen           : int, number of GA generations
    pop_size        : int, population size
    random_state    : int

    Returns
    -------
    selected_features : list of str
    best_fitness      : float  (negative RMSE, higher = better)
    hof               : DEAP HallOfFame
    log               : DEAP Logbook
    """
    np.random.seed(random_state)
    random.seed(random_state)

    X_np = X.values if hasattr(X, "values") else np.asarray(X)
    n_features = X_np.shape[1]

    # ----------------------------------------------------------------
    # CV and base model for fitness
    # ----------------------------------------------------------------
    cv = KFold(n_splits=GA_CV_FOLDS, shuffle=True, random_state=random_state)

    # HistGBR: native NaN support, fast on large datasets, no need to scale
    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.05,
        max_iter=200,
        random_state=random_state
    )

    # ----------------------------------------------------------------
    # Fitness cache
    # ----------------------------------------------------------------
    fitness_cache = {}

    # ----------------------------------------------------------------
    # Fitness function
    # ----------------------------------------------------------------
    def evaluate_individual(individual):
        key = tuple(individual)
        if key in fitness_cache:
            return fitness_cache[key]

        if len(fitness_cache) > GA_MAX_CACHE:
            fitness_cache.clear()

        n_selected = sum(individual)
        MIN_FEATURES = max(5, int(0.01 * n_features))

        if n_selected < MIN_FEATURES:
            fitness_cache[key] = (-1e6,)
            return (-1e6,)

        selected_idx = [i for i, bit in enumerate(individual) if bit == 1]
        X_sel = X_np[:, selected_idx]

        try:
            # n_jobs=-1: safe because DEAP evaluates sequentially
            scores = cross_val_score(
                model, X_sel, y,
                cv=cv,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1
            )

            mean_score = scores.mean()   # negative RMSE (higher = better)
            std_score  = scores.std()

            rmse = -mean_score           # positive RMSE

            # Penalización ligera por número de features
            complexity_penalty = GA_LAMBDA_PENALTY * n_selected

            # Penalización por exceder límite
            excess_penalty = 0.0
            if n_selected > GA_MAX_FEATURES:
                excess_penalty = 0.05 * (n_selected - GA_MAX_FEATURES)

            # Penalización por inestabilidad entre folds
            stability_penalty = 0.1 * std_score

            fitness = -(rmse + complexity_penalty + excess_penalty + stability_penalty)

        except Exception:
            fitness = -1e6

        fitness_cache[key] = (fitness,)
        return (fitness,)

    # ----------------------------------------------------------------
    # DEAP setup (notebook-safe: guard creator classes)
    # ----------------------------------------------------------------
    if not hasattr(creator, "FitnessMaxReg"):
        creator.create("FitnessMaxReg", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "IndividualReg"):
        creator.create("IndividualReg", list, fitness=creator.FitnessMaxReg)

    toolbox = base.Toolbox()

    # ----------------------------------------------------------------
    # Population initialization (Gaussian around 10-20% of features)
    # ----------------------------------------------------------------
    MIN_FEATURES = max(5, int(0.01 * n_features))

    def init_individual():
        ind = [0] * n_features
        k = int(np.random.normal(loc=0.15 * n_features, scale=0.07 * n_features))
        k = int(np.clip(k, MIN_FEATURES, int(0.45 * n_features)))
        for i in random.sample(range(n_features), k):
            ind[i] = 1
        return creator.IndividualReg(ind)

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # ----------------------------------------------------------------
    # Genetic operators
    # ----------------------------------------------------------------
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxTwoPoint)

    indpb = min(0.05, 3.0 / n_features)
    toolbox.register("mutate", tools.mutFlipBit, indpb=indpb)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # ----------------------------------------------------------------
    # Run GA (eaMuPlusLambda for elitism)
    # ----------------------------------------------------------------
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("min", np.min)
    stats.register("std", np.std)

    pop, log = algorithms.eaMuPlusLambda(
        pop, toolbox,
        mu=pop_size,
        lambda_=pop_size * 2,
        cxpb=GA_CXPB,
        mutpb=GA_MUTPB,
        ngen=n_gen,
        stats=stats,
        halloffame=hof,
        verbose=True
    )

    # ----------------------------------------------------------------
    # Results
    # ----------------------------------------------------------------
    best_ind     = hof[0]
    best_fitness = best_ind.fitness.values[0]

    selected_features = [
        descriptor_names[i]
        for i, bit in enumerate(best_ind)
        if bit == 1
    ]

    print("\n===== RESULTADO FINAL =====")
    print(f"Fitness  : {best_fitness:.4f}  (≈ RMSE = {-best_fitness:.4f})")
    print(f"Features : {len(selected_features)}")

    return selected_features, best_fitness, hof, log
