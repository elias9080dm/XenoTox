from deap import base, creator, tools, algorithms
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
import random

# ==========================================================
# GA Feature Selection (Regresión - Optimizado para precisión)
# ==========================================================


def ga_feature_selection(
    X,
    y,
    descriptor_names,
    n_gen=40,
    pop_size=60,
    random_state=42
):

    # ======================================================
    # Seeds
    # ======================================================
    np.random.seed(random_state)
    random.seed(random_state)

    X_np = X.values if hasattr(X, "values") else X
    n_features = X_np.shape[1]

    # ======================================================
    # CV (importante)
    # ======================================================
    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state
    )

    # ======================================================
    # Modelo base (NO lineal)
    # ======================================================
    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.05,
        max_iter=300,
        random_state=random_state
    )

    # ======================================================
    # Cache
    # ======================================================
    fitness_cache = {}
    MAX_CACHE = 10000

    # ======================================================
    # Fitness function
    # ======================================================
    def evaluate_individual(individual):

        key = tuple(individual)
        if key in fitness_cache:
            return fitness_cache[key]

        if len(fitness_cache) > MAX_CACHE:
            fitness_cache.clear()

        n_selected = sum(individual)

        MIN_FEATURES = max(5, int(0.01 * n_features))
        if n_selected < MIN_FEATURES:
            return (-1e6,)

        selected_idx = [i for i, bit in enumerate(individual) if bit == 1]
        X_sel = X_np[:, selected_idx]

        try:
            scores = cross_val_score(
                model,
                X_sel,
                y,
                cv=cv,
                scoring="neg_root_mean_squared_error",
                n_jobs=1
            )

            mean_score = scores.mean()   # negativo
            std_score = scores.std()

            rmse = -mean_score  # positivo

            # -----------------------------
            # Penalización ligera
            # -----------------------------
            lambda_penalty = 0.0005
            complexity_penalty = lambda_penalty * n_selected

            # -----------------------------
            # Penalización por inestabilidad
            # -----------------------------
            stability_penalty = 0.1 * std_score

            # -----------------------------
            # Fitness (maximizar)
            # -----------------------------
            fitness = -(rmse + complexity_penalty + stability_penalty)

        except Exception:
            fitness = -1e6

        fitness_cache[key] = (fitness,)
        return (fitness,)

    # ======================================================
    # DEAP setup
    # ======================================================
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # ======================================================
    # Inicialización mejorada (NO sesgada)
    # ======================================================
    MIN_FEATURES = max(5, int(0.01 * n_features))

    def init_individual():
        ind = [0] * n_features

        k = int(np.random.normal(
            loc=0.2 * n_features,
            scale=0.1 * n_features
        ))

        k = np.clip(k, MIN_FEATURES, n_features)

        selected = random.sample(range(n_features), int(k))

        for i in selected:
            ind[i] = 1

        return creator.Individual(ind)

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # ======================================================
    # Operadores genéticos
    # ======================================================
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxTwoPoint)

    indpb = min(0.05, 3.0 / n_features)
    toolbox.register("mutate", tools.mutFlipBit, indpb=indpb)

    toolbox.register("select", tools.selTournament, tournsize=3)

    # ======================================================
    # Ejecución GA
    # ======================================================
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("min", np.min)
    stats.register("std", np.std)

    pop, log = algorithms.eaMuPlusLambda(
        pop,
        toolbox,
        mu=pop_size,
        lambda_=pop_size * 3,
        cxpb=0.6,
        mutpb=0.4,
        ngen=n_gen,
        stats=stats,
        halloffame=hof,
        verbose=True
    )

    # ======================================================
    # Resultado final
    # ======================================================
    best_ind = hof[0]
    best_fitness = best_ind.fitness.values[0]

    selected_features = [
        descriptor_names[i]
        for i, bit in enumerate(best_ind)
        if bit == 1
    ]

    print("\n===== RESULTADO FINAL =====")
    print(f"Fitness: {best_fitness:.4f}")
    print(f"Número de features: {len(selected_features)}")

    return selected_features, best_fitness, hof, log
