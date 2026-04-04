from deap import base, creator, tools, algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import StratifiedKFold
import random


def ga_feature_selection(X, y, descriptor_names):

    def evaluate_individual(individual, X, y):
        n_selected = sum(individual)
        MIN_FEATURES = max(5, int(0.01 * len(individual)))

        # Penalization for too few features
        if n_selected < MIN_FEATURES:
            return -1.0,

        selected_indices = [i for i, bit in enumerate(individual) if bit == 1]
        X_selected = X[:, selected_indices]

        try:
            model = LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="liblinear",
                class_weight="balanced",
                max_iter=5000
            )

            mcc_scorer = make_scorer(matthews_corrcoef)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            scores = cross_val_score(model, X_selected, y,
                                     cv=cv, scoring=mcc_scorer)

            # Penalization complexity
            alpha = 0.5
            fitness = scores.mean() - alpha * (n_selected / len(individual))
            return fitness,

        except Exception as e:
            print("Error in evaluate_individual:", repr(e))
        raise

    n_features = X.shape[1]

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_bool, n=n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_individual,
                     X=X.values, y=y)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.005)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Genetic Algorithm parameters
    np.random.seed(42)
    random.seed(42)
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("min", np.min)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3,
                                   ngen=30, stats=stats, halloffame=hof, verbose=True)

    # Display best solution
    best_ind = hof[0]

    selected_features = [
        descriptor_names[i]
        for i, bit in enumerate(best_ind)
        if bit == 1
    ]

    # Summary
    print(f"\n Selected features: {len(selected_features)}")
    print(" Descriptors:")
    for i, f in enumerate(selected_features, 1):
        print(f"{i:2d}. {f}")

    return selected_features
