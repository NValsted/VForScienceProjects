import random
from typing import Tuple

import numpy as np
from tqdm import tqdm


def _fitness_func(
    solution: dict[str, int], components: dict[str, np.ndarray], target: np.ndarray
) -> int:
    output = np.add.reduce([components[k][v] for k, v in solution.items()])
    fitness = 1.0 / np.sum(np.abs(target - output))
    return fitness


def _mutate(solution: dict[str, int], lower_bound: int, upper_bound: int) -> list[int]:
    return {
        k: max(min(v + random.randint(-1, 1), upper_bound), lower_bound)
        for k, v in solution.items()
    }


def fit_combination(
    target: np.ndarray,
    components: np.ndarray,
    iterations: int = 100,
    num_populations: int = 10,
    population_size: int = 25,
) -> Tuple[float, dict]:

    upper_bound = len(list(components.values())[0]) - 1
    populations = [
        [
            {k: random.randint(0, upper_bound) for k in components.keys()}
            for _ in range(population_size)
        ]
        for _ in range(num_populations)
    ]

    for _ in tqdm(range(iterations)):
        new_populations = []

        for population in populations:
            _, solution = max(
                (_fitness_func(solution, components, target), solution)
                for solution in population
            )
            new_populations.append(
                [_mutate(solution, 0, upper_bound) for _ in range(population_size)]
            )

        populations = new_populations

    best_score, best_solution = max(
        max(
            (_fitness_func(solution, components, target), solution)
            for solution in population
        )
        for population in populations
    )

    return best_score, best_solution
