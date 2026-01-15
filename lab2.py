import random
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import pandas as pd
import streamlit as st


@dataclass
class GAProblem:
    name: str
    chromosome_type: str
    dim: int
    bounds: Tuple[float, float] | None
    fitness_fn: Callable[[np.ndarray], float]


def make_custom_problem(dim: int = 80) -> GAProblem:
   
    def fitness(x: np.ndarray) -> float:
        ones = np.sum(x)
        return float(80 - abs(ones - 40))

    return GAProblem(
        name="80-bit Genetic Algorithm Problem",
        chromosome_type="bit",
        dim=dim,
        bounds=None,
        fitness_fn=fitness,
    )



def init_population(problem: GAProblem, pop_size: int, rng: np.random.Generator):
    return rng.integers(0, 2, size=(pop_size, problem.dim), dtype=np.int8)


def evaluate_population(pop, problem: GAProblem):
    return np.array([problem.fitness_fn(ind) for ind in pop])


def tournament_selection(fitness, k, rng):
    idx = rng.integers(0, len(fitness), size=k)
    return idx[np.argmax(fitness[idx])]


def one_point_crossover(p1, p2, rng):
    point = rng.integers(1, len(p1))
    c1 = np.concatenate([p1[:point], p2[point:]])
    c2 = np.concatenate([p2[:point], p1[point:]])
    return c1, c2


def bit_mutation(x, rate, rng):
    mask = rng.random(x.shape) < rate
    x = x.copy()
    x[mask] = 1 - x[mask]
    return x



def run_ga(problem, pop_size, generations):
    rng = np.random.default_rng(42)

    crossover_rate = 0.9
    mutation_rate = 0.01
    tournament_k = 3
    elitism = 2

    population = init_population(problem, pop_size, rng)
    fitness = evaluate_population(population, problem)

    history = {"Best": [], "Average": [], "Worst": []}

    for gen in range(generations):
       
        best = np.max(fitness)
        avg = np.mean(fitness)
        worst = np.min(fitness)

        history["Best"].append(best)
        history["Average"].append(avg)
        history["Worst"].append(worst)

       
        elite_idx = np.argsort(fitness)[-elitism:]
        elites = population[elite_idx]

       
        new_pop = []

        while len(new_pop) < pop_size - elitism:
            i1 = tournament_selection(fitness, tournament_k, rng)
            i2 = tournament_selection(fitness, tournament_k, rng)

            p1, p2 = population[i1], population[i2]

            if rng.random() < crossover_rate:
                c1, c2 = one_point_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = bit_mutation(c1, mutation_rate, rng)
            c2 = bit_mutation(c2, mutation_rate, rng)

            new_pop.append(c1)
            if len(new_pop) < pop_size - elitism:
                new_pop.append(c2)

        population = np.vstack([new_pop, elites])
        fitness = evaluate_population(population, problem)

    best_idx = np.argmax(fitness)
    return population[best_idx], fitness[best_idx], pd.DataFrame(history)



st.set_page_config(page_title="Genetic Algorithm Web App", page_icon="🧬", layout="wide")
st.title("Genetic Algorithm – Bit Pattern Generator")
st.write("**Population:** 300 | **Chromosome Length:** 80 | **Generations:** 50")
st.write("**Fitness peaks at 40 ones (Max Fitness = 80)**")

if st.button("Run Genetic Algorithm"):
    problem = make_custom_problem(80)
    best_solution, best_fitness, history_df = run_ga(
        problem=problem,
        pop_size=300,
        generations=50,
    )

    st.subheader("Fitness Progress Across Generations")
    st.line_chart(history_df)

    st.subheader("Best Chromosome Found")
    bitstring = "".join(map(str, best_solution.tolist()))
    st.code(bitstring, language="text")

    st.write("**Number of Ones:**", int(np.sum(best_solution)))
    st.write("**Best Fitness:**", best_fitness)
