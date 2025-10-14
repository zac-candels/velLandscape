import numpy as np
from scipy.interpolate import Rbf
import pygad

# === Data ===
data = np.array([
    [100, 0.1, 0.00034609637384060474],
    [100, 0.3, -7.733846095047767e-07],
    [100, 0.5, -6.627373288151749e-07],
    [100, 0.7, 2.205559072785262e-07],
    [100, 0.9, 0.0002231715135631163],
    [80, 0.1, -1.211636337027777e-06],
    [80, 0.3, -1.969628517048235e-06],
    [80, 0.5, -2.342576732758492e-06],
    [80, 0.7, -8.828512976445789e-07],
    [80, 0.9, 2.46810229113222e-07],
    [85, 0.1, 0.00023883504985304312],
    [85, 0.3, -1.7567583401030129e-06],
    [85, 0.5, -1.3308129640324209e-06],
    [85, 0.7, -5.133976234629423e-07],
    [85, 0.9, -2.8542131355770607e-06],
    [90, 0.1, 0.00015319164872920762],
    [90, 0.3, -1.5671575882178282e-06],
    [90, 0.5, -9.75300760431573e-07],
    [90, 0.7, -2.3704087914270885e-07],
    [90, 0.9, 0.0001512297371032265],
    [95, 0.1, 0.00020006669346882188],
    [95, 0.3, -1.3998249420941369e-06],
    [95, 0.5, -7.72534840580562e-07],
    [95, 0.7, -4.0176301084497454e-08],
    [95, 0.9, 1.7985358529366716e-05]
])

# === Build RBF interpolator ===
theta = data[:, 0]
postFrac = data[:, 1]
f = data[:, 2]

rbf = Rbf(theta, postFrac, f, function='multiquadric', smooth=0)

# === Fitness function (PyGAD ≥ 2.20.0) ===
def fitness_func(ga_instance, solution, solution_idx):
    theta_val, postFrac_val = solution

    # Penalize out-of-bounds solutions
    if not (80 <= theta_val <= 100 and 0.1 <= postFrac_val <= 0.9):
        return -1e6

    # Evaluate the RBF interpolant
    value = float(rbf(theta_val, postFrac_val))

    # PyGAD maximizes fitness
    return value

# === GA setup ===
ga_instance = pygad.GA(
    num_generations=200,
    num_parents_mating=10,
    fitness_func=fitness_func,
    sol_per_pop=30,
    num_genes=2,
    gene_space=[{'low': 80, 'high': 100}, {'low': 0.1, 'high': 0.9}],
    mutation_percent_genes=20,
    mutation_type="random",
    mutation_by_replacement=True,
    crossover_type="single_point",
    parent_selection_type="rws",
    keep_parents=2,
    random_seed=42
)

# === Run the optimization ===
ga_instance.run()

# === Best solution ===
solution, solution_fitness, solution_idx = ga_instance.best_solution()
theta_opt, postFrac_opt = solution

print("=== PyGAD optimization result ===")
print(f"Best theta: {theta_opt:.4f}")
print(f"Best postFrac: {postFrac_opt:.4f}")
print(f"Maximum fitness value: {solution_fitness:.6e}")
print(f"Generations completed: {ga_instance.generations_completed}")

# === Plot convergence ===
ga_instance.plot_fitness()
