import subprocess
import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D 
from scipy.interpolate import Rbf
from io import StringIO

data_file = open("velLandscape_lx100_pts90.txt", "r")
data_str = data_file.read()

# Convert to numpy array
data = np.loadtxt(StringIO(data_str), delimiter=",")

# Extract columns
theta = data[:,0]
postFrac = data[:,1]
f = data[:,2]

# Create grid for interpolation
theta_grid = np.linspace(theta.min(), theta.max(), 100)
postFrac_grid = np.linspace(postFrac.min(), postFrac.max(), 100)
Theta, PostFrac = np.meshgrid(theta_grid, postFrac_grid)

# Interpolate data onto grid
F_grid = griddata((theta, postFrac), f, (Theta, PostFrac), method='cubic')

# Build smooth interpolating function using Radial Basis Functions
rbf = Rbf(theta, postFrac, f, function='multiquadric', smooth=0)

def run_simulation(theta: float, postFrac: float) -> float:
    return -rbf(theta, postFrac)


# --- scikit-optimize libraries ---
from skopt import gp_minimize
from skopt.space import Real
from skopt.callbacks import VerboseCallback

def print_best_so_far(res):
    current_best_idx = int(np.argmin(res.func_vals))
    current_best_val = -res.func_vals[current_best_idx]
    current_best_params = res.x_iters[current_best_idx]
    theta, postFrac = current_best_params
    print(f"[Iter {len(res.func_vals)}] Best objective so far: {current_best_val:.6f} at (theta={theta:.2f}, postFrac={postFrac:.2f})")


# Search space
search_space = [
    Real(80, 100, name='theta'),
    Real(0.1, 0.9, name='postFrac')
]

def objective_sk(params):
    theta, postFrac = params
    try:
        val = float(run_simulation(theta, postFrac))
    except Exception as e:
        print(f"Error at {params}: {e}")
        return 1e6
    return val  # minimize negative


def run_skopt():
    n_calls = 200 # Number of function evaluations
    
    xi=0.01 # controls exploration vs exploitation
    # Default value of xi is 0.01
    # Larger values of xi result in more exploration
    result = gp_minimize(func=objective_sk, dimensions=search_space,
                         acq_func='EI', n_initial_points=20,
                         n_calls=n_calls, random_state=41,
                         callback=[print_best_so_far], xi=xi)
    theta, postFrac = result.x
    print("== skopt best ==")
    print(f"theta={theta:.2e}, postFrac={postFrac:.2e}")
    print(f"Max value={-result.fun:.4f}")
    print("Number of iterations", n_calls)
    print("xi =", xi)


run_skopt()

#%% Plot surface
import matplotlib.pyplot as plt

# Create grid for interpolation
theta_grid = np.linspace(theta.min(), theta.max(), 100)
postFrac_grid = np.linspace(postFrac.min(), postFrac.max(), 100)
Theta, PostFrac = np.meshgrid(theta_grid, postFrac_grid)

# Interpolate data onto grid
F_grid = griddata((theta, postFrac), f, (Theta, PostFrac), method='cubic')

# Create surface plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# Plot surface
surf = ax.plot_surface(Theta, PostFrac, F_grid, cmap='viridis', edgecolor='none')
#wire = ax.plot_wireframe(Theta, PostFrac, F_grid, color='navy', linewidth=0.8)

# Labels and title
ax.set_xlabel(r'$\theta_E$')
ax.set_ylabel('postFraction')
ax.set_zlabel('droplet velocity')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=8, label='velocity')

plt.tight_layout()
plt.show()

# Build smooth interpolating function using Radial Basis Functions
rbf = Rbf(theta, postFrac, f, function='multiquadric', smooth=0)

# Create fine grid
theta_grid = np.linspace(theta.min(), theta.max(), 500)
postFrac_grid = np.linspace(postFrac.min(), postFrac.max(), 500)
Theta, PostFrac = np.meshgrid(theta_grid, postFrac_grid)

# Evaluate interpolating function
F_interp = rbf(Theta, PostFrac)

# Find maximum
max_index = np.unravel_index(np.argmax(F_interp), F_interp.shape)
theta_max = Theta[max_index]
postFrac_max = PostFrac[max_index]
f_max = F_interp[max_index]

print(f"Maximum interpolated value: {f_max:.6e}")
print(f"At theta = {theta_max:.3f}, postFrac = {postFrac_max:.3f}")
