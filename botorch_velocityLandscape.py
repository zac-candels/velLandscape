import torch
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.double)

torch.manual_seed(2)

from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.acquisition import qExpectedImprovement
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
import subprocess 
import os
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D 
from scipy.interpolate import Rbf
from io import StringIO
import numpy as np

import warnings
warnings.filterwarnings("ignore")

data_file = open("velLandscape_lx128_pts25.txt", "r")
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

# Define search space bounds
bounds = torch.tensor([[80, 0.1], [100, 0.9]])

input_tf = Normalize(
    d=2,                        # dimension of input
    bounds=bounds )   

# Define the objective function
def objective(X: torch.Tensor) -> torch.Tensor:
    """
    X: (batch_size x 2) tensor of x,y coordinates
    returns: (batch_size x 1) tensor of -simulation_value
    """
    results = []
    for x in X:
        # extract scalar floats
        x0 = float(x[0].item())
        x1 = float(x[1].item())

        val = float(rbf(x0, x1))

        # store the *negative* of the objective
        results.append(val)

    # stack into a (batch_size x 1) tensor
    return torch.tensor(results, dtype=X.dtype).unsqueeze(-1)


# Set random seed for reproducibility
torch.manual_seed(10)

# Initialize with random points
n_init = 20
X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n_init, 2)
Y = objective(X)

# Optimization loop parameters
n_iterations = 2
batch_size = 4

# Optimization loop
for i in range(n_iterations):
    # Fit a GP model to the current data
    gp = SingleTaskGP(
    train_X=X,               # shape (n,2)
    train_Y=Y,               # shape (n,1)
    input_transform=input_tf,
    outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    
    # Define the q-EI acquisition function
    best_f = Y.max().item()
    qEI = qLogExpectedImprovement(gp, best_f=best_f)
    
    # Optimize the acquisition function to get the next batch of points
    candidates, _ = optimize_acqf(
        qEI,
        bounds=bounds,
        q=batch_size,
        num_restarts=20,
        raw_samples=2000,
    )
    
    # Evaluate the objective at the new points
    new_Y = objective(candidates)
    
    # Update the dataset
    X = torch.cat([X, candidates], dim=0)
    Y = torch.cat([Y, new_Y], dim=0)
    
    # Print progress
    print(f"Iteration {i+1}, Best observed value: {Y.max().item():.6f}")
    best_idx = Y.argmax()
    best_X = X[best_idx]
    best_Y = Y[best_idx]
    print(f"Best point found: ({best_X[0]:.4f}, {best_X[1]:.4f})\n\n")

# Report the final result
best_idx = Y.argmax()
best_X = X[best_idx]
best_Y = Y[best_idx]
print(f"\nBest point found: ({best_X[0]:.4f}, {best_X[1]:.4f})")
print(f"Maximum value: {best_Y.item():.6f}")

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