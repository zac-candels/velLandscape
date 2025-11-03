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
from scipy.interpolate import CloughTocher2DInterpolator

import warnings
warnings.filterwarnings("ignore")

data_file = open("CA_landscape.txt", "r")
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

interp = CloughTocher2DInterpolator(list(zip(theta, postFrac)), f)

def CA_interp(theta, postFrac):
    val = interp(theta, postFrac)
    return val

# Define search space bounds
bounds = torch.tensor([[30, 0.2], [130, 0.8]])

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

        val = float(CA_interp(x0, x1))

        # store the *negative* of the objective
        results.append(val)

    # stack into a (batch_size x 1) tensor
    return torch.tensor(results, dtype=X.dtype).unsqueeze(-1)


# Set random seed for reproducibility
torch.manual_seed(10)

# Initialize with random points
n_init = 30
X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n_init, 2)
Y = objective(X)

# Optimization loop parameters
n_iterations = 100
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
    qEI = qExpectedImprovement(gp, best_f=best_f)
    
    # Optimize the acquisition function to get the next batch of points
    candidates, _ = optimize_acqf(
        qEI,
        bounds=bounds,
        q=batch_size,
        num_restarts=200,
        raw_samples=40000,
    )
    
    # Evaluate the objective at the new points
    new_Y = objective(candidates)
    
    # Update the dataset
    X = torch.cat([X, candidates], dim=0)
    Y = torch.cat([Y, new_Y], dim=0)
    
    # Print progress
    current_params = candidates[-1]
    theta, postFrac = current_params
    
    best_idx = Y.argmax()
    best_X = X[best_idx]
    best_Y = Y[best_idx]
    
    best_theta, best_postFrac = best_X
    print(f"Iteration {i+1}, Best value in current iteration: {new_Y[-1].item():.6f} at (theta={theta:.3f}, postFrac={postFrac:.3f})\n")
    print(f"Iteration {i+1}, Best value (all iterations): {Y.max().item():.6f} at theta=({best_theta:.3f}, {best_postFrac:.3f})\n")
    
    error_percent = 100*abs( Y.max().item() - 168.7883) / 168.7883
    print("percent error =", error_percent,"%\n\n\n")
    
    {}

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

# Create fine grid
theta_grid = np.linspace(theta.min(), theta.max(), 500)
postFrac_grid = np.linspace(postFrac.min(), postFrac.max(), 500)
Theta, PostFrac = np.meshgrid(theta_grid, postFrac_grid)

# Evaluate interpolating function
F_interp = CA_interp(Theta, PostFrac)

# Find maximum
max_index = np.unravel_index(np.argmax(F_interp), F_interp.shape)
theta_max = Theta[max_index]
postFrac_max = PostFrac[max_index]
f_max = F_interp[max_index]

print(f"Maximum interpolated value: {f_max:.6e}")
print(f"At theta = {theta_max:.3f}, postFrac = {postFrac_max:.3f}")