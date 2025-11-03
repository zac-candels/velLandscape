import numpy as np
from skopt.space import Real
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern
from scipy.interpolate import CloughTocher2DInterpolator
from io import StringIO
from scipy.interpolate import griddata

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
# Use your existing interpolation
interp = CloughTocher2DInterpolator(list(zip(theta, postFrac)), f)

def run_simulation(theta, postFrac):
    val = interp(theta, postFrac)
    if np.isnan(val):
        return 1e6
    return -val  # negative for minimization

# --- Quasi-sequential batch BO parameters ---
bounds = [(30, 130), (0.2, 0.8)]
n_init = 20
batch_size = 10
n_iterations = 100  # outer iterations
xi = 0.01  # exploration-exploitation tradeoff

# Initialize random points
X = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], size=(n_init, 2))
Y = np.array([run_simulation(x[0], x[1]) for x in X])

# Use a GP model manually
kernel = Matern(nu=2.5)
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
gp.fit(X, Y)

def expected_improvement(X_candidate, X_train, Y_train, gp, xi=0.01):
    mu, sigma = gp.predict(X_candidate, return_std=True)
    mu_sample_opt = np.min(Y_train)
    
    with np.errstate(divide='warn'):
        imp = mu_sample_opt - mu - xi
        Z = imp / sigma
        from scipy.stats import norm
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    return ei

def propose_next_point(gp, X_train, Y_train, bounds):
    # Random restarts
    n_restarts = 50
    candidates = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], size=(n_restarts, 2))
    eis = expected_improvement(candidates, X_train, Y_train, gp, xi)
    best_idx = np.argmax(eis)
    return candidates[best_idx].reshape(1, -1)

# --- Quasi-sequential batch BO loop ---
for iteration in range(n_iterations):
    batch_X = []
    temp_X = X.copy()
    temp_Y = Y.copy()
    
    # Select q points sequentially with fantasy updates
    for b in range(batch_size):
        x_next = propose_next_point(gp, temp_X, temp_Y, bounds)
        batch_X.append(x_next.flatten())
        # Fantasy update using GP mean
        y_fantasy = gp.predict(x_next)
        temp_X = np.vstack([temp_X, x_next])
        temp_Y = np.append(temp_Y, y_fantasy)
        # Optionally, refit GP on temp_X/temp_Y to emulate sequential updates
        gp.fit(temp_X, temp_Y)
    
    batch_X = np.array(batch_X)
    
    # Evaluate all batch points in parallel (simulated here)
    batch_Y = np.array([run_simulation(x[0], x[1]) for x in batch_X])
    
    # Update the GP with the real outcomes
    X = np.vstack([X, batch_X])
    Y = np.append(Y, batch_Y)
    gp.fit(X, Y)
    
    # Print best so far
    best_idx = np.argmin(Y)
    best_val = -Y[best_idx]
    best_params = X[best_idx]
    print(f"[Iter {(iteration+1)}] Best so far: {best_val:.6f} at theta={best_params[0]:.2f}, postFrac={best_params[1]:.2f}")
    
    error_percent = 100*abs( best_val - 168.7883) / 168.7883
    print("percent error =", error_percent)

print("\nFinal best point:", X[np.argmin(Y)])
print("Final best value:", -np.min(Y))
