import numpy as np
import matplotlib.pyplot as plt
import emcee

# ==========================================
# 1. Data Setup (Reading from data.txt)
# ==========================================
try:
    data = np.loadtxt("data.txt")
    d = data[:, 0]      # First column is the data vector 'd'
    sigma = data[:, 1]  # Second column is the error vector 'sigma'
except FileNotFoundError:
    print("Error: Could not find 'data.txt'. Please ensure it is in the same folder as this script.")
    raise

# The index array 'i' of length n
n = len(d)
i_vals = np.arange(n) 

# ==========================================
# Q2: Python Model Functions
# ==========================================
def model_1(p, n):
    """Model 1: d_i = p0 * i + p1"""
    i = np.arange(n)
    return p[0] * i + p[1]

def model_2(p, n):
    """Model 2: d_i = p0 * i^{p1}"""
    i = np.arange(n)
    # Using np.errstate to ignore warnings when evaluating 0 to a power
    with np.errstate(divide='ignore', invalid='ignore'):
        return p[0] * (i ** p[1])

# ==========================================
# Q3: Gaussian Log-Likelihood Functions
# ==========================================
def log_likelihood_1(p, d, sigma):
    m = model_1(p, len(d))
    return -0.5 * np.sum(((d - m) / sigma) ** 2)

def log_likelihood_2(p, d, sigma):
    m = model_2(p, len(d))
    return -0.5 * np.sum(((d - m) / sigma) ** 2)

# Define uniform priors to keep walkers in sensible regions
def log_prior(p):
    p0, p1 = p
    if -10.0 < p0 < 10.0 and 0.0 < p1 < 10.0: # p1 > 0 to prevent 0^negative errors
        return 0.0
    return -np.inf

# Full log-probability (Prior + Likelihood)
def log_probability_1(p, d, sigma):
    lp = log_prior(p)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_1(p, d, sigma)

def log_probability_2(p, d, sigma):
    lp = log_prior(p)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_2(p, d, sigma)

# ==========================================
# Q4: MCMC Sampling using emcee
# ==========================================
nwalkers = 16
ndim = 2
nsteps = 2000

# Initialize walkers
initial_p1 = [0.3, 0.1] + 1e-4 * np.random.randn(nwalkers, ndim)
sampler1 = emcee.EnsembleSampler(nwalkers, ndim, log_probability_1, args=(d, sigma))
sampler1.run_mcmc(initial_p1, nsteps, progress=True)

initial_p2 = [1.0, 0.5] + 1e-4 * np.random.randn(nwalkers, ndim)
sampler2 = emcee.EnsembleSampler(nwalkers, ndim, log_probability_2, args=(d, sigma))
sampler2.run_mcmc(initial_p2, nsteps, progress=True)

# Discard the first 500 steps as "burn-in"
burnin = 500
flat_samples1 = sampler1.get_chain(discard=burnin, flat=True)
flat_samples2 = sampler2.get_chain(discard=burnin, flat=True)

# Get the best fit parameters (using median)
best_p1 = np.median(flat_samples1, axis=0)
best_p2 = np.median(flat_samples2, axis=0)

# ==========================================
# Q5: Plot Data & Theory, Determine Best Fit
# ==========================================
plt.figure(figsize=(10, 6))
plt.errorbar(i_vals, d, yerr=sigma, fmt=".k", capsize=3, label="Data")

theory1 = model_1(best_p1, n)
theory2 = model_2(best_p2, n)

plt.plot(i_vals, theory1, "r-", label=f"Model 1: $p_0$={best_p1[0]:.2f}, $p_1$={best_p1[1]:.2f}")
plt.plot(i_vals, theory2, "b--", label=f"Model 2: $p_0$={best_p2[0]:.2f}, $p_1$={best_p2[1]:.2f}")

plt.xlabel("Index $i$")
plt.ylabel("$d_i$")
plt.legend()
plt.title("Question 5: Data vs Theory")
plt.show()

# Calculate Chi-Squared
chi2_1 = np.sum(((d - theory1) / sigma) ** 2)
chi2_2 = np.sum(((d - theory2) / sigma) ** 2)
print(f"Chi-Squared for Model 1: {chi2_1:.2f}")
print(f"Chi-Squared for Model 2: {chi2_2:.2f}")
if chi2_1 < chi2_2:
    print("-> Model 1 is the best-fitting model overall.")
else:
    print("-> Model 2 is the best-fitting model overall.")

# ==========================================
# Q6: Plot flattened chain to verify burn-in
# ==========================================
chain1 = sampler1.get_chain()

plt.figure(figsize=(10, 4))
plt.plot(chain1[:, :, 0], color="k", alpha=0.3)
plt.axvline(burnin, color="red", linestyle="--", label="Burn-in threshold")
plt.xlabel("Step Number")
plt.ylabel("$p_0$ (Model 1)")
plt.title("Question 6: MCMC Trace (Verifying Burn-in)")
plt.legend()
plt.show()

# ==========================================
# Q7: Posterior Histogram for Model 2
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(flat_samples2[:, 0], bins=30, color="blue", alpha=0.7)
axes[0].set_xlabel("$p_0$")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Posterior for $p_0$ (Model 2)")

axes[1].hist(flat_samples2[:, 1], bins=30, color="green", alpha=0.7)
axes[1].set_xlabel("$p_1$")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Posterior for $p_1$ (Model 2)")

plt.suptitle("Question 7: Posterior Probabilities for Model 2")
plt.tight_layout()
plt.show()