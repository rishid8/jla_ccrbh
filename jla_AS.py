import os
import numpy as np
from numpy.linalg import inv, pinv
import scipy.linalg as la
import matplotlib.pyplot as plt
import emcee
import corner
import time
import scipy.integrate as integrate
from multiprocessing import Pool
from scipy.optimize import minimize
import scipy.stats as stats
from scipy.special import lambertw
plt.rcParams['text.usetex'] = True


# ------------- LOAD DESI BAO, HUBBLE CC and PANTHEON+SH0ES DATA------------


# Load DESI data

z_desi_bao_eff = np.array([0.295, 0.510, 0.706, 0.934, 1.321, 1.484, 2.330])  # this is the redshift we are using to calculate the theory observables.
data_bao = np.loadtxt("data/desi_mean.txt")
dist_dataBAO = data_bao[:, 1]
cov_bao = np.loadtxt("data/desi_cov.txt") # this is covariance matrix.
inv_cov_bao = inv(cov_bao) #Here, we are taking the inverse of it.


# Load CC data

data_H = np.loadtxt("data/CC.txt")
z_dataH = data_H[:, 0]
H_obs_data = data_H[:,1]
cov_H = np.loadtxt("data/CovMat.txt")
inv_cov_H = la.inv(cov_H)

# Load Patheon+SH0ES data


data_SN = np.loadtxt("data/Pantheon+SH0ES.dat", skiprows = 1, usecols = (2, 8))
z_dataSN = data_SN[:, 0]
m_dataSN = data_SN[:, 1]        # Disclaimer: This is m_B
cov_SN = np.loadtxt("data/Pantheon+SH0ES_STAT+SYS.cov")
reshaped_cov_SN = cov_SN.reshape(1701, 1701)
inv_cov_SN = la.inv(reshaped_cov_SN)




# ------------------------- COSMOLOGY -------------------------


# We set the speed of light in km/s
c_km_s = 299792.458

def E_z(z, Om, A):
    """Dimensionless expansion E(z) = H(z)/H0 for flat LCDM"""
    x = np.exp(A)
    Obh = (1.0 / x) * (np.exp(x * (1.0 - Om)) - 1.0)
    Ode = (1.0 / x) * np.log(1.0 + x * Obh * (1.0 + z)**3)
    return np.sqrt(Om * (1.0 + z)**3 + Ode)




def H_of_z(z, H0, Om, A):
    """The Hubble parameter at redshift z"""
    return H0 * E_z(z, Om, A)


def comoving_distance(z, H0, Om, A):
    """Comoving distance in Mpc for flat universe"""
    integrand = lambda zp: 1.0 / E_z(zp, Om, A)
    I, _ = integrate.quad(integrand, 0.0, z, epsabs=1e-8, epsrel=1e-8, limit=200)
    return (c_km_s / H0) * I


def luminosity_distance(z, H0, Om, A):
    return (1.0 + z) * comoving_distance(z, H0, Om, A)


def mu_theory(z, H0, Om, A):
    """Distance modulus: mu = 5 log10(D_L/Mpc) + 25 + M
    M acts as a free nuisance that shifts mu (this encapsulates
    absolute magnitude and calibration offsets). For now, I have
    set M = 0 and in the theory calculation and analytically
    marginalizng over M in the likelihood.
    """
    D_L = np.array([luminosity_distance(zi, H0, Om, A) for zi in np.atleast_1d(z)])
    mu = 5.0 * np.log10(D_L) + 25.0
    return mu


def DM_rd(z, H0, Om, A, rd):
    return comoving_distance(z, H0, Om, A) / rd


def DH_rd(z, H0, Om, A, rd):
    h = H0/100.0
    return c_km_s / H_of_z(z, H0, Om, A) / rd

def DV_rd(z, H0, Om, A, rd):
    return (z * DM_rd(z, H0, Om, A, rd)**2 * DH_rd(z, H0, Om, A, rd))**(1/3)



# ------------------------- LIKELIHOODS -------------------------


# BAO likelihood

def log_likelihood_BAO(H0, Om, A, rd):
    """
    Likelihood function for the DESI BAO data.
    The theory vector must be flattened to match the order of the desi_mean.txt data.
    """

    # Construct the full theory vector (must match the structure of desi_mean.txt)

    val_bao = np.array([
        DV_rd(z_desi_bao_eff[0], H0, Om, A, rd),
        DM_rd(z_desi_bao_eff[1], H0, Om, A, rd), DH_rd(z_desi_bao_eff[1], H0, Om, A, rd),
        DM_rd(z_desi_bao_eff[2], H0, Om, A, rd), DH_rd(z_desi_bao_eff[2], H0, Om, A, rd),
        DM_rd(z_desi_bao_eff[3], H0, Om, A, rd), DH_rd(z_desi_bao_eff[3], H0, Om, A, rd),
        DM_rd(z_desi_bao_eff[4], H0, Om, A, rd), DH_rd(z_desi_bao_eff[4], H0, Om, A, rd),
        DM_rd(z_desi_bao_eff[5], H0, Om, A, rd), DH_rd(z_desi_bao_eff[5], H0, Om, A, rd),
        DM_rd(z_desi_bao_eff[6], H0, Om, A, rd), DH_rd(z_desi_bao_eff[6], H0, Om, A, rd)
    ])

    # Calculate the residual vector (Data - Theory)
    delta_BAO = dist_dataBAO - val_bao

    # Calculate the Chi-squared value
    # Chi2 = Delta^T * C^{-1} * Delta
    Chi2_BAO = delta_BAO @ inv_cov_bao @ delta_BAO

    return -0.5 * Chi2_BAO



# SN likelihood

ones = np.ones(z_dataSN.size)

def log_likelihood_SN(H0, Om, A):
    # Analytically marginalizing over M, saves time
    mu0 = mu_theory(z_dataSN, H0, Om, A)
    r = m_dataSN - mu0

    # Compute a, b, c
    a = float(r @ (inv_cov_SN @ r))
    b = float(ones @ (inv_cov_SN @ r))
    c = float(ones @ (inv_cov_SN @ ones))

    chi2_marg = a - (b*b) / c

    lnL = -0.5 * chi2_marg - 0.5 * np.log(c)
    return lnL

# CC likelihood

def log_likelihood_CC(H0, Om, A):
    """
    Likelihood function for the CC data.
    """
    theory_H = np.array([H_of_z(z, H0, Om, A) for z in z_dataH])

    # Calculate the residual vector (Data - Theory)
    delta_CC = H_obs_data - theory_H

    # Calculate the Chi-squared value
    # Chi2 = Delta^T * C^{-1} * Delta
    Chi2_CC = delta_CC @ inv_cov_H @ delta_CC

    # Return the log-likelihood
    return -0.5 * Chi2_CC


# Add them all

def log_likelihood_total(params):
    # Assuming this parameter order for the MCMC sampler
    H0, Om, A, rd = params

    if not (40.0 < H0 < 100.0 and 0.0 < Om < 0.5 and -20.0 < A < 5.0 and 100.0 < rd < 200.0):
        return -np.inf # Return negative infinity for non-physical parameters

    lnL_BAO = log_likelihood_BAO(H0, Om, A, rd)
    lnL_CC = log_likelihood_CC(H0, Om, A)
    lnL_SN = log_likelihood_SN(H0, Om, A)
    #lnP_rd = log_prior_rd(rd)

    return lnL_BAO + lnL_CC + lnL_SN




# ----------------------- MCMC + postprocessing -----------------------


# --- Sampler settings ---

nwalkers = 20
nsteps = 200
burnin = 20
processes = max(1, min(6, (os.cpu_count() or 4) - 1))  # use up to 6 or cpu_count-1
seed = 12345
out_prefix = "AS"
save_results_to = "output/as/"

np.random.seed(seed)

# ---------- helper chi2 functions (for reporting) ----------

def chi2_BAO(H0, Om, A, rd):
    val_bao = np.array([
        DV_rd(z_desi_bao_eff[0], H0, Om, A, rd),
        DM_rd(z_desi_bao_eff[1], H0, Om, A, rd), DH_rd(z_desi_bao_eff[1], H0, Om, A, rd),
        DM_rd(z_desi_bao_eff[2], H0, Om, A, rd), DH_rd(z_desi_bao_eff[2], H0, Om, A, rd),
        DM_rd(z_desi_bao_eff[3], H0, Om, A, rd), DH_rd(z_desi_bao_eff[3], H0, Om, A, rd),
        DM_rd(z_desi_bao_eff[4], H0, Om, A, rd), DH_rd(z_desi_bao_eff[4], H0, Om, A, rd),
        DM_rd(z_desi_bao_eff[5], H0, Om, A, rd), DH_rd(z_desi_bao_eff[5], H0, Om, A, rd),
        DM_rd(z_desi_bao_eff[6], H0, Om, A, rd), DH_rd(z_desi_bao_eff[6], H0, Om, A, rd)
    ])
    delta = dist_dataBAO - val_bao
    return float(delta @ inv_cov_bao @ delta)

def chi2_SN(H0, Om, A):
    mu0 = mu_theory(z_dataSN, H0, Om, A)   # NOTE: this mu_theory must NOT subtract M
    r = m_dataSN - mu0
    # Compute a, b, c
    a = float(r @ (inv_cov_SN @ r))
    b = float(ones @ (inv_cov_SN @ r))
    c = float(ones @ (inv_cov_SN @ ones))
    chi2_marg = a - (b*b) / c
    return chi2_marg

def chi2_CC(H0, Om, A):
    H_model = np.array([H_of_z(z, H0, Om, A) for z in z_dataH])
    delta = H_obs_data - H_model
    return float(delta @ inv_cov_H @ delta)


# ---------- posterior wrapper (prior + likelihood) ----------

def log_prior(params):
    H0, Om, A, rd = params
    if not (40.0 < H0 < 100.0 and 0.0 < Om < 0.5 and -20.0 < A < 5.0 and 100.0 < rd < 200.0):
        return -np.inf

    # Strong Gaussian likelihood from Planck
    rd_PLANCK = 147.09
    rd_SIGMA = 0.26
    logP_rd = -0.5 * ((rd - rd_PLANCK) / rd_SIGMA)**2

    return logP_rd
    #return 0.0

def log_posterior(params):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf

    ll = log_likelihood_total(params)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll



# ---------- find MAP (maximize log posterior) for initialization ----------

x0 = np.array([70.0, 0.30, 2.0, 147.0])
def neglogpost(x):
    lp = log_posterior(x)
    if not np.isfinite(lp):
        return 1e30
    return -lp

# Using a robust method: Powell/Nelder-Mead, sometimes Powell might not work
res = minimize(neglogpost, x0, method='Powell', options={'maxiter':2000, 'disp': False})
if not res.success:
    print("Optimizer warning:", res.message)
p_map = res.x
print("MAP estimate (from optimizer):", p_map, "logpost =", -res.fun)



# ---------- prepare initial walker positions around MAP ----------

ndim = 4
p0_std = np.array([0.5, 0.02, 0.1, 0.5])   # small spread around MAP; tweak if necessary
p0 = p_map + p0_std * np.random.randn(nwalkers, ndim)

# ensure all p0 are inside prior bounds; if any outside, clipping/pushing them in
for i in range(nwalkers):
    H0, Om, A, rd = p0[i]
    p0[i,0] = np.clip(p0[i,0], 41.0, 99.0)
    p0[i,1] = np.clip(p0[i,1], 1e-3, 0.999)
    p0[i,2] = np.clip(p0[i,2], -19.9, 4.9)
    p0[i,3] = np.clip(p0[i,3], 101.0, 199.0)

# ---------- configure mixed moves ----------

moves = [
    (emcee.moves.StretchMove(a=2.0), 0.30),
    (emcee.moves.DEMove(), 0.50),
    (emcee.moves.DESnookerMove(), 0.20)
]

# ---------- run emcee with Pool ----------

print(f"Running emcee with {nwalkers} walkers, {nsteps} steps, processes={processes}")
tstart = time.time()
with Pool(processes=processes) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, pool=pool, moves=moves)
    sampler.run_mcmc(p0, nsteps, progress=True)
tend = time.time()
print("Sampling finished in {:.1f} s".format(tend - tstart))

# ---------- save chain and lnprob ----------

chain = sampler.get_chain()                        # (nsteps, nwalkers, ndim)
lnprob = sampler.get_log_prob()                    # (nsteps, nwalkers)
np.save(save_results_to + out_prefix + "_chain.npy", chain)
np.save(save_results_to + out_prefix + "_lnprob.npy", lnprob)

# ---------- flatten and discard burn-in ----------

flat = sampler.get_chain(discard=burnin, thin=1, flat=True)
flat_lnprob = sampler.get_log_prob(discard=burnin, thin=1, flat=True)

# ---------- best-fit from samples (highest posterior) ----------

best_idx = np.nanargmax(flat_lnprob)
best = flat[best_idx]
best_H0, best_Om, best_A, best_rd = best
print("Best sample (max posterior) from chain:", best)
print("Log-posterior at best sample:", float(flat_lnprob[best_idx]))

# ---------- also evaluate MAP/logpost from optimizer (compare) ----------

print("Optimizer MAP:", p_map, "logpost:", -res.fun)

# ---------- compute chi2 and goodness-of-fit metrics ----------

chi2_bao = chi2_BAO(best_H0, best_Om, best_A, best_rd)
chi2_cc  = chi2_CC(best_H0, best_Om, best_A)
chi2_sn  = chi2_SN(best_H0, best_Om, best_A)

chi2_tot = chi2_bao + chi2_cc + chi2_sn

Ndata = len(dist_dataBAO) + len(z_dataH) + len(z_dataSN)
k = ndim  # number of free parameters
ndof = Ndata - k
pval = 1.0 - stats.chi2.cdf(chi2_tot, ndof)

# For comparision with other models
AIC = chi2_tot + 2.0 * k
BIC = chi2_tot + k * np.log(Ndata)

print(f"Chi2: BAO = {chi2_bao:.2f}, CC = {chi2_cc:.2f}, SN = {chi2_sn:.2f}, total = {chi2_tot:.2f}")
print(f"Ndata = {Ndata}, ndof = {ndof}, k = {k}")
print(f"AIC = {AIC:.2f}, BIC = {BIC:.2f}, reduced-chi2 = {chi2_tot/ndof:.3f}, P-Value = {pval:.2f}")

# ---------- produce trace plots ----------

def plot_traces(chain, names, outname):
    nsteps, nwalkers, ndim = chain.shape
    fig, axes = plt.subplots(ndim, 1, figsize=(10, 2.2*ndim), sharex=True)
    x = np.arange(nsteps)
    for i in range(ndim):
        ax = axes[i]
        for w in range(nwalkers):
            ax.plot(x, chain[:, w, i], alpha=0.3)
        ax.set_ylabel(names[i])
    axes[-1].set_xlabel('step')
    plt.tight_layout()
    fig.savefig(outname, dpi=150)
    plt.close(fig)

plot_traces(chain, ['H0', 'Om', 'A', 'rd'], save_results_to + out_prefix + "_traces.png")

# ---------- corner plot of posterior marginalized samples ----------

custom_levels = [0.6827, 0.9545]
labels = [r'$H_0$', r'$\Omega_m$', r'$log(Q_{AS})$', r'$r_d$']
fig = corner.corner(flat, labels=labels, show_titles=True, title_fmt=".3f", plot_datapoints=False, levels=custom_levels)
fig.savefig(save_results_to + out_prefix + "_corner.png")
plt.close(fig)
# ---------- data vs model quick-plots ----------

# H(z)
z_plot = np.linspace(0.0, max(z_dataH)*1.05, 200)
H_mod_plot = np.array([H_of_z(z, best_H0, best_Om, best_A) for z in z_plot])

plt.figure(figsize=(8,5))
plt.errorbar(z_dataH, H_obs_data, yerr=np.sqrt(np.diag(cov_H)), fmt='o', markersize=3, label='CC data')
plt.plot(z_plot, H_mod_plot, '-', label=f'model H(z) (H0={best_H0:.2f}, Om={best_Om:.3f}, A={best_A:.4f})')
plt.xlabel('z'); plt.ylabel('H(z) [km/s/Mpc]'); plt.legend(); plt.grid(True)
plt.savefig(save_results_to + out_prefix + "_Hz.png", dpi=150)
plt.close()

# SN mu
z_sn_plot = np.linspace(0.0, max(z_dataSN)*1.05, 300)
mu_mod_plot = mu_theory(z_sn_plot, best_H0, best_Om, best_A)
mu0_best = mu_theory(z_dataSN, best_H0, best_Om, best_A)
r_best = m_dataSN - mu0_best
xx = la.solve(reshaped_cov_SN, r_best, assume_a='pos')
b = float(ones @ xx)
yy = la.solve(reshaped_cov_SN, ones, assume_a='pos')
c = float(ones @ yy)
M_hat = b / c

# corrected observed mu for plotting
mu_obs_plot = m_dataSN - M_hat

# plot
plt.figure(figsize=(8,5))
plt.errorbar(z_dataSN, mu_obs_plot, yerr=np.sqrt(np.diag(reshaped_cov_SN)), fmt='o', markersize=2, alpha=0.6)
plt.plot(z_sn_plot, mu_mod_plot, '-', label=f'model mu(z) (H0={best_H0:.2f}, Om={best_Om:.3f})')
plt.xlabel('z'); plt.ylabel('mu'); plt.legend(); plt.grid(True)
plt.savefig(save_results_to + out_prefix + "_mu.png", dpi=150)
plt.close()

# BAO comparison (the observed vector vs theory)
theory_bao_val = np.array([
    DV_rd(z_desi_bao_eff[0], best_H0, best_Om, best_A, best_rd),
    DM_rd(z_desi_bao_eff[1], best_H0, best_Om, best_A, best_rd), DH_rd(z_desi_bao_eff[1], best_H0, best_Om, best_A, best_rd),
    DM_rd(z_desi_bao_eff[2], best_H0, best_Om, best_A, best_rd), DH_rd(z_desi_bao_eff[2], best_H0, best_Om, best_A, best_rd),
    DM_rd(z_desi_bao_eff[3], best_H0, best_Om, best_A, best_rd), DH_rd(z_desi_bao_eff[3], best_H0, best_Om, best_A, best_rd),
    DM_rd(z_desi_bao_eff[4], best_H0, best_Om, best_A, best_rd), DH_rd(z_desi_bao_eff[4], best_H0, best_Om, best_A, best_rd),
    DM_rd(z_desi_bao_eff[5], best_H0, best_Om, best_A, best_rd), DH_rd(z_desi_bao_eff[5], best_H0, best_Om, best_A, best_rd),
    DM_rd(z_desi_bao_eff[6], best_H0, best_Om, best_A, best_rd), DH_rd(z_desi_bao_eff[6], best_H0, best_Om, best_A, best_rd)
])
plt.figure(figsize=(10,4))
plt.plot(dist_dataBAO, 'o', label='DESI BAO data vector')
plt.plot(theory_bao_val, 'x', label='theory vector')
plt.legend(); plt.grid(True)
plt.xlabel('index (flattened BAO vector)'); plt.ylabel('observable (rd-scaled)')
plt.savefig(save_results_to + out_prefix + "_bao_compare.png", dpi=150)
plt.close()

# ---------- save best-fit and stats ----------

np.savetxt(save_results_to + out_prefix + "_bestfit.txt", np.append(best, [chi2_tot, chi2_tot/ndof, AIC, BIC]).reshape(1,-1),
           header="H0 Om A rd chi2 chi2_red AIC BIC")

print("Outputs saved with prefix:", out_prefix)

# --------- median and 1sigma credible intervals ---------

param_names = ["H0", "Om", "A", "rd"]      # A is log(x)
H0_s, Om_s, A_s, rd_s = flat[:,0], flat[:,1], flat[:,2], flat[:,3]

# ---------- compute summary statistics ----------
out_lines = []

# --- 1. Median ± 1σ for H0 and Om ---
for name, samples in zip(["H0", "Om", "rd"], [H0_s, Om_s, rd_s]):
    median = np.percentile(samples, 50)
    lo = median - np.percentile(samples, 16)
    hi = np.percentile(samples, 84) - median
    out_lines.append(f"{name}  {median:.6g}  {lo:.6g}  {hi:.6g}")

# --- 2. Lower bound only for x = exp(A) ---
A_samples = A_s
x_samples = np.exp(A_samples)

# 95% lower bound → 5th percentile
x95 = np.percentile(x_samples, 5.0)
out_lines.append(f"x_lower_95   {x95:.6g}")

# ---------- Save to file ----------
header = "parameter   median   minus_1sigma   plus_1sigma\n" \
         "(for x only: value is 95% lower bound)\n"

with open(save_results_to + out_prefix + "_summary.txt", "w") as f:
    f.write(header)
    for line in out_lines:
        f.write(line + "\n")

print("Saved:", out_prefix + "_summary.txt")





def gelman_rubin(chains):
    """
    Compute Gelman-Rubin R-hat for MCMC chains.

    Parameters
    ----------
    chains : ndarray
        Shape = (nchains, nsamples, ndim)
        nchains independent chains or walkers.

    Returns
    -------
    rhat : ndarray
        R-hat values for each parameter (ndim).
    """
    chains = np.array(chains)
    nchains, nsamples, ndim = chains.shape

    # Mean of each chain
    chain_means = np.mean(chains, axis=1)    # shape (nchains, ndim)

    # Variance of each chain
    chain_vars = np.var(chains, axis=1, ddof=1)    # shape (nchains, ndim)

    # Overall mean
    mean_total = np.mean(chain_means, axis=0)      # shape (ndim,)

    # Between-chain variance B
    B = nsamples * np.var(chain_means, axis=0, ddof=1)

    # Within-chain variance W
    W = np.mean(chain_vars, axis=0)

    # Estimate of marginal posterior variance
    var_hat = ( (nsamples - 1) * W + B ) / nsamples

    # Gelman–Rubin R-hat
    R_hat = np.sqrt(var_hat / W)

    return R_hat

chains = np.swapaxes(chain, 0, 1)    # now (nwalkers, nsteps, ndim)

rhat = gelman_rubin(chains)
print("R_hat:", rhat)





