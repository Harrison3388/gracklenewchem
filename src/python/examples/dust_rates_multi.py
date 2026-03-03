"""
dust_rates_multi.py
Multi-metallicity dust rates analysis.
Each metallicity gets its own row (2 panels: timescales + rates),
all sharing the same x-axis.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yt
import warnings
import os
warnings.filterwarnings("ignore")

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9,
                     "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 6})

# -----------------------------------------------------------------------
# Constants & dust model parameters
# -----------------------------------------------------------------------
mass_hydrogen_cgs    = 1.67262171e-24
sec_per_Myr          = 3.155e13
sec_per_year         = 3.155e7
dust_grainsize       = 0.1
dust_growth_tauref   = 0.004
dust_growth_densref  = 2.3e-22
SolarMetalFractionByMass = 0.01295
t_ref                = 20.0
sne_coeff            = 1.0
sne_shockspeed       = 100.0
dust_destruction_eff = 0.3
M_sun_cgs            = 1.989e33
cm_per_mpc           = 3.086e24
sne_rate_default     = 1.0

density_units = mass_hydrogen_cgs
tbase1        = sec_per_Myr
urho          = density_units
uxyz          = cm_per_mpc
Ms100_code    = (6800.0 * sne_coeff * (100.0/sne_shockspeed)**2
                 * M_sun_cgs / (urho * uxyz**3))
tau_ref_s     = dust_growth_tauref * 1e9 * sec_per_year

# -----------------------------------------------------------------------
# Dataset list: (filename, metallicity label)
# -----------------------------------------------------------------------
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
met_indices = [-1, -2, -3, -4, -5, -6, 0]
datasets = []
for mi in met_indices:
    fname = os.path.join(DATA_DIR, f"freefalltest_{mi}.h5")
    if os.path.exists(fname):
        datasets.append((fname, mi))

print(f"Found {len(datasets)} datasets: {[d[1] for d in datasets]}")

# -----------------------------------------------------------------------
# Helper: compute all quantities for one dataset
# -----------------------------------------------------------------------
def compute_dust_data(filepath):
    ds = yt.load(filepath)
    d  = ds.data
    def v(field):
        return np.array(d[("data", field)])

    time_s   = v("time")
    density  = v("density")
    dust_den = v("dust_density")
    metal_den = v("metal_density")
    gas_den  = density
    nH_gas   = gas_den / mass_hydrogen_cgs
    T_gas    = v("temperature")
    cool_time = v("cooling_time")

    dt_s = np.diff(time_s, prepend=time_s[0])
    dt_s[0] = dt_s[1]

    # tau_sput
    tau_sput_s = (1.7e8 * sec_per_year * (dust_grainsize / 0.1)
                  * (1.0e-27 / gas_den)
                  * (np.power(2.0e6 / T_gas, 2.5) + 1.0))
    tau_sput_yr = tau_sput_s / sec_per_year

    # tau_accr
    rho_metal_code = metal_den / mass_hydrogen_cgs
    rho_metal_code_safe = np.maximum(rho_metal_code, 1e-20)
    tau_accr_s = (tau_ref_s
                  * (dust_growth_densref / mass_hydrogen_cgs)
                  * np.sqrt(t_ref / T_gas)
                  * (SolarMetalFractionByMass / rho_metal_code_safe))
    tau_accr_yr = tau_accr_s / sec_per_year

    # tau_dest
    try:
        sne_rate_arr = v("sne_rate")
    except Exception:
        sne_rate_arr = np.full_like(gas_den, sne_rate_default)
    rho_gas_code = gas_den / density_units
    tau_dest_code = rho_gas_code / (Ms100_code * sne_rate_arr * dust_destruction_eff)
    tau_dest_s = tau_dest_code * tbase1
    tau_dest_yr = tau_dest_s / sec_per_year

    # Reference timescales
    t_ff_yr = np.sqrt(3*np.pi / (32 * 6.674e-8 * density)) / sec_per_year
    cool_time_yr = np.abs(cool_time) / sec_per_year

    # Rates
    frac_metal_avail = np.where(
        metal_den > 0,
        np.clip(metal_den / (dust_den + metal_den), 0, 1), 0.0)
    dM_growth_rate = frac_metal_avail * dust_den / tau_accr_s
    dM_sput_rate   = dust_den / tau_sput_s * 3.0
    dM_dest_rate   = np.where(sne_rate_arr > 0,
                               np.minimum(dust_den / tau_dest_s, dust_den / dt_s),
                               0.0)
    dM_total_rate  = dM_growth_rate - dM_sput_rate - dM_dest_rate

    return dict(
        x=nH_gas, tau_sput_yr=tau_sput_yr, tau_accr_yr=tau_accr_yr,
        tau_dest_yr=tau_dest_yr, t_ff_yr=t_ff_yr, cool_time_yr=cool_time_yr,
        dM_growth_rate=dM_growth_rate, dM_sput_rate=dM_sput_rate,
        dM_dest_rate=dM_dest_rate, dM_total_rate=dM_total_rate,
        nsteps=len(nH_gas))

# -----------------------------------------------------------------------
# Load all datasets
# -----------------------------------------------------------------------
all_data = []
for fname, mi in datasets:
    print(f"  Loading Z = 10^{mi} Z_sun  ({fname}) ...")
    all_data.append((mi, compute_dust_data(fname)))

# -----------------------------------------------------------------------
# Figure: N_rows x 2 cols, shared x-axis
# -----------------------------------------------------------------------
n_rows = len(all_data)
fig, axes = plt.subplots(n_rows, 2, figsize=(14, 2.8 * n_rows),
                         sharex=True, squeeze=False)
fig.subplots_adjust(hspace=0.08)
lw = 0.9

for row, (mi, dd) in enumerate(all_data):
    x = dd["x"]
    ax1 = axes[row, 0]
    ax2 = axes[row, 1]

    Z_label = f"$Z = 10^{{{mi}}}\\,Z_\\odot$" if mi != 0 else r"$Z = Z_\odot$"

    # --- Left: Timescales ---
    ax1.loglog(x, dd["tau_sput_yr"], color="tab:red", lw=lw,
               label=r"$\tau_{\rm sput}$")
    ax1.loglog(x, dd["tau_accr_yr"], color="tab:blue", lw=lw, ls="--",
               label=r"$\tau_{\rm accr}$")
    ax1.loglog(x, np.clip(dd["tau_dest_yr"], 1e-20, None), color="tab:orange",
               lw=lw, ls=":", label=r"$\tau_{\rm dest}$ (SNe)")
    ax1.loglog(x, dd["t_ff_yr"], color="gray", lw=0.6, ls=":",
               label=r"$t_{\rm ff}$")
    ax1.loglog(x, dd["cool_time_yr"], color="tab:green", lw=0.6, ls="-.",
               label=r"$t_{\rm cool}$")
    ax1.set_ylabel("Timescale [yr]")
    ax1.text(0.02, 0.95, Z_label, transform=ax1.transAxes,
             fontsize=8, va="top", ha="left",
             bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8, pad=2))
    if row == 0:
        ax1.legend(loc="upper right")
    if row < n_rows - 1:
        ax1.tick_params(labelbottom=False)

    # --- Right: Rates ---
    ax2.loglog(x, np.clip(dd["dM_growth_rate"], 1e-80, None), color="tab:blue",
               lw=lw, label=r"growth $\dot{M}_{\rm accr}$")
    ax2.loglog(x, np.clip(dd["dM_sput_rate"], 1e-80, None), color="tab:red",
               lw=lw, ls="--", label=r"sputtering $\dot{M}_{\rm sput}$")
    ax2.loglog(x, np.clip(dd["dM_dest_rate"], 1e-80, None), color="tab:orange",
               lw=lw, ls=":", label=r"SNe $\dot{M}_{\rm dest}$")

    # Net total
    net_pos = np.where(dd["dM_total_rate"] > 0, dd["dM_total_rate"], np.nan)
    net_neg = np.where(dd["dM_total_rate"] < 0, -dd["dM_total_rate"], np.nan)
    ax2.loglog(x, np.clip(net_pos, 1e-80, None), color="black", lw=lw*1.3,
               label=r"net total (growth)")
    ax2.loglog(x, np.clip(net_neg, 1e-80, None), color="black", lw=lw*1.3,
               ls="--", label=r"net total (destruction)")

    # Transition lines & shading
    sign_changes = np.where(np.diff(np.sign(dd["dM_total_rate"])))[0]
    for idx in sign_changes:
        ax2.axvline(x[idx], color="gray", lw=0.8, ls="--", alpha=0.7)
    boundaries = np.concatenate([[x[0]], x[sign_changes], [x[-1]]])
    for i in range(len(boundaries) - 1):
        mid_idx = np.searchsorted(x, 0.5*(boundaries[i] + boundaries[i+1]))
        mid_idx = min(mid_idx, len(dd["dM_total_rate"]) - 1)
        if dd["dM_total_rate"][mid_idx] > 0:
            ax2.axvspan(boundaries[i], boundaries[i+1], alpha=0.05, color="blue")
        else:
            ax2.axvspan(boundaries[i], boundaries[i+1], alpha=0.05, color="red")

    ax2.set_ylabel(r"Rate [g cm$^{-3}$ s$^{-1}$]")
    # ax2.text(0.02, 0.95, Z_label, transform=ax2.transAxes,
    #          fontsize=8, va="top", ha="left",
    #          bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8, pad=2))
    if row == 0:
        ax2.legend(loc="upper left")
    if row < n_rows - 1:
        ax2.tick_params(labelbottom=False)

# Shared x-axis labels on bottom row only
axes[-1, 0].set_xlabel(r"$n_H$ [cm$^{-3}$]")
axes[-1, 1].set_xlabel(r"$n_H$ [cm$^{-3}$]")

fig.suptitle("Dust timescales & rates — multi-metallicity freefall tests",
             fontsize=12)
# Column titles on top row only
axes[0, 0].set_title("Timescales", fontsize=9)
axes[0, 1].set_title("Rates", fontsize=9)

outfile = "dust_rates_multi.png"
plt.savefig(outfile, dpi=150, bbox_inches="tight")
print(f"Saved: {outfile}")
plt.show()
