"""
dust_rates_overview.py
Presentation-quality overview of dust processes across metallicities.

Panel layout (2x2):
  Top-left:     Overlay of dominant timescale for each Z (color = metallicity)
  Top-right:    Overlay of net dust rate for each Z
  Bottom-left:  Heatmap — dominant process vs (nH, Z)
  Bottom-right: Heatmap — log10(net rate) vs (nH, Z)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import yt
import warnings
import os
warnings.filterwarnings("ignore")

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10,
                     "axes.labelsize": 9, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 7})

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
# Dataset list
# -----------------------------------------------------------------------
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
met_indices = [-6, -5, -4, -3, -2, -1, 0]   # sorted for heatmap y-axis
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

    time_s    = v("time")
    density   = v("density")
    dust_den  = v("dust_density")
    metal_den = v("metal_density")
    gas_den   = density
    nH_gas    = gas_den / mass_hydrogen_cgs
    T_gas     = v("temperature")
    cool_time = v("cooling_time")

    dt_s = np.diff(time_s, prepend=time_s[0])
    dt_s[0] = dt_s[1]

    # tau_sput
    tau_sput_s = (1.7e8 * sec_per_year * (dust_grainsize / 0.1)
                  * (1.0e-27 / gas_den)
                  * (np.power(2.0e6 / T_gas, 2.5) + 1.0))

    # tau_accr
    rho_metal_code_safe = np.maximum(metal_den / mass_hydrogen_cgs, 1e-20)
    tau_accr_s = (tau_ref_s
                  * (dust_growth_densref / mass_hydrogen_cgs)
                  * np.sqrt(t_ref / T_gas)
                  * (SolarMetalFractionByMass / rho_metal_code_safe))

    # tau_dest
    try:
        sne_rate_arr = v("sne_rate")
    except Exception:
        sne_rate_arr = np.full_like(gas_den, sne_rate_default)
    rho_gas_code = gas_den / density_units
    tau_dest_s = (rho_gas_code / (Ms100_code * sne_rate_arr * dust_destruction_eff)) * tbase1

    # Reference
    t_ff_s = np.sqrt(3*np.pi / (32 * 6.674e-8 * density))

    # Rates [g/cm³/s]
    frac_metal_avail = np.where(
        metal_den > 0,
        np.clip(metal_den / (dust_den + metal_den), 0, 1), 0.0)
    dM_growth  = frac_metal_avail * dust_den / tau_accr_s
    dM_sput    = dust_den / tau_sput_s * 3.0
    dM_dest    = np.where(sne_rate_arr > 0,
                           np.minimum(dust_den / tau_dest_s, dust_den / dt_s), 0.0)
    dM_net     = dM_growth - dM_sput - dM_dest

    return dict(
        x=nH_gas,
        tau_sput_s=tau_sput_s, tau_accr_s=tau_accr_s,
        tau_dest_s=tau_dest_s, t_ff_s=t_ff_s,
        cool_time_s=np.abs(cool_time),
        dM_growth=dM_growth, dM_sput=dM_sput,
        dM_dest=dM_dest, dM_net=dM_net,
        T_gas=T_gas)

# -----------------------------------------------------------------------
# Load all datasets
# -----------------------------------------------------------------------
all_data = []
for fname, mi in datasets:
    print(f"  Loading Z = 10^{mi} Z_sun ...")
    all_data.append((mi, compute_dust_data(fname)))

# -----------------------------------------------------------------------
# Color map for metallicities
# -----------------------------------------------------------------------
cmap_met = plt.cm.viridis
met_values = [mi for mi, _ in all_data]
norm_met = mcolors.Normalize(vmin=min(met_values) - 0.5, vmax=max(met_values) + 0.5)

def Z_label(mi):
    return f"$10^{{{mi}}}\\,Z_\\odot$" if mi != 0 else r"$Z_\odot$"

# -----------------------------------------------------------------------
# Figure: 2x2
# -----------------------------------------------------------------------
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 11))

# =======================================================================
# Top-left: Overlay timescales (all Z)
# =======================================================================
for mi, dd in all_data:
    c = cmap_met(norm_met(mi))
    ax1.loglog(dd["x"], dd["tau_sput_s"]/sec_per_year, color=c, lw=0.8, alpha=0.8)
    ax1.loglog(dd["x"], dd["tau_accr_s"]/sec_per_year, color=c, lw=0.8, ls="--", alpha=0.8)
    ax1.loglog(dd["x"], np.clip(dd["tau_dest_s"]/sec_per_year, 1e-20, None),
               color=c, lw=0.8, ls=":", alpha=0.8)

# Reference: t_ff and t_cool from first dataset (same initial density)
dd0 = all_data[0][1]
ax1.loglog(dd0["x"], dd0["t_ff_s"]/sec_per_year, color="gray", lw=1.5, ls=":",
           label=r"$t_{\rm ff}$")
ax1.loglog(dd0["x"], dd0["cool_time_s"]/sec_per_year, color="gray", lw=1.5, ls="-.",
           label=r"$t_{\rm cool}$")

# Legend entries for line styles
from matplotlib.lines import Line2D
style_handles = [
    Line2D([0], [0], color="gray", lw=1, ls="-", label=r"$\tau_{\rm sput}$"),
    Line2D([0], [0], color="gray", lw=1, ls="--", label=r"$\tau_{\rm accr}$"),
    Line2D([0], [0], color="gray", lw=1, ls=":", label=r"$\tau_{\rm dest}$ (SNe)"),
    Line2D([0], [0], color="gray", lw=1.5, ls=":", label=r"$t_{\rm ff}$"),
    Line2D([0], [0], color="gray", lw=1.5, ls="-.", label=r"$t_{\rm cool}$"),
]
ax1.legend(handles=style_handles, loc="upper right")
ax1.set_xlabel(r"$n_H$ [cm$^{-3}$]")
ax1.set_ylabel("Timescale [yr]")
ax1.set_title("Dust Timescales (color = metallicity)")

# Colorbar for metallicity
sm = plt.cm.ScalarMappable(cmap=cmap_met, norm=norm_met)
sm.set_array([])
cb1 = fig.colorbar(sm, ax=ax1, pad=0.02)
cb1.set_label(r"$\log_{10}(Z / Z_\odot)$")
cb1.set_ticks(met_values)
cb1.set_ticklabels([str(m) for m in met_values])

# =======================================================================
# Top-right: Overlay net rates (all Z)
# =======================================================================
for mi, dd in all_data:
    c = cmap_met(norm_met(mi))
    net_pos = np.where(dd["dM_net"] > 0, dd["dM_net"], np.nan)
    net_neg = np.where(dd["dM_net"] < 0, -dd["dM_net"], np.nan)
    ax2.loglog(dd["x"], np.clip(net_pos, 1e-80, None), color=c, lw=1.0, alpha=0.9)
    ax2.loglog(dd["x"], np.clip(net_neg, 1e-80, None), color=c, lw=1.0, ls="--", alpha=0.9)

style_handles2 = [
    Line2D([0], [0], color="gray", lw=1, ls="-", label="net growth"),
    Line2D([0], [0], color="gray", lw=1, ls="--", label="net destruction"),
]
ax2.legend(handles=style_handles2, loc="upper left")
ax2.set_xlabel(r"$n_H$ [cm$^{-3}$]")
ax2.set_ylabel(r"|net rate| [g cm$^{-3}$ s$^{-1}$]")
ax2.set_title("Net Dust Rate (color = metallicity)")

cb2 = fig.colorbar(sm, ax=ax2, pad=0.02)
cb2.set_label(r"$\log_{10}(Z / Z_\odot)$")
cb2.set_ticks(met_values)
cb2.set_ticklabels([str(m) for m in met_values])

# =======================================================================
# Bottom-left: Heatmap — dominant process
# Process codes: 0=growth, 1=sputtering, 2=SNe destruction
# =======================================================================
# Build common nH grid (log-spaced)
nH_min = min(dd["x"].min() for _, dd in all_data)
nH_max = max(dd["x"].max() for _, dd in all_data)
nH_grid = np.logspace(np.log10(nH_min), np.log10(nH_max), 300)
log_nH_grid = np.log10(nH_grid)

dominant_map = np.full((len(all_data), len(nH_grid)), np.nan)
for row, (mi, dd) in enumerate(all_data):
    growth_interp = np.interp(log_nH_grid, np.log10(dd["x"]), dd["dM_growth"])
    sput_interp   = np.interp(log_nH_grid, np.log10(dd["x"]), dd["dM_sput"])
    dest_interp   = np.interp(log_nH_grid, np.log10(dd["x"]), dd["dM_dest"])
    # Dominant = largest magnitude process
    rates = np.array([growth_interp, sput_interp, dest_interp])
    dominant_map[row, :] = np.argmax(rates, axis=0)

dom_cmap = mcolors.ListedColormap(["tab:blue", "tab:red", "tab:orange"])
dom_norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], dom_cmap.N)

im3 = ax3.pcolormesh(log_nH_grid, np.arange(len(all_data)),
                     dominant_map, cmap=dom_cmap, norm=dom_norm, shading="nearest")
ax3.set_yticks(range(len(all_data)))
ax3.set_yticklabels([Z_label(mi) for mi, _ in all_data])
ax3.set_xlabel(r"$\log_{10}\,n_H$ [cm$^{-3}$]")
ax3.set_title("Dominant Process")
legend_patches = [
    Patch(facecolor="tab:blue", label="Growth"),
    Patch(facecolor="tab:red", label="Sputtering"),
    Patch(facecolor="tab:orange", label="SNe destruction"),
]
ax3.legend(handles=legend_patches, loc="upper left", fontsize=7)

# =======================================================================
# Bottom-right: Heatmap — log10(|net rate|), signed
# Positive = net growth (blue), Negative = net destruction (red)
# =======================================================================
net_rate_map = np.full((len(all_data), len(nH_grid)), np.nan)
for row, (mi, dd) in enumerate(all_data):
    net_interp = np.interp(log_nH_grid, np.log10(dd["x"]), dd["dM_net"])
    # Signed log: positive values stay positive, negative stay negative
    net_rate_map[row, :] = np.sign(net_interp) * np.log10(np.clip(np.abs(net_interp), 1e-80, None))

vmax = np.nanmax(np.abs(net_rate_map[np.isfinite(net_rate_map)]))
vmin = -vmax

im4 = ax4.pcolormesh(log_nH_grid, np.arange(len(all_data)),
                     net_rate_map, cmap="RdBu", shading="nearest",
                     vmin=vmin, vmax=vmax)
ax4.set_yticks(range(len(all_data)))
ax4.set_yticklabels([Z_label(mi) for mi, _ in all_data])
ax4.set_xlabel(r"$\log_{10}\,n_H$ [cm$^{-3}$]")
ax4.set_title(r"Net Rate: sign$\times\log_{10}|\dot{M}|$ (blue=growth, red=destruction)")
cb4 = fig.colorbar(im4, ax=ax4, pad=0.02)
cb4.set_label(r"sign $\times$ $\log_{10}|\dot{M}_{\rm net}|$ [g cm$^{-3}$ s$^{-1}$]")

fig.suptitle("Dust processes overview — multi-metallicity freefall tests",
             fontsize=13, y=1.01)
plt.tight_layout()

outfile = "dust_rates_overview.png"
plt.savefig(outfile, dpi=150, bbox_inches="tight")
print(f"Saved: {outfile}")
plt.show()
