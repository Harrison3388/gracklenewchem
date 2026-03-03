"""
dust_rates_analysis.py
Focused 2-panel analysis of dust timescales and rates from a freefall test.
Builds on the same data loading as freefalltest_analysis.py.
"""
import numpy as np
import matplotlib.pyplot as plt
import yt
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10,
                     "axes.labelsize": 9, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 7})

# -----------------------------------------------------------------------
# Load
# -----------------------------------------------------------------------
INFILE = "freefalltest_new.h5"

ds = yt.load(INFILE)
d  = ds.data

mass_hydrogen_cgs = 1.67262171e-24   # g
sec_per_Myr       = 3.155e13         # s/Myr
sec_per_year      = 3.155e7          # s/yr

# -----------------------------------------------------------------------
# Dust model parameters (defaults from grackle_chemistry_data_fields.def)
# -----------------------------------------------------------------------
dust_grainsize      = 0.1            # micron
dust_growth_tauref  = 0.004          # Gyr
dust_growth_densref = 2.3e-22        # g/cm³
SolarMetalFractionByMass = 0.01295
t_ref               = 20.0          # K

# SNe destruction parameters
sne_coeff           = 1.0
sne_shockspeed      = 100.0          # km/s
dust_destruction_eff = 0.3
M_sun_cgs           = 1.989e33       # g
cm_per_mpc          = 3.086e24       # cm
sne_rate            = 1.0            # default if not in HDF5

def v(field):
    """Return plain numpy array for a field name (strip units)."""
    return np.array(d[("data", field)])

# -----------------------------------------------------------------------
# Primary arrays
# -----------------------------------------------------------------------
time_s      = v("time")
density     = v("density")                      # g/cm³ (gas + metal, no dust)
dust_den    = v("dust_density")
metal_den   = v("metal_density")
gas_den     = density                           # d does not include dust

nH_gas      = gas_den / mass_hydrogen_cgs
T_gas       = v("temperature")
cool_time   = v("cooling_time")                 # s

# --- simulation dt ---
dt_s = np.diff(time_s, prepend=time_s[0])
dt_s[0] = dt_s[1]

# -----------------------------------------------------------------------
# Dust timescales (physical cgs)
# -----------------------------------------------------------------------
# tau_sput [s]
tau_sput_s = (1.7e8 * sec_per_year
              * (dust_grainsize / 0.1)
              * (1.0e-27 / gas_den)
              * (np.power(2.0e6 / T_gas, 2.5) + 1.0))
tau_sput_yr = tau_sput_s / sec_per_year

# tau_accr [s]
tau_ref_s = dust_growth_tauref * 1e9 * sec_per_year
rho_metal_code = metal_den / mass_hydrogen_cgs
rho_metal_code_safe = np.maximum(rho_metal_code, 1e-20)
tau_accr_s = (tau_ref_s
              * (dust_growth_densref / mass_hydrogen_cgs)
              * np.sqrt(t_ref / T_gas)
              * (SolarMetalFractionByMass / rho_metal_code_safe))
tau_accr_yr = tau_accr_s / sec_per_year

# tau_dest [s] (SNe)
density_units = mass_hydrogen_cgs
tbase1        = sec_per_Myr
urho          = density_units
uxyz          = cm_per_mpc
Ms100_code    = (6800.0 * sne_coeff * (100.0/sne_shockspeed)**2
                 * M_sun_cgs / (urho * uxyz**3))
try:
    sne_rate_arr = v("sne_rate")
except Exception:
    sne_rate_arr = np.full_like(gas_den, sne_rate)
rho_gas_code  = gas_den / density_units
tau_dest_code = rho_gas_code / (Ms100_code * sne_rate_arr * dust_destruction_eff)
tau_dest_s    = tau_dest_code * tbase1
tau_dest_yr   = tau_dest_s / sec_per_year

# Reference timescales
t_ff_yr = np.sqrt(3*np.pi / (32 * 6.674e-8 * density)) / sec_per_year
cool_time_yr = np.abs(cool_time) / sec_per_year

# -----------------------------------------------------------------------
# Instantaneous rates [g/cm³/s]
# -----------------------------------------------------------------------
frac_metal_avail = np.where(
    metal_den > 0,
    np.clip(metal_den / (dust_den + metal_den), 0, 1),
    0.0)
dM_growth_rate = frac_metal_avail * dust_den / tau_accr_s
dM_sput_rate   = dust_den / tau_sput_s * 3.0     # factor 3 from C++
dM_dest_rate   = np.where(sne_rate_arr > 0,
                           np.minimum(dust_den / tau_dest_s, dust_den / dt_s),
                           0.0)

# Total net rate: growth is positive, destruction is negative
dM_total_rate = dM_growth_rate - dM_sput_rate - dM_dest_rate

# -----------------------------------------------------------------------
# Figure: 1 row x 2 cols
# -----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
lw = 1.0
x = nH_gas

# -----------------------------------------------------------------------
# Left panel: Timescales
# -----------------------------------------------------------------------
ax1.loglog(x, tau_sput_yr, color="tab:red",   lw=lw, label=r"$\tau_{\rm sput}$")
ax1.loglog(x, tau_accr_yr, color="tab:blue",  lw=lw, linestyle="--",
           label=r"$\tau_{\rm accr}$")
ax1.loglog(x, np.clip(tau_dest_yr, 1e-20, None), color="tab:orange", lw=lw,
           linestyle=":", label=r"$\tau_{\rm dest}$ (SNe)")
ax1.loglog(x, t_ff_yr, color="gray", lw=0.7, ls=":", label=r"$t_{\rm ff}$")
ax1.loglog(x, cool_time_yr, color="tab:green", lw=0.7, ls="-.",
           label=r"$t_{\rm cool}$")
ax1.set_xlabel(r"$n_H$ [cm$^{-3}$]")
ax1.set_ylabel("Timescale [yr]")
ax1.set_title("Dust & Reference Timescales")
ax1.legend()

# -----------------------------------------------------------------------
# Right panel: Rates (with total)
# -----------------------------------------------------------------------
ax2.loglog(x, np.clip(dM_growth_rate, 1e-80, None), color="tab:blue", lw=lw,
           label=r"growth $\dot{M}_{\rm accr}$")
ax2.loglog(x, np.clip(dM_sput_rate, 1e-80, None), color="tab:red", lw=lw,
           linestyle="--", label=r"sputtering $\dot{M}_{\rm sput}$")
ax2.loglog(x, np.clip(dM_dest_rate, 1e-80, None), color="tab:orange", lw=lw,
           linestyle=":", label=r"SNe $\dot{M}_{\rm dest}$")
# Total: plot positive (net growth) and negative (net destruction) separately
net_positive = np.where(dM_total_rate > 0, dM_total_rate, np.nan)
net_negative = np.where(dM_total_rate < 0, -dM_total_rate, np.nan)
ax2.loglog(x, np.clip(net_positive, 1e-80, None), color="black", lw=lw*1.3,
           label=r"net total (growth)")
ax2.loglog(x, np.clip(net_negative, 1e-80, None), color="black", lw=lw*1.3,
           linestyle="--", label=r"net total (destruction)")
# Mark transition points where net rate changes sign
sign_changes = np.where(np.diff(np.sign(dM_total_rate)))[0]
for idx in sign_changes:
    nH_cross = x[idx]
    ax2.axvline(nH_cross, color="gray", lw=0.8, ls="--", alpha=0.7)
    ax2.annotate(f"$n_H$={nH_cross:.1e}", xy=(nH_cross, ax2.get_ylim()[1]),
                 fontsize=6, rotation=90, va="top", ha="right", color="gray")
# Shade background: green for net growth, red for net destruction
ylim = ax2.get_ylim()
# Build boundary array including plot edges
boundaries = np.concatenate([[x[0]], x[sign_changes], [x[-1]]])
for i in range(len(boundaries) - 1):
    mid_idx = np.searchsorted(x, 0.5*(boundaries[i] + boundaries[i+1]))
    mid_idx = min(mid_idx, len(dM_total_rate) - 1)
    if dM_total_rate[mid_idx] > 0:
        ax2.axvspan(boundaries[i], boundaries[i+1], alpha=0.05, color="blue")
    else:
        ax2.axvspan(boundaries[i], boundaries[i+1], alpha=0.05, color="red")

ax2.set_xlabel(r"$n_H$ [cm$^{-3}$]")
ax2.set_ylabel(r"Rate [g cm$^{-3}$ s$^{-1}$]")
ax2.set_title("Dust Growth vs Destruction Rates")
ax2.legend()

fig.suptitle(
    f"Dust rates analysis ({INFILE}, {len(x)} timesteps)",
    fontsize=11, y=1.02)
plt.tight_layout()

outfile = "dust_rates_analysis.png"
plt.savefig(outfile, dpi=150, bbox_inches="tight")
print(f"Saved: {outfile}")
plt.show()
