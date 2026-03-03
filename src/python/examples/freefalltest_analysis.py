"""
freefalltest_analysis.py
Detailed multi-panel analysis of a freefalltest_new.h5 output.
X-axis: gas number density nH [cm⁻³]  (density / mH)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yt
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({"font.size": 7, "axes.titlesize": 8,
                     "axes.labelsize": 7, "xtick.labelsize": 6,
                     "ytick.labelsize": 6, "legend.fontsize": 6})

# -----------------------------------------------------------------------
# Load
# -----------------------------------------------------------------------
INFILE = "freefalltest_new.h5"

ds = yt.load(INFILE)
d  = ds.data   # shorthand

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
t_ref               = 20.0          # K (reference temperature for growth)

# SNe destruction parameters
sne_coeff           = 1.0
sne_shockspeed      = 100.0          # km/s
dust_destruction_eff = 0.3
M_sun_cgs           = 1.989e33       # g
cm_per_mpc          = 3.086e24       # cm
sne_rate            = 1.0            # SNe per cell per code_time (from freefalltest.py)

def v(field):
    """Return plain numpy array for a field name (strip units)."""
    return np.array(d[("data", field)])

# -----------------------------------------------------------------------
# Primary arrays
# -----------------------------------------------------------------------
time_s      = v("time")                         # s
density     = v("density")                      # g/cm³  (gas + metal, no dust)
dust_den    = v("dust_density")                 # g/cm³
metal_den   = v("metal_density")                # g/cm³
gas_den     = density                           # g/cm³  (d does not include dust)

nH          = density / mass_hydrogen_cgs       # cm⁻³
nH_gas      = gas_den / mass_hydrogen_cgs       # cm⁻³
time_Myr    = time_s / sec_per_Myr

T_gas       = v("temperature")                  # K
T_dust      = v("dust_temperature")             # K
e_int       = v("internal_energy")              # erg/g
pressure    = v("pressure")                     # dyn/cm²
gamma       = v("gamma")
mu          = v("mean_molecular_weight")
cool_rate   = v("cooling_rate")                 # erg cm³/s  (negative = cooling)
cool_time   = v("cooling_time")                 # s

# --- species (all g/cm³) ---
HI_den      = v("HI_density")
HII_den     = v("HII_density")
H2I_den     = v("H2I_density")
H2II_den    = v("H2II_density")
HM_den      = v("HM_density")
HeI_den     = v("HeI_density")
HeII_den    = v("HeII_density")
HeIII_den   = v("HeIII_density")
e_den       = v("e_density")
force_fac   = v("force_factor")

# --- derived mass fractions (relative to gas density) ---
fHI         = HI_den   / gas_den
fHII        = HII_den  / gas_den
fH2         = H2I_den  / gas_den       # H2I mass fraction
fH2II       = H2II_den / gas_den
fHM         = HM_den   / gas_den
fHeI        = HeI_den  / gas_den
fHeII       = HeII_den / gas_den
fHeIII      = HeIII_den/ gas_den
f_e         = e_den    / gas_den

dust_to_gas = dust_den  / gas_den
metal_to_gas= metal_den / gas_den
dust_to_metal = np.where(metal_den > 0, dust_den / metal_den, np.nan)

# --- simulation dt (from consecutive time snapshots) ---
dt_s = np.diff(time_s, prepend=time_s[0])
dt_s[0] = dt_s[1]   # first element has no predecessor

# -----------------------------------------------------------------------
# Recomputed dust timescales (physical cgs units)
# Matches formulas in dust_growth_and_destruction.cpp
# -----------------------------------------------------------------------
# tau_sput [s]: thermal sputtering timescale
tau_sput_s = (1.7e8 * sec_per_year
              * (dust_grainsize / 0.1)
              * (1.0e-27 / gas_den)
              * (np.power(2.0e6 / T_gas, 2.5) + 1.0))
tau_sput_yr = tau_sput_s / sec_per_year

# tau_accr [s]: dust growth (accretion) timescale
#   tau_accr0 = tau_ref_s * (densref / rho_gas) * sqrt(t_ref / T)
#   tau_accr  = tau_accr0 * (Z_solar / (rho_metal / mH))
#   where (rho_metal / mH) is rho_metal in code units (density_units = mH)
tau_ref_s = dust_growth_tauref * 1e9 * sec_per_year
rho_metal_code = metal_den / mass_hydrogen_cgs         # code units
rho_metal_code_safe = np.maximum(rho_metal_code, 1e-20)
tau_accr_s = (tau_ref_s
              * (dust_growth_densref / mass_hydrogen_cgs)   # densref in code units
              * np.sqrt(t_ref / T_gas)
              * (SolarMetalFractionByMass / rho_metal_code_safe))
tau_accr_yr = tau_accr_s / sec_per_year

# Instantaneous rates [g/cm³/s]
frac_metal_avail = np.where(
    metal_den > 0,
    np.clip(metal_den / (dust_den + metal_den), 0, 1),
    0.0)
dM_growth_rate  = frac_metal_avail * dust_den / tau_accr_s   # g/cm³/s
dM_sput_rate    = dust_den / tau_sput_s * 3.0                # g/cm³/s (factor 3 from C++)

# tau_dest [s]: SNe shock destruction timescale
#   C++ formula: tau_dest_code = rho_gas_code / (Ms100_code * sne_rate * eff)
#   Physical: tau_dest_s = tau_dest_code * tbase1
#   where tbase1 = time_units / a_units = sec_per_Myr
density_units = mass_hydrogen_cgs
tbase1        = sec_per_Myr
urho          = density_units      # for z=0, a=1
uxyz          = cm_per_mpc
Ms100_code    = (6800.0 * sne_coeff * (100.0/sne_shockspeed)**2
                 * M_sun_cgs / (urho * uxyz**3))
# Try to read sne_rate from HDF5, else use default
try:
    sne_rate_arr = v("sne_rate")
except Exception:
    sne_rate_arr = np.full_like(gas_den, sne_rate)
rho_gas_code  = gas_den / density_units
tau_dest_code = rho_gas_code / (Ms100_code * sne_rate_arr * dust_destruction_eff)
tau_dest_s    = tau_dest_code * tbase1
tau_dest_yr   = tau_dest_s / sec_per_year

# SNe destruction rate [g/cm³/s]
dM_dest_rate  = np.where(sne_rate_arr > 0,
                         np.minimum(dust_den / tau_dest_s, dust_den / dt_s),
                         0.0)

# -----------------------------------------------------------------------
# Figure layout: 5 rows x 3 cols = 15 panels
# -----------------------------------------------------------------------
fig = plt.figure(figsize=(14, 20))
gs  = gridspec.GridSpec(5, 3, figure=fig, hspace=0.55, wspace=0.42)
axes = [fig.add_subplot(gs[r, c]) for r in range(5) for c in range(3)]

lw   = 0.9
x    = nH_gas    # gas nH as the x-axis throughout

# -----------------------------------------------------------------------
# Panel 0: Gas & dust temperature
# -----------------------------------------------------------------------
ax = axes[0]
ax.loglog(x, T_gas,  color="tab:red",  lw=lw, label="$T_{\\rm gas}$")
ax.loglog(x, T_dust, color="tab:blue", lw=lw, linestyle="--",
          label="$T_{\\rm dust}$")
ax.loglog(x, np.full_like(x, 2.73), color="gray", lw=0.6, ls=":", label="CMB 2.73 K")
ax.set_xlabel(r"$n_H$ [cm$^{-3}$]")
ax.set_ylabel("T [K]")
ax.set_title("Gas & Dust Temperature")
ax.legend(fontsize=6)

# -----------------------------------------------------------------------
# Panel 1: T_gas / T_dust ratio
# -----------------------------------------------------------------------
ax = axes[1]
ratio = T_gas / T_dust
ax.loglog(x, ratio, color="tab:purple", lw=lw)
ax.axhline(1.0, color="gray", lw=0.7, ls=":")
ax.set_xlabel(r"$n_H$ [cm$^{-3}$]")
ax.set_ylabel(r"$T_{\rm gas}/T_{\rm dust}$")
ax.set_title("Gas-to-Dust Temperature Ratio")

# -----------------------------------------------------------------------
# Panel 2: Internal energy & pressure
# -----------------------------------------------------------------------
ax = axes[2]
ax2_twin = ax.twinx()
ax.loglog(x, e_int,    color="tab:orange", lw=lw, label="$e$ [erg/g]")
ax2_twin.loglog(x, pressure, color="tab:green",  lw=lw, linestyle="--",
                label="$P$ [dyn/cm²]")
ax.set_xlabel(r"$n_H$ [cm$^{-3}$]")
ax.set_ylabel("Internal energy [erg/g]", color="tab:orange")
ax2_twin.set_ylabel("Pressure [dyn/cm²]", color="tab:green")
ax.set_title("Internal Energy & Pressure")
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2_twin.get_legend_handles_labels()
ax.legend(lines1+lines2, labs1+labs2, fontsize=6)

# -----------------------------------------------------------------------
# Panel 3: Effective adiabatic index γ & mean molecular weight μ
# -----------------------------------------------------------------------
ax = axes[3]
ax2_twin = ax.twinx()
ax.semilogx(x, gamma, color="tab:red",  lw=lw, label=r"$\gamma_{\rm eff}$")
ax.axhline(5/3, color="tab:red", lw=0.6, ls=":", alpha=0.5)
ax.axhline(7/5, color="tab:red", lw=0.6, ls="--", alpha=0.5)
ax2_twin.semilogx(x, mu, color="tab:blue", lw=lw, linestyle="--",
                  label=r"$\mu$")
ax.set_xlabel(r"$n_H$ [cm$^{-3}$]")
ax.set_ylabel(r"$\gamma_{\rm eff}$", color="tab:red")
ax2_twin.set_ylabel(r"$\mu$", color="tab:blue")
ax.set_title(r"Adiabatic Index $\gamma$ & Mean Molecular Weight $\mu$")
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2_twin.get_legend_handles_labels()
ax.legend(lines1+lines2, labs1+labs2, fontsize=6)

# -----------------------------------------------------------------------
# Panel 4: Hydrogen species mass fractions
# -----------------------------------------------------------------------
ax = axes[4]
ax.loglog(x, fHI,   color="tab:blue",   lw=lw, label="HI")
ax.loglog(x, fHII,  color="tab:orange", lw=lw, label="HII")
ax.loglog(x, fH2,   color="tab:green",  lw=lw, label="H$_2$I")
ax.loglog(x, np.clip(fH2II, 1e-40, None), color="tab:red",
          lw=lw, linestyle="--", label="H$_2$II")
ax.loglog(x, np.clip(fHM, 1e-40, None),  color="tab:purple",
          lw=lw, linestyle=":", label="H$^-$")
ax.set_xlabel(r"$n_H$ [cm$^{-3}$]")
ax.set_ylabel("Mass fraction")
ax.set_title("Hydrogen Species Fractions")
ax.set_ylim(bottom=1e-15)
ax.legend(fontsize=6, ncol=2)

# -----------------------------------------------------------------------
# Panel 5: He species & electron fraction
# -----------------------------------------------------------------------
ax = axes[5]
ax.loglog(x, fHeI,  color="tab:blue",   lw=lw, label="HeI")
ax.loglog(x, np.clip(fHeII,  1e-40, None), color="tab:orange",
          lw=lw, label="HeII")
ax.loglog(x, np.clip(fHeIII, 1e-40, None), color="tab:red",
          lw=lw, linestyle="--", label="HeIII")
ax.loglog(x, np.clip(f_e, 1e-40, None), color="tab:green",
          lw=lw, linestyle=":", label=r"$e^-$")
ax.set_xlabel(r"$n_H$ [cm$^{-3}$]")
ax.set_ylabel("Mass fraction")
ax.set_title("Helium Species & Electron Fraction")
ax.set_ylim(bottom=1e-20)
ax.legend(fontsize=6, ncol=2)

# -----------------------------------------------------------------------
# Panel 6: Dust density & dust-to-gas ratio
# -----------------------------------------------------------------------
ax = axes[6]
ax2_twin = ax.twinx()
ax.loglog(x, dust_den,    color="tab:brown", lw=lw, label=r"$\rho_{\rm dust}$")
ax.loglog(x, gas_den,     color="tab:blue",  lw=lw, linestyle="--",
          label=r"$\rho_{\rm gas}$")
ax2_twin.loglog(x, dust_to_gas, color="tab:orange", lw=lw,
                label="dust/gas")
ax.set_xlabel(r"$n_H$ [cm$^{-3}$]")
ax.set_ylabel(r"Density [g/cm$^3$]")
ax2_twin.set_ylabel("Dust-to-gas ratio", color="tab:orange")
ax.set_title("Dust & Gas Density")
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2_twin.get_legend_handles_labels()
ax.legend(lines1+lines2, labs1+labs2, fontsize=6)

# -----------------------------------------------------------------------
# Panel 7: Metal density & metallicity
# -----------------------------------------------------------------------
ax = axes[7]
ax2_twin = ax.twinx()
ax.loglog(x, metal_den,   color="tab:cyan",  lw=lw, label=r"$\rho_{\rm metal}$")
ax2_twin.loglog(x, metal_to_gas, color="tab:green", lw=lw,
                linestyle="--", label="Z (metal/gas)")
ax2_twin.loglog(x, dust_to_metal, color="tab:orange", lw=lw,
                linestyle=":", label="dust/metal")
ax.set_xlabel(r"$n_H$ [cm$^{-3}$]")
ax.set_ylabel(r"Metal density [g/cm$^3$]")
ax2_twin.set_ylabel("Ratio")
ax.set_title("Metal Density & Ratios")
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2_twin.get_legend_handles_labels()
ax.legend(lines1+lines2, labs1+labs2, fontsize=6)

# -----------------------------------------------------------------------
# Panel 8: Cooling rate & cooling time
# -----------------------------------------------------------------------
ax = axes[8]
ax2_twin = ax.twinx()
ax.loglog(x, np.abs(cool_rate), color="tab:red", lw=lw,
          label=r"$|\Lambda|$ [erg cm³/s]")
cool_time_Myr = np.abs(cool_time) / sec_per_Myr
ax2_twin.loglog(x, cool_time_Myr, color="tab:blue", lw=lw,
                linestyle="--", label=r"$|t_{\rm cool}|$ [Myr]")
ax.set_xlabel(r"$n_H$ [cm$^{-3}$]")
ax.set_ylabel(r"$|\Lambda|$ [erg cm$^3$/s]", color="tab:red")
ax2_twin.set_ylabel(r"$|t_{\rm cool}|$ [Myr]", color="tab:blue")
ax.set_title("Cooling Rate & Cooling Time")
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2_twin.get_legend_handles_labels()
ax.legend(lines1+lines2, labs1+labs2, fontsize=6)

# -----------------------------------------------------------------------
# Panel 9: Pressure–density EOS (with reference isothermal & adiabatic)
# -----------------------------------------------------------------------
ax = axes[9]
ax.loglog(x, pressure, color="black", lw=lw, label="Simulation")
# Reference slopes
n_ref   = np.array([x.min(), x.max()])
P_iso   = pressure[0] * (n_ref / x[0])            # gamma=1 (isothermal)
P_adi   = pressure[0] * (n_ref / x[0])**(5/3)     # gamma=5/3 (adiabatic)
ax.loglog(n_ref, P_iso, "b:", lw=0.8, label=r"$\gamma=1$ (isothermal)")
ax.loglog(n_ref, P_adi, "r:", lw=0.8, label=r"$\gamma=5/3$ (adiabatic)")
ax.set_xlabel(r"$n_H$ [cm$^{-3}$]")
ax.set_ylabel(r"$P$ [dyn/cm$^2$]")
ax.set_title("Equation of State: $P$–$n_H$")
ax.legend(fontsize=6)

# -----------------------------------------------------------------------
# Panel 10: Force factor (pressure support)
# -----------------------------------------------------------------------
ax = axes[10]
ax.semilogx(x, force_fac, color="tab:green", lw=lw)
ax.axhline(0.0, color="gray", lw=0.7, ls=":")
ax.axhline(1.0, color="gray", lw=0.7, ls="--")
ax.set_xlabel(r"$n_H$ [cm$^{-3}$]")
ax.set_ylabel("Force factor")
ax.set_title("Force Factor (pressure support fraction)")
ax.set_ylim(-0.05, 1.05)

# -----------------------------------------------------------------------
# Panel 11: Time vs density (collapse trajectory)
# -----------------------------------------------------------------------
ax = axes[11]
ax.loglog(x, time_Myr + 1e-10, color="tab:gray", lw=lw)
ax.set_xlabel(r"$n_H$ [cm$^{-3}$]")
ax.set_ylabel("Time [Myr]")
ax.set_title("Collapse Time vs Density")

# -----------------------------------------------------------------------
# Panel 12: Dust timescales (tau_sput, tau_accr)
# -----------------------------------------------------------------------
ax = axes[12]
ax.loglog(x, tau_sput_yr, color="tab:red",   lw=lw, label=r"$\tau_{\rm sput}$")
ax.loglog(x, tau_accr_yr, color="tab:blue",  lw=lw, linestyle="--",
          label=r"$\tau_{\rm accr}$")
ax.loglog(x, np.clip(tau_dest_yr, 1e-20, None), color="tab:orange", lw=lw,
          linestyle=":", label=r"$\tau_{\rm dest}$ (SNe)")
# freefall time for reference
t_ff_yr = np.sqrt(3*np.pi / (32 * 6.674e-8 * density)) / sec_per_year
ax.loglog(x, t_ff_yr, color="gray", lw=0.6, ls=":", label=r"$t_{\rm ff}$")
# cooling time
cool_time_yr = np.abs(cool_time) / sec_per_year
ax.loglog(x, cool_time_yr, color="tab:green", lw=0.6, ls="-.", label=r"$t_{\rm cool}$")
ax.set_xlabel(r"$n_H$ [cm$^{-3}$]")
ax.set_ylabel("Timescale [yr]")
ax.set_title("Dust Timescales")
ax.legend(fontsize=6)

# -----------------------------------------------------------------------
# Panel 13: Instantaneous dust rates (growth vs destruction)
# -----------------------------------------------------------------------
ax = axes[13]
ax.loglog(x, np.clip(dM_growth_rate, 1e-80, None), color="tab:blue", lw=lw,
          label=r"growth $\dot{M}_{\rm accr}$")
ax.loglog(x, np.clip(dM_sput_rate, 1e-80, None), color="tab:red", lw=lw,
          linestyle="--", label=r"sputtering $\dot{M}_{\rm sput}$")
ax.loglog(x, np.clip(dM_dest_rate, 1e-80, None), color="tab:orange", lw=lw,
          linestyle=":", label=r"SNe $\dot{M}_{\rm dest}$")
ax.set_xlabel(r"$n_H$ [cm$^{-3}$]")
ax.set_ylabel(r"Rate [g cm$^{-3}$ s$^{-1}$]")
ax.set_title("Dust Growth vs Destruction Rate")
ax.legend(fontsize=6)

# -----------------------------------------------------------------------
# Panel 14: dt / tau ratios (resolution of dust processes)
# -----------------------------------------------------------------------
ax = axes[14]
dt_over_tau_sput = dt_s / tau_sput_s
dt_over_tau_accr = dt_s / tau_accr_s
dt_over_tau_dest = dt_s / tau_dest_s
ax.loglog(x, np.clip(dt_over_tau_sput, 1e-30, None), color="tab:red", lw=lw,
          label=r"$\Delta t / \tau_{\rm sput}$")
ax.loglog(x, np.clip(dt_over_tau_accr, 1e-30, None), color="tab:blue", lw=lw,
          linestyle="--", label=r"$\Delta t / \tau_{\rm accr}$")
ax.loglog(x, np.clip(dt_over_tau_dest, 1e-30, None), color="tab:orange", lw=lw,
          linestyle=":", label=r"$\Delta t / \tau_{\rm dest}$")
ax.axhline(1.0, color="gray", lw=0.7, ls=":")
ax.set_xlabel(r"$n_H$ [cm$^{-3}$]")
ax.set_ylabel(r"$\Delta t / \tau$")
ax.set_title(r"Timestep Resolution of Dust Processes")
ax.legend(fontsize=6)

# -----------------------------------------------------------------------
# Suptitle & save
# -----------------------------------------------------------------------
fig.suptitle(
    f"Freefall test — detailed analysis\n({INFILE},  {len(x)} timesteps,  "
    f"$n_{{H,0}}={x[0]:.2e}$ → $n_{{H,f}}={x[-1]:.2e}$ cm$^{{-3}}$)",
    fontsize=10, y=0.995
)

outfile = "freefalltest_new_analysis.png"
plt.savefig(outfile, dpi=150, bbox_inches="tight")
print(f"Saved: {outfile}")
plt.show()
