import numpy as np
import matplotlib.pyplot as plt
from unyt import unyt_array

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------
mass_hydrogen_cgs = 1.67262171e-24  # g
sec_per_year      = 3.155e7         # s/yr
M_sun_cgs         = 1.989e33        # g

# -----------------------------------------------------------------------
# Parameters (matching defaults in dust_growth_and_destruction.cpp)
# -----------------------------------------------------------------------
grainsize      = 0.1    # micron
rho_ref        = 1.0e-27  # g/cm^3  (normalisation in tau_sput)
sne_coeff      = 1.0
sne_shockspeed = 100.0    # km/s
dust_dest_eff  = 0.3

# Swept mass per SNe [g]
Ms100_Msun = 6800.0 * sne_coeff * (100.0 / sne_shockspeed)**2
Ms100_cgs  = Ms100_Msun * M_sun_cgs

# -----------------------------------------------------------------------
# Code unit setup (matching cooling_cell.py)
#   density_units = mH  [g/cm³]
#   length_units  = 1 Mpc = cm_per_mpc  [cm]
#   time_units    = 1 Myr = 1e6 yr
# -----------------------------------------------------------------------
cm_per_mpc     = 3.086e24          # cm
density_units  = mass_hydrogen_cgs # g/cm³  →  rho_code = rho_phys/density_units = nH
vol_units_cm3  = cm_per_mpc**3     # cm³ per code_vol (1 Mpc³)
time_units_yr  = 1e6               # yr per code_time (1 Myr)

# Ms100 in code units: [code_density per SNe]  =  g/SNe / (g/cm³ × cm³)
Ms100_code = Ms100_cgs / (density_units * vol_units_cm3)

T   = unyt_array(np.logspace(0, 8, 81), "K")
nH  = unyt_array(np.logspace(5, 13, 81), "cm**-3")
SNR = np.logspace(5, 30, 81)   # dimensionless code units

rho = unyt_array(nH.v * mass_hydrogen_cgs, "g/cm**3")

# --- grids for tau_sput: axes = T (x), rho_gas (y) ---
T_g,   rho_g = np.meshgrid(T,   rho)

# --- grids for tau_dest: axes = SNR (x), rho_gas (y) ---
SNR_g, rho_g2 = np.meshgrid(SNR, rho)

# -----------------------------------------------------------------------
# tau_sput [yr]  =  1.7e8 * (a/0.1) * (rho_ref/rho) * ((T_sput/T)^2.5 + 1)
# -----------------------------------------------------------------------
tau_sput_yr = (1.7e8
               * (grainsize / 0.1)
               * (rho_ref / rho_g.to("g/cm**3").v)
               * ((2.0e6/ T_g.to("K").v)**2.5 + 1.0))

# -----------------------------------------------------------------------
# tau_dest [code_time]  =  nH_code / (Ms100_code * SNR_code * eff) * dt
# tau_dest [yr]         =  tau_dest_code * time_units_yr
#
# rho_gas_code = rho_gas [g/cm³] / density_units [g/cm³]
# SNR_code = dimensionless field value (N_SNe per cell per code_time)
# dt = 2.137248e-04  [code_time]  (representative timestep, as in C++ formula)
# -----------------------------------------------------------------------
dt_code      = 2.137248e-04
rho_code2    = rho_g2.to("g/cm**3").v / density_units   # rho_gas in code units
tau_dest_yr  = (rho_code2 / (Ms100_code * SNR_g * dust_dest_eff)) * dt_code * time_units_yr

# -----------------------------------------------------------------------
# tau_sput normalised by a representative dt  [dimensionless]
# -----------------------------------------------------------------------
dt_sput_ref  = 4.204491e-09          # code time
tau_sput_norm = tau_sput_yr / (dt_sput_ref * time_units_yr)

# -----------------------------------------------------------------------
# Plot: 1 row, 3 panels
# -----------------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5))
ax2, ax3 = ax3, ax2  # normalized sputtering next to sputtering

contour_levels = [1, 2,4,6]   # log10 yr labels to draw

# --- Left: tau_sput (T vs nH) ---
tau_s = np.log10(np.where(tau_sput_yr > 0, tau_sput_yr, np.nan))
im1 = ax1.pcolormesh(np.log10(T_g.to("K").v),
                     np.log10(rho_g.to("g/cm**3").v / mass_hydrogen_cgs),
                     tau_s, cmap="plasma_r", vmin=0, vmax=12)
plt.colorbar(im1, ax=ax1, label=r"$\log_{10}(\tau_{\rm sput})$ [yr]")

cs1 = ax1.contour(np.log10(T_g.to("K").v),
                  np.log10(rho_g.to("g/cm**3").v / mass_hydrogen_cgs),
                  tau_s, levels=contour_levels,
                  colors="white", linewidths=0.8, linestyles=":")
ax1.clabel(cs1, fmt=lambda v: f"$10^{{{int(v)}}}$ yr", fontsize=7)
ax1.set_xlabel(r"$\log_{10}(T\ [\mathrm{K}])$")
ax1.set_ylabel(r"$\log_{10}(n_H\ [\mathrm{cm}^{-3}])$")
ax1.set_title(r"$\tau_{\rm sput}$ — thermal sputtering")
ax1.legend(fontsize=8)

# --- Right: tau_dest (SNR vs nH) ---
tau_d = np.log10(np.where(tau_dest_yr > 0, tau_dest_yr, np.nan))
im2 = ax2.pcolormesh(np.log10(SNR_g),
                     np.log10(rho_g2.to("g/cm**3").v / mass_hydrogen_cgs),
                     tau_d, cmap="plasma_r", vmin=0, vmax=12)
plt.colorbar(im2, ax=ax2, label=r"$\log_{10}(\tau_{\rm dest})$ [yr]")

cs2 = ax2.contour(np.log10(SNR_g),
                  np.log10(rho_g2.to("g/cm**3").v / mass_hydrogen_cgs),
                  tau_d, levels=contour_levels,
                  colors="white", linewidths=0.8, linestyles=":")
ax2.clabel(cs2, fmt=lambda v: f"$10^{{{int(v)}}}$ yr", fontsize=7)
ax2.set_xlabel(r"$\log_{10}(\mathrm{SNR}\ [\mathrm{dimensionless}])$")
ax2.set_ylabel(r"$\log_{10}(n_H\ [\mathrm{cm}^{-3}])$")
ax2.set_title(r"$\tau_{\rm dest}$ — SNe shock destruction")

# --- Third: tau_sput / dt_ref (T vs nH) ---
tau_sn = np.log10(np.where(tau_sput_norm > 0, 1.0 / tau_sput_norm, np.nan))
im3 = ax3.pcolormesh(np.log10(T_g.to("K").v),
                     np.log10(rho_g.to("g/cm**3").v / mass_hydrogen_cgs),
                     tau_sn, cmap="plasma_r")
plt.colorbar(im3, ax=ax3, label=r"$\log_{10}(\Delta t\,/\,\tau_{\rm sput})$")

cs3 = ax3.contour(np.log10(T_g.to("K").v),
                  np.log10(rho_g.to("g/cm**3").v / mass_hydrogen_cgs),
                  tau_sn, levels=[-12, -11, -10, -9, -8, -7, -6],
                  colors="white", linewidths=0.8, linestyles=":")
ax3.clabel(cs3, fmt=lambda v: f"$10^{{{int(v)}}}$", fontsize=7, inline=True)
ax3.set_xlabel(r"$\log_{10}(T\ [\mathrm{K}])$")
ax3.set_ylabel(r"$\log_{10}(n_H\ [\mathrm{cm}^{-3}])$")
ax3.set_title(rf"$\Delta t/\tau_{{\rm sput}}$  ($\Delta t={dt_sput_ref:.3e}$ code time)")
ax3.set_xlim([0, 2])
ax3.set_ylim([10, 13])

plt.suptitle(
    rf"Dust destruction timescales  ($a={grainsize}\,\mu$m, "
    rf"$v_s={sne_shockspeed}$ km/s, $\epsilon={dust_dest_eff}$, "
    rf"$M_{{s}}={Ms100_Msun:.0f}\,M_\odot$)",
    fontsize=11
)
plt.tight_layout()
outfile = "dust_timescales.png"
plt.savefig(outfile, dpi=150, bbox_inches="tight")
print(f"Saved: {outfile}")
plt.show()
