"""
Plotting script for analyzing saved grackle simulation data.
Loads .h5 files produced by freefalltest.py or cooling_cell.py
and generates diagnostic plots without rerunning the simulation.

Usage:
    python testing.py <h5_file> [--type freefall|cooling]
"""

import sys
import yt
import matplotlib.pyplot as plt
import numpy as np

from gracklepy.utilities.physical_constants import \
    mass_hydrogen_cgs, \
    sec_per_Myr

def plot_freefall(ds, output_prefix):
    """Plots for freefall collapse data (x-axis: number density)."""
    data = ds.data
    n_H = data["density"] / mass_hydrogen_cgs  # number density

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Temperature vs density
    ax = axes[0, 0]
    ax.loglog(n_H, data["temperature"], color="black")
    ax.set_xlabel("n$_H$ [cm$^{-3}$]")
    ax.set_ylabel("T [K]")
    ax.set_title("Temperature")

    # 2. Dust-to-gas ratio vs density
    ax = axes[0, 1]
    dust_to_gas = data["dust_density"] / data["density"]
    ax.loglog(n_H, dust_to_gas, color="blue")
    ax.set_xlabel("n$_H$ [cm$^{-3}$]")
    ax.set_ylabel("dust / gas")
    ax.set_title("Dust-to-Gas Ratio")

    # 3. Metal fraction and dust fraction vs density
    ax = axes[1, 0]
    metal_frac = data["metal_density"] / data["density"]
    dust_frac = data["dust_density"] / data["density"]
    ax.loglog(n_H, metal_frac, color="green", label="metal/gas")
    ax.loglog(n_H, dust_frac, color="brown", label="dust/gas")
    ax.loglog(n_H, metal_frac + dust_frac, color="gray",
              linestyle="--", label="(metal+dust)/gas")
    ax.set_xlabel("n$_H$ [cm$^{-3}$]")
    ax.set_ylabel("mass fraction")
    ax.set_title("Metal & Dust Fractions")
    ax.legend(fontsize=8)

    # 4. H2 fraction vs density (if available)
    ax = axes[1, 1]
    available = [str(f[1]) for f in ds.field_list]
    if "H2I_density" in available:
        h2_frac = data["H2I_density"] / data["density"]
        ax.loglog(n_H, h2_frac, color="red")
        ax.set_ylabel("H$_2$ / gas")
        ax.set_title("H$_2$ Fraction")
    else:
        # fallback: dust-to-metal ratio
        dust_to_metal = data["dust_density"] / data["metal_density"]
        ax.loglog(n_H, dust_to_metal, color="purple")
        ax.set_ylabel("dust / metal")
        ax.set_title("Dust-to-Metal Ratio")
    ax.set_xlabel("n$_H$ [cm$^{-3}$]")

    fig.suptitle(f"Freefall: {output_prefix}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_analysis.png", dpi=200,
                bbox_inches="tight")
    print(f"Saved {output_prefix}_analysis.png")
    plt.show()


def plot_cooling(ds, output_prefix):
    """Plots for cooling cell data (x-axis: time)."""
    data = ds.data
    time_myr = data["time"].to("Myr")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Temperature vs time
    ax = axes[0, 0]
    # ax.semilogy(time_myr, data["H2I_density"] / data["density"], color="black")
    ax.semilogy(time_myr, data["temperature"], color="black")
    ax.set_xlabel("Time [Myr]")
    ax.set_ylabel("T [K]")
    ax.set_title("Temperature")

    # 2. Dust and metal density vs time
    ax = axes[0, 1]
    ax.semilogy(time_myr, data["dust_density"] / mass_hydrogen_cgs,
                color="brown", label="dust")
    ax.semilogy(time_myr, data["metal_density"] / mass_hydrogen_cgs,
                color="green", label="metal")
    ax.set_xlabel("Time [Myr]")
    ax.set_ylabel("density [cm$^{-3}$]")
    ax.set_title("Dust & Metal Density")
    ax.legend(fontsize=8)

    # 3. Dust-to-gas ratio vs time
    ax = axes[1, 0]
    dust_to_gas = data["dust_density"] / data["density"]
    ax.plot(time_myr, dust_to_gas, color="blue")
    ax.set_xlabel("Time [Myr]")
    ax.set_ylabel("dust / gas")
    ax.set_title("Dust-to-Gas Ratio")

    # 4. Mass conservation check
    ax = axes[1, 1]
    total = data["dust_density"] + data["metal_density"]
    ax.plot(time_myr, total / total[0], color="gray")
    ax.set_xlabel("Time [Myr]")
    ax.set_ylabel("(dust + metal) / initial")
    ax.set_title("Mass Conservation Check")
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)

    fig.suptitle(f"Cooling Cell: {output_prefix}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_analysis.png", dpi=200,
                bbox_inches="tight")
    print(f"Saved {output_prefix}_analysis.png")
    plt.show()


def detect_type(ds):
    """Auto-detect whether this is a freefall or cooling cell run."""
    data = ds.data
    density_vals = np.array(data["density"])
    # freefall: density spans many orders of magnitude
    if density_vals.max() / density_vals.min() > 100:
        return "freefall"
    return "cooling"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python testing.py <h5_file> [--type freefall|cooling]")
        print("Example: python testing.py freefalltest_new.h5")
        print("         python testing.py cooling_cell.h5 --type cooling")
        sys.exit(1)

    h5_file = sys.argv[1]
    output_prefix = h5_file.replace(".h5", "")

    # parse optional --type argument
    plot_type = None
    if "--type" in sys.argv:
        idx = sys.argv.index("--type")
        if idx + 1 < len(sys.argv):
            plot_type = sys.argv[idx + 1]

    ds = yt.load(h5_file)

    # print available fields for reference
    print("Available fields:", [str(f[1]) for f in ds.field_list])

    if plot_type is None:
        plot_type = detect_type(ds)
    print(f"Detected plot type: {plot_type}")

    if plot_type == "freefall":
        plot_freefall(ds, output_prefix)
    else:
        plot_cooling(ds, output_prefix)
