#!/usr/bin/env python
"""
Add UV-background photo-ionization rates for C I and O I to a Grackle
UV background table.

This script computes the optically-thin photo-ionization rates

    k(z) = int 4*pi*J(nu, z) / (h*nu) * sigma(nu) dnu   [1/s]

for
    C I + gamma -> C II + e-   (threshold 11.26 eV)
    O I + gamma -> O II + e-   (threshold 13.62 eV)

on the same redshift grid as the existing /UVBRates/Chemistry/k24 dataset
and writes them as two new datasets,

    /UVBRates/Chemistry/kphCI
    /UVBRates/Chemistry/kphOI

into a COPY of an existing Grackle UV background table (the input table is
never modified). These datasets are read by Grackle when metal_chemistry > 0
(missing datasets are tolerated; the rates are then zero).

Inputs
------
1. --table: an existing Grackle UV background table
   (e.g. CloudyData_UVB=HM2012.h5).

2. --spectrum: the Haardt & Madau (2012, ApJ, 746, 125) UV background
   spectrum ("UVB.out" from the CUBA web site). Expected plain-text format:
     - comment lines start with '#' (possibly preceded by whitespace)
     - the first non-comment line lists the N_z sampling redshifts
     - each following line holds a rest-frame wavelength in Angstroms
       followed by N_z values of the background intensity J in
       erg s^-1 cm^-2 Hz^-1 sr^-1 at the sampling redshifts
     - wavelengths appear twice at absorption edges (the two rows give J on
       either side of the discontinuity)

3. --sigma-ci / --sigma-oi: photo-ionization cross sections of atomic C and
   atomic O from the Leiden photodissociation and photoionization database
   described by Heays, Bosman & van Dishoeck (2017, A&A, 602, A105;
   arXiv:1701.04459). The HDF5 cross-section files (C.h5, O.h5) are
   available from
     https://home.strw.leidenuniv.nl/~ewine/photo/
   (Downloads -> "all_cross_sections_h5.zip"). Expected HDF5 format:
     - dataset 'wavelength':      wavelength in nm
     - dataset 'photoionisation': cross section in cm^2
   The C cross section originates from TOPbase (Mendoza 1996; Nahar &
   Pradhan 1997); the O cross section from Mendoza (1996)/TOPbase and
   Huebner & Mukherjee (2015).

The integral runs from each species' ionization threshold up to the
highest photon energy tabulated in the spectrum. The cross sections are
taken to be zero outside the wavelength range tabulated in the Leiden files
(for the HM2012 spectrum the neglected X-ray contribution is negligible).

Only numpy and h5py are required.

Example
-------
    python scripts/add_metal_UVB_rates.py \\
        --table input/CloudyData_UVB=HM2012.h5 \\
        --spectrum UVB.out \\
        --sigma-ci C.h5 --sigma-oi O.h5 \\
        --output CloudyData_UVB=HM2012_metal.h5
"""

import argparse
import os
import shutil
import sys

import h5py
import numpy as np

# physical constants (CGS)
h_planck = 6.62607015e-27   # erg s
ev2erg = 1.60217653e-12     # matches src/clib/update_UVbackground_rates.cpp
c_cgs = 2.99792458e10       # cm/s

# ionization thresholds
E_TH_EV = {"CI": 11.26, "OI": 13.62}

# a tiny relative wavelength shift used to keep duplicated wavelengths
# (absorption edges) strictly increasing without smearing the edge
EDGE_EPS = 1.0e-9

J_FLOOR = 1.0e-300


def read_hm12_spectrum(fname):
    """Read a Haardt & Madau (2012) "UVB.out" spectrum file.

    Returns (z, lam, J): redshifts (N_z,), wavelengths in Angstroms (N_lam,)
    sorted ascending with duplicated edge wavelengths nudged apart by a
    negligible amount, and J in erg/s/cm^2/Hz/sr with shape (N_lam, N_z).
    """
    rows = []
    with open(fname) as fh:
        for line in fh:
            stripped = line.strip()
            if (not stripped) or stripped.startswith("#"):
                continue
            rows.append(np.array(stripped.split(), dtype=float))
    z = rows[0]
    data = np.array(rows[1:])
    lam = data[:, 0]
    J = data[:, 1:]
    if J.shape[1] != z.size:
        raise ValueError(
            f"expected {z.size} intensity columns per wavelength row in "
            f"{fname}, found {J.shape[1]}"
        )
    order = np.argsort(lam, kind="stable")
    lam, J = lam[order], J[order]
    # nudge duplicated wavelengths (absorption edges) so lam is strictly
    # increasing; the stable sort keeps the two edge rows in file order
    for i in range(1, lam.size):
        if lam[i] <= lam[i - 1]:
            lam[i] = lam[i - 1] * (1.0 + EDGE_EPS)
    return z, lam, J


def read_leiden_cross_section(fname):
    """Read wavelength (Angstroms) and photo-ionization cross section (cm^2)
    from a Leiden database HDF5 cross-section file."""
    with h5py.File(fname, "r") as f:
        lam = f["wavelength"][:] * 10.0  # nm -> Angstrom
        sigma = f["photoionisation"][:]
    order = np.argsort(lam, kind="stable")
    lam, sigma = lam[order], sigma[order]
    for i in range(1, lam.size):
        if lam[i] <= lam[i - 1]:
            lam[i] = lam[i - 1] * (1.0 + EDGE_EPS)
    return lam, sigma


def photoionization_rates(z, lam_spec, J, lam_sig, sigma, threshold_ev):
    """Compute k(z) = int 4 pi J/(h nu) sigma dnu in 1/s for each redshift.

    The integral is evaluated in wavelength space,
        k = (4 pi / h) * int J(lam) sigma(lam) dlam / lam ,
    on the union of the spectrum and cross-section wavelength grids between
    the shortest tabulated wavelength and the ionization threshold.
    """
    lam_th = h_planck * c_cgs / (threshold_ev * ev2erg) * 1.0e8  # Angstrom

    lam_lo = max(lam_spec[0], lam_sig[0])
    grid = np.union1d(lam_spec, lam_sig)
    grid = np.union1d(grid, [lam_th])
    grid = grid[(grid >= lam_lo) & (grid <= lam_th)]

    # cross section: linear interpolation, zero outside its tabulated range
    # and beyond the ionization threshold
    sig_grid = np.interp(grid, lam_sig, sigma, left=0.0, right=0.0)

    rates = np.empty(z.size)
    log_lam_spec = np.log(lam_spec)
    log_grid = np.log(grid)
    for iz in range(z.size):
        # intensity: power-law (log-log) interpolation between spectrum points
        logJ = np.log(np.maximum(J[:, iz], J_FLOOR))
        J_grid = np.exp(np.interp(log_grid, log_lam_spec, logJ))
        J_grid[J_grid <= 10.0 * J_FLOOR] = 0.0
        integrand = J_grid * sig_grid / grid
        rates[iz] = 4.0 * np.pi / h_planck * np.trapz(integrand, grid)
    return rates


def interp_to_table_redshifts(z_table, z_spec, rates):
    """Interpolate rates (log-log in 1+z) onto the table redshift grid.

    Redshifts where the spectrum is empty (zero rate, e.g. beyond the
    reionization sources) are excluded; they must lie outside the table's
    redshift range.
    """
    positive = rates > 0.0
    z_spec, rates = z_spec[positive], rates[positive]
    if z_table.min() < z_spec.min() - 1e-8 or z_table.max() > z_spec.max() + 1e-8:
        raise ValueError(
            "the table redshift grid extends beyond the redshifts with a "
            f"non-zero spectrum; table: [{z_table.min()}, {z_table.max()}], "
            f"spectrum: [{z_spec.min()}, {z_spec.max()}]"
        )
    return np.exp(
        np.interp(np.log1p(z_table), np.log1p(z_spec), np.log(rates))
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
    )
    parser.add_argument("--table", required=True,
                        help="existing Grackle UV background table (HDF5)")
    parser.add_argument("--spectrum", required=True,
                        help="Haardt & Madau (2012) UVB.out spectrum file")
    parser.add_argument("--sigma-ci", required=True,
                        help="Leiden database HDF5 cross-section file for C")
    parser.add_argument("--sigma-oi", required=True,
                        help="Leiden database HDF5 cross-section file for O")
    parser.add_argument("--output", required=True,
                        help="output HDF5 file (a copy of --table plus the "
                             "new datasets; must differ from --table)")
    parser.add_argument("--clobber", action="store_true",
                        help="overwrite --output if it exists")
    args = parser.parse_args()

    if os.path.abspath(args.output) == os.path.abspath(args.table):
        parser.error("--output must differ from --table "
                     "(the input table is never modified)")
    if os.path.exists(args.output) and not args.clobber:
        parser.error(f"{args.output} exists (use --clobber to overwrite)")

    z_spec, lam_spec, J = read_hm12_spectrum(args.spectrum)

    with h5py.File(args.table, "r") as f:
        z_table = f["/UVBRates/z"][:]
        k24 = f["/UVBRates/Chemistry/k24"][:]

    provenance = (
        "computed by scripts/add_metal_UVB_rates.py: Haardt & Madau (2012, "
        "ApJ, 746, 125) spectrum; photo-ionization cross section from the "
        "Leiden database (Heays, Bosman & van Dishoeck 2017, A&A, 602, A105)"
    )
    new_datasets = {}
    for species, sigma_file in (("CI", args.sigma_ci), ("OI", args.sigma_oi)):
        lam_sig, sigma = read_leiden_cross_section(sigma_file)
        rates = photoionization_rates(
            z_spec, lam_spec, J, lam_sig, sigma, E_TH_EV[species])
        new_datasets[f"kph{species}"] = interp_to_table_redshifts(
            z_table, z_spec, rates)

    shutil.copy(args.table, args.output)
    with h5py.File(args.output, "r+") as f:
        for name, rates in new_datasets.items():
            dset = f.create_dataset(
                f"/UVBRates/Chemistry/{name}", data=rates.astype(np.float64))
            dset.attrs["units"] = "s^-1"
            dset.attrs["provenance"] = provenance

    # sanity output
    kph_ci = new_datasets["kphCI"]
    kph_oi = new_datasets["kphOI"]
    print(f"wrote {args.output}")
    print(f"{'z':>6} {'kphCI [1/s]':>12} {'kphOI [1/s]':>12} {'k24 [1/s]':>12}")
    log1pz = np.log1p(z_table)
    for z0 in (0.0, 1.0, 2.0, 3.0, 6.0):
        vals = [np.exp(np.interp(np.log1p(z0), log1pz, np.log(y)))
                for y in (kph_ci, kph_oi, k24)]
        print(f"{z0:>6.1f} {vals[0]:>12.4e} {vals[1]:>12.4e} {vals[2]:>12.4e}")

    bad = kph_oi > k24
    if np.any(bad):
        zbad = z_table[bad]
        print(f"WARNING: kphOI > k24 at {bad.sum()} redshift(s) "
              f"(z = {zbad.min():.3f} .. {zbad.max():.3f}); "
              "kphOI is expected to be smaller than k24", file=sys.stderr)


if __name__ == "__main__":
    main()
