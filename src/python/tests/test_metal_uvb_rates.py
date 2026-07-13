########################################################################
#
# Tests for the UV-background photo-ionization rates of metal species
# (kphCI/kphOI datasets in the UV background table, applied to the
# C I and O I species when metal_chemistry > 0).
#
# Copyright (c) 2013, Enzo/Grackle Development Team.
#
# Distributed under the terms of the Enzo Public Licence.
#
# The full license is in the file LICENSE, distributed with this
# software.
########################################################################

import os
import shutil

import h5py
import numpy as np

from gracklepy import chemistry_data, FluidContainer
from gracklepy.utilities.data_path import grackle_data_dir
from gracklepy.utilities.physical_constants import (
    mass_hydrogen_cgs,
    sec_per_Myr,
    cm_per_mpc,
)

STOCK_TABLE = os.path.join(grackle_data_dir, "CloudyData_UVB=HM2012.h5")

# dummy redshift-independent photo-ionization rates (in 1/s) written into
# copies of the UV background table by make_table_with_rates
K_UVB_CI = 1.0e-12
K_UVB_OI = 3.0e-13

# dummy radiative-transfer photo-ionization rates (in 1/s)
K_RT_CI = 2.0e-12
K_RT_OI = 5.0e-13

# time step (in s) for the single backward-Euler step taken by each test;
# chosen so that k*dt ~ 1e-2 (well above the comparison tolerance, but
# small enough that a single step is well converged)
DT_SECONDS = 1.0e10

# relative tolerance when comparing measured species ratios against the
# analytic single-step estimate (residual coupling through the electron
# density perturbs the ratios at the ~1e-4 level)
RATIO_ATOL = 2.0e-3


def make_table_with_rates(tmp_path, k_ci, k_oi):
    """Copy the stock UV background table and add constant kphCI/kphOI
    datasets with the given values (in 1/s)."""
    table_copy = str(tmp_path / "HM2012_with_metal_rates.h5")
    shutil.copy(STOCK_TABLE, table_copy)
    with h5py.File(table_copy, "r+") as f:
        z = f["/UVBRates/z"][:]
        f["/UVBRates/Chemistry/kphCI"] = np.full_like(z, k_ci)
        f["/UVBRates/Chemistry/kphOI"] = np.full_like(z, k_oi)
    return table_copy


def setup_fluid_container(data_file, use_rt=False, rt_ci=0.0, rt_oi=0.0):
    """Return an initialized (chemistry_data, FluidContainer) pair with
    metal_chemistry enabled, the UV background on, and (optionally) the
    radiative-transfer metal ionization rates set (in 1/s)."""
    my_chemistry = chemistry_data()
    my_chemistry.use_grackle = 1
    my_chemistry.with_radiative_cooling = 0
    my_chemistry.primordial_chemistry = 2
    my_chemistry.metal_cooling = 1
    my_chemistry.UVbackground = 1
    my_chemistry.grackle_data_file = data_file
    my_chemistry.metal_chemistry = 1
    my_chemistry.dust_chemistry = 0
    my_chemistry.dust_species = 0
    my_chemistry.multi_metals = 0
    if use_rt:
        my_chemistry.use_radiative_transfer = 1
        my_chemistry.radiative_transfer_metal_ionization = 1

    # Set units (proper frame, z = 0)
    my_chemistry.comoving_coordinates = 0
    my_chemistry.a_units = 1.0
    my_chemistry.a_value = 1.0
    my_chemistry.density_units = mass_hydrogen_cgs
    my_chemistry.length_units = cm_per_mpc
    my_chemistry.time_units = sec_per_Myr
    my_chemistry.set_velocity_units()

    assert my_chemistry.initialize() == 1

    fc = FluidContainer(my_chemistry, n_vals=1)
    fc["density"][:] = 1.0
    fc["HI_density"][:] = 1.0e-4
    fc["HII_density"][:] = 0.759 - 1.0e-4
    fc["HeI_density"][:] = 1.0e-5
    fc["HeII_density"][:] = 1.0e-5
    fc["HeIII_density"][:] = 0.24 - 2.0e-5
    fc["e_density"][:] = fc["HII_density"] + \
        fc["HeII_density"] / 4.0 + fc["HeIII_density"] / 2.0
    for name in fc.density_fields:
        if fc[name][0] == 0.0:
            fc[name][:] = 1.0e-20
    fc["metal_density"][:] = 1.0e-3
    fc["CI_density"][:] = 1.0e-6
    fc["OI_density"][:] = 1.0e-6
    fc["internal_energy"][:] = \
        1.0e13 / my_chemistry.get_velocity_units() ** 2 * 1.0e4
    if use_rt:
        for name in ("RT_heating_rate", "RT_HI_ionization_rate",
                     "RT_HeI_ionization_rate", "RT_HeII_ionization_rate",
                     "RT_H2_dissociation_rate"):
            fc[name][:] = 0.0
        # the RT rate fields are in code units (1 / time_units)
        fc["RT_CI_ionization_rate"][:] = rt_ci * my_chemistry.time_units
        fc["RT_OI_ionization_rate"][:] = rt_oi * my_chemistry.time_units
    return my_chemistry, fc


def evolve_ratios(data_file, use_rt=False, rt_ci=0.0, rt_oi=0.0):
    """Take a single chemistry step of DT_SECONDS and return the
    (CI_final/CI_initial, OI_final/OI_initial) density ratios."""
    my_chemistry, fc = setup_fluid_container(
        data_file, use_rt=use_rt, rt_ci=rt_ci, rt_oi=rt_oi)
    ci_initial = fc["CI_density"][0].item()
    oi_initial = fc["OI_density"][0].item()
    fc.solve_chemistry(DT_SECONDS / my_chemistry.time_units)
    return (fc["CI_density"][0].item() / ci_initial,
            fc["OI_density"][0].item() / oi_initial)


def expected_ratio(k_total):
    # the solver takes a single backward-Euler step:
    #   n_final = n_initial / (1 + k * dt)
    return 1.0 / (1.0 + k_total * DT_SECONDS)


def test_uvb_metal_photoionization(tmp_path):
    # a table with kphCI/kphOI datasets must ionize C I and O I at the
    # tabulated rates; normalizing by an otherwise-identical run without
    # the datasets isolates the photo-ionization terms from the rest of
    # the metal network (collisional terms, renormalization).
    metal_table = make_table_with_rates(tmp_path, K_UVB_CI, K_UVB_OI)

    ci_base, oi_base = evolve_ratios(STOCK_TABLE)
    ci_uvb, oi_uvb = evolve_ratios(metal_table)

    assert np.isclose(ci_uvb / ci_base, expected_ratio(K_UVB_CI),
                      rtol=0.0, atol=RATIO_ATOL)
    assert np.isclose(oi_uvb / oi_base, expected_ratio(K_UVB_OI),
                      rtol=0.0, atol=RATIO_ATOL)


def test_rt_and_uvb_rates_are_summed(tmp_path):
    # the UV background rates must be added on top of the local
    # radiative-transfer ionization rates, never replace them
    metal_table = make_table_with_rates(tmp_path, K_UVB_CI, K_UVB_OI)

    ci_base, oi_base = evolve_ratios(STOCK_TABLE, use_rt=True)
    ci_rt, oi_rt = evolve_ratios(
        STOCK_TABLE, use_rt=True, rt_ci=K_RT_CI, rt_oi=K_RT_OI)
    ci_both, oi_both = evolve_ratios(
        metal_table, use_rt=True, rt_ci=K_RT_CI, rt_oi=K_RT_OI)

    # RT alone still works
    assert np.isclose(ci_rt / ci_base, expected_ratio(K_RT_CI),
                      rtol=0.0, atol=RATIO_ATOL)
    assert np.isclose(oi_rt / oi_base, expected_ratio(K_RT_OI),
                      rtol=0.0, atol=RATIO_ATOL)
    # RT + UV background destroy C I / O I at the summed rate
    assert np.isclose(ci_both / ci_base, expected_ratio(K_RT_CI + K_UVB_CI),
                      rtol=0.0, atol=RATIO_ATOL)
    assert np.isclose(oi_both / oi_base, expected_ratio(K_RT_OI + K_UVB_OI),
                      rtol=0.0, atol=RATIO_ATOL)


def test_old_table_backward_compatible(tmp_path, capfd):
    # a UV background table without the kphCI/kphOI datasets (all tables
    # predating their introduction) must initialize successfully, warn
    # about the missing datasets, and behave exactly like a table with
    # all-zero rates
    zero_table = make_table_with_rates(tmp_path, 0.0, 0.0)

    ci_old, oi_old = evolve_ratios(STOCK_TABLE)
    captured = capfd.readouterr()
    assert "kphCI" in captured.err
    assert "kphOI" in captured.err

    ci_zero, oi_zero = evolve_ratios(zero_table)
    captured = capfd.readouterr()
    assert "kphCI" not in captured.err
    assert "kphOI" not in captured.err

    assert np.isclose(ci_old, ci_zero, rtol=1.0e-12, atol=0.0)
    assert np.isclose(oi_old, oi_zero, rtol=1.0e-12, atol=0.0)
