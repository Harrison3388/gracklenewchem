/***********************************************************************
/
/ Calculate additive cooling/heating contribution fields.
/
/ This is an experimental diagnostic API for inspecting the terms that are
/ accumulated into edot by cool1d_multi_g and rate_timestep_g.
/
************************************************************************/

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include "cool1d_multi_g.hpp"
#include "cooling_rate_contributions.hpp"
#include "full_rxn_rate_buf.hpp"
#include "gas_props.hpp"
#include "grackle.h"
#include "inject_model/grain_metal_inject_pathways.hpp"
#include "inject_model/misc.hpp"
#include "internal_types.hpp"
#include "internal_units.hpp"
#include "lookup_cool_rates1d.hpp"
#include "rate_timestep_g.hpp"
#include "scale_fields.hpp"
#include "self_shielding_err_check.hpp"
#include "support/index_helper.hpp"
#include "update_UVbackground_rates.hpp"
#include "utils-cpp.hpp"

#include "ceiling_species.hpp"

namespace {

using grackle::impl::CoolingContributionFieldCount;
using grackle::impl::CoolingContributionResidual;
using grackle::impl::CoolingContributionScratch;
using grackle::impl::CoolingContributionTotal;

std::array<gr_float*, CoolingContributionFieldCount>
contribution_output_fields(grackle_cooling_rate_contribution_data* data) {
  return {
      data->total,
      data->residual,
      data->collisional_excitation,
      data->collisional_ionisation,
      data->recombination_cooling,
      data->bremsstrahlung,
      data->h2_line,
      data->h2_cie,
      data->hd,
      data->dust_gas_grain,
      data->photoelectric,
      data->dust_recombination,
      data->photoionization_heating,
      data->cloudy_primordial,
      data->compton,
      data->rt_photoheating,
      data->cloudy_metal,
      data->ci,
      data->cii,
      data->oi,
      data->co,
      data->oh,
      data->h2o,
      data->volumetric_heating,
      data->specific_heating,
      data->chemistry_hi_hei,
      data->chemistry_heii,
      data->chemistry_h2_gas,
      data->chemistry_h2_dust,
      data->chemistry_hi_collisional_ionization,
      data->chemistry_hii_recombination,
      data->chemistry_hei_collisional_ionization,
      data->chemistry_heii_recombination,
      data->chemistry_heii_collisional_ionization,
      data->chemistry_heiii_recombination,
      data->chemistry_h2_hminus_formation,
      data->chemistry_h2_threebody_formation,
      data->chemistry_h2_collisional_dissociation,
  };
}

photo_rate_storage contribution_uvb_rates(chemistry_data* my_chemistry,
                                          chemistry_data_storage* my_rates,
                                          code_units* my_units,
                                          int* ierr) {
  photo_rate_storage my_uvb_rates;

  my_uvb_rates.k24 = my_uvb_rates.k25 = my_uvb_rates.k26 =
      my_uvb_rates.k27 = my_uvb_rates.k28 = my_uvb_rates.k29 =
          my_uvb_rates.k30 = my_uvb_rates.k31 = my_uvb_rates.piHI =
              my_uvb_rates.piHeI = my_uvb_rates.piHeII =
                  my_uvb_rates.crsHI = my_uvb_rates.crsHeI =
                      my_uvb_rates.crsHeII = my_uvb_rates.comp_xray =
                          my_uvb_rates.temp_xray = 0.;

  if (my_chemistry->UVbackground == 1) {
    if (grackle::impl::update_UVbackground_rates(
            my_chemistry, my_rates, &my_uvb_rates, my_units) != GR_SUCCESS) {
      std::fprintf(stderr, "Error in update_UVbackground_rates.\n");
      *ierr = GR_FAIL;
    }
  } else {
    my_uvb_rates.k24 = my_rates->k24;
    my_uvb_rates.k25 = my_rates->k25;
    my_uvb_rates.k26 = my_rates->k26;
    my_uvb_rates.k27 = my_rates->k27;
    my_uvb_rates.k28 = my_rates->k28;
    my_uvb_rates.k29 = my_rates->k29;
    my_uvb_rates.k30 = my_rates->k30;
    my_uvb_rates.k31 = my_rates->k31;
    my_uvb_rates.piHI = my_rates->piHI;
    my_uvb_rates.piHeI = my_rates->piHeI;
    my_uvb_rates.piHeII = my_rates->piHeII;
    my_uvb_rates.crsHI = my_rates->crsHI;
    my_uvb_rates.crsHeI = my_rates->crsHeI;
    my_uvb_rates.crsHeII = my_rates->crsHeII;
    my_uvb_rates.comp_xray = my_rates->comp_xray;
    my_uvb_rates.temp_xray = my_rates->temp_xray;
  }

  return my_uvb_rates;
}

}  // namespace

extern "C" int local_calculate_cooling_rate_contributions(
    chemistry_data* my_chemistry, chemistry_data_storage* my_rates,
    code_units* my_units, grackle_field_data* my_fields, double dt_value,
    grackle_cooling_rate_contribution_data* contributions) {
  if (!my_chemistry->use_grackle) {
    return GR_SUCCESS;
  }

  if (contributions == nullptr) {
    std::fprintf(stderr, "Cooling contribution output struct is null.\n");
    return GR_FAIL;
  }

  int ierr = GR_SUCCESS;
  photo_rate_storage my_uvb_rates =
      contribution_uvb_rates(my_chemistry, my_rates, my_units, &ierr);
  if (ierr == GR_FAIL) {
    return GR_FAIL;
  }

  if (grackle::impl::self_shielding_err_check(my_chemistry, my_fields,
                                              "local_calculate_cooling_rate_"
                                              "contributions") !=
      GR_SUCCESS) {
    return GR_FAIL;
  }

  const int imetal = (my_fields->metal_density != NULL) ? TRUE : FALSE;
  GRIMPL_NS::InternalGrUnits internalu = GRIMPL_NS::new_internalu_(my_units);

  const gr_mask_type anydust =
      ((my_chemistry->h2_on_dust > 0) || (my_chemistry->dust_chemistry > 0))
          ? MASK_TRUE
          : MASK_FALSE;
  const double dom = GRIMPL_NS::internalu_calc_dom_(internalu);
  const double chunit = GRIMPL_NS::internalu_get_chunit_(internalu);
  const double dx_cgs = my_fields->grid_dx * internalu.xbase1;
  const double c_ljeans =
      GRIMPL_NS::internalu_calc_coef_ljeans_(internalu, my_chemistry->Gamma);

  const auto output_fields = contribution_output_fields(contributions);

  if (internalu.extfields_in_comoving == 1) {
    const gr_float factor = (gr_float)(std::pow(internalu.a_value, -3));
    GRIMPL_NS::scale_fields(
        imetal, factor, my_chemistry, my_fields,
        GRIMPL_NS::get_n_inject_pathway_density_ptrs(my_rates));
  }

  GRIMPL_NS::ceiling_species(imetal, my_chemistry, my_fields);

  const GRIMPL_NS::IndexHelper idx_helper =
      GRIMPL_NS::build_index_helper_(my_fields);

  OMP_PRAGMA("omp parallel")
  {
    GRIMPL_NS::GrainSpeciesCollection grain_temperatures =
        GRIMPL_NS::new_GrainSpeciesCollection(my_fields->grid_dimension[0]);
    GRIMPL_NS::LnTLinInterpBuf logTlininterp_buf =
        GRIMPL_NS::new_LnTLinInterpBuf(my_fields->grid_dimension[0]);
    GRIMPL_NS::Cool1DMultiScratchBuf cool1dmulti_buf =
        GRIMPL_NS::new_Cool1DMultiScratchBuf(my_fields->grid_dimension[0]);
    GRIMPL_NS::CoolHeatScratchBuf coolingheating_buf =
        GRIMPL_NS::new_CoolHeatScratchBuf(my_fields->grid_dimension[0]);
    GRIMPL_NS::InternalDustPropBuf internal_dust_prop_scratch_buf =
        GRIMPL_NS::new_InternalDustPropBuf(
            my_fields->grid_dimension[0],
            GRIMPL_NS::GrainMetalInjectPathways_get_n_log10Tdust_vals(
                my_rates->opaque_storage->inject_pathway_props));
    std::vector<double> dedot(my_fields->grid_dimension[0]);
    std::vector<double> HIdot(my_fields->grid_dimension[0]);
    GRIMPL_NS::FullRxnRateBuf rxn_rate_buf =
        GRIMPL_NS::new_FullRxnRateBuf(my_fields->grid_dimension[0]);
    GRIMPL_NS::ChemHeatingRates chemheatrates_buf =
        GRIMPL_NS::new_ChemHeatingRates(my_fields->grid_dimension[0]);

    std::vector<double> tgas(my_fields->grid_dimension[0]);
    std::vector<double> mmw(my_fields->grid_dimension[0]);
    std::vector<double> nelec_times_mH(my_fields->grid_dimension[0]);
    std::vector<double> tdust(my_fields->grid_dimension[0]);
    std::vector<double> metallicity(my_fields->grid_dimension[0]);
    std::vector<double> dust2gas(my_fields->grid_dimension[0]);
    std::vector<double> rhoH(my_fields->grid_dimension[0]);
    std::vector<double> edot(my_fields->grid_dimension[0]);

    std::vector<gr_mask_type> itmask(my_fields->grid_dimension[0]);
    std::vector<gr_mask_type> itmask_metal(my_fields->grid_dimension[0]);

    std::array<std::vector<double>, CoolingContributionFieldCount>
        contribution_storage;
    CoolingContributionScratch contribution_scratch;
    for (int channel = 0; channel < CoolingContributionFieldCount; channel++) {
      contribution_storage[channel].resize(my_fields->grid_dimension[0]);
      contribution_scratch.fields[channel] =
          contribution_storage[channel].data();
    }

    OMP_PRAGMA("omp for")
    for (int t = 0; t < idx_helper.outer_ind_size; t++) {
      const GRIMPL_NS::IndexRange idx_range =
          GRIMPL_NS::make_idx_range_(t, &idx_helper);
      const int k = idx_range.k;
      const int j = idx_range.j;

      for (int channel = 0; channel < CoolingContributionFieldCount; channel++) {
        std::fill(contribution_storage[channel].begin() + idx_range.i_start,
                  contribution_storage[channel].begin() + idx_range.i_stop,
                  0.0);
      }

      for (int i = idx_range.i_start; i < idx_range.i_stop; i++) {
        itmask[i] = MASK_TRUE;
      }

      GRIMPL_NS::extended_gas_props(
          tgas.data(), mmw.data(), rhoH.data(), metallicity.data(),
          nelec_times_mH.data(), logTlininterp_buf, imetal, itmask.data(),
          my_chemistry, &my_rates->cloudy_primordial, my_fields, internalu,
          idx_range, nullptr);

      GRIMPL_NS::cool1d_multi_g(
          imetal, edot.data(), tgas.data(), mmw.data(), tdust.data(),
          metallicity.data(), dust2gas.data(), rhoH.data(),
          nelec_times_mH.data(), itmask.data(), itmask_metal.data(),
          my_chemistry, my_rates, my_fields, my_uvb_rates, internalu,
          idx_range, grain_temperatures, logTlininterp_buf, cool1dmulti_buf,
          coolingheating_buf, &contribution_scratch);

      if (my_chemistry->primordial_chemistry > 0) {
        GRIMPL_NS::lookup_cool_rates1d(
            idx_range, anydust, tgas.data(), mmw.data(), tdust.data(),
            dust2gas.data(), dom, dx_cgs, c_ljeans, itmask.data(),
            itmask_metal.data(), dt_value, my_chemistry, my_rates, my_fields,
            my_uvb_rates, internalu, grain_temperatures, logTlininterp_buf,
            rxn_rate_buf, chemheatrates_buf,
            internal_dust_prop_scratch_buf);

        GRIMPL_NS::rate_timestep_g(
            dedot.data(), HIdot.data(), anydust, rhoH.data(),
            itmask.data(), edot.data(), chunit, dom, my_chemistry, my_fields,
            idx_range, chemheatrates_buf, rxn_rate_buf,
            &contribution_scratch);
      }

      for (int i = idx_range.i_start; i < idx_range.i_stop; i++) {
        contribution_scratch.fields[CoolingContributionTotal][i] = edot[i];
        contribution_scratch.fields[CoolingContributionResidual][i] =
            edot[i] -
            GRIMPL_NS::cooling_contribution_sum_active(&contribution_scratch,
                                                       i);

        for (int channel = 0; channel < CoolingContributionFieldCount;
             channel++) {
          if (output_fields[channel] != nullptr) {
            GRIMPL_NS::View<gr_float***> output(
                output_fields[channel], my_fields->grid_dimension[0],
                my_fields->grid_dimension[1], my_fields->grid_dimension[2]);
            output(i, j, k) =
                (gr_float)contribution_scratch.fields[channel][i];
          }
        }
      }
    }

    GRIMPL_NS::drop_GrainSpeciesCollection(&grain_temperatures);
    GRIMPL_NS::drop_LnTLinInterpBuf(&logTlininterp_buf);
    GRIMPL_NS::drop_Cool1DMultiScratchBuf(&cool1dmulti_buf);
    GRIMPL_NS::drop_CoolHeatScratchBuf(&coolingheating_buf);
    GRIMPL_NS::drop_InternalDustPropBuf(&internal_dust_prop_scratch_buf);
    GRIMPL_NS::drop_FullRxnRateBuffer(&rxn_rate_buf);
    GRIMPL_NS::drop_ChemHeatingRates(&chemheatrates_buf);
  }

  if (internalu.extfields_in_comoving == 1) {
    const gr_float factor = (gr_float)(std::pow(internalu.a_value, 3));
    GRIMPL_NS::scale_fields(
        imetal, factor, my_chemistry, my_fields,
        GRIMPL_NS::get_n_inject_pathway_density_ptrs(my_rates));
  }

  return GR_SUCCESS;
}
