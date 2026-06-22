//===----------------------------------------------------------------------===//
//
// See the LICENSE file for license and copyright information
// SPDX-License-Identifier: NCSA AND BSD-3-Clause
//
//===----------------------------------------------------------------------===//
///
/// @file
/// Helpers for collecting additive cooling/heating source terms.
///
//===----------------------------------------------------------------------===//

#ifndef COOLING_RATE_CONTRIBUTIONS_HPP
#define COOLING_RATE_CONTRIBUTIONS_HPP

#include "grackle.h"

#include <cmath>

namespace grackle::impl {

enum CoolingContributionChannel {
  CoolingContributionTotal = 0,
  CoolingContributionResidual = 1,
  CoolingContributionCollisionalExcitation = 2,
  CoolingContributionCollisionalIonisation = 3,
  CoolingContributionRecombinationCooling = 4,
  CoolingContributionBremsstrahlung = 5,
  CoolingContributionH2Line = 6,
  CoolingContributionH2CIE = 7,
  CoolingContributionHD = 8,
  CoolingContributionDustGasGrain = 9,
  CoolingContributionPhotoelectric = 10,
  CoolingContributionDustRecombination = 11,
  CoolingContributionPhotoionizationHeating = 12,
  CoolingContributionCloudyPrimordial = 13,
  CoolingContributionCompton = 14,
  CoolingContributionRTPhotoheating = 15,
  CoolingContributionCloudyMetal = 16,
  CoolingContributionCI = 17,
  CoolingContributionCII = 18,
  CoolingContributionOI = 19,
  CoolingContributionCO = 20,
  CoolingContributionOH = 21,
  CoolingContributionH2O = 22,
  CoolingContributionVolumetricHeating = 23,
  CoolingContributionSpecificHeating = 24,
  CoolingContributionChemistryHIHeI = 25,
  CoolingContributionChemistryHeII = 26,
  CoolingContributionChemistryH2Gas = 27,
  CoolingContributionChemistryH2Dust = 28,
  CoolingContributionChemistryHICollisionalIonization = 29,
  CoolingContributionChemistryHIIRecombination = 30,
  CoolingContributionChemistryHeICollisionalIonization = 31,
  CoolingContributionChemistryHeIIRecombination = 32,
  CoolingContributionChemistryHeIICollisionalIonization = 33,
  CoolingContributionChemistryHeIIIRecombination = 34,
  CoolingContributionChemistryH2HminusFormation = 35,
  CoolingContributionChemistryH2ThreeBodyFormation = 36,
  CoolingContributionChemistryH2CollisionalDissociation = 37,
  CoolingContributionFieldCount = 38
};

struct CoolingContributionScratch {
  double* fields[CoolingContributionFieldCount] = {};
};

inline void cooling_contribution_add(CoolingContributionScratch* contributions,
                                     CoolingContributionChannel channel, int i,
                                     double term) {
  if ((contributions != nullptr) && (contributions->fields[channel] != nullptr)) {
    contributions->fields[channel][i] += term;
  }
}

inline bool cooling_contribution_is_primitive(int channel) {
  return (channel != CoolingContributionTotal) &&
         (channel != CoolingContributionResidual) &&
         (channel != CoolingContributionChemistryHIHeI) &&
         (channel != CoolingContributionChemistryHeII) &&
         (channel != CoolingContributionChemistryH2Gas);
}

inline void cooling_contribution_scale_active(
    CoolingContributionScratch* contributions, int i, double factor) {
  if (contributions == nullptr) {
    return;
  }
  for (int channel = CoolingContributionCollisionalExcitation;
       channel < CoolingContributionFieldCount; channel++) {
    if (cooling_contribution_is_primitive(channel) &&
        contributions->fields[channel] != nullptr) {
      contributions->fields[channel][i] *= factor;
    }
  }
}

inline double cooling_contribution_sum_active(
    const CoolingContributionScratch* contributions, int i) {
  if (contributions == nullptr) {
    return 0.0;
  }

  double sum = 0.0;
  for (int channel = CoolingContributionCollisionalExcitation;
       channel < CoolingContributionFieldCount; channel++) {
    if (cooling_contribution_is_primitive(channel) &&
        contributions->fields[channel] != nullptr) {
      sum += contributions->fields[channel][i];
    }
  }
  return sum;
}

}  // namespace grackle::impl

#endif /* COOLING_RATE_CONTRIBUTIONS_HPP */
