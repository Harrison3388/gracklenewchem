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

// The channel enum is generated from the GRACKLE_COOLING_CONTRIBUTION_CHANNELS
// X-macro defined in grackle.h, so the enum values stay in lock-step with the
// grackle_cooling_rate_contribution_data struct and the output-field table.
// Total is value 0 and Residual is value 1 (relied on below); every channel
// after them is "primitive" unless excluded by cooling_contribution_is_primitive.
enum CoolingContributionChannel {
#define GRACKLE_COOLING_CONTRIBUTION_ENUM(enum_suffix, field) \
  CoolingContribution##enum_suffix,
  GRACKLE_COOLING_CONTRIBUTION_CHANNELS(GRACKLE_COOLING_CONTRIBUTION_ENUM)
#undef GRACKLE_COOLING_CONTRIBUTION_ENUM
  CoolingContributionFieldCount
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
