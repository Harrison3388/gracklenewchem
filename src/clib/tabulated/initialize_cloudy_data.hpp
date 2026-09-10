//===----------------------------------------------------------------------===//
//
// See the LICENSE file for license and copyright information
// SPDX-License-Identifier: NCSA AND BSD-3-Clause
//
//===----------------------------------------------------------------------===//
///
/// @file
/// Declares the function to initialize the cloudy data
///
//===----------------------------------------------------------------------===//

#ifndef TABULATED_INITIALIZE_CLOUDY_DATA_HPP
#define TABULATED_INITIALIZE_CLOUDY_DATA_HPP

#include "grackle.h"
#include "../support/config.hpp"

namespace GRIMPL_NAMESPACE_DECL {

// initialize cloudy cooling data
int initialize_cloudy_data(chemistry_data* my_chemistry,
                           chemistry_data_storage* my_rates,
                           cloudy_data* my_cloudy, const char* group_name,
                           code_units* my_units, int read_data);

int free_cloudy_data(cloudy_data* my_cloudy, chemistry_data* my_chemistry,
                     int primordial);

/// Reads the "HydrogenFractionByMass" root attribute of the data file: the
/// H mass fraction assumed when its heating/cooling tables were generated.
///
/// @param[in]  fname Path to the data file
/// @param[out] hfrac Set to the attribute's value when it is present
///
/// @returns 1 if the attribute was read, 0 if the file has no such attribute
///     (older data files), and -1 on error (a message is printed)
int read_cloudy_HydrogenFractionByMass(const char* fname, double* hfrac);

}  // namespace GRIMPL_NAMESPACE_DECL

#endif /* TABULATED_INITIALIZE_CLOUDY_DATA_HPP */
