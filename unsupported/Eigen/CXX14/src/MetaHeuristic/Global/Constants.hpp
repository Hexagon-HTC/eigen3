// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2022 Shawn Li <tokinobug@163.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_HEU_CONSTANTS_HPP
#define EIGEN_HEU_CONSTANTS_HPP

#include <stdint.h>
#include <limits>

#include "InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

/**
 * \ingroup CXX14_METAHEURISTIC
 * \brief Positive infinity float
 *
 */
const float pinfF = std::numeric_limits<float>::infinity();

/**
 * \ingroup CXX14_METAHEURISTIC
 * \brief Positive infinity double
 *
 */
const double pinfD = std::numeric_limits<double>::infinity();

/**
 * \ingroup CXX14_METAHEURISTIC
 * \brief Negativev infinity float
 *
 */
const float ninfF = -pinfF;

/**
 * \ingroup CXX14_METAHEURISTIC
 * \brief Negative infinity double
 *
 */
const double ninfD = -pinfD;

}  //  namespace internal

}  //  namespace Eigen

#endif  // EIGEN_HEU_CONSTANTS_HPP