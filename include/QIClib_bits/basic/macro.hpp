/*
 * QIClib (Quantum information and computation library)
 *
 * Copyright (c) 2015 - 2019  Titas Chanda (titas.chanda@gmail.com)
 *
 * This file is part of QIClib.
 *
 * QIClib is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * QIClib is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QIClib.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef _QICLIB_MACRO_HPP_
#define _QICLIB_MACRO_HPP_

#include <armadillo>

// GCC unused variable warning
#if (__GNUC__ && !__clang__)
#define _QICLIB_UNUSED_ __attribute__((unused))
#else
#define _QICLIB_UNUSED_
#endif

// Maximum qudit count
#ifndef QICLIB_MAXQDIT_COUNT
#define QICLIB_MAXQDIT_COUNT 40
#endif

#ifndef QICLIB_DC_USE_LIMIT
#define QICLIB_DC_USE_LIMIT 20
#endif

// floating point precision
#ifndef QICLIB_FLOAT_PRECISION
#define QICLIB_FLOAT_PRECISION (100.0 * std::numeric_limits<float>::epsilon())
#endif
#ifndef QICLIB_DOUBLE_PRECISION
#define QICLIB_DOUBLE_PRECISION (100.0 * std::numeric_limits<double>::epsilon())
#endif

// init on or off
#ifndef QICLIB_NO_INIT_MESSAGE
#define QICLIB_INIT
#endif

// SPM on or off
#ifndef QICLIB_NO_SPM
#define QICLIB_SPM
#endif

// GATES on or off
#ifndef QICLIB_NO_GATES
#define QICLIB_GATES
#endif

// round_off for uword/sword, don't change
#ifdef ARMA_64BIT_WORD
#define QICLIB_ROUND_OFF std::llround
#else
#define QICLIB_ROUND_OFF std::lround
#endif

// nlopt features on or off
#ifndef QICLIB_DONT_USE_NLOPT
#define QICLIB_NLOPT
#endif

// use old or new discord
#ifndef QICLIB_USE_OLD_DISCORD
#define QICLIB_NEW_DISCORD
#endif

// openmp parallelization
#if defined(QICLIB_USE_OPENMP) && defined(_OPENMP)
#define QICLIB_OPENMP_FOR _Pragma("omp parallel for")
#define QICLIB_OPENMP_FOR_COLLAPSE_2 _Pragma("omp parallel for collapse(2)")
#define QICLIB_OPENMP_CRITICAL _Pragma("omp critical")
#else
#define QICLIB_OPENMP_FOR
#define QICLIB_OPENMP_FOR_COLLAPSE_2
#define QICLIB_OPENMP_CRITICAL
#define QICLIB_USE_SERIAL_TRX
#define QICLIB_USE_SERIAL_TX
#define QICLIB_USE_SERIAL_SYSPERM
#define QICLIB_USE_SERIAL_APPLY
#define QICLIB_USE_SERIAL_MEASURE
#define QICLIB_USE_SERIAL_MAKE_CTRL
#endif

#if defined(QICLIB_USE_OPENMP_TRX) && defined(_OPENMP)
#undef QICLIB_USE_SERIAL_TRX
#endif

#if defined(QICLIB_USE_OPENMP_TX) && defined(_OPENMP)
#undef QICLIB_USE_SERIAL_TX
#endif

#if defined(QICLIB_USE_OPENMP_SYSPERM) && defined(_OPENMP)
#undef QICLIB_USE_SERIAL_SYSPERM
#endif

#if defined(QICLIB_USE_OPENMP_APPLY) && defined(_OPENMP)
#undef QICLIB_USE_SERIAL_APPLY
#endif

#if defined(QICLIB_USE_OPENMP_MEASURE) && defined(_OPENMP)
#undef QICLIB_USE_SERIAL_MEASURE
#endif

#if defined(QICLIB_USE_OPENMP_MAKE_CTRL) && defined(_OPENMP)
#undef QICLIB_USE_SERIAL_MAKE_CTRL
#endif

#endif
