/*
 * QIClib (Quantum information and computation library)
 *
 * Copyright (c) 2015 - 2016  Titas Chanda (titas.chanda@gmail.com)
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

// floating point precision
#ifndef QICLIB_FLOAT_PRECISION
#define QICLIB_FLOAT_PRECISION (100.0 * std::numeric_limits<float>::epsilon())
#endif
#ifndef QICLIB_DOUBLE_PRECISION
#define QICLIB_DOUBLE_PRECISION (100.0 * std::numeric_limits<double>::epsilon())
#endif

// init on or off
#define QICLIB_INIT
#ifdef QICLIB_NO_INIT_MESSAGE
#undef QICLIB_INIT
#endif

// nlopt features on or off
#define QICLIB_NLOPT
#ifdef QICLIB_DONT_USE_NLOPT
#undef QICLIB_NLOPT
#endif


#ifdef QICLIB_USE_OLD_DISCORD
#undef QICLIB_NEW_DISCORD
#endif


// openmp parallelisation
#ifdef QICLIB_PARALLEL
#define QICLIB_OPENMP_FOR _Pragma("omp parallel for")
#define QICLIB_OPENMP_FOR_COLLAPSE_2 _Pragma("omp parallel for collapse(2)")
#define QICLIB_OPENMP_CRITICAL _Pragma("omp critical")
#else
#define QICLIB_OPENMP_FOR
#define QICLIB_OPENMP_FOR_COLLAPSE_2
#define QICLIB_OPENMP_CRITICAL
#endif
