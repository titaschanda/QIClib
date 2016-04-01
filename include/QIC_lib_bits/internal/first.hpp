/*
 * QIC_lib (Quantum information and computation library)
 *
 * Copyright (c) 2015 - 2016  Titas Chanda (titas.chanda@gmail.com)
 *
 * This file is part of QIC_lib.
 *
 * QIC_lib is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * QIC_lib is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QIC_lib.  If not, see <http://www.gnu.org/licenses/>.
 */

namespace qic {

namespace _internal {

namespace protect_subs {

//******************************************************************************

template <typename T> struct cond_I { static constexpr T value = 0; };

template <typename T> struct cond_I<std::complex<T> > {
  static constexpr std::complex<T> value = {0, 1};
};

template <typename T1> constexpr T1 cond_I<T1>::value;

template <typename T1>
constexpr std::complex<T1> cond_I<std::complex<T1> >::value;

//******************************************************************************

#ifndef QIC_LIB_MAXQDIT_COUNT
constexpr arma::uword MAXQDIT = 24;
#else
constexpr arma::uword MAXQDIT = QIC_LIB_MAXQDIT_COUNT;
#endif

//******************************************************************************

}  // namespace protect_subs

}  // namespace _internal

}  // namespace qic
