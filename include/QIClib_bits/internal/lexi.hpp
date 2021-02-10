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

#ifndef _QICLIB_INTERNAL_LEXI_HPP_
#define _QICLIB_INTERNAL_LEXI_HPP_

#include <armadillo>

namespace qic {

//************************************************************************

namespace _internal {

//******************************************************************************

inline void num_to_lexi(arma::uword n, const arma::uvec& dim,
                        arma::uword* result) noexcept {
  for (arma::uword i = 1; i < dim.n_elem; ++i) {
    result[dim.n_elem - i] = n % (dim.at(dim.n_elem - i));
    n /= (dim.at(dim.n_elem - i));
  }
  result[0] = n;
}

//******************************************************************************
inline arma::uword lexi_to_num(const arma::uword* index,
                               const arma::uvec& dim) noexcept {
  arma::uword product(1);
  arma::uword I = 0;

  for (arma::uword i = 1; i < dim.n_elem; ++i) {
    product *= dim.at(dim.n_elem - i);
    I += product * index[dim.n_elem - i - 1];
  }
  return I + index[dim.n_elem - 1];
}

//******************************************************************************

}  // namespace _internal

}  // namespace qic

#endif
