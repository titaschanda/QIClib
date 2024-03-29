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

#ifndef _QICLIB_GENERATOR_HPP_
#define _QICLIB_GENERATOR_HPP_

#include "../basic/type_traits.hpp"
#include "../class/exception.hpp"
#include "../internal/as_arma.hpp"
#include <armadillo>

namespace qic {

//******************************************************************************

template <typename T1 = double>

inline arma::Col<T1> mket(const arma::uvec& mask, const arma::uvec& dim) {
  const arma::uword m = mask.n_elem;
  const arma::uword D = arma::prod(dim);

#ifndef QICLIB_NO_DEBUG
  if (m == 0)
    throw Exception("qic::mket", Exception::type::ZERO_SIZE);

  if (dim.n_elem == 0 || D == 0)
    throw Exception("qic::mket", Exception::type::INVALID_DIMS);

  if (m != dim.n_elem)
    throw Exception("qic::mket", Exception::type::SUBSYS_MISMATCH_DIMS);

  for (arma::uword i = 0; i < m; ++i)
    if (mask.at(i) >= dim.at(i))
      throw Exception("qic::mket", Exception::type::SUBSYS_MISMATCH_DIMS);
#endif

  arma::uword product(1);
  arma::uword index = 0;

  for (arma::uword i = 1; i < m; ++i) {
    product *= dim.at(m - i);
    index += product * mask.at(m - 1 - i);
  }

  index += mask.at(m - 1);

  arma::Col<T1> ret(D, arma::fill::zeros);
  ret.at(index) = static_cast<T1>(1);
  return ret;
}

//******************************************************************************

template <typename T1 = double>

inline arma::Col<T1> mket(const arma::uvec& mask, arma::uword d = 2) {
  arma::uvec dim(mask.n_elem);
  dim.fill(d);
  return mket<T1>(mask, dim);
}

//******************************************************************************

template <typename T1 = double>

inline arma::Mat<T1> mproj(const arma::uvec& mask, const arma::uvec& dim) {
  const arma::uword m = mask.n_elem;
  const arma::uword D = arma::prod(dim);

#ifndef QICLIB_NO_DEBUG
  if (m == 0)
    throw Exception("qic::mproj", Exception::type::ZERO_SIZE);

  if (dim.n_elem == 0 || D == 0)
    throw Exception("qic::mproj", Exception::type::INVALID_DIMS);

  if (m != dim.n_elem)
    throw Exception("qic::mproj", Exception::type::SUBSYS_MISMATCH_DIMS);

  for (arma::uword i = 0; i < m; ++i)
    if (mask.at(i) >= dim.at(i))
      throw Exception("qic::mproj", Exception::type::SUBSYS_MISMATCH_DIMS);
#endif

  arma::uword product(1);
  arma::uword index = 0;

  for (arma::uword i = 1; i < m; ++i) {
    product *= dim.at(m - i);
    index += product * mask.at(m - 1 - i);
  }
  index += mask.at(m - 1);

  arma::Mat<T1> ret(D, D, arma::fill::zeros);
  ret.at(index, index) = static_cast<T1>(1);
  return ret;
}

//******************************************************************************

template <typename T1 = double>

inline arma::Mat<T1> mproj(const arma::uvec& mask, arma::uword d = 2) {
  arma::uvec dim(mask.n_elem);
  dim.fill(d);
  return mproj<T1>(mask, dim);
}

//******************************************************************************

}  // namespace qic

#endif
