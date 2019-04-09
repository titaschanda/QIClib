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

#ifndef _QICLIB_NEG_HPP_
#define _QICLIB_NEG_HPP_

#include "../basic/type_traits.hpp"
#include "../class/exception.hpp"
#include "../internal/as_arma.hpp"
#include <armadillo>

namespace qic {

//******************************************************************************

template <typename T1,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1> >::value, trait::pT<T1> >::type>

inline TR neg(const T1& rho1, arma::uvec subsys, arma::uvec dim) {
  const auto& rho = _internal::as_Mat(rho1);

#ifndef QICLIB_NO_DEBUG
  const bool checkV = (rho.n_cols != 1);

  if (rho.n_elem == 0)
    throw Exception("qic::neg", Exception::type::ZERO_SIZE);

  if (checkV)
    if (rho.n_rows != rho.n_cols)
      throw Exception("qic::neg",
                      Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);

  if (dim.n_elem == 0 || arma::any(dim == 0))
    throw Exception("qic::neg", Exception::type::INVALID_DIMS);

  if (arma::prod(dim) != rho.n_rows)
    throw Exception("qic::neg", Exception::type::DIMS_MISMATCH_MATRIX);

  if (dim.n_elem < subsys.n_elem || arma::any(subsys == 0) ||
      arma::any(subsys > dim.n_elem) ||
      subsys.n_elem != arma::unique(subsys).eval().n_elem)
    throw Exception("qic::neg", Exception::type::INVALID_SUBSYS);

#endif

  auto rho_T = Tx(rho, std::move(subsys), std::move(dim));
  auto eigval = arma::eig_sym(rho_T);
  trait::pT<T1> Neg = 0.0;

  for (const auto& i : eigval)
    Neg += (std::abs(i) >= _precision::eps<trait::pT<T1> >::value)
             ? 0.5 * (std::abs(i) - i)
             : 0;
  return Neg;
}

//******************************************************************************

template <typename T1,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1> >::value, trait::pT<T1> >::type>

inline TR log_neg(const T1& rho1, arma::uvec subsys, arma::uvec dim) {
  return std::log2(2.0 * neg(rho1, std::move(subsys), std::move(dim)) + 1.0);
}

//******************************************************************************

template <typename T1,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1> >::value, trait::pT<T1> >::type>

inline TR neg(const T1& rho1, arma::uvec subsys, arma::uword dim = 2) {
  const auto& rho = _internal::as_Mat(rho1);

#ifndef QICLIB_NO_DEBUG
  bool checkV = (rho.n_cols != 1);

  if (rho.n_elem == 0)
    throw Exception("qic::neg", Exception::type::ZERO_SIZE);

  if (checkV)
    if (rho.n_rows != rho.n_cols)
      throw Exception("qic::neg",
                      Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);

  if (dim == 0)
    throw Exception("qic::neg", Exception::type::INVALID_DIMS);
#endif

  arma::uword n = static_cast<arma::uword>(
    QICLIB_ROUND_OFF(std::log(rho.n_rows) / std::log(dim)));

  arma::uvec dim2(n);
  dim2.fill(dim);
  return neg(rho, std::move(subsys), std::move(dim2));
}

//******************************************************************************

template <typename T1,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1> >::value, trait::pT<T1> >::type>

inline TR log_neg(const T1& rho1, arma::uvec subsys, arma::uword dim = 2) {
  return std::log2(2.0 * neg(rho1, std::move(subsys), dim) + 1.0);
}

//******************************************************************************

}  // namespace qic

#endif
