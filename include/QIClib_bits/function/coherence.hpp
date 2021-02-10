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

#ifndef _QICLIB_COHERENCE_HPP_
#define _QICLIB_COHERENCE_HPP_

#include "../basic/type_traits.hpp"
#include "../class/exception.hpp"
#include "../internal/as_arma.hpp"
#include "../internal/conj2.hpp"
#include <armadillo>

namespace qic {

//******************************************************************************

template <typename T1,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1> >::value, trait::pT<T1> >::type>

inline TR l1_coh(const T1& rho1) {
  const auto& rho = _internal::as_Mat(rho1);
  const bool checkV = (rho.n_cols != 1);

#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::l1_coh", Exception::type::ZERO_SIZE);

  if (checkV)
    if (rho.n_rows != rho.n_cols)
      throw Exception("qic::l1_coh",
                      Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);
#endif

  trait::pT<T1> ret(0.0);

  if (checkV) {
    for (arma::uword ii = 0; ii < rho.n_cols; ++ii) {
      for (arma::uword jj = 0; jj < rho.n_rows; ++jj) {
        ret += ii != jj ? std::abs(rho.at(jj, ii)) : 0.0;
      }
    }

  } else {

    for (arma::uword ii = 0; ii < rho.n_rows; ++ii) {
      for (arma::uword jj = 0; jj < rho.n_rows; ++jj) {
        ret +=
          ii != jj ? std::abs(rho.at(ii) * _internal::conj2(rho.at(jj))) : 0.0;
      }
    }
  }

  return ret;
}

//******************************************************************************

template <typename T1, typename T2,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1>, trait::pT<T2> >::value ||
              is_same_pT_var<T1, T2>::value,
            trait::pT<T1> >::type>

inline TR l1_coh(const T1& rho1, const T2& U1) {
  const auto& rho = _internal::as_Mat(rho1);
  const auto& U = _internal::as_Mat(U1);
  const bool checkV = (rho.n_cols != 1);

#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0 || U.n_elem == 0)
    throw Exception("qic::l1_coh", Exception::type::ZERO_SIZE);

  if (checkV)
    if (rho.n_rows != rho.n_cols)
      throw Exception("qic::l1_coh",
                      Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);

  if (U.n_rows != U.n_cols)
    throw Exception("qic::l1_coh", Exception::type::MATRIX_NOT_SQUARE);

  if (rho.n_rows != U.n_rows)
    throw Exception("qic::l1_coh", Exception::type::MATRIX_SIZE_MISMATCH);
#endif

  if (checkV)
    return l1_coh((U * rho * U.t()).eval());
  else
    return l1_coh((U * rho).eval());
}

//******************************************************************************

template <typename T1,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1> >::value, trait::pT<T1> >::type>

inline TR rel_entropy_coh(const T1& rho1) {
  const auto& rho = _internal::as_Mat(rho1);
  const bool checkV = (rho.n_cols != 1);

#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::rel_entropy_coh", Exception::type::ZERO_SIZE);

  if (checkV)
    if (rho.n_rows != rho.n_cols)
      throw Exception("qic::rel_entropy_coh",
                      Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);
#endif

  if (checkV)
    return shannon(arma::real(arma::diagvec(rho))) - entropy(rho);

  else
    return shannon(arma::pow(arma::abs(rho), 2.0));
}

//******************************************************************************

template <typename T1, typename T2,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1>, trait::pT<T2> >::value ||
              is_same_pT_var<T1, T2>::value,
            trait::pT<T1> >::type>

inline TR rel_entropy_coh(const T1& rho1, const T2& U1) {
  const auto& rho = _internal::as_Mat(rho1);
  const auto& U = _internal::as_Mat(U1);
  const bool checkV = (rho.n_cols != 1);

#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0 || U.n_elem == 0)
    throw Exception("qic::rel_entropy_coh", Exception::type::ZERO_SIZE);

  if (checkV)
    if (rho.n_rows != rho.n_cols)
      throw Exception("qic::rel_entropy_coh",
                      Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);

  if (U.n_rows != U.n_cols)
    throw Exception("qic::rel_entropy_coh", Exception::type::MATRIX_NOT_SQUARE);

  if (rho.n_rows != U.n_rows)
    throw Exception("qic::rel_entropy_coh",
                    Exception::type::MATRIX_SIZE_MISMATCH);
#endif

  if (checkV)
    return rel_entropy_coh((U * rho * U.t()).eval());
  else
    return rel_entropy_coh((U * rho).eval());
}

//******************************************************************************

}  // namespace qic

#endif
