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

#ifndef _QICLIB_IS_UNITARY_HPP_
#define _QICLIB_IS_UNITARY_HPP_

#include "../class/constants.hpp"
#include "../internal/as_arma.hpp"
#include "is_equal.hpp"
#include "type_traits.hpp"
#include <armadillo>

namespace qic {

template <typename T1,
          typename = typename std::enable_if<
            is_floating_point_var<trait::pT<T1> >::value, void>::type>
inline bool
is_Unitary(const T1& rho1,
           const trait::pT<T1>& atol = _precision::eps<trait::pT<T1> >::value,
           const trait::pT<T1>& rtol = 10 *
                                       _precision::eps<trait::pT<T1> >::value) {
  const auto& rho = _internal::as_Mat(rho1);

  const arma::uword n = rho.n_rows;
  const arma::uword m = rho.n_cols;

  if (n != m) {
    return false;

  } else {
    arma::Mat<trait::eT<T1> > eye1 = rho * rho.t();
    arma::Mat<trait::eT<T1> > eye2 = rho.t() * rho;
    arma::Mat<trait::eT<T1> > eye3 =
      arma::eye<arma::Mat<trait::eT<T1> > >(n, m);

    return is_equal(eye1, eye3, false, atol, rtol) &&
           is_equal(eye2, eye3, false, atol, rtol);
  }
}

}  // namespace qic

#endif
