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

namespace qic {

template <typename T1, typename T2,
          typename = typename std::enable_if<is_arma_type_var<T1, T2>::value &&
                                               is_same_pT_var<T1, T2>::value,
                                             void>::type>
inline bool
is_equal(const T1& rho11, const T2& rho12, bool typecheck = false,
         const trait::pT<T1>& atol = _precision::eps<trait::pT<T1> >::value,
         const trait::pT<T1>& rtol = 10 *
                                     _precision::eps<trait::pT<T1> >::value) {
  const auto& rho1 = _internal::as_Mat(rho11);
  const auto& rho2 = _internal::as_Mat(rho12);

  const arma::uword n1 = rho1.n_rows;
  const arma::uword m1 = rho1.n_cols;
  const arma::uword n2 = rho2.n_rows;
  const arma::uword m2 = rho2.n_cols;

  if (n1 != n2 || m1 != m2 ||
      (typecheck && !std::is_same<trait::eT<T1>, trait::eT<T2> >::value)) {
    return false;
  } else {
    const arma::uword N = rho1.n_elem;
    arma::uword ii, jj;

    for (ii = 0, jj = 1; jj < N; ii += 2, jj += 2) {
      if (std::abs(rho1[ii] - rho2[ii]) >
          atol + rtol * std::max(std::abs(rho1[ii]), std::abs(rho2[ii])))
        return false;

      if (std::abs(rho1[jj] - rho2[jj]) >
          atol + rtol * std::max(std::abs(rho1[jj]), std::abs(rho2[jj])))
        return false;
    }
    if (ii < N) {
      if (std::abs(rho1[ii] - rho2[ii]) >
          atol + rtol * std::max(std::abs(rho1[ii]), std::abs(rho2[ii])))
        return false;
    }

    return true;
  }
}

}  // namespace qic
