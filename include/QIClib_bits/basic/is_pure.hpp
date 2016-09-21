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

template <typename T1,
          typename = typename std::enable_if<
            is_floating_point_var<trait::pT<T1> >::value, void>::type>
inline bool
is_pure(const T1& rho1, bool check_norm = true,
        const trait::pT<T1>& tol = _precision::eps<trait::pT<T1> >::value) {
  const auto& rho = _internal::as_Mat(rho1);

  if ((rho.n_rows == 1) || (rho.n_cols == 1)) {
    if (std::abs(arma::norm(rho) - 1) < tol || !check_norm)
      return true;
    else
      return false;
  } else if (!is_Hermitian(rho, tol, 10 * tol)) {
    return false;
  } else if (!is_equal(rho, rho * rho, false, tol, 10 * tol)) {
    return false;
  } else {
    if (std::abs(std::abs(arma::trace(rho)) - 1.0) < tol || !check_norm)
      return true;
    else
      return false;
  }
}

}  // namespace qic
