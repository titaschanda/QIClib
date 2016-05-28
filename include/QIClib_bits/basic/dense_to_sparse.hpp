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

//******************************************************************************

template <typename T1,
          typename TR = typename std::enable_if<
            is_arma_type_var<T1>::value, arma::SpMat<trait::eT<T1> > >::type>
inline TR
dense_to_sparse(const T1& rho1,
                trait::pT<T1> tol = _precision::eps<trait::pT<T1> >::value) {
  const auto& rho = as_Mat(rho1);
  const arma::uword N = rho.n_elem;
  arma::SpMat<trait::eT<T1> > ret(rho.n_rows, rho.n_cols);
  arma::uword ii, jj;

  for (ii = 0, jj = 1; jj < N; ii += 2, jj += 2) {
    if (std::abs(rho[ii]) > tol) {
      arma::uword i = ii % rho.n_rows;
      arma::uword j = ii / rho.n_rows;
      ret.at(i, j) = rho[ii];
    }

    if (std::abs(rho[jj]) > tol) {
      arma::uword i = jj % rho.n_rows;
      arma::uword j = jj / rho.n_rows;
      ret.at(i, j) = rho[jj];
    }
  }
  if (ii < N) {
    if (std::abs(rho[ii]) > tol) {
      arma::uword i = ii % rho.n_rows;
      arma::uword j = ii / rho.n_rows;
      ret.at(i, j) = rho[ii];
    }
  }

  return ret;
}

//******************************************************************************

}  // namespace qic
