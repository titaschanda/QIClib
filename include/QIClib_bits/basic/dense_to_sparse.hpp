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
  const auto& rho = _internal::as_Mat(rho1);
  const arma::uword N = rho.n_elem;
  arma::umat Index(2, N);
  arma::Col<trait::eT<T1> > value(N);

  arma::uword ii, jj, count(0);

  for (ii = 0, jj = 1; jj < N; ii += 2, jj += 2) {
    if (std::abs(rho[ii]) > tol) {
      arma::uword i = ii % rho.n_rows;
      arma::uword j = ii / rho.n_rows;
      Index.at(0, count) = i;
      Index.at(1, count) = j;
      value.at(count) = rho[ii];
      ++count;
    }

    if (std::abs(rho[jj]) > tol) {
      arma::uword i = jj % rho.n_rows;
      arma::uword j = jj / rho.n_rows;
      Index.at(0, count) = i;
      Index.at(1, count) = j;
      value.at(count) = rho[jj];
      ++count;
    }
  }
  if (ii < N) {
    if (std::abs(rho[ii]) > tol) {
      arma::uword i = ii % rho.n_rows;
      arma::uword j = ii / rho.n_rows;
      Index.at(0, count) = i;
      Index.at(1, count) = j;
      value.at(count) = rho[ii];
      ++count;
    }
  }

  if (count == 0)
    return arma::SpMat<trait::eT<T1> >(rho.n_rows, rho.n_cols);
  else if (count == N)
    return arma::SpMat<trait::eT<T1> >(false, Index, value, rho.n_rows,
                                       rho.n_cols, false, false);

  else
    return arma::SpMat<trait::eT<T1> >(false, Index.cols(0, count - 1),
                                       value.rows(0, count - 1), rho.n_rows,
                                       rho.n_cols, false, false);
}

//******************************************************************************

}  // namespace qic
