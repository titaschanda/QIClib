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
            is_arma_type_var<T1>::value, arma::Mat<trait::eT<T1> > >::type>
inline TR sysperm(const T1& rho1, const arma::uvec& sys,
                   const arma::uvec& dim) {
  const auto& rho = as_Mat(rho1);
  const arma::uword n = dim.n_elem;

  bool checkV = true;
  if (rho.n_cols == 1)
    checkV = false;

#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::sysperm", Exception::type::ZERO_SIZE);

  if (checkV)
    if (rho.n_rows != rho.n_cols)
      throw Exception("qic::sysperm",
                      Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);

  if (dim.n_elem == 0 || arma::any(dim == 0))
    throw Exception("qic::sysperm", Exception::type::INVALID_DIMS);

  if (arma::prod(dim) != rho.n_rows)
    throw Exception("qic::sysperm", Exception::type::DIMS_MISMATCH_MATRIX);

  if (n != sys.n_elem || arma::any(sys == 0) || arma::any(sys > n) ||
      sys.n_elem != arma::unique(sys).eval().n_elem)
    throw Exception("qic::sysperm", Exception::type::PERM_INVALID);
#endif

  arma::uword product[_internal::MAXQDIT];
  product[n-1] = 1;
  for (arma::sword i = n - 2; i >= 0; --i)
    product[i] = product[i + 1] * dim.at(i + 1);

  arma::uword productr[_internal::MAXQDIT];
  productr[n-1] = 1;
  for (arma::sword i = n - 2; i >= 0; --i)
    productr[i] = productr[i + 1] * dim.at(sys.at(i + 1) - 1);

  if (checkV) {
    arma::Mat<trait::eT<T1> > rho_ret(rho.n_rows, rho.n_cols, arma::fill::zeros);

    const arma::uword loop_no = 2 * n;
    constexpr auto loop_no_buffer = 2 * _internal::MAXQDIT + 1;
    arma::uword loop_counter[loop_no_buffer] = {0};
    arma::uword MAX[loop_no_buffer];

    for (arma::uword i = 0; i < n; ++i) {
      MAX[i] = dim.at(i);
      MAX[i + n] = dim.at(i);
    }
    MAX[loop_no] = 2;

    arma::uword p1 = 0;

    while (loop_counter[loop_no] == 0) {
      arma::uword I(0), J(0), K(0), L(0);
      for (arma::uword i = 0; i < n; ++i) {
        I += product[i] * loop_counter[i];
        J += product[i] * loop_counter[i + n];
        K += productr[i] * loop_counter[sys.at(i) - 1];
        L += productr[i] * loop_counter[sys.at(i) + n - 1];
      }

      rho_ret.at(K, L) = rho.at(I, J);

      ++loop_counter[0];
      while (loop_counter[p1] == MAX[p1]) {
        loop_counter[p1] = 0;
        loop_counter[++p1]++;
        if (loop_counter[p1] != MAX[p1])
          p1 = 0;
      }
    }
    return rho_ret;

  } else {
    arma::Col<trait::eT<T1> > rho_ret(rho.n_rows, arma::fill::zeros);

    const arma::uword loop_no = n;
    constexpr auto loop_no_buffer = _internal::MAXQDIT + 1;
    arma::uword loop_counter[loop_no_buffer] = {0};
    arma::uword MAX[loop_no_buffer];

    for (arma::uword i = 0; i < n; ++i) MAX[i] = dim.at(i);
    MAX[loop_no] = 2;

    for (arma::uword i = 0; i < loop_no + 1; ++i) loop_counter[i] = 0;

    arma::uword p1 = 0;

    while (loop_counter[loop_no] == 0) {
      arma::uword I(0), K(0);
      for (arma::uword i = 0; i < n; ++i) {
        I += product[i] * loop_counter[i];
        K += productr[i] * loop_counter[sys.at(i) - 1];
      }

      rho_ret.at(K) = rho.at(I);

      ++loop_counter[0];
      while (loop_counter[p1] == MAX[p1]) {
        loop_counter[p1] = 0;
        loop_counter[++p1]++;
        if (loop_counter[p1] != MAX[p1])
          p1 = 0;
      }
    }

    return rho_ret;
  }
}

//******************************************************************************

template <typename T1,
          typename TR = typename std::enable_if<
            is_arma_type_var<T1>::value, arma::Mat<trait::eT<T1> > >::type>
inline TR sysperm(const T1& rho1, const arma::uvec& sys, arma::uword dim = 2) {
  const auto& rho = as_Mat(rho1);

#ifndef QICLIB_NO_DEBUG
  bool checkV = true;
  if (rho.n_cols == 1)
    checkV = false;

  if (rho.n_elem == 0)
    throw Exception("qic::sysperm", Exception::type::ZERO_SIZE);

  if (checkV)
    if (rho.n_rows != rho.n_cols)
      throw Exception("qic::sysperm",
                      Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);

  if (dim == 0)
    throw Exception("qic::sysperm", Exception::type::INVALID_DIMS);
#endif

  arma::uword n =
    static_cast<arma::uword>(std::llround(std::log(rho.n_rows) / std::log(dim)));

  arma::uvec dim2(n);
  dim2.fill(dim);
  return sysperm(rho, sys, dim2);
}

//******************************************************************************

}  //  namespace qic