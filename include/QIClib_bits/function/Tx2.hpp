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
inline TR Tx2(const T1& rho1, arma::uvec sys, arma::uvec dim) {
  auto p = as_Mat(rho1);

  bool checkV = true;
  if (p.n_cols == 1)
    checkV = false;

#ifndef QICLIB_NO_DEBUG
  if (p.n_elem == 0)
    throw Exception("qic::Tx", Exception::type::ZERO_SIZE);

  if (checkV)
    if (p.n_rows != p.n_cols)
      throw Exception("qic::Tx", Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);

  if (dim.n_elem == 0 || arma::any(dim == 0))
    throw Exception("qic::Tx", Exception::type::INVALID_DIMS);

  if (arma::prod(dim) != p.n_rows)
    throw Exception("qic::Tx", Exception::type::DIMS_MISMATCH_MATRIX);

  if (dim.n_elem < sys.n_elem || arma::any(sys == 0) ||
      arma::any(sys > dim.n_elem) ||
      sys.n_elem != arma::find_unique(sys, false).eval().n_elem)
    throw Exception("qic::Tx", Exception::type::INVALID_SUBSYS);
#endif

  if (!checkV)
    p *= p.t();

  if (sys.n_elem == dim.n_elem)
    return p.st();

  if (sys.n_elem == 0)
    return p;
  
  _internal::dim_collapse_sys(dim, sys);
  const arma::uword n = dim.n_elem;

  arma::uword product[_internal::MAXQDIT];
  product[n-1] = 1;
  for (arma::sword i = n - 2; i >= 0; --i)
    product[i] = product[i + 1] * dim.at(i + 1);

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

      if (arma::any(sys == i + 1)) {
        K += product[i] * loop_counter[i + n];
        L += product[i] * loop_counter[i];
      } else {
        K += product[i] * loop_counter[i];
        L += product[i] * loop_counter[i + n];
      }
    }

    if (I > K)
      std::swap(p.at(I, J), p.at(K, L));

    ++loop_counter[0];
    while (loop_counter[p1] == MAX[p1]) {
      loop_counter[p1] = 0;
      loop_counter[++p1]++;
      if (loop_counter[p1] != MAX[p1])
        p1 = 0;
    }
  }
  return p;
}

//******************************************************************************

template <typename T1,
          typename TR = typename std::enable_if<
            is_arma_type_var<T1>::value, arma::Mat<trait::eT<T1> > >::type>
inline TR Tx2(const T1& rho1, arma::uvec sys, arma::uword dim = 2) {
  const auto& p = as_Mat(rho1);

#ifndef QICLIB_NO_DEBUG
  bool checkV = true;
  if (p.n_cols == 1)
    checkV = false;

  if (p.n_elem == 0)
    throw Exception("qic::Tx", Exception::type::ZERO_SIZE);

  if (checkV)
    if (p.n_rows != p.n_cols)
      throw Exception("qic::Tx", Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);

  if (dim == 0)
    throw Exception("qic::Tx", Exception::type::INVALID_DIMS);
#endif

  arma::uword n =
    static_cast<arma::uword>(std::llround(std::log(p.n_rows) / std::log(dim)));

  arma::uvec dim2(n);
  dim2.fill(dim);
  return Tx(p, std::move(sys), std::move(dim2));
}

//******************************************************************************

}  // namespace qic
