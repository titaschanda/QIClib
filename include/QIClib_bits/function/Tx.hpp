/*
 * QIClib (Quantum information and computation library)
 *
 * Copyright (c) 2015 - 2017  Titas Chanda (titas.chanda@gmail.com)
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

#ifdef QICLIB_USE_SERIAL_TX
// USE SERIAL ALGORITHM

//******************************************************************************

template <typename T1,
          typename TR = typename std::enable_if<
            is_arma_type_var<T1>::value, arma::Mat<trait::eT<T1> > >::type>

inline TR Tx(const T1& rho1, arma::uvec subsys, arma::uvec dim,
             bool is_Hermitian = false) {
  auto rho = _internal::as_Mat(rho1);  // force copy
  const bool checkV = (rho.n_cols != 1);
  (void)is_Hermitian;  // do not need this in serial algorithm

#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::Tx", Exception::type::ZERO_SIZE);

  if (checkV)
    if (rho.n_rows != rho.n_cols)
      throw Exception("qic::Tx", Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);

  if (dim.n_elem == 0 || arma::any(dim == 0))
    throw Exception("qic::Tx", Exception::type::INVALID_DIMS);

  if (arma::prod(dim) != rho.n_rows)
    throw Exception("qic::Tx", Exception::type::DIMS_MISMATCH_MATRIX);

  if (dim.n_elem < subsys.n_elem || arma::any(subsys == 0) ||
      arma::any(subsys > dim.n_elem) ||
      subsys.n_elem != arma::unique(subsys).eval().n_elem)
    throw Exception("qic::Tx", Exception::type::INVALID_SUBSYS);
#endif

  if (!checkV)
    rho *= rho.t();

  if (subsys.n_elem == dim.n_elem)
    return rho.st();

  if (subsys.n_elem == 0)
    return rho;

  _internal::dim_collapse_sys(dim, subsys);
  const arma::uword n = dim.n_elem;

  arma::uword product[_internal::MAXQDIT];
  product[n - 1] = 1;
  for (arma::uword i = 1; i < n; ++i)
    product[n - 1 - i] = product[n - i] * dim.at(n - i);

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

      if (arma::any(subsys == i + 1)) {
        K += product[i] * loop_counter[i + n];
        L += product[i] * loop_counter[i];
      } else {
        K += product[i] * loop_counter[i];
        L += product[i] * loop_counter[i + n];
      }
    }

    if (I > K)
      std::swap(rho.at(I, J), rho.at(K, L));

    ++loop_counter[0];
    while (loop_counter[p1] == MAX[p1]) {
      loop_counter[p1] = 0;
      loop_counter[++p1]++;
      if (loop_counter[p1] != MAX[p1])
        p1 = 0;
    }
  }
  return rho;
}

//******************************************************************************

#else
// USE PARALLEL ALGORITHM

//******************************************************************************

template <typename T1,
          typename TR = typename std::enable_if<
            is_arma_type_var<T1>::value, arma::Mat<trait::eT<T1> > >::type>

inline TR Tx(const T1& rho1, arma::uvec subsys, arma::uvec dim,
             bool is_Hermitian = false) {
  const auto& rho = _internal::as_Mat(rho1);
  const bool checkV = (rho.n_cols != 1);

#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::Tx", Exception::type::ZERO_SIZE);

  if (checkV)
    if (rho.n_rows != rho.n_cols)
      throw Exception("qic::Tx", Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);

  if (dim.n_elem == 0 || arma::any(dim == 0))
    throw Exception("qic::Tx", Exception::type::INVALID_DIMS);

  if (arma::prod(dim) != rho.n_rows)
    throw Exception("qic::Tx", Exception::type::DIMS_MISMATCH_MATRIX);

  if (dim.n_elem < subsys.n_elem || arma::any(subsys == 0) ||
      arma::any(subsys > dim.n_elem) ||
      subsys.n_elem != arma::unique(subsys).eval().n_elem)
    throw Exception("qic::Tx", Exception::type::INVALID_SUBSYS);
#endif

  if (subsys.n_elem == dim.n_elem) {
    if (checkV)
      return rho.st();
    else
      return (rho * rho.t()).st();
  }

  if (subsys.n_elem == 0) {
    if (checkV)
      return rho;
    else
      return rho * rho.t();
  }

  _internal::dim_collapse_sys(dim, subsys);
  const arma::uword n = dim.n_elem;

  arma::uword product[_internal::MAXQDIT];
  product[n - 1] = 1;
  for (arma::uword i = 1; i < n; ++i)
    product[n - 1 - i] = product[n - i] * dim.at(n - i);

  auto worker = [n, checkV, &dim, &subsys, &product,
                 &rho](arma::uword I, arma::uword J) noexcept -> trait::eT<T1> {
    arma::uword K(0), L(0);

    for (arma::uword i = 1; i < n; ++i) {
      arma::uword Iindex = I % dim.at(n - i);
      arma::uword Jindex = J % dim.at(n - i);
      I /= dim.at(n - i);
      J /= dim.at(n - i);

      if (arma::any(subsys == n - i + 1)) {
        K += product[n - i] * Jindex;
        L += product[n - i] * Iindex;
      } else {
        K += product[n - i] * Iindex;
        L += product[n - i] * Jindex;
      }
    }

    if (arma::any(subsys == 1)) {
      K += product[0] * J;
      L += product[0] * I;
    } else {
      K += product[0] * I;
      L += product[0] * J;
    }

    if (checkV)
      return rho.at(K, L);
    else
      return rho.at(K) * _internal::conj2(rho.at(L));
  };

  arma::Mat<trait::eT<T1> > tr_rho(rho.n_rows, rho.n_rows);

  if (is_Hermitian) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (arma::uword JJ = 0; JJ < rho.n_rows; ++JJ) {
      for (arma::uword II = 0; II <= JJ; ++II)
        tr_rho.at(II, JJ) = worker(II, JJ);
    }

    
    for (arma::uword JJ = 0; JJ < rho.n_rows; ++JJ) {
      for (arma::uword II = JJ + 1; II < rho.n_rows; ++II)
        tr_rho.at(II, JJ) = _internal::conj2(tr_rho.at(JJ, II));
    }

  } else {

#if defined(_OPENMP)
#pragma omp parallel for collapse(2)
#endif
    for (arma::uword JJ = 0; JJ < rho.n_rows; ++JJ) {
      for (arma::uword II = 0; II < rho.n_rows; ++II)
        tr_rho.at(II, JJ) = worker(II, JJ);
    }
  }

  return tr_rho;
}

//******************************************************************************

#endif

//******************************************************************************

template <typename T1,
          typename TR = typename std::enable_if<
            is_arma_type_var<T1>::value, arma::Mat<trait::eT<T1> > >::type>

inline TR Tx(const T1& rho1, arma::uvec subsys, arma::uword dim = 2,
             bool is_Hermitian = false) {
  const auto& rho = _internal::as_Mat(rho1);

#ifndef QICLIB_NO_DEBUG
  bool checkV = (rho.n_cols != 1);

  if (rho.n_elem == 0)
    throw Exception("qic::Tx", Exception::type::ZERO_SIZE);

  if (checkV)
    if (rho.n_rows != rho.n_cols)
      throw Exception("qic::Tx", Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);

  if (dim == 0)
    throw Exception("qic::Tx", Exception::type::INVALID_DIMS);
#endif

  const arma::uword n = static_cast<arma::uword>(
    QICLIB_ROUND_OFF(std::log(rho.n_rows) / std::log(dim)));

  arma::uvec dim2(n);
  dim2.fill(dim);
  return Tx(rho, std::move(subsys), std::move(dim2), is_Hermitian);
}

//******************************************************************************

}  // namespace qic
