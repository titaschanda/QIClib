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

#ifndef _QICLIB_APPLY_HPP_
#define _QICLIB_APPLY_HPP_

#include "../basic/type_traits.hpp"
#include "../class/exception.hpp"
#include "../internal/as_arma.hpp"
#include <armadillo>

namespace qic {

//******************************************************************************

template <typename T1, typename T2,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1>, trait::pT<T2> >::value &&
              is_same_pT_var<T1, T2>::value,
            arma::Mat<typename eT_promoter_var<T1, T2>::type> >::type>

inline TR apply(const T1& rho1, const T2& A, arma::uvec subsys,
                arma::uvec dim) {
  const auto& rho = _internal::as_Mat(rho1);
  const auto& A1 = _internal::as_Mat(A);

#ifndef QICLIB_NO_DEBUG
  const bool checkV = (rho.n_cols != 1);

  if (rho.n_elem == 0)
    throw Exception("qic::apply", Exception::type::ZERO_SIZE);

  if (A1.n_elem == 0)
    throw Exception("qic::apply", Exception::type::ZERO_SIZE);

  if (checkV)
    if (rho.n_rows != rho.n_cols)
      throw Exception("qic::apply",
                      Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);

  if (A1.n_rows != A1.n_cols)
    throw Exception("qic::apply", Exception::type::MATRIX_NOT_SQUARE);

  if (dim.n_elem == 0 || arma::any(dim == 0))
    throw Exception("qic::apply", Exception::type::INVALID_DIMS);

  if (arma::prod(dim) != rho.n_rows)
    throw Exception("qic::apply", Exception::type::DIMS_MISMATCH_MATRIX);

  if (arma::prod(dim(subsys - 1)) != A1.n_rows)
    throw Exception("qic::apply", Exception::type::DIMS_MISMATCH_MATRIX);

  if (subsys.n_elem > dim.n_elem ||
      arma::unique(subsys).eval().n_elem != subsys.n_elem ||
      arma::any(subsys > dim.n_elem) || arma::any(subsys == 0))
    throw Exception("qic::apply", Exception::type::INVALID_SUBSYS);
#endif

  return apply_ctrl(rho, A1, {}, std::move(subsys), std::move(dim));
}

//******************************************************************************

template <typename T1, typename T2,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1>, trait::pT<T2> >::value &&
              is_same_pT_var<T1, T2>::value,
            arma::Mat<typename eT_promoter_var<T1, T2>::type> >::type>

inline TR apply(const T1& rho1, const T2& A, arma::uvec subsys,
                arma::uword dim = 2) {
  const auto& rho = _internal::as_Mat(rho1);

#ifndef QICLIB_NO_DEBUG
  const bool checkV = (rho.n_cols != 1);

  if (rho.n_elem == 0)
    throw Exception("qic::apply", Exception::type::ZERO_SIZE);

  if (checkV)
    if (rho.n_rows != rho.n_cols)
      throw Exception("qic::apply",
                      Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);

  if (dim == 0)
    throw Exception("qic::apply", Exception::type::INVALID_DIMS);
#endif

  const arma::uword n = static_cast<arma::uword>(
    QICLIB_ROUND_OFF(std::log(rho.n_rows) / std::log(dim)));

  arma::uvec dim2(n);
  dim2.fill(dim);
  return apply(rho, A, std::move(subsys), std::move(dim2));
}

//******************************************************************************

template <typename T1, typename T2,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1>, trait::GPT<T2> >::value &&
              is_all_same<trait::pT<T1>, trait::GPT<T2> >::value,
            arma::Mat<typename promote_var<trait::eT<T1>, T2>::type> >::type>

inline TR apply(const T1& rho1, const std::vector<arma::Mat<T2> >& Ks) {
  const auto& rho = _internal::as_Mat(rho1);
  const bool checkV = (rho.n_cols != 1);

#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::apply", Exception::type::ZERO_SIZE);

  if (Ks.size() == 0)
    throw Exception("qic::apply", Exception::type::ZERO_SIZE);

  if (checkV)
    if (rho.n_rows != rho.n_cols)
      throw Exception("qic::apply",
                      Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);

  for (const auto& k : Ks)
    if (k.n_rows != k.n_cols)
      throw Exception("qic::apply", Exception::type::MATRIX_NOT_SQUARE);

  for (const auto& k : Ks)
    if ((k.n_rows != Ks[0].n_rows) || (k.n_cols != Ks[0].n_cols))
      throw Exception("qic::apply", Exception::type::DIMS_NOT_EQUAL);

  if (Ks[0].n_rows != rho.n_rows)
    throw Exception("qic::apply", Exception::type::DIMS_MISMATCH_MATRIX);
#endif

  using mattype = arma::Mat<typename promote_var<trait::eT<T1>, T2>::type>;
  mattype ret(rho.n_rows, rho.n_rows, arma::fill::zeros);

#if (defined(QICLIB_USE_OPENMP) || defined(QICLIB_USE_OPENMP_APPLY)) &&        \
  defined(_OPENMP)
#pragma omp parallel for
#endif
  for (arma::uword i = 0; i < Ks.size(); ++i) {
#if (defined(QICLIB_USE_OPENMP) || defined(QICLIB_USE_OPENMP_APPLY)) &&        \
  defined(_OPENMP)
#pragma omp critical
#endif
    {
      ret +=
        checkV ? Ks[i] * rho * Ks[i].t() : Ks[i] * rho * rho.t() * Ks[i].t();
    }
  }
  return ret;
}

//******************************************************************************

template <typename T1, typename T2,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1>, trait::GPT<T2> >::value &&
              is_all_same<trait::pT<T1>, trait::GPT<T2> >::value,
            arma::Mat<typename promote_var<trait::eT<T1>, T2>::type> >::type>

inline TR apply(const T1& rho1, const arma::field<arma::Mat<T2> >& Ks) {
  const auto& rho = _internal::as_Mat(rho1);
  const bool checkV = (rho.n_cols != 1);

#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::apply", Exception::type::ZERO_SIZE);

  if (Ks.n_elem == 0)
    throw Exception("qic::apply", Exception::type::ZERO_SIZE);

  if (checkV)
    if (rho.n_rows != rho.n_cols)
      throw Exception("qic::apply",
                      Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);

  for (const auto& k : Ks)
    if (k.n_rows != k.n_cols)
      throw Exception("qic::apply", Exception::type::MATRIX_NOT_SQUARE);

  for (const auto& k : Ks)
    if ((k.n_rows != Ks.at(0).n_rows) || (k.n_cols != Ks.at(0).n_cols))
      throw Exception("qic::apply", Exception::type::DIMS_NOT_EQUAL);

  if (Ks.at(0).n_rows != rho.n_rows)
    throw Exception("qic::apply", Exception::type::DIMS_MISMATCH_MATRIX);
#endif

  using mattype = arma::Mat<typename promote_var<trait::eT<T1>, T2>::type>;
  mattype ret(rho.n_rows, rho.n_rows, arma::fill::zeros);

#if (defined(QICLIB_USE_OPENMP) || defined(QICLIB_USE_OPENMP_APPLY)) &&        \
  defined(_OPENMP)
#pragma omp parallel for
#endif
  for (arma::uword i = 0; i < Ks.n_elem; ++i) {
#if (defined(QICLIB_USE_OPENMP) || defined(QICLIB_USE_OPENMP_APPLY)) &&        \
  defined(_OPENMP)
#pragma omp critical
#endif
    {
      ret += checkV ? Ks.at(i) * rho * Ks.at(i).t()
                    : Ks.at(i) * rho * rho.t() * Ks.at(i).t();
    }
  }
  return ret;
}

//******************************************************************************

template <typename T1, typename T2,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1>, trait::GPT<T2> >::value &&
              is_all_same<trait::pT<T1>, trait::GPT<T2> >::value,
            arma::Mat<typename promote_var<trait::eT<T1>, T2>::type> >::type>

inline TR apply(const T1& rho1,
                const std::initializer_list<arma::Mat<T2> >& Ks) {
  const auto& rho = _internal::as_Mat(rho1);
  return apply(rho, static_cast<std::vector<arma::Mat<T2> > >(Ks));
}

//******************************************************************************

template <typename T1, typename T2,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1>, trait::GPT<T2> >::value &&
              is_all_same<trait::pT<T1>, trait::GPT<T2> >::value,
            arma::Mat<typename promote_var<trait::eT<T1>, T2>::type> >::type>

inline TR apply(const T1& rho1, const std::vector<arma::Mat<T2> >& Ks,
                arma::uvec subsys, arma::uvec dim) {
  const auto& rho = _internal::as_Mat(rho1);
  const bool checkV = (rho.n_cols != 1);

#ifndef QICLIB_NO_DEBUG
  const arma::uword D = arma::prod(dim);
  const arma::uword Dsys = arma::prod(dim(subsys - 1));

  if (rho.n_elem == 0)
    throw Exception("qic::apply", Exception::type::ZERO_SIZE);

  if (Ks.size() == 0)
    throw Exception("qic::apply", Exception::type::ZERO_SIZE);

  if (checkV)
    if (rho.n_rows != rho.n_cols)
      throw Exception("qic::apply",
                      Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);

  for (const auto& k : Ks)
    if (k.n_rows != k.n_cols)
      throw Exception("qic::apply", Exception::type::MATRIX_NOT_SQUARE);

  for (const auto& k : Ks)
    if ((k.n_rows != Ks[0].n_rows) || (k.n_cols != Ks[0].n_cols))
      throw Exception("qic::apply", Exception::type::DIMS_NOT_EQUAL);

  if (dim.n_elem == 0 || arma::any(dim == 0))
    throw Exception("qic::apply", Exception::type::INVALID_DIMS);

  if (D != rho.n_rows)
    throw Exception("qic::apply", Exception::type::DIMS_MISMATCH_MATRIX);

  if (Dsys != Ks[0].n_rows)
    throw Exception("qic::apply", Exception::type::DIMS_MISMATCH_MATRIX);

  if (subsys.n_elem > dim.n_elem ||
      arma::unique(subsys).eval().n_elem != subsys.n_elem ||
      arma::any(subsys > dim.n_elem) || arma::any(subsys == 0))
    throw Exception("qic::apply", Exception::type::INVALID_SUBSYS);
#endif

  using mattype = arma::Mat<typename promote_var<trait::eT<T1>, T2>::type>;
  mattype ret(rho.n_rows, rho.n_rows, arma::fill::zeros);

#if (defined(QICLIB_USE_OPENMP) || defined(QICLIB_USE_OPENMP_APPLY)) &&        \
  defined(_OPENMP)
#pragma omp parallel for
#endif
  for (arma::uword i = 0; i < Ks.size(); ++i) {
    auto tmp = apply(rho, Ks[i], subsys, dim);
#if (defined(QICLIB_USE_OPENMP) || defined(QICLIB_USE_OPENMP_APPLY)) &&        \
  defined(_OPENMP)
#pragma omp critical
#endif
    { ret += checkV ? tmp : tmp * tmp.t(); }
  }
  return ret;
}

//******************************************************************************

template <typename T1, typename T2,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1>, trait::GPT<T2> >::value &&
              is_all_same<trait::pT<T1>, trait::GPT<T2> >::value,
            arma::Mat<typename promote_var<trait::eT<T1>, T2>::type> >::type>

inline TR apply(const T1& rho1, const arma::field<arma::Mat<T2> >& Ks,
                arma::uvec subsys, arma::uvec dim) {
  const auto& rho = _internal::as_Mat(rho1);
  const bool checkV = (rho.n_cols != 1);

#ifndef QICLIB_NO_DEBUG
  const arma::uword D = arma::prod(dim);
  const arma::uword Dsys = arma::prod(dim(subsys - 1));

  if (rho.n_elem == 0)
    throw Exception("qic::apply", Exception::type::ZERO_SIZE);

  if (Ks.n_elem == 0)
    throw Exception("qic::apply", Exception::type::ZERO_SIZE);

  if (checkV)
    if (rho.n_rows != rho.n_cols)
      throw Exception("qic::apply",
                      Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);

  for (const auto& k : Ks)
    if (k.n_rows != k.n_cols)
      throw Exception("qic::apply", Exception::type::MATRIX_NOT_SQUARE);

  for (const auto& k : Ks)
    if ((k.n_rows != Ks.at(0).n_rows) || (k.n_cols != Ks.at(0).n_cols))
      throw Exception("qic::apply", Exception::type::DIMS_NOT_EQUAL);

  if (dim.n_elem == 0 || arma::any(dim == 0))
    throw Exception("qic::apply", Exception::type::INVALID_DIMS);

  if (D != rho.n_rows)
    throw Exception("qic::apply", Exception::type::DIMS_MISMATCH_MATRIX);

  if (Dsys != Ks.at(0).n_rows)
    throw Exception("qic::apply", Exception::type::DIMS_MISMATCH_MATRIX);

  if (subsys.n_elem > dim.n_elem ||
      arma::unique(subsys).eval().n_elem != subsys.n_elem ||
      arma::any(subsys > dim.n_elem) || arma::any(subsys == 0))
    throw Exception("qic::apply", Exception::type::INVALID_SUBSYS);
#endif

  using mattype = arma::Mat<typename promote_var<trait::eT<T1>, T2>::type>;
  mattype ret(rho.n_rows, rho.n_rows, arma::fill::zeros);

#if (defined(QICLIB_USE_OPENMP) || defined(QICLIB_USE_OPENMP_APPLY)) &&        \
  defined(_OPENMP)
#pragma omp parallel for
#endif
  for (arma::uword i = 0; i < Ks.n_elem; ++i) {
    auto tmp = apply(rho, Ks.at(i), subsys, dim);
#if (defined(QICLIB_USE_OPENMP) || defined(QICLIB_USE_OPENMP_APPLY)) &&        \
  defined(_OPENMP)
#pragma omp critical
#endif
    { ret += checkV ? tmp : tmp * tmp.t(); }
  }
  return ret;
}

//******************************************************************************

template <typename T1, typename T2,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1>, trait::GPT<T2> >::value &&
              is_all_same<trait::pT<T1>, trait::GPT<T2> >::value,
            arma::Mat<typename promote_var<trait::eT<T1>, T2>::type> >::type>

inline TR apply(const T1& rho1, const std::initializer_list<arma::Mat<T2> >& Ks,
                arma::uvec subsys, arma::uvec dim) {
  return apply(rho1, static_cast<std::vector<arma::Mat<T2> > >(Ks),
               std::move(subsys), std::move(dim));
}

//******************************************************************************

template <typename T1, typename T2,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1>, trait::GPT<T2> >::value &&
              is_all_same<trait::pT<T1>, trait::GPT<T2> >::value,
            arma::Mat<typename promote_var<trait::eT<T1>, T2>::type> >::type>

inline TR apply(const T1& rho1, const std::vector<arma::Mat<T2> >& Ks,
                arma::uvec subsys, arma::uword dim = 2) {
  const auto& rho = _internal::as_Mat(rho1);

#ifndef QICLIB_NO_DEBUG
  const bool checkV = (rho.n_cols != 1);

  if (rho.n_elem == 0)
    throw Exception("qic::apply", Exception::type::ZERO_SIZE);

  if (checkV)
    if (rho.n_rows != rho.n_cols)
      throw Exception("qic::apply",
                      Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);

  if (dim == 0)
    throw Exception("qic::apply", Exception::type::INVALID_DIMS);
#endif

  const arma::uword n = static_cast<arma::uword>(
    QICLIB_ROUND_OFF(std::log(rho.n_rows) / std::log(dim)));

  arma::uvec dim2(n);
  dim2.fill(dim);

  return apply(rho, Ks, std::move(subsys), std::move(dim2));
}

//******************************************************************************

template <typename T1, typename T2,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1>, trait::GPT<T2> >::value &&
              is_all_same<trait::pT<T1>, trait::GPT<T2> >::value,
            arma::Mat<typename promote_var<trait::eT<T1>, T2>::type> >::type>

inline TR apply(const T1& rho1, const arma::field<arma::Mat<T2> >& Ks,
                arma::uvec subsys, arma::uword dim = 2) {
  const auto& rho = _internal::as_Mat(rho1);

#ifndef QICLIB_NO_DEBUG
  const bool checkV = (rho.n_cols != 1);

  if (rho.n_elem == 0)
    throw Exception("qic::apply", Exception::type::ZERO_SIZE);

  if (checkV)
    if (rho.n_rows != rho.n_cols)
      throw Exception("qic::apply",
                      Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);

  if (dim == 0)
    throw Exception("qic::apply", Exception::type::INVALID_DIMS);
#endif

  const arma::uword n = static_cast<arma::uword>(
    QICLIB_ROUND_OFF(std::log(rho.n_rows) / std::log(dim)));

  arma::uvec dim2(n);
  dim2.fill(dim);

  return apply(rho, Ks, std::move(subsys), std::move(dim2));
}

//******************************************************************************

template <typename T1, typename T2,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1>, trait::GPT<T2> >::value &&
              is_all_same<trait::pT<T1>, trait::GPT<T2> >::value,
            arma::Mat<typename promote_var<trait::eT<T1>, T2>::type> >::type>

inline TR apply(const T1& rho1, const std::initializer_list<arma::Mat<T2> >& Ks,
                arma::uvec subsys, arma::uword dim = 2) {
  const auto& rho = _internal::as_Mat(rho1);
  return apply(rho, static_cast<std::vector<arma::Mat<T2> > >(Ks),
               std::move(subsys), std::move(dim));
}

//******************************************************************************

}  // namespace qic

#endif
