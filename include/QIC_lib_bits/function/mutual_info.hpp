/*
 * QIC_lib (Quantum information and computation library)
 *
 * Copyright (c) 2015 - 2016  Titas Chanda (titas.chanda@gmail.com)
 *
 * This file is part of QIC_lib.
 *
 * QIC_lib is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * QIC_lib is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QIC_lib.  If not, see <http://www.gnu.org/licenses/>.
 */

namespace qic {

//******************************************************************************

template <typename T1,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1> >::value, trait::pT<T1> >::type>
inline TR mutual_info(const T1& rho1, arma::uvec dim) {
  const auto& rho = as_Mat(rho1);

#ifndef QIC_LIB_NO_DEBUG
  bool checkV = true;
  if (rho.n_cols == 1)
    checkV = false;

  if (rho.n_elem == 0)
    throw Exception("qic::mutual_info", Exception::type::ZERO_SIZE);

  if (checkV)
    if (rho.n_rows != rho.n_cols)
      throw Exception("qic::mutual_info",
                      Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);

  if (arma::any(dim == 0))
    throw Exception("qic::mutual_info", Exception::type::INVALID_DIMS);

  if (arma::prod(dim) != rho.n_rows)
    throw Exception("qic::mutual_info", Exception::type::DIMS_MISMATCH_MATRIX);

  if (dim.n_elem != 2)
    throw Exception("qic::mutual_info", Exception::type::NOT_BIPARTITE);
#endif

  auto rho_A = TrX(rho, {2}, dim);
  auto rho_B = TrX(rho, {1}, std::move(dim));

  auto S_A = entropy(rho_A);
  auto S_B = entropy(rho_B);
  auto S_A_B = entropy(rho);

  return S_A + S_B - S_A_B;
}

//******************************************************************************

template <typename T1,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1> >::value, trait::pT<T1> >::type>
inline TR mutual_info(const T1& rho1, arma::uvec sys1, arma::uvec sys2,
                      arma::uvec dim) {
  const auto& rho = as_Mat(rho1);

  arma::uvec sys12 = arma::join_cols(sys1, sys2);

#ifndef QIC_LIB_NO_DEBUG
  bool checkV = true;
  if (rho.n_cols == 1)
    checkV = false;

  if (rho.n_elem == 0)
    throw Exception("qic::mutual_info", Exception::type::ZERO_SIZE);

  if (checkV)
    if (rho.n_rows != rho.n_cols)
      throw Exception("qic::mutual_info",
                      Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);

  if (arma::any(dim == 0))
    throw Exception("qic::mutual_info", Exception::type::INVALID_DIMS);

  if (arma::prod(dim) != rho.n_rows)
    throw Exception("qic::mutual_info", Exception::type::DIMS_MISMATCH_MATRIX);

  if (sys12.n_elem > dim.n_elem ||
      arma::find_unique(sys12).eval().n_elem != sys12.n_elem ||
      arma::any(sys12 > dim.n_elem) || arma::any(sys12 == 0))
    throw Exception("qic::mutual_info", Exception::type::INVALID_SUBSYS);

#endif

  const arma::uword n = dim.n_elem;

  arma::uvec sys1bar(n - sys1.n_elem);
  arma::uword count1(0);
  for (arma::uword run = 0; run < n; ++run) {
    if (!arma::any(sys1 == run + 1)) {
      sys1bar.at(count1) = run + 1;
      ++count1;
    }
  }

  arma::uvec sys2bar(n - sys2.n_elem);
  arma::uword count2(0);
  for (arma::uword run = 0; run < n; ++run) {
    if (!arma::any(sys2 == run + 1)) {
      sys2bar.at(count2) = run + 1;
      ++count2;
    }
  }

  arma::uvec sys12bar(n - sys12.n_elem);
  arma::uword count12(0);
  for (arma::uword run = 0; run < n; ++run) {
    if (!arma::any(sys12 == run + 1)) {
      sys12bar.at(count12) = run + 1;
      ++count12;
    }
  }

  auto rho_A = TrX(rho, std::move(sys1bar), dim);
  auto rho_B = TrX(rho, std::move(sys2bar), dim);
  auto rho_AB = TrX(rho, std::move(sys12bar), std::move(dim));

  auto S_A = entropy(rho_A);
  auto S_B = entropy(rho_B);
  auto S_A_B = entropy(rho_AB);

  return S_A + S_B - S_A_B;
}

//******************************************************************************

template <typename T1,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1> >::value, trait::pT<T1> >::type>
inline TR mutual_info(const T1& rho1, arma::uvec sys1, arma::uvec sys2,
                      arma::uword dim = 2) {
  const auto& rho = as_Mat(rho1);

#ifndef QIC_LIB_NO_DEBUG
  bool checkV = true;
  if (rho.n_cols == 1)
    checkV = false;

  if (rho.n_elem == 0)
    throw Exception("qic::mutual_info", Exception::type::ZERO_SIZE);

  if (checkV)
    if (rho.n_rows != rho.n_cols)
      throw Exception("qic::mutual_info",
                      Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);

  if (dim == 0)
    throw Exception("qic::mutual_info", Exception::type::INVALID_DIMS);
#endif

  arma::uword n = static_cast<arma::uword>(
    std::llround(std::log(rho.n_rows) / std::log(dim)));

  arma::uvec dim2(n);
  dim2.fill(dim);
  return mutual_info(rho, std::move(sys1), std::move(sys2), std::move(dim2));
}

//******************************************************************************

}  // namespace qic
