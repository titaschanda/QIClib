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
inline TR neg(const T1& rho1, arma::uvec sys, arma::uvec dim) {
  const auto& p = as_Mat(rho1);

#ifndef QIC_LIB_NO_DEBUG
  bool checkV = true;
  if (p.n_cols == 1)
    checkV = false;

  if (p.n_elem == 0)
    throw Exception("qic::neg", Exception::type::ZERO_SIZE);

  if (checkV)
    if (p.n_rows != p.n_cols)
      throw Exception("qic::neg",
                      Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);

  if (dim.n_elem == 0 || arma::any(dim == 0))
    throw Exception("qic::neg", Exception::type::INVALID_DIMS);

  if (arma::prod(dim) != p.n_rows)
    throw Exception("qic::neg", Exception::type::DIMS_MISMATCH_MATRIX);

  if (dim.n_elem < sys.n_elem || arma::any(sys == 0) ||
      arma::any(sys > dim.n_elem) ||
      sys.n_elem != arma::find_unique(sys, false).eval().n_elem)
    throw Exception("qic::neg", Exception::type::INVALID_SUBSYS);

#endif

  auto rho_T = Tx(p, std::move(sys), std::move(dim));
  auto eigval = arma::eig_sym(rho_T);
  trait::pT<T1> Neg = 0.0;

  for (const auto& i : eigval)
    Neg += (std::abs(i) >= _precision::eps<trait::pT<T1> >::value)
             ? 0.5 * (std::abs(i) - i)
             : 0;
  return Neg;
}

//******************************************************************************

template <typename T1,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1> >::value, trait::pT<T1> >::type>
inline TR log_neg(const T1& rho1, arma::uvec sys, arma::uvec dim) {
  return std::log2(2.0 * neg(rho1, std::move(sys), std::move(dim)) + 1.0);
}

//******************************************************************************

template <typename T1,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1> >::value, trait::pT<T1> >::type>
inline TR neg(const T1& rho1, arma::uvec sys, arma::uword dim = 2) {
  const auto& p = as_Mat(rho1);

#ifndef QIC_LIB_NO_DEBUG
  bool checkV = true;
  if (p.n_cols == 1)
    checkV = false;

  if (p.n_elem == 0)
    throw Exception("qic::neg", Exception::type::ZERO_SIZE);

  if (checkV)
    if (p.n_rows != p.n_cols)
      throw Exception("qic::neg",
                      Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);

  if (dim == 0)
    throw Exception("qic::neg", Exception::type::INVALID_DIMS);
#endif

  arma::uword n =
    static_cast<arma::uword>(std::llround(std::log(p.n_rows) / std::log(dim)));

  arma::uvec dim2(n);
  dim2.fill(dim);
  return neg(rho1, std::move(sys), std::move(dim2));
}

//******************************************************************************

template <typename T1,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1> >::value, trait::pT<T1> >::type>
inline TR log_neg(const T1& rho1, arma::uvec sys, arma::uword dim = 2) {
  return std::log2(2.0 * neg(rho1, std::move(sys), dim) + 1.0);
}

//******************************************************************************

}  // namespace qic
