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

template <typename T1, typename TR = typename std::enable_if<
                         is_floating_point_var<trait::pT<T1> >::value,
                         arma::Mat<trait::pT<T1> > >::type>
inline TR std_to_HS(const T1& rho1) {
  const auto& rho = _internal::as_Mat(rho1);

#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::std_to_HS", Exception::type::ZERO_SIZE);

  if (rho.n_rows != rho.n_cols)
    throw Exception("qic::std_to_HS", Exception::type::MATRIX_NOT_SQUARE);

  if (rho.n_rows != 4)
    throw Exception("qic::std_to_HS", Exception::type::NOT_QUBIT_SUBSYS);
#endif

  auto& S = SPM<trait::pT<T1> >::get_instance().S;

  arma::Mat<trait::pT<T1> > ret(4, 4);

  for (arma::uword j = 0; j < 4; ++j) {
    for (arma::uword i = 0; i < 4; ++i)
      ret.at(i, j) = std::real(arma::trace(arma::kron(S.at(i), S.at(j)) * rho));
  }
  return ret;
}

//******************************************************************************

template <typename T1, typename TR = typename std::enable_if<
                         is_floating_point_var<trait::pT<T1> >::value,
                         arma::Mat<std::complex<trait::pT<T1> > > >::type>
inline TR HS_to_std(const T1& rho1) {
  const auto& rho = _internal::as_Mat(rho1);

#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::HS_to_std", Exception::type::ZERO_SIZE);

  if (!std::is_same<trait::eT<T1>, trait::pT<T1> >::value)
    throw Exception("qic::HS_to_std", "Matrix is not real");

  if (rho.n_rows != rho.n_cols)
    throw Exception("qic::HS_to_std", Exception::type::MATRIX_NOT_SQUARE);

  if (rho.n_rows != 4)
    throw Exception("qic::HS_to_std", Exception::type::NOT_QUBIT_SUBSYS);
#endif

  auto& S = SPM<trait::pT<T1> >::get_instance().S;

  arma::Mat<std::complex<trait::pT<T1> > > ret(4, 4, arma::fill::zeros);

  for (arma::uword j = 0; j < 4; ++j) {
    for (arma::uword i = 0; i < 4; ++i)
      ret += rho.at(i, j) * arma::kron(S.at(i), S.at(j)) * 0.25;
  }
  return ret;
}

//******************************************************************************

}  // namespace qic
