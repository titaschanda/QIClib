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
                         arma::Mat<trait::eT<T1> > >::type>
inline TR absm(const T1& rho1) {
  const auto& rho = as_Mat(rho1);

#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::absm", Exception::type::ZERO_SIZE);

  if (rho.n_rows != rho.n_cols)
    throw Exception("qic::absm", Exception::type::MATRIX_NOT_SQUARE);
#endif

  return arma::real(sqrtm_sym((rho.t() * rho).eval()));
}

//******************************************************************************

}  // namespace qic
