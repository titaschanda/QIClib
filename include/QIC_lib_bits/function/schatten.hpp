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
inline TR schatten(const T1& rho1, trait::pT<T1> p) {
  const auto& rho = as_Mat(rho1);

#ifndef QIC_LIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::schatten", Exception::type::ZERO_SIZE);

  if (p < 0)
    throw Exception("qic::schatten", Exception::type::OUT_OF_RANGE);
#endif

  if (p == 0)
    return arma::rank(rho);

  if (p == arma::Datum<trait::pT<T1> >::inf)
    return arma::svd(rho).at(0);

  else
    return std::pow(arma::accu(arma::pow(arma::svd(rho), p)), 1.0 / p);
}

//******************************************************************************

}  // namespace qic
