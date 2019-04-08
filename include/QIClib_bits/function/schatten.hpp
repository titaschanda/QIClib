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

template <typename T1, typename TR = trait::pT<T1> >

inline TR schatten(const T1& rho1, const trait::pT<T1>& p) {
  const auto& rho = _internal::as_Mat(rho1);

#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::schatten", Exception::type::ZERO_SIZE);

  if (rho.n_rows != rho.n_cols)
    throw Exception("qic::schatten", Exception::type::MATRIX_NOT_SQUARE);
    
  if (p < 0)
    throw Exception("qic::schatten", Exception::type::OUT_OF_RANGE);
#endif

  if (std::abs(p - 0.0) < _precision::eps<trait::pT<T1> >::value)
    return arma::rank(rho);

  if (p == arma::Datum<trait::pT<T1> >::inf)
    return arma::svd(rho).at(0);

  else {

    arma::Mat<trait::eT<T1> > rhoH = rho.t() * rho;
    arma::Col<trait::pT<T1> > eigval;
    arma::Mat<trait::eT<T1> > eigvec;

  if (rhoH.n_rows > 20) {
    bool check = arma::eig_sym(eigval, eigvec, rhoH, "dc");
    if (!check)
      throw std::runtime_error("qic::schatten(): Decomposition failed!");

  } else {
    bool check = arma::eig_sym(eigval, eigvec, rhoH, "std");
    if (!check)
      throw std::runtime_error("qic::schatten(): Decomposition failed!");
  }

  return std::pow(
    std::real(arma::trace(
      eigvec *
      arma::diagmat(arma::pow(
        _internal::as_type<arma::Col<std::complex<trait::pT<T1> > > >::from(
          eigval),
        p / 2.0)) *
      eigvec.t())),
    1.0 / p);
  }
  //return std::pow(std::real(arma::trace(powm_sym(absm(rho), p))), 1.0 / p);
}

//******************************************************************************

}  // namespace qic
