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
namespace {
namespace protect {

//******************************************************************************

template <typename T1> struct TO_PASS {
  const arma::Mat<trait::eT<T1> >& rho;
  const arma::Mat<trait::pT<T1> >& eye2;
  const arma::Mat<trait::pT<T1> >& eye3;
  const arma::Mat<trait::pT<T1> >& eye4;
  arma::uword nodal;
  arma::uword party_no;

  TO_PASS(const T1& a, const arma::Mat<trait::pT<T1> >& c,
          const arma::Mat<trait::pT<T1> >& d,
          const arma::Mat<trait::pT<T1> >& e, arma::uword f, arma::uword g)
      : rho(a), eye2(c), eye3(d), eye4(e), nodal(f), party_no(g) {}
  ~TO_PASS() {}
};

//******************************************************************************

}}  // namespace protect

//******************************************************************************

}  // namespace qic
