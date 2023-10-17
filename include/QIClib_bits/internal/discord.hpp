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

#ifndef _QICLIB_INTERNAL_DISCORD_HPP_
#define _QICLIB_INTERNAL_DISCORD_HPP_

#include "../basic/type_traits.hpp"
#include "../class/constants.hpp"
#include <armadillo>

namespace qic {

//******************************************************************************

namespace _internal {

//******************************************************************************

template <typename T1> struct TO_PASS {
  T1& rho;
  arma::Mat<trait::pT<T1> >& eye2;
  arma::Mat<trait::pT<T1> >& eye3;
  arma::Mat<trait::pT<T1> >& eye4;
  arma::uvec& dim;
  arma::uword nodal;
  arma::uword party_no;

  TO_PASS(T1& a, arma::Mat<trait::pT<T1> >& c, arma::Mat<trait::pT<T1> >& d,
          arma::Mat<trait::pT<T1> >& e, arma::uvec& f, arma::uword g,
          arma::uword h)
      : rho(a), eye2(c), eye3(d), eye4(e), dim(f), nodal(g), party_no(h) {}

  ~TO_PASS() = default;

  TO_PASS() = delete;
  TO_PASS(const TO_PASS&) = delete;
  TO_PASS& operator=(const TO_PASS&) = delete;
};

//******************************************************************************

template <typename T1>
inline double disc_nlopt2(const std::vector<double>& x,
                          std::vector<double>& grad, void* my_func_data) {
  (void)grad;
  std::complex<trait::pT<T1> > I(0.0, 1.0);
  trait::pT<T1> theta = static_cast<trait::pT<T1> >(x[0]);
  trait::pT<T1> phi = static_cast<trait::pT<T1> >(x[1]);

  TO_PASS<T1>* pB = static_cast<TO_PASS<T1>*>(my_func_data);

  auto& u = SPM<trait::pT<T1> >::get_instance().basis2.at(0, 0);
  auto& d = SPM<trait::pT<T1> >::get_instance().basis2.at(1, 0);

  arma::Mat<std::complex<trait::pT<T1> > > proj1 =
    std::cos(static_cast<trait::pT<T1> >(0.5) * theta) * u +
    std::exp(I * phi) * std::sin(static_cast<trait::pT<T1> >(0.5) * theta) * d;

  arma::Mat<std::complex<trait::pT<T1> > > proj2 =
    std::sin(static_cast<trait::pT<T1> >(0.5) * theta) * u -
    std::exp(I * phi) * std::cos(static_cast<trait::pT<T1> >(0.5) * theta) * d;

  proj1 *= proj1.t();
  proj2 *= proj2.t();

  if ((*pB).nodal == 1) {
    proj1 = kron(proj1, (*pB).eye2);
    proj2 = kron(proj2, (*pB).eye2);
  } else if ((*pB).party_no == (*pB).nodal) {
    proj1 = kron((*pB).eye2, proj1);
    proj2 = kron((*pB).eye2, proj2);
  } else {
    proj1 = kron(kron((*pB).eye3, proj1), (*pB).eye4);
    proj2 = kron(kron((*pB).eye3, proj2), (*pB).eye4);
  }

  arma::Mat<std::complex<trait::pT<T1> > > rho_1 =
    (proj1 * ((*pB).rho) * proj1);
  arma::Mat<std::complex<trait::pT<T1> > > rho_2 =
    (proj2 * ((*pB).rho) * proj2);

  trait::pT<T1> p1 = std::real(arma::trace(rho_1));
  trait::pT<T1> p2 = std::real(arma::trace(rho_2));

  trait::pT<T1> S_max = 0.0;
  if (p1 > _precision::eps<trait::pT<T1> >::value) {
    rho_1 /= p1;
    S_max += p1 * entropy(TrX(rho_1, {pB->nodal}, pB->dim));
  }

  if (p2 > _precision::eps<trait::pT<T1> >::value) {
    rho_2 /= p2;
    S_max += p2 * entropy(TrX(rho_2, {pB->nodal}, pB->dim));
  }
  return static_cast<double>(S_max);
}

//******************************************************************************

template <typename T1>
inline double disc_nlopt3(const std::vector<double>& x,
                          std::vector<double>& grad, void* my_func_data) {
  (void)grad;
  std::complex<trait::pT<T1> > I(0.0, 1.0);

  trait::pT<T1> theta1 = static_cast<trait::pT<T1> >(0.5 * x[0]);
  trait::pT<T1> theta2 = static_cast<trait::pT<T1> >(0.5 * x[1]);
  trait::pT<T1> theta3 = static_cast<trait::pT<T1> >(0.5 * x[2]);
  trait::pT<T1> phi1 = static_cast<trait::pT<T1> >(x[3]);
  trait::pT<T1> phi2 = static_cast<trait::pT<T1> >(-x[3]);
  trait::pT<T1> del = static_cast<trait::pT<T1> >(x[4]);

  TO_PASS<T1>* pB = static_cast<TO_PASS<T1>*>(my_func_data);

  auto& U = SPM<trait::pT<T1> >::get_instance().basis3.at(0, 0);
  auto& M = SPM<trait::pT<T1> >::get_instance().basis3.at(1, 0);
  auto& D = SPM<trait::pT<T1> >::get_instance().basis3.at(2, 0);

  arma::Mat<std::complex<trait::pT<T1> > > proj1 =
    std::cos(theta1) * std::cos(theta2) * U -
    std::exp(I * phi1) *
      (std::exp(I * del) * std::sin(theta1) * std::cos(theta2) *
         std::cos(theta3) +
       std::sin(theta2) * std::sin(theta3)) *
      M +
    std::exp(I * phi2) *
      (-std::exp(I * del) * std::sin(theta1) * std::cos(theta2) *
         std::sin(theta3) +
       std::sin(theta2) * std::cos(theta3)) *
      D;

  arma::Mat<std::complex<trait::pT<T1> > > proj2 =
    std::exp(-I * del) * std::sin(theta1) * U +
    std::exp(I * phi1) * std::cos(theta1) * std::cos(theta3) * M +
    std::exp(I * phi2) * std::cos(theta1) * std::sin(theta3) * D;

  arma::Mat<std::complex<trait::pT<T1> > > proj3 =
    std::cos(theta1) * std::sin(theta2) * U +
    std::exp(I * phi1) *
      (-std::exp(I * del) * std::sin(theta1) * std::sin(theta2) *
         std::cos(theta3) +
       std::cos(theta2) * std::sin(theta3)) *
      M -
    std::exp(I * phi2) *
      (std::exp(I * del) * std::sin(theta1) * std::sin(theta2) *
         std::sin(theta3) +
       std::cos(theta2) * std::cos(theta3)) *
      D;

  proj1 *= proj1.t();
  proj2 *= proj2.t();
  proj3 *= proj3.t();

  if ((*pB).nodal == 1) {
    proj1 = kron(proj1, (*pB).eye2);
    proj2 = kron(proj2, (*pB).eye2);
    proj3 = kron(proj3, (*pB).eye2);
  } else if ((*pB).party_no == (*pB).nodal) {
    proj1 = kron((*pB).eye2, proj1);
    proj2 = kron((*pB).eye2, proj2);
    proj3 = kron((*pB).eye2, proj3);
  } else {
    proj1 = kron(kron((*pB).eye3, proj1), (*pB).eye4);
    proj2 = kron(kron((*pB).eye3, proj2), (*pB).eye4);
    proj3 = kron(kron((*pB).eye3, proj3), (*pB).eye4);
  }

  arma::Mat<std::complex<trait::pT<T1> > > rho_1 =
    (proj1 * ((*pB).rho) * proj1);
  arma::Mat<std::complex<trait::pT<T1> > > rho_2 =
    (proj2 * ((*pB).rho) * proj2);
  arma::Mat<std::complex<trait::pT<T1> > > rho_3 =
    (proj3 * ((*pB).rho) * proj3);

  trait::pT<T1> p1 = std::real(arma::trace(rho_1));
  trait::pT<T1> p2 = std::real(arma::trace(rho_2));
  trait::pT<T1> p3 = std::real(arma::trace(rho_3));

  trait::pT<T1> S_max = 0.0;
  if (p1 > _precision::eps<trait::pT<T1> >::value) {
    rho_1 /= p1;
    S_max += p1 * entropy(TrX(rho_1, {pB->nodal}, pB->dim));
  }
  if (p2 > _precision::eps<trait::pT<T1> >::value) {
    rho_2 /= p2;
    S_max += p2 * entropy(TrX(rho_2, {pB->nodal}, pB->dim));
  }
  if (p3 > _precision::eps<trait::pT<T1> >::value) {
    rho_3 /= p3;
    S_max += p3 * entropy(TrX(rho_3, {pB->nodal}, pB->dim));
  }

  return static_cast<double>(S_max);
}

//******************************************************************************

template <typename T1>
double def_nlopt2(const std::vector<double>& x, std::vector<double>& grad,
                  void* my_func_data) {
  (void)grad;
  std::complex<trait::pT<T1> > I(0.0, 1.0);
  trait::pT<T1> theta = static_cast<trait::pT<T1> >(x[0]);
  trait::pT<T1> phi = static_cast<trait::pT<T1> >(x[1]);

  TO_PASS<T1>* pB = static_cast<TO_PASS<T1>*>(my_func_data);

  auto& u = SPM<trait::pT<T1> >::get_instance().basis2.at(0, 0);
  auto& d = SPM<trait::pT<T1> >::get_instance().basis2.at(1, 0);

  arma::Mat<std::complex<trait::pT<T1> > > proj1 =
    std::cos(static_cast<trait::pT<T1> >(0.5) * theta) * u +
    std::exp(I * phi) * std::sin(static_cast<trait::pT<T1> >(0.5) * theta) * d;

  arma::Mat<std::complex<trait::pT<T1> > > proj2 =
    std::sin(static_cast<trait::pT<T1> >(0.5) * theta) * u -
    std::exp(I * phi) * std::cos(static_cast<trait::pT<T1> >(0.5) * theta) * d;

  proj1 *= proj1.t();
  proj2 *= proj2.t();

  if ((*pB).nodal == 1) {
    proj1 = kron(proj1, (*pB).eye2);
    proj2 = kron(proj2, (*pB).eye2);

  } else if ((*pB).party_no == (*pB).nodal) {
    proj1 = kron((*pB).eye2, proj1);
    proj2 = kron((*pB).eye2, proj2);

  } else {
    proj1 = kron(kron((*pB).eye3, proj1), (*pB).eye4);
    proj2 = kron(kron((*pB).eye3, proj2), (*pB).eye4);
  }

  arma::Mat<std::complex<trait::pT<T1> > > rho_1 =
    (proj1 * ((*pB).rho) * proj1);
  arma::Mat<std::complex<trait::pT<T1> > > rho_2 =
    (proj2 * ((*pB).rho) * proj2);

  rho_1 += rho_2;
  trait::pT<T1> S_max = entropy(rho_1);
  return static_cast<double>(S_max);
}

//******************************************************************************

template <typename T1>
double def_nlopt3(const std::vector<double>& x, std::vector<double>& grad,
                  void* my_func_data) {
  (void)grad;
  std::complex<trait::pT<T1> > I(0.0, 1.0);

  trait::pT<T1> theta1 = static_cast<trait::pT<T1> >(0.5 * x[0]);
  trait::pT<T1> theta2 = static_cast<trait::pT<T1> >(0.5 * x[1]);
  trait::pT<T1> theta3 = static_cast<trait::pT<T1> >(0.5 * x[2]);
  trait::pT<T1> phi1 = static_cast<trait::pT<T1> >(x[3]);
  trait::pT<T1> phi2 = static_cast<trait::pT<T1> >(-x[3]);
  trait::pT<T1> del = static_cast<trait::pT<T1> >(x[4]);

  TO_PASS<T1>* pB = static_cast<TO_PASS<T1>*>(my_func_data);

  auto& U = SPM<trait::pT<T1> >::get_instance().basis3.at(0, 0);
  auto& M = SPM<trait::pT<T1> >::get_instance().basis3.at(1, 0);
  auto& D = SPM<trait::pT<T1> >::get_instance().basis3.at(2, 0);

  arma::Mat<std::complex<trait::pT<T1> > > proj1 =
    std::cos(theta1) * std::cos(theta2) * U -
    std::exp(I * phi1) *
      (std::exp(I * del) * std::sin(theta1) * std::cos(theta2) *
         std::cos(theta3) +
       std::sin(theta2) * std::sin(theta3)) *
      M +
    std::exp(I * phi2) *
      (-std::exp(I * del) * std::sin(theta1) * std::cos(theta2) *
         std::sin(theta3) +
       std::sin(theta2) * std::cos(theta3)) *
      D;

  arma::Mat<std::complex<trait::pT<T1> > > proj2 =
    std::exp(-I * del) * std::sin(theta1) * U +
    std::exp(I * phi1) * std::cos(theta1) * std::cos(theta3) * M +
    std::exp(I * phi2) * std::cos(theta1) * std::sin(theta3) * D;

  arma::Mat<std::complex<trait::pT<T1> > > proj3 =
    std::cos(theta1) * std::sin(theta2) * U +
    std::exp(I * phi1) *
      (-std::exp(I * del) * std::sin(theta1) * std::sin(theta2) *
         std::cos(theta3) +
       std::cos(theta2) * std::sin(theta3)) *
      M -
    std::exp(I * phi2) *
      (std::exp(I * del) * std::sin(theta1) * std::sin(theta2) *
         std::sin(theta3) +
       std::cos(theta2) * std::cos(theta3)) *
      D;

  proj1 *= proj1.t();
  proj2 *= proj2.t();
  proj3 *= proj3.t();

  if ((*pB).nodal == 1) {
    proj1 = kron(proj1, (*pB).eye2);
    proj2 = kron(proj2, (*pB).eye2);
    proj3 = kron(proj3, (*pB).eye2);

  } else if ((*pB).party_no == (*pB).nodal) {
    proj1 = kron((*pB).eye2, proj1);
    proj2 = kron((*pB).eye2, proj2);
    proj3 = kron((*pB).eye2, proj3);

  } else {
    proj1 = kron(kron((*pB).eye3, proj1), (*pB).eye4);
    proj2 = kron(kron((*pB).eye3, proj2), (*pB).eye4);
    proj3 = kron(kron((*pB).eye3, proj3), (*pB).eye4);
  }

  arma::Mat<std::complex<trait::pT<T1> > > rho_1 =
    (proj1 * ((*pB).rho) * proj1);
  arma::Mat<std::complex<trait::pT<T1> > > rho_2 =
    (proj2 * ((*pB).rho) * proj2);
  arma::Mat<std::complex<trait::pT<T1> > > rho_3 =
    (proj3 * ((*pB).rho) * proj3);

  rho_1 += rho_2 + rho_3;
  trait::pT<T1> S_max = entropy(rho_1);
  return (static_cast<double>(S_max));
}

//******************************************************************************

}  // namespace _internal

//******************************************************************************

}  // namespace qic

#endif
