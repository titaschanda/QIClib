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

#ifndef _QICLIB_GATES_HPP_
#define _QICLIB_GATES_HPP_

#include "../basic/macro.hpp"
#include "../basic/type_traits.hpp"
#include "../internal/as_arma.hpp"
#include "../internal/constants.hpp"
#include "../internal/singleton.hpp"
#include "constants.hpp"
#include "exception.hpp"
#include <armadillo>

namespace qic {

//******************************************************************************

template <typename T1, typename Enable = typename std::enable_if<
                         std::is_floating_point<T1>::value, void>::type>
class GATES final : public _internal::Singleton<const GATES<T1> > {
  friend class _internal::Singleton<const GATES<T1> >;

 public:
  typename arma::Mat<T1>::template fixed<2, 2> X{arma::fill::zeros};

  typename arma::Mat<std::complex<T1> >::template fixed<2, 2> Y{
    arma::fill::zeros};

  typename arma::Mat<T1>::template fixed<2, 2> Z{arma::fill::zeros};
  typename arma::Mat<T1>::template fixed<2, 2> Had{arma::fill::zeros};

  typename arma::Mat<T1>::template fixed<4, 4> CNOT{arma::fill::zeros};
  typename arma::Mat<T1>::template fixed<4, 4> CZ{arma::fill::zeros};
  typename arma::Mat<T1>::template fixed<4, 4> swap{arma::fill::zeros};

  typename arma::Mat<T1>::template fixed<8, 8> Tof{arma::fill::zeros};
  typename arma::Mat<T1>::template fixed<8, 8> Fred{arma::fill::zeros};

 private:
  GATES()
      : X(arma::real(SPM<T1>::get_instance().S.at(1))),
        Y(SPM<T1>::get_instance().S.at(2)),
        Z(arma::real(SPM<T1>::get_instance().S.at(3))) {

    Had  = {{T1(std::sqrt(0.5)), T1(std::sqrt(0.5))},
            {T1(std::sqrt(0.5)), T1(-std::sqrt(0.5))}};

    //**************************************************************************

    CNOT.at(0, 0) = CNOT.at(1, 1) = CNOT.at(2, 3) = CNOT.at(3, 2) = 1.0;

    CZ.at(0, 0) = CZ.at(1, 1) = CZ.at(2, 2) = 1.0;
    CZ.at(3, 3) = -1.0;

    swap.at(0, 0) = swap.at(1, 2) = swap.at(2, 1) = swap.at(3, 3) = 1.0;

    //**************************************************************************

    Tof.at(0, 0) = Tof.at(1, 1) = Tof.at(2, 2) = Tof.at(3, 3) = Tof.at(4, 4) =
      Tof.at(5, 5) = Tof.at(6, 7) = Tof.at(7, 6) = 1.0;

    Fred.at(0, 0) = Fred.at(1, 1) = Fred.at(2, 2) = Fred.at(3, 3) =
      Fred.at(4, 4) = Fred.at(5, 6) = Fred.at(6, 5) = Fred.at(7, 7) = 1.0;

    //**************************************************************************
  }
  ~GATES() = default;

 public:
  //**************************************************************************

  typename arma::Mat<std::complex<T1> >::template fixed<2, 2>
  U2(T1 theta, const arma::Col<T1>& unitV) const {
#ifndef QICLIB_NO_DEBUG
    if (unitV.size() != 3)
      throw Exception("qic::GATES::U2", "Vector is not 3-dimensional!");

    if (std::abs(arma::norm(unitV) - 1.0) > _precision::eps<T1>::value)
      throw Exception("qic::GATES::U2", "Vector is not unit vector!");
#endif

    const auto& I = _internal::cond_I<std::complex<T1> >::value;
    typename arma::Mat<std::complex<T1> >::template fixed<2, 2> ret(
      arma::fill::eye);

    return std::cos(0.5 * theta) * ret +
           I * std::sin(0.5 * theta) *
             (unitV.at(0) * X + unitV.at(1) * Y + unitV.at(2) * Z);
  }

  //**************************************************************************

  typename arma::Mat<std::complex<T1> >::template fixed<2, 2> PS(T1 phi) const {

    const auto& I = _internal::cond_I<std::complex<T1> >::value;
    typename arma::Mat<std::complex<T1> >::template fixed<2, 2> ret(
      arma::fill::zeros);

    ret.at(0, 0) = 1;
    ret.at(1, 1) = std::exp(I * phi);
    return ret;
  }

  //**************************************************************************

  arma::Mat<std::complex<T1> > qft(arma::uword dim) const {
#ifndef QICLIB_NO_DEBUG
    if (dim == 0)
      throw Exception("qic::GATES::qft", Exception::type::INVALID_DIMS);
#endif

    arma::Mat<std::complex<T1> > ret(dim, dim);
    const auto& I = _internal::cond_I<std::complex<T1> >::value;

    QICLIB_OPENMP_FOR_COLLAPSE_2
    for (arma::uword j = 0; j < dim; ++j) {
      for (arma::uword i = 0; i < dim; ++i) {
        ret.at(i, j) = 1.0 / std::sqrt(static_cast<T1>(dim)) *
                       std::exp(2.0 * I * arma::Datum<T1>::pi *
                                static_cast<T1>(i * j) / static_cast<T1>(dim));
      }
    }

    return ret;
  }

  //**************************************************************************
};

//******************************************************************************

#ifdef QICLIB_GATES

static const GATES<double>& gates _QICLIB_UNUSED_ =
  GATES<double>::get_instance();
static const GATES<float>& gatesf _QICLIB_UNUSED_ =
  GATES<float>::get_instance();

#endif

//******************************************************************************

}  // namespace qic

#endif
