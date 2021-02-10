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

#ifndef _QICLIB_CONSTANTS_HPP_
#define _QICLIB_CONSTANTS_HPP_

#include "../basic/macro.hpp"
#include "../basic/type_traits.hpp"
#include "../internal/as_arma.hpp"
#include "../internal/singleton.hpp"
#include <armadillo>

namespace qic {

//******************************************************************************

namespace _precision {

//******************************************************************************

template <typename T,
          typename Enable = typename std::enable_if<
            std::is_arithmetic<trait::DECAY<T> >::value, void>::type>
struct eps;

template <typename T> struct eps<T> {
  static constexpr T value =
    std::is_integral<trait::DECAY<T> >::value
      ? 0
      : (std::is_same<trait::DECAY<T>, float>::value
           ? QICLIB_FLOAT_PRECISION  // std::numeric_limits<T>::epsilon()
           : (std::is_same<trait::DECAY<T>, double>::value
                ? QICLIB_DOUBLE_PRECISION  // std::numeric_limits<T>::epsilon()
                : 100.0 * std::numeric_limits<trait::DECAY<T> >::epsilon()));
};

template <typename T> constexpr T eps<T>::value;

//******************************************************************************

}  // namespace _precision

//******************************************************************************

template <typename T1, typename Enable = typename std::enable_if<
                         std::is_floating_point<T1>::value, void>::type>
class SPM final : public _internal::Singleton<const SPM<T1> > {
  friend class _internal::Singleton<const SPM<T1> >;

 public:
  arma::field<typename arma::Mat<std::complex<T1> >::template fixed<2, 2> > S{
    4};
  arma::field<typename arma::Col<std::complex<T1> >::template fixed<2> > basis2{
    2, 4};
  arma::field<typename arma::Col<std::complex<T1> >::template fixed<3> > basis3{
    3, 4};
  arma::field<typename arma::Mat<std::complex<T1> >::template fixed<2, 2> >
    proj2{2, 4};
  arma::field<typename arma::Mat<std::complex<T1> >::template fixed<3, 3> >
    proj3{3, 4};

  struct {
    typename arma::Col<T1>::template fixed<4> phim{};
    typename arma::Col<T1>::template fixed<4> phip{};
    typename arma::Col<T1>::template fixed<4> psim{};
    typename arma::Col<T1>::template fixed<4> psip{};
  } bell;

 private:
  SPM() : bell() {

    S.at(0) = {{std::complex<T1>(1.0), std::complex<T1>(0.0)},
               {std::complex<T1>(0.0), std::complex<T1>(1.0)}};

    S.at(1) = {{std::complex<T1>(0.0), std::complex<T1>(1.0)},
               {std::complex<T1>(1.0), std::complex<T1>(0.0)}};

    S.at(2) = {{std::complex<T1>(0.0), std::complex<T1>(0.0, -1.0)},
               {std::complex<T1>(0.0, 1.0), std::complex<T1>(0.0)}};

    S.at(3) = {{std::complex<T1>(1.0), std::complex<T1>(0.0)},
               {std::complex<T1>(0.0), std::complex<T1>(-1.0)}};

    //**************************************************************************

    basis2.at(0, 0) = {std::complex<T1>(1.0), std::complex<T1>(0.0)};
    basis2.at(1, 0) = {std::complex<T1>(0.0), std::complex<T1>(1.0)};

    basis2.at(0, 1) = {std::complex<T1>(std::sqrt(0.5)), std::complex<T1>(std::sqrt(0.5))};
    basis2.at(1, 1) = {std::complex<T1>(std::sqrt(0.5)), std::complex<T1>(-std::sqrt(0.5))};

    basis2.at(0, 2) = {std::complex<T1>(std::sqrt(0.5)), std::complex<T1>(0.0, std::sqrt(0.5))};
    basis2.at(1, 2) = {std::complex<T1>(std::sqrt(0.5)), std::complex<T1>(0.0, -std::sqrt(0.5))};

    basis2.at(0, 3) = basis2.at(0, 0);
    basis2.at(1, 3) = basis2.at(1, 0);

    //**************************************************************************

    basis3.at(0, 0) = {1.0, 0.0, 0.0};
    basis3.at(1, 0) = {0.0, 1.0, 0.0};
    basis3.at(2, 0) = {0.0, 0.0, 1.0};

    basis3.at(0, 1) = {0.5, std::sqrt(0.5), 0.5};
    basis3.at(1, 1) = {-std::sqrt(0.5), 0.0, std::sqrt(0.5)};
    basis3.at(2, 1) = {0.5, -std::sqrt(0.5), 0.5};

    basis3.at(0, 2) = {-0.5, std::complex<T1>(0.0, -std::sqrt(0.5)), 0.5};
    basis3.at(1, 2) = {std::sqrt(0.5), 0.0, std::sqrt(0.5)};
    basis3.at(2, 2) = {-0.5, std::complex<T1>(0.0, std::sqrt(0.5)), 0.5};

    basis3.at(0, 3) = basis3.at(0, 0);
    basis3.at(1, 3) = basis3.at(1, 0);
    basis3.at(2, 3) = basis3.at(2, 0);

    //**************************************************************************

    proj2.at(0, 0) = basis2.at(0, 0) * basis2.at(0, 0).t();
    proj2.at(1, 0) = basis2.at(1, 0) * basis2.at(1, 0).t();
    proj2.at(0, 1) = basis2.at(0, 1) * basis2.at(0, 1).t();
    proj2.at(1, 1) = basis2.at(1, 1) * basis2.at(1, 1).t();
    proj2.at(0, 2) = basis2.at(0, 2) * basis2.at(0, 2).t();
    proj2.at(1, 2) = basis2.at(1, 2) * basis2.at(1, 2).t();
    proj2.at(0, 3) = proj2.at(0, 0);
    proj2.at(1, 3) = proj2.at(1, 0);

    //**************************************************************************

    proj3.at(0, 0) = basis3.at(0, 0) * basis3.at(0, 0).t();
    proj3.at(1, 0) = basis3.at(1, 0) * basis3.at(1, 0).t();
    proj3.at(2, 0) = basis3.at(2, 0) * basis3.at(2, 0).t();
    proj3.at(0, 1) = basis3.at(0, 1) * basis3.at(0, 1).t();
    proj3.at(1, 1) = basis3.at(1, 1) * basis3.at(1, 1).t();
    proj3.at(2, 1) = basis3.at(2, 1) * basis3.at(2, 1).t();
    proj3.at(0, 2) = basis3.at(0, 2) * basis3.at(0, 2).t();
    proj3.at(1, 2) = basis3.at(1, 2) * basis3.at(1, 2).t();
    proj3.at(2, 2) = basis3.at(2, 2) * basis3.at(2, 2).t();
    proj3.at(0, 3) = proj3.at(0, 0);
    proj3.at(1, 3) = proj3.at(1, 0);
    proj3.at(2, 3) = proj3.at(2, 0);

    //**************************************************************************

    bell.phim = {T1(std::sqrt(0.5)), 0.0, 0.0, T1(-std::sqrt(0.5))};
    bell.phip = {T1(std::sqrt(0.5)), 0.0, 0.0, T1(std::sqrt(0.5))};
    bell.psim = {0.0, T1(std::sqrt(0.5)), T1(-std::sqrt(0.5)), 0.0};
    bell.psip = {0.0, T1(std::sqrt(0.5)), T1(std::sqrt(0.5)), 0.0};
  }
  ~SPM() = default;
};

//******************************************************************************

#ifdef QICLIB_SPM

static const SPM<double>& spm _QICLIB_UNUSED_ = SPM<double>::get_instance();
static const SPM<float>& spmf _QICLIB_UNUSED_ = SPM<float>::get_instance();

#endif

//******************************************************************************

}  // namespace qic

#endif
