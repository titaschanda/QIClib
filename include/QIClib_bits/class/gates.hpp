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

template <typename T1, typename Enable = typename std::enable_if<
                         std::is_floating_point<T1>::value, void>::type>
class GATES final : public _internal::Singleton<const GATES<T1> > {
  friend class _internal::Singleton<const GATES<T1> >;

 public:
  typename arma::Mat<T1>::template fixed<2, 2> X{0};
  typename arma::Mat<std::complex<T1> >::template fixed<2, 2> Y{0};
  typename arma::Mat<T1>::template fixed<2, 2> Z{0};
  typename arma::Mat<T1>::template fixed<2, 2> Had{0};

  typename arma::Mat<T1>::template fixed<4, 4> CNOT{0};
  typename arma::Mat<T1>::template fixed<4, 4> CZ{0};
  typename arma::Mat<T1>::template fixed<4, 4> swap{0};

  typename arma::Mat<T1>::template fixed<8, 8> Tof{0};
  typename arma::Mat<T1>::template fixed<8, 8> Fred{0};

 private:
  GATES()
      : X(arma::real(SPM<T1>::get_instance().S.at(1))),
        Y(SPM<T1>::get_instance().S.at(2)),
        Z(arma::real(SPM<T1>::get_instance().S.at(3))) {

    Had << std::sqrt(0.5) << std::sqrt(0.5) << arma::endr << std::sqrt(0.5)
        << -std::sqrt(0.5) << arma::endr;

    //**************************************************************************

    CNOT.fill(0.0);
    CNOT.at(0, 0) = CNOT.at(1, 1) = CNOT.at(2, 3) = CNOT.at(3, 2) = 1.0;

    CZ.fill(0.0);
    CZ.at(0, 0) = CZ.at(1, 1) = CZ.at(2, 2) = 1.0;
    CZ.at(3, 3) = -1.0;

    swap.fill(0.0);
    swap.at(0, 0) = swap.at(1, 2) = swap.at(2, 1) = swap.at(3, 3) = 1.0;

    //**************************************************************************

    Tof.fill(0.0);
    Tof.at(0, 0) = Tof.at(1, 1) = Tof.at(2, 2) = Tof.at(3, 3) = Tof.at(4, 4) =
      Tof.at(5, 5) = Tof.at(6, 7) = Tof.at(7, 6) = 1.0;

    Fred.fill(0.0);
    Fred.at(0, 0) = Fred.at(1, 1) = Fred.at(2, 2) = Fred.at(3, 3) =
      Fred.at(4, 4) = Fred.at(5, 6) = Fred.at(6, 5) = Fred.at(7, 7) = 1.0;

    //**************************************************************************
  }
  ~GATES() = default;

 public:
  //**************************************************************************

  typename arma::Mat<std::complex<T1> >::template fixed<2, 2>
  U2(T1 theta, const arma::Col<T1>& unit) const {
#ifndef QICLIB_NO_DEBUG
    if (unit.size() != 3)
      throw Exception("qic::GATES::U2", "Vector is not 3-dimensional!");

    if (std::abs(arma::norm(unit) - 1.0) > _precision::eps<T1>::value)
      throw Exception("qic::GATES::U2", "Vector is not unit vector!");
#endif

    const auto& I = _internal::cond_I<std::complex<T1> >::value;
    return std::cos(0.5 * theta) *
             arma::eye<arma::Mat<std::complex<T1> > >(2, 2) +
           I * std::sin(0.5 * theta) *
             (unit.at(0) * X + unit.at(1) * Y + unit.at(2) * Z);
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

static const GATES<double>& gates _QICLIB_UNUSED_ =
  GATES<double>::get_instance();
static const GATES<float>& gatesf _QICLIB_UNUSED_ =
  GATES<float>::get_instance();

//******************************************************************************

}  // namespace qic
