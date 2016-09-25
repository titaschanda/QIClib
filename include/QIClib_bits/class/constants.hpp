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

namespace _precision {

//******************************************************************************

template <typename T, typename Enable = typename std::enable_if<
                        std::is_arithmetic<trait::RCV<T> >::value, void>::type>
struct eps;

template <typename T> struct eps<T> {
  static constexpr T value =
    std::is_integral<trait::RCV<T> >::value
      ? 0
      : (std::is_same<trait::RCV<T>, float>::value
           ? QICLIB_FLOAT_PRECISION  // std::numeric_limits<T>::epsilon()
           : (std::is_same<trait::RCV<T>, double>::value
                ? QICLIB_DOUBLE_PRECISION  // std::numeric_limits<T>::epsilon()
                : 100.0 * std::numeric_limits<trait::RCV<T> >::epsilon()));
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
    const typename arma::Col<T1>::template fixed<4> phim{
      static_cast<T1>(std::sqrt(0.5)), static_cast<T1>(0.0),
      static_cast<T1>(0.0), static_cast<T1>(-std::sqrt(0.5))};

    const typename arma::Col<T1>::template fixed<4> phip{
      static_cast<T1>(std::sqrt(0.5)), static_cast<T1>(0.0),
      static_cast<T1>(0.0), static_cast<T1>(std::sqrt(0.5))};

    const typename arma::Col<T1>::template fixed<4> psim{
      static_cast<T1>(0.0), static_cast<T1>(std::sqrt(0.5)),
      static_cast<T1>(-std::sqrt(0.5)), static_cast<T1>(0.0)};

    const typename arma::Col<T1>::template fixed<4> psip{
      static_cast<T1>(0.0), static_cast<T1>(std::sqrt(0.5)),
      static_cast<T1>(std::sqrt(0.5)), static_cast<T1>(0.0)};
  } bell;

 private:
  SPM() : bell() {

    S.at(0) << 1.0 << 0.0 << arma::endr << 0.0 << 1.0 << arma::endr;

    S.at(1) << 0.0 << 1.0 << arma::endr << 1.0 << 0.0 << arma::endr;

    S.at(2) << 0.0 << std::complex<T1>(0.0, -1.0) << arma::endr
            << std::complex<T1>(0.0, 1.0) << arma::endr;

    S.at(3) << 1.0 << 0.0 << arma::endr << 0.0 << -1.0 << arma::endr;

    //**************************************************************************

    basis2.at(0, 0) << 1.0 << 0.0;
    basis2.at(1, 0) << 0.0 << 1.0;

    basis2.at(0, 1) << std::sqrt(0.5) << std::sqrt(0.5);
    basis2.at(1, 1) << std::sqrt(0.5) << -std::sqrt(0.5);

    basis2.at(0, 2) << std::sqrt(0.5) << std::complex<T1>(0.0, std::sqrt(0.5));
    basis2.at(1, 2) << std::sqrt(0.5) << std::complex<T1>(0.0, -std::sqrt(0.5));

    basis2.at(0, 3) = basis2.at(0, 0);
    basis2.at(1, 3) = basis2.at(1, 0);

    //**************************************************************************

    basis3.at(0, 0) << 1.0 << 0.0 << 0.0;
    basis3.at(1, 0) << 0.0 << 1.0 << 0.0;
    basis3.at(2, 0) << 0.0 << 0.0 << 1.0;

    basis3.at(0, 1) << 0.5 << std::sqrt(0.5) << 0.5;
    basis3.at(1, 1) << -std::sqrt(0.5) << 0.0 << std::sqrt(0.5);
    basis3.at(2, 1) << 0.5 << -std::sqrt(0.5) << 0.5;

    basis3.at(0, 2) << -0.5 << std::complex<T1>(0.0, -std::sqrt(0.5)) << 0.5;
    basis3.at(1, 2) << std::sqrt(0.5) << 0.0 << std::sqrt(0.5);
    basis3.at(2, 2) << -0.5 << std::complex<T1>(0.0, std::sqrt(0.5)) << 0.5;

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

    // bell.phim << std::sqrt(0.5) << 0.0 << 0.0 << -std::sqrt(0.5);
    // bell.phip << std::sqrt(0.5) << 0.0 << 0.0 << std::sqrt(0.5);
    // bell.psim << 0.0 << std::sqrt(0.5) << -std::sqrt(0.5) << 0.0;
    // bell.psip << 0.0 << std::sqrt(0.5) << std::sqrt(0.5) << 0.0;
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
