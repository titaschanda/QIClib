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

//****************************************************************************

template <
  typename T1 = double,
  typename TR = typename std::enable_if<
    is_floating_point_var<T1>::value || is_complex_fp<T1>::value, T1>::type>
inline TR randU(
  const arma::Col<typename arma::get_pod_type<T1>::result>& range = {0, 1}) {
#ifndef QICLIB_NO_DEBUG
  if (range.n_elem != 2 || range.at(0) > range.at(1))
    throw Exception("qic::randU", "Not proper range!");
#endif

  std::uniform_real_distribution<typename arma::get_pod_type<T1>::result> dis(
    range.at(0), range.at(1));

  if (is_floating_point_var<T1>::value) {
    return dis(rdevs.rng);

  } else {
    auto& I = _internal::cond_I<T1>::value;
    return dis(rdevs.rng) + I * dis(rdevs.rng);
  }
}

//****************************************************************************

template <typename T1 = arma::vec,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1> >::value &&
              (arma::is_Col<T1>::value || arma::is_Row<T1>::value),
            T1>::type>
inline TR randU(const arma::uword& N,
                const arma::Col<trait::pT<T1> >& range = {0, 1}) {
#ifndef QICLIB_NO_DEBUG
  if (range.n_elem != 2 || range.at(0) > range.at(1))
    throw Exception("qic::randU", "Not proper range!");
#endif

  std::uniform_real_distribution<trait::pT<T1> > dis(range.at(0), range.at(1));
  T1 ret(N);

  if (std::is_same<trait::eT<T1>, trait::pT<T1> >::value) {
    ret.imbue([&dis]() { return dis(rdevs.rng); });

  } else {
    auto& I = _internal::cond_I<trait::eT<T1> >::value;
    ret.imbue([&dis, &I]() { return dis(rdevs.rng) + I * dis(rdevs.rng); });
  }
  return ret;
}

//****************************************************************************

template <typename T1 = arma::mat,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1> >::value &&
              arma::is_Mat_only<T1>::value,
            arma::Mat<trait::eT<T1> > >::type>
inline TR randU(const arma::uword& m, const arma::uword& n,
                const arma::Col<trait::pT<T1> >& range = {0, 1}) {
#ifndef QICLIB_NO_DEBUG
  if (range.n_elem != 2 || range.at(0) > range.at(1))
    throw Exception("qic::randU", "Not proper range!");
#endif

  std::uniform_real_distribution<trait::pT<T1> > dis(range.at(0), range.at(1));
  arma::Mat<trait::eT<T1> > ret(m, n);

  if (std::is_same<trait::eT<T1>, trait::pT<T1> >::value) {
    ret.imbue([&dis]() { return dis(rdevs.rng); });

  } else {
    auto& I = _internal::cond_I<trait::eT<T1> >::value;
    ret.imbue([&dis, &I]() { return dis(rdevs.rng) + I * dis(rdevs.rng); });
  }
  return ret;
}

//****************************************************************************

template <
  typename T1 = double,
  typename TR = typename std::enable_if<
    is_floating_point_var<T1>::value || is_complex_fp<T1>::value, T1>::type>
inline TR randN(
  const arma::Col<typename arma::get_pod_type<T1>::result>& meansd = {0, 1}) {
#ifndef QICLIB_NO_DEBUG
  if (meansd.n_elem != 2 ||
      meansd.at(1) <
        _precision::eps<typename arma::get_pod_type<T1>::result>::value)
    throw Exception("qic::randN", "Not proper mean and standard deviation!");
#endif

  std::normal_distribution<typename arma::get_pod_type<T1>::result> dis(
    meansd.at(0), meansd.at(1));

  if (is_floating_point_var<T1>::value) {
    return dis(rdevs.rng);

  } else {
    auto& I = _internal::cond_I<T1>::value;
    return dis(rdevs.rng) + I * dis(rdevs.rng);
  }
}

//****************************************************************************

template <typename T1 = arma::vec,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1> >::value &&
              (arma::is_Col<T1>::value || arma::is_Row<T1>::value),
            T1>::type>
inline TR randN(const arma::uword& N,
                const arma::Col<trait::pT<T1> >& meansd = {0, 1}) {
#ifndef QICLIB_NO_DEBUG
  if (meansd.n_elem != 2 ||
      meansd.at(1) < _precision::eps<trait::pT<T1> >::value)
    throw Exception("qic::randN", "Not proper mean and standard deviation!");
#endif

  std::normal_distribution<trait::pT<T1> > dis(meansd.at(0), meansd.at(1));
  T1 ret(N);

  if (std::is_same<trait::eT<T1>, trait::pT<T1> >::value) {
    ret.imbue([&dis]() { return dis(rdevs.rng); });

  } else {
    auto& I = _internal::cond_I<trait::eT<T1> >::value;
    ret.imbue([&dis, &I]() { return dis(rdevs.rng) + I * dis(rdevs.rng); });
  }
  return ret;
}

//****************************************************************************

template <typename T1 = arma::mat,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1> >::value &&
              arma::is_Mat_only<T1>::value,
            arma::Mat<trait::eT<T1> > >::type>
inline TR randN(const arma::uword& m, const arma::uword& n,
                const arma::Col<trait::pT<T1> >& meansd = {0, 1}) {
#ifndef QICLIB_NO_DEBUG
  if (meansd.n_elem != 2 ||
      meansd.at(1) < _precision::eps<trait::pT<T1> >::value)
    throw Exception("qic::randN", "Not proper mean and standard deviation!");
#endif

  std::normal_distribution<trait::pT<T1> > dis(meansd.at(0), meansd.at(1));
  arma::Mat<trait::eT<T1> > ret(m, n);

  if (std::is_same<trait::eT<T1>, trait::pT<T1> >::value) {
    ret.imbue([&dis]() { return dis(rdevs.rng); });

  } else {
    auto& I = _internal::cond_I<trait::eT<T1> >::value;
    ret.imbue([&dis, &I]() { return dis(rdevs.rng) + I * dis(rdevs.rng); });
  }
  return ret;
}

//****************************************************************************

template <typename T1 = arma::sword,
          typename TR = typename std::enable_if<
            std::is_arithmetic<T1>::value || is_complex<T1>::value, T1>::type,
          typename TA = typename std::conditional<
            std::is_unsigned<typename arma::get_pod_type<T1> >::value,
            arma::uword, arma::sword>::type>
inline TR randI(const arma::Col<TA>& range = {0, 1000}) {
#ifndef QICLIB_NO_DEBUG
  if (range.n_elem != 2 || range.at(0) > range.at(1))
    throw Exception("qic::randI", "Not proper range!");

  if (std::is_unsigned<typename arma::get_pod_type<T1> >::value &&
      arma::any(range < 0))
    throw Exception("qic::randI", "Negative range for unsigned type!");
#endif

  std::uniform_int_distribution<TA> dis(range.at(0), range.at(1));

  if (is_floating_point_var<T1>::value) {
    return static_cast<typename arma::get_pod_type<T1>::result>(dis(rdevs.rng));

  } else {
    auto& I = _internal::cond_I<T1>::value;
    return static_cast<typename arma::get_pod_type<T1>::result>(
             dis(rdevs.rng)) +
           I * static_cast<typename arma::get_pod_type<T1>::result>(
                 dis(rdevs.rng));
  }
}

//****************************************************************************

template <typename T1 = arma::ivec,
          typename TR = typename std::enable_if<is_arma_type_var<T1>::value &&
                                                  (arma::is_Col<T1>::value ||
                                                   arma::is_Row<T1>::value),
                                                T1>::type,
          typename TA =
            typename std::conditional<std::is_unsigned<trait::pT<T1> >::value,
                                      arma::uword, arma::sword>::type>
inline TR randI(const arma::uword& N, const arma::Col<TA>& range = {0, 1000}) {
#ifndef QICLIB_NO_DEBUG
  if (range.n_elem != 2 || range.at(0) > range.at(1))
    throw Exception("qic::randI", "Not proper range!");

  if (std::is_unsigned<trait::pT<T1> >::value && arma::any(range < 0))
    throw Exception("qic::randI", "Negative range for unsigned type!");
#endif

  std::uniform_int_distribution<TA> dis(range.at(0), range.at(1));
  T1 ret(N);

  if (std::is_same<trait::eT<T1>, trait::pT<T1> >::value) {
    ret.imbue([&dis]() { return static_cast<trait::pT<T1> >(dis(rdevs.rng)); });

  } else {
    auto& I = _internal::cond_I<trait::eT<T1> >::value;
    ret.imbue([&dis, &I]() {
      return static_cast<trait::pT<T1> >(dis(rdevs.rng)) +
             I * static_cast<trait::pT<T1> >(dis(rdevs.rng));
    });
  }
  return ret;
}

//****************************************************************************

template <
  typename T1 = arma::imat,
  typename TR = typename std::enable_if<is_arma_type_var<T1>::value &&
                                          arma::is_Mat_only<T1>::value,
                                        arma::Mat<trait::eT<T1> > >::type,
  typename TA = typename std::conditional<
    std::is_unsigned<trait::pT<T1> >::value, arma::uword, arma::sword>::type>
inline TR randI(const arma::uword& m, const arma::uword& n,
                const arma::Col<TA>& range = {0, 1000}) {
#ifndef QICLIB_NO_DEBUG
  if (range.n_elem != 2 || range.at(0) > range.at(1))
    throw Exception("qic::randI", "Not proper range!");

  if (std::is_unsigned<trait::pT<T1> >::value && arma::any(range < 0))
    throw Exception("qic::randI", "Negative range for unsigned type!");
#endif

  std::uniform_int_distribution<TA> dis(range.at(0), range.at(1));
  arma::Mat<trait::eT<T1> > ret(m, n);

  if (std::is_same<trait::eT<T1>, trait::pT<T1> >::value) {
    ret.imbue([&dis]() { return static_cast<trait::pT<T1> >(dis(rdevs.rng)); });

  } else {
    auto& I = _internal::cond_I<trait::eT<T1> >::value;
    ret.imbue([&dis, &I]() {
      return static_cast<trait::pT<T1> >(dis(rdevs.rng)) +
             I * static_cast<trait::pT<T1> >(dis(rdevs.rng));
    });
  }

  return ret;
}

//****************************************************************************

template <typename T1 = arma::cx_double,
          typename = typename std::enable_if<is_complex_fp<T1>::value>::type>
inline arma::Mat<T1> randHermitian(const arma::uword& m) {
  auto& I = _internal::cond_I<T1>::value;
  arma::Mat<T1> ret =
    2.0 * randU<arma::Mat<T1> >(m, m) -
    (static_cast<typename arma::get_pod_type<T1>::result>(1.0) + I) *
      arma::ones<arma::Mat<T1> >(m, m);

  return ret + ret.t();
}

//****************************************************************************

template <typename T1 = arma::cx_double,
          typename = typename std::enable_if<is_complex_fp<T1>::value>::type>
inline arma::Col<T1> randPsi(const arma::uword& m) {
  auto ret = randN<arma::Col<T1> >(m);
  return ret / arma::norm(ret);
}

//****************************************************************************

template <typename T1 = arma::cx_double,
          typename = typename std::enable_if<is_complex_fp<T1>::value>::type>
inline arma::Mat<T1> randRho(const arma::uword& m) {
  arma::Mat<T1> ret = 10.0 * randHermitian<T1>(m);
  ret *= ret.t();
  return ret / arma::trace(ret);
}

//****************************************************************************

template <typename T1 = arma::cx_double,
          typename = typename std::enable_if<is_complex_fp<T1>::value>::type>
inline arma::Mat<T1> randUnitary(const arma::uword& m) {
  arma::Mat<T1> A =
    randN<arma::Mat<T1> >(m, m) * std::sqrt(static_cast<T1>(0.5));

  arma::Mat<T1> Q, R;
  bool check = arma::qr(Q, R, A);

  if (!check)
    throw std::runtime_error("qic::randUnitary(): Decomposition failed!");

  arma::Col<T1> P = R.diag() / arma::abs(R.diag());

  return Q * arma::diagmat(P);
}

//****************************************************************************

inline arma::uvec randPerm(arma::uword n, arma::uword start = 0) {
#ifndef QICLIB_NO_DEBUG
  if (n == 0)
    throw Exception("qic::randPerm", Exception::type::INVALID_PERM);
#endif

  arma::uvec ret(n);
  std::iota(ret.begin(), ret.end(), start);
  std::shuffle(ret.begin(), ret.end(), rdevs.rng);

  return ret;
}

//****************************************************************************

}  // namespace qic
