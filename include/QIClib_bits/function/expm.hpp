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

template <typename T1, typename T2,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1> >::value,
            arma::Mat<std::complex<trait::pT<T1> > > >::type>
inline TR expm_sym(const T1& H1, const std::complex<T2>& a) {
  const auto& H = _internal::as_Mat(H1);

#ifndef QICLIB_NO_DEBUG
  if (H.n_elem == 0)
    throw Exception("qic::expm_sym", Exception::type::ZERO_SIZE);

  if (H.n_rows != H.n_cols)
    throw Exception("qic::expm_sym", Exception::type::MATRIX_NOT_SQUARE);
#endif

  arma::Col<trait::pT<T1> > eigval;
  arma::Mat<trait::eT<T1> > eigvec;

  if (H.n_rows > 20) {
    bool check = arma::eig_sym(eigval, eigvec, H, "dc");
    if (!check)
      throw std::runtime_error("qic::expm_sym(): Decomposition failed!");

  } else {
    bool check = arma::eig_sym(eigval, eigvec, H, "std");
    if (!check)
      throw std::runtime_error("qic::expm_sym(): Decomposition failed!");
  }

  return eigvec * arma::diagmat(arma::exp(a * eigval)) * eigvec.t();
}

//******************************************************************************

template <typename T1, typename T2,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1> >::value &&
              std::is_arithmetic<T2>::value,
            arma::Mat<trait::eT<T1> > >::type>
inline TR expm_sym(const T1& H1, const T2& a) {
  const auto& H = _internal::as_Mat(H1);

#ifndef QICLIB_NO_DEBUG
  if (H.n_elem == 0)
    throw Exception("qic::expm_sym", Exception::type::ZERO_SIZE);

  if (H.n_rows != H.n_cols)
    throw Exception("qic::expm_sym", Exception::type::MATRIX_NOT_SQUARE);
#endif

  arma::Col<trait::pT<T1> > eigval;
  arma::Mat<trait::eT<T1> > eigvec;

  if (H.n_rows > 20) {
    bool check = arma::eig_sym(eigval, eigvec, H, "dc");
    if (!check)
      throw std::runtime_error("qic::expm_sym(): Decomposition failed!");

  } else {
    bool check = arma::eig_sym(eigval, eigvec, H, "std");
    if (!check)
      throw std::runtime_error("qic::expm_sym(): Decomposition failed!");
  }

  return eigvec *
         arma::diagmat(arma::exp(
           a * _internal::as_type<arma::Col<trait::eT<T1> > >::from(eigval))) *
         eigvec.t();
}

//******************************************************************************

template <typename T1, typename TR = typename std::enable_if<
                         is_floating_point_var<trait::pT<T1> >::value,
                         arma::Mat<trait::eT<T1> > >::type>
inline TR expm_sym(const T1& H1) {
  const auto& H = _internal::as_Mat(H1);

#ifndef QICLIB_NO_DEBUG
  if (H.n_elem == 0)
    throw Exception("qic::expm_sym", Exception::type::ZERO_SIZE);

  if (H.n_rows != H.n_cols)
    throw Exception("qic::expm_sym", Exception::type::MATRIX_NOT_SQUARE);
#endif

  arma::Col<trait::pT<T1> > eigval;
  arma::Mat<trait::eT<T1> > eigvec;

  if (H.n_rows > 20) {
    bool check = arma::eig_sym(eigval, eigvec, H, "dc");
    if (!check)
      throw std::runtime_error("qic::expm_sym(): Decomposition failed!");

  } else {
    bool check = arma::eig_sym(eigval, eigvec, H, "std");
    if (!check)
      throw std::runtime_error("qic::expm_sym(): Decomposition failed!");
  }

  return eigvec *
         arma::diagmat(arma::exp(
           _internal::as_type<arma::Col<trait::eT<T1> > >::from(eigval))) *
         eigvec.t();
}

//******************************************************************************

template <typename T1, typename TR = typename std::enable_if<
                         is_floating_point_var<trait::pT<T1> >::value,
                         arma::Mat<trait::eT<T1> > >::type>
inline TR expm_gen(const T1& A1) {
  auto A = _internal::as_Mat(A1);

#ifndef QICLIB_NO_DEBUG
  if (A.n_elem == 0)
    throw Exception("qic::expm_gen", Exception::type::ZERO_SIZE);

  if (A.n_cols != A.n_rows)
    throw Exception("qic::expm_gen", Exception::type::MATRIX_NOT_SQUARE);
#endif

  const trait::pT<T1> norm_val = arma::norm(A, "inf");
  const double log2_val = (norm_val > static_cast<trait::pT<T1> >(0))
                            ? static_cast<double>(std::log2(norm_val))
                            : static_cast<double>(0);

  int exponent = static_cast<int>(0);
  std::frexp(log2_val, &exponent);
  const arma::uword s = static_cast<arma::uword>(
    std::max(static_cast<int>(0), exponent + static_cast<int>(1)));

  A /= (std::pow(2.0, static_cast<trait::pT<T1> >(s)));

  trait::pT<T1> c(0.5);

  arma::Mat<trait::eT<T1> > E(A.n_rows, A.n_cols, arma::fill::eye);
  E += c * A;
  arma::Mat<trait::eT<T1> > D(A.n_rows, A.n_cols, arma::fill::eye);
  D -= c * A;

  const arma::uword q(6);
  bool p(true);
  arma::Mat<trait::eT<T1> > X(A);

  for (arma::uword k = 2; k <= q; ++k) {
    c *= static_cast<trait::pT<T1> >(q - k + 1) /
         static_cast<trait::pT<T1> >(k * (2 * q - k + 1));
    X = A * X;
    E += c * X;

    if (p)
      D += c * X;
    else
      D -= c * X;

    p = !p;
  }

  E = arma::solve(D, E);

  for (arma::uword k = 1; k <= s; ++k) E *= E;

  return E;
}

//******************************************************************************

}  // namespace qic
