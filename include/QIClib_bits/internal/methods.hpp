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

#ifndef _QICLIB_INTERNAL_METHODS_HPP_
#define _QICLIB_INTERNAL_METHODS_HPP_

#include "../basic/type_traits.hpp"
#include "../class/exception.hpp"
#include "../internal/as_arma.hpp"
#include <armadillo>

namespace qic {

//******************************************************************************

namespace _internal {

//******************************************************************************

// More efficient than iterative counterpart, as iterative one needs
// extra copy constructor
template <typename T1, typename T2, typename TR = arma::Mat<trait::eT<T1> > >

inline TR POWM_GEN_INT(const T1& rho, const T2& P) {
  if (P == 0) {
    arma::uword n = rho.n_rows;
    return arma::eye<arma::Mat<trait::eT<T1> > >(n, n);

  } else if (P == 1) {
    return rho;

  } else if (P == 2) {
    return rho * rho;

  } else {
    if (P % 2 == 0)
      return POWM_GEN_INT(POWM_GEN_INT(rho, 2), P / 2);
    else
      return POWM_GEN_INT(POWM_GEN_INT(rho, 2), (P - 1) / 2) * rho;
  }
}

//******************************************************************************

template <typename T1> struct int_tag {
  using type = int_tag;
  using ret_type = arma::Mat<trait::eT<T1> >;
};

template <typename T1> struct uint_tag {
  using type = uint_tag;
  using ret_type = arma::Mat<trait::eT<T1> >;
};

template <typename T1> struct nonint_tag {
  using type = nonint_tag;
  using ret_type = arma::Mat<std::complex<trait::pT<T1> > >;
};

template <typename T1, typename T2>
struct powm_tag : std::conditional<
                    std::is_integral<T2>::value,
                    typename std::conditional<std::is_unsigned<T2>::value,
                                              uint_tag<T1>, int_tag<T1> >::type,
                    nonint_tag<T1> >::type {};

//******************************************************************************

template <typename T1, typename T2,
          typename TR = arma::Mat<std::complex<trait::pT<T1> > > >

inline TR powm_gen_implement(const T1& rho, const T2& P, nonint_tag<T1>) {
#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::powm_gen", Exception::type::ZERO_SIZE);

  if (rho.n_rows != rho.n_cols)
    throw Exception("qic::powm_gen", Exception::type::MATRIX_NOT_SQUARE);
#endif

  arma::Col<std::complex<trait::pT<T1> > > eigval;
  arma::Mat<std::complex<trait::pT<T1> > > eigvec;
  bool check = arma::eig_gen(eigval, eigvec, rho);
  if (!check)
    throw std::runtime_error("qic::powm_gen(): Decomposition failed!");

  return eigvec * diagmat(arma::pow(eigval, P)) * eigvec.i();
}

//******************************************************************************

template <typename T1, typename T2, typename TR = arma::Mat<trait::eT<T1> > >

inline TR powm_gen_implement(const T1& rho, const T2& P, int_tag<T1>) {

#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::powm_gen", Exception::type::ZERO_SIZE);

  if (rho.n_rows != rho.n_cols)
    throw Exception("qic::powm_gen", Exception::type::MATRIX_NOT_SQUARE);
#endif

  if (P < 0)
    return POWM_GEN_INT(rho.i().eval(), -P);
  else
    return POWM_GEN_INT(rho, P);
}

//******************************************************************************

template <typename T1, typename T2, typename TR = arma::Mat<trait::eT<T1> > >

inline TR powm_gen_implement(const T1& rho, const T2& P, uint_tag<T1>) {
#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::powm_gen", Exception::type::ZERO_SIZE);

  if (rho.n_rows != rho.n_cols)
    throw Exception("qic::powm_gen", Exception::type::MATRIX_NOT_SQUARE);
#endif

  return POWM_GEN_INT(rho, P);
}

//******************************************************************************

template <typename T1, typename T2,
          typename TR = arma::Mat<std::complex<trait::pT<T1> > > >

inline TR powm_sym_implement(const T1& rho, const T2& P, nonint_tag<T1>) {
#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::powm_sym", Exception::type::ZERO_SIZE);

  if (rho.n_rows != rho.n_cols)
    throw Exception("qic::powm_sym", Exception::type::MATRIX_NOT_SQUARE);
#endif

  arma::Col<trait::pT<T1> > eigval;
  arma::Mat<trait::eT<T1> > eigvec;

  if (rho.n_rows > 20) {
    bool check = arma::eig_sym(eigval, eigvec, rho, "dc");
    if (!check)
      throw std::runtime_error("qic::powm_sym(): Decomposition failed!");

  } else {
    bool check = arma::eig_sym(eigval, eigvec, rho, "std");
    if (!check)
      throw std::runtime_error("qic::powm_sym(): Decomposition failed!");
  }

  return eigvec *
         arma::diagmat(arma::pow(
           _internal::as_type<arma::Col<std::complex<trait::pT<T1> > > >::from(
             eigval),
           P)) *
         eigvec.t();
}

//******************************************************************************

template <typename T1, typename T2, typename TR = arma::Mat<trait::eT<T1> > >

inline TR powm_sym_implement(const T1& rho, const T2& P, int_tag<T1>) {
#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::powm_sym", Exception::type::ZERO_SIZE);

  if (rho.n_rows != rho.n_cols)
    throw Exception("qic::powm_sym", Exception::type::MATRIX_NOT_SQUARE);
#endif
  if (P < 0)
    return POWM_GEN_INT(rho.i().eval(), -P);
  else
    return POWM_GEN_INT(rho, P);
}

//******************************************************************************

template <typename T1, typename T2, typename TR = arma::Mat<trait::eT<T1> > >

inline TR powm_sym_implement(const T1& rho1, const T2& P, uint_tag<T1>) {
  const auto& rho = as_Mat(rho1);

#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::powm_sym", Exception::type::ZERO_SIZE);

  if (rho.n_rows != rho.n_cols)
    throw Exception("qic::powm_sym", Exception::type::MATRIX_NOT_SQUARE);
#endif

  return POWM_GEN_INT(rho, P);
}

//******************************************************************************

template <typename T1, typename TR = arma::Mat<trait::eT<T1> > >

inline TR TENSOR_POW(const T1& rho, arma::uword n) {
  if (n == 1) {
    return rho;

  } else if (n == 2) {
    return arma::kron(rho, rho);

  } else {
    if (n % 2 == 0)
      return TENSOR_POW(TENSOR_POW(rho, 2), n / 2);
    else
      return arma::kron(TENSOR_POW(TENSOR_POW(rho, 2), (n - 1) / 2), rho);
  }
}

//******************************************************************************

template <typename T1> struct real_tag { using type = real_tag; };

template <typename T1> struct cplx_tag { using type = cplx_tag; };

template <typename T1>
struct absm_tag
    : std::conditional<std::is_same<trait::eT<T1>, trait::pT<T1> >::value,
                       real_tag<T1>, cplx_tag<T1> >::type {};

//******************************************************************************

template <typename T1, typename TR = arma::Mat<trait::eT<T1> > >

inline TR absm_implement(const T1& rho, real_tag<T1>) {

#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::absm", Exception::type::ZERO_SIZE);

  if (rho.n_rows != rho.n_cols)
    throw Exception("qic::absm", Exception::type::MATRIX_NOT_SQUARE);
#endif

  return arma::real(sqrtm_sym((rho.t() * rho).eval()));
}

//******************************************************************************

template <typename T1, typename TR = arma::Mat<trait::eT<T1> > >

inline TR absm_implement(const T1& rho, cplx_tag<T1>) {

#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::absm", Exception::type::ZERO_SIZE);

  if (rho.n_rows != rho.n_cols)
    throw Exception("qic::absm", Exception::type::MATRIX_NOT_SQUARE);
#endif

  return sqrtm_sym((rho.t() * rho).eval());
}

//******************************************************************************

}  // namespace _internal

//******************************************************************************

}  // namespace qic

#endif
