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

template <typename T1, typename TR = typename std::enable_if<
                         is_floating_point_var<trait::pT<T1> >::value,
                         arma::Col<trait::pT<T1> > >::type>
inline TR schmidt(const T1& rho1, const arma::uvec& dim) {
  const auto& rho = _internal::as_Mat(rho1);

  bool checkV = true;
  if (rho.n_cols == 1)
    checkV = false;

#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::schmidt", Exception::type::ZERO_SIZE);

  if (checkV)
    if (rho.n_rows != rho.n_cols)
      throw Exception("qic::schmidt",
                      Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);

  if (arma::any(dim == 0))
    throw Exception("qic::schmidt", Exception::type::INVALID_DIMS);

  if (arma::prod(dim) != rho.n_rows)
    throw Exception("qic::schmidt", Exception::type::DIMS_MISMATCH_MATRIX);

  if (dim.n_elem != 2)
    throw Exception("qic::schmidt", Exception::type::NOT_BIPARTITE);
#endif

  if (checkV)
    return arma::svd(
      arma::reshape(conv_to_pure(rho), dim.at(1), dim.at(0)).st());
  else
    return arma::svd(arma::reshape(rho, dim.at(1), dim.at(0)).st());
}

//******************************************************************************

template <typename T1,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1> >::value, void>::type>
inline bool schmidt(const T1& rho1, const arma::uvec& dim,
                    arma::Col<trait::pT<T1> >& S, arma::Mat<trait::eT<T1> >& U,
                    arma::Mat<trait::eT<T1> >& V) {
  const auto& rho = _internal::as_Mat(rho1);

  bool checkV = true;
  if (rho.n_cols == 1)
    checkV = false;

#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    return false;

  if (checkV)
    if (rho.n_rows != rho.n_cols)
      return false;

  if (arma::any(dim == 0))
    return false;

  if (arma::prod(dim) != rho.n_rows)
    return false;

  if ((dim.n_elem) != 2)
    return false;
#endif

  if (checkV) {
    bool ret = arma::svd_econ(
      U, S, V, arma::reshape(conv_to_pure(rho), dim.at(1), dim.at(0)).st(),
      "both", "std");

    if (ret == true)
      V = arma::conj(V);
    return (ret);

  } else {
    bool ret = arma::svd_econ(
      U, S, V, arma::reshape(rho, dim.at(1), dim.at(0)).st(), "both", "std");

    if (ret == true)
      V = arma::conj(V);
    return (ret);
  }
}

//******************************************************************************

template <typename T1,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1> >::value, void>::type>
inline bool schmidt_full(const T1& rho1, const arma::uvec& dim,
                         arma::Col<trait::pT<T1> >& S,
                         arma::Mat<trait::eT<T1> >& U,
                         arma::Mat<trait::eT<T1> >& V) {
  const auto& rho = _internal::as_Mat(rho1);

  bool checkV = true;
  if (rho.n_cols == 1)
    checkV = false;

#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    return false;

  if (checkV)
    if (rho.n_rows != rho.n_cols)
      return false;

  if (arma::any(dim == 0))
    return false;

  if (arma::prod(dim) != rho.n_rows)
    return false;

  if ((dim.n_elem) != 2)
    return false;
#endif

  if (checkV) {
    bool ret = arma::svd(
      U, S, V, arma::reshape(conv_to_pure(rho), dim.at(1), dim.at(0)).st(),
      "both", "std");

    if (ret == true)
      V = arma::conj(V);
    return (ret);

  } else {
    bool ret = arma::svd(
      U, S, V, arma::reshape(rho, dim.at(1), dim.at(0)).st(), "both", "std");

    if (ret == true)
      V = arma::conj(V);
    return (ret);
  }
}

//******************************************************************************

template <typename T1, typename TR = typename std::enable_if<
                         is_floating_point_var<trait::pT<T1> >::value,
                         arma::Mat<trait::eT<T1> > >::type>
inline TR schmidtA(const T1& rho1, const arma::uvec& dim) {
  const auto& rho = _internal::as_Mat(rho1);

  bool checkV = true;
  if (rho.n_cols == 1)
    checkV = false;

#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::schmidtA", Exception::type::ZERO_SIZE);

  if (checkV)
    if (rho.n_rows != rho.n_cols)
      throw Exception("qic::schmidtA",
                      Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);

  if (arma::any(dim == 0))
    throw Exception("qic::schmidtA", Exception::type::INVALID_DIMS);

  if (arma::prod(dim) != rho.n_rows)
    throw Exception("qic::schmidtA", Exception::type::DIMS_MISMATCH_MATRIX);

  if ((dim.n_elem) != 2)
    throw Exception("qic::schmidtA", Exception::type::NOT_BIPARTITE);
#endif

  arma::Mat<trait::eT<T1> > U, V;
  arma::Col<trait::pT<T1> > S;

  if (checkV) {
    bool check = arma::svd_econ(
      U, S, V, arma::reshape(conv_to_pure(rho), dim.at(1), dim.at(0)).st(),
      "left", "std");
    if (!check)
      throw std::runtime_error("qic::schmidtA(): Decomposition failed!");
    return U;

  } else {
    bool check = arma::svd_econ(
      U, S, V, arma::reshape(rho, dim.at(1), dim.at(0)).st(), "left", "std");
    if (!check)
      throw std::runtime_error("qic::schmidtA(): Decomposition failed!");
    return U;
  }
}

//******************************************************************************

template <typename T1, typename TR = typename std::enable_if<
                         is_floating_point_var<trait::pT<T1> >::value,
                         arma::Mat<trait::eT<T1> > >::type>
inline TR schmidtB(const T1& rho1, const arma::uvec& dim) {
  const auto& rho = _internal::as_Mat(rho1);

  bool checkV = true;
  if (rho.n_cols == 1)
    checkV = false;

#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::schmidtB", Exception::type::ZERO_SIZE);

  if (checkV)
    if (rho.n_rows != rho.n_cols)
      throw Exception("qic::schmidtB",
                      Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);

  if (arma::any(dim == 0))
    throw Exception("qic::schmidtB", Exception::type::INVALID_DIMS);

  if (arma::prod(dim) != rho.n_rows)
    throw Exception("qic::schmidtB", Exception::type::DIMS_MISMATCH_MATRIX);

  if ((dim.n_elem) != 2)
    throw Exception("qic::schmidtB", Exception::type::NOT_BIPARTITE);
#endif

  arma::Mat<trait::eT<T1> > U, V;
  arma::Col<trait::pT<T1> > S;

  if (checkV) {
    bool check = arma::svd_econ(
      U, S, V, arma::reshape(conv_to_pure(rho), dim.at(1), dim.at(0)).st(),
      "right", "std");
    if (!check)
      throw std::runtime_error("qic::schmidtB(): Decomposition failed!");
    return arma::conj(V);

  } else {
    bool check = arma::svd(
      U, S, V, arma::reshape(rho, dim.at(1), dim.at(0)).st(), "right", "std");
    if (!check)
      throw std::runtime_error("qic::schmidtB(): Decomposition failed!");
    return arma::conj(V);
  }
}

//******************************************************************************

template <typename T1, typename TR = typename std::enable_if<
                         is_floating_point_var<trait::pT<T1> >::value,
                         arma::field<arma::Mat<trait::eT<T1> > > >::type>
inline TR schmidtAB(const T1& rho1, const arma::uvec& dim) {
  const auto& rho = _internal::as_Mat(rho1);

  bool checkV = true;
  if (rho.n_cols == 1)
    checkV = false;

#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::schmidtAB", Exception::type::ZERO_SIZE);

  if (checkV)
    if (rho.n_rows != rho.n_cols)
      throw Exception("qic::schmidtAB",
                      Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);

  if (arma::any(dim == 0))
    throw Exception("qic::schmidtAB", Exception::type::INVALID_DIMS);

  if (arma::prod(dim) != rho.n_rows)
    throw Exception("qic::schmidtAB", Exception::type::DIMS_MISMATCH_MATRIX);

  if ((dim.n_elem) != 2)
    throw Exception("qic::schmidtAB", Exception::type::NOT_BIPARTITE);
#endif

  arma::Mat<trait::eT<T1> > U, V;
  arma::Col<trait::pT<T1> > S;
  arma::field<arma::Mat<trait::eT<T1> > > ret(2);

  if (checkV) {
    bool check = arma::svd_econ(
      U, S, V, arma::reshape(conv_to_pure(rho), dim.at(1), dim.at(0)).st(),
      "both", "std");
    if (!check)
      throw std::runtime_error("qic::schmidtAB(): Decomposition failed!");
    ret.at(0) = std::move(U);
    ret.at(1) = std::move(arma::conj(V));
    return ret;

  } else {
    bool check = arma::svd_econ(
      U, S, V, arma::reshape(rho, dim.at(1), dim.at(0)).st(), "both", "std");
    if (!check)
      throw std::runtime_error("qic::schmidtAB(): Decomposition failed!");
    ret.at(0) = std::move(U);
    ret.at(1) = std::move(arma::conj(V));
    return ret;
  }
}

//******************************************************************************

template <typename T1, typename TR = typename std::enable_if<
                         is_floating_point_var<trait::pT<T1> >::value,
                         arma::Mat<trait::eT<T1> > >::type>
inline TR schmidtA_full(const T1& rho1, const arma::uvec& dim) {
  const auto& rho = _internal::as_Mat(rho1);

  bool checkV = true;
  if (rho.n_cols == 1)
    checkV = false;

#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::schmidtA_full", Exception::type::ZERO_SIZE);

  if (checkV)
    if (rho.n_rows != rho.n_cols)
      throw Exception("qic::schmidtA_full",
                      Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);

  if (arma::any(dim == 0))
    throw Exception("qic::schmidtA_full", Exception::type::INVALID_DIMS);

  if (arma::prod(dim) != rho.n_rows)
    throw Exception("qic::schmidtA_full",
                    Exception::type::DIMS_MISMATCH_MATRIX);

  if ((dim.n_elem) != 2)
    throw Exception("qic::schmidtA_full", Exception::type::NOT_BIPARTITE);
#endif

  arma::Mat<trait::eT<T1> > U, V;
  arma::Col<trait::pT<T1> > S;

  if (checkV) {
    bool check = arma::svd(
      U, S, V, arma::reshape(conv_to_pure(rho), dim.at(1), dim.at(0)).st(),
      "std");
    if (!check)
      throw std::runtime_error("qic::schmidtA_full(): Decomposition failed!");
    return U;

  } else {
    bool check =
      arma::svd(U, S, V, arma::reshape(rho, dim.at(1), dim.at(0)).st(), "std");
    if (!check)
      throw std::runtime_error("qic::schmidtA_full(): Decomposition failed!");
    return U;
  }
}

//******************************************************************************

template <typename T1, typename TR = typename std::enable_if<
                         is_floating_point_var<trait::pT<T1> >::value,
                         arma::Mat<trait::eT<T1> > >::type>
inline TR schmidtB_full(const T1& rho1, const arma::uvec& dim) {
  const auto& rho = _internal::as_Mat(rho1);

  bool checkV = true;
  if (rho.n_cols == 1)
    checkV = false;

#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::schmidtB_full", Exception::type::ZERO_SIZE);

  if (checkV)
    if (rho.n_rows != rho.n_cols)
      throw Exception("qic::schmidtB_full",
                      Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);

  if (arma::any(dim == 0))
    throw Exception("qic::schmidtB_full", Exception::type::INVALID_DIMS);

  if (arma::prod(dim) != rho.n_rows)
    throw Exception("qic::schmidtB_full",
                    Exception::type::DIMS_MISMATCH_MATRIX);

  if ((dim.n_elem) != 2)
    throw Exception("qic::schmidtB_full", Exception::type::NOT_BIPARTITE);
#endif

  arma::Mat<trait::eT<T1> > U, V;
  arma::Col<trait::pT<T1> > S;

  if (checkV) {
    bool check = arma::svd(
      U, S, V, arma::reshape(conv_to_pure(rho), dim.at(1), dim.at(0)).st(),
      "std");
    if (!check)
      throw std::runtime_error("qic::schmidtB_full(): Decomposition failed!");
    return arma::conj(V);

  } else {
    bool check =
      arma::svd(U, S, V, arma::reshape(rho, dim.at(1), dim.at(0)).st(), "std");
    if (!check)
      throw std::runtime_error("qic::schmidtB_full(): Decomposition failed!");
    return arma::conj(V);
  }
}

//******************************************************************************

template <typename T1, typename TR = typename std::enable_if<
                         is_floating_point_var<trait::pT<T1> >::value,
                         arma::field<arma::Mat<trait::eT<T1> > > >::type>
inline TR schmidtAB_full(const T1& rho1, const arma::uvec& dim) {
  const auto& rho = _internal::as_Mat(rho1);

  bool checkV = true;
  if (rho.n_cols == 1)
    checkV = false;

#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::schmidtAB_full", Exception::type::ZERO_SIZE);

  if (checkV)
    if (rho.n_rows != rho.n_cols)
      throw Exception("qic::schmidtAB_full",
                      Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);

  if (arma::any(dim == 0))
    throw Exception("qic::schmidtAB_full", Exception::type::INVALID_DIMS);

  if (arma::prod(dim) != rho.n_rows)
    throw Exception("qic::schmidtAB_full",
                    Exception::type::DIMS_MISMATCH_MATRIX);

  if ((dim.n_elem) != 2)
    throw Exception("qic::schmidtAB_full", Exception::type::NOT_BIPARTITE);
#endif

  arma::Mat<trait::eT<T1> > U, V;
  arma::Col<trait::pT<T1> > S;
  arma::field<arma::Mat<trait::eT<T1> > > ret(2);

  if (checkV) {
    bool check = arma::svd(
      U, S, V, arma::reshape(conv_to_pure(rho), dim.at(1), dim.at(0)).st(),
      "std");
    if (!check)
      throw std::runtime_error("qic::schmidtAB_full(): Decomposition failed!");
    ret.at(0) = std::move(U);
    ret.at(1) = std::move(arma::conj(V));
    return ret;

  } else {
    bool check =
      arma::svd(U, S, V, arma::reshape(rho, dim.at(1), dim.at(0)).st(), "std");
    if (!check)
      throw std::runtime_error("qic::schmidtAB_full(): Decomposition failed!");
    ret.at(0) = std::move(U);
    ret.at(1) = std::move(arma::conj(V));
    return ret;
  }
}

//******************************************************************************

}  // namespace qic
