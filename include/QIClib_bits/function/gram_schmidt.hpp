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
                         arma::Mat<trait::eT<T1> > >::type>
inline TR gram_schmidt(const T1& rho1, bool normalize = true) {
  const auto& rho = _internal::as_Mat(rho1);

#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::gram_schmidt", Exception::type::ZERO_SIZE);

  if (rho.n_cols > rho.n_rows)
    throw Exception("qic::gram_schmidt", "Invalid number of columns!");
#endif

  TR ret = rho;
  TR ret2(rho.n_rows, rho.n_cols);
  arma::uword count(0);

  for (arma::uword i = 0; i < rho.n_cols; ++i) {
    trait::pT<T1> norm1 = arma::norm(ret.col(i));

    if (norm1 > _precision::eps<trait::pT<T1> >::value) {
      ret2.col(count) =
        normalize ? (ret.col(i) / norm1).eval() : ret.col(i).eval();
      ++count;
    } else
      continue;

    for (arma::uword j = i + 1; j < rho.n_cols; ++j) {
      trait::eT<T1> r =
        normalize ? cdot(ret2.col(count - 1), ret.col(j))
                  : cdot(ret2.col(count - 1), ret.col(j)) / (norm1 * norm1);
      ret.col(j) -= r * ret2.col(count - 1);
    }
  }

  return ret2.cols(0, count - 1);
}

//******************************************************************************

template <typename T1, typename TR = typename std::enable_if<
                         is_floating_point_var<trait::pT<T1> >::value,
                         std::vector<arma::Col<trait::eT<T1> > > >::type>
inline TR gram_schmidt(const std::vector<T1>& rho, bool normalize = true) {
#ifndef QICLIB_NO_DEBUG
  if (rho.size() == 0)
    throw Exception("qic::gram_schmidt", Exception::type::ZERO_SIZE);

  for (auto&& ii : rho)
    if (ii.eval().n_elem == 0)
      throw Exception("qic::gram_schmidt", Exception::type::ZERO_SIZE);

  if (rho[0].eval().n_cols != 1)
    throw Exception("qic::gram_schmidt", Exception::type::MATRIX_NOT_CVECTOR);

  for (auto&& ii : rho)
    if (ii.eval().n_rows != rho[0].eval().n_rows || ii.eval().n_cols != 1)
      throw Exception("qic::gram_schmidt", Exception::type::DIMS_NOT_EQUAL);

  if (rho[0].eval().n_rows < rho.size())
    throw Exception("qic::gram_schmidt", "Invalid number of column vectors!");
#endif

  TR ret = rho;
  TR ret2(rho.size());
  arma::uword count(0);

  for (arma::uword i = 0; i < rho.size(); ++i) {
    trait::pT<T1> norm1 = arma::norm(ret[i]);

    if (norm1 > _precision::eps<trait::pT<T1> >::value) {
      ret2[count] =
        normalize ? (ret[i] / norm1).eval() : ret[i].eval();
      ++count;
    } else
      continue;

    for (arma::uword j = i + 1; j < rho.size(); ++j) {
      trait::eT<T1> r =
        normalize ? cdot(ret2[count - 1], ret[j])
                  : cdot(ret2[count - 1], ret[j]) / (norm1 * norm1);
      ret[j] -= r * ret2[count - 1];
    }
  }

  auto first = ret2.begin();
  auto last = first + count;
  
  return TR(first, last);
}

//******************************************************************************

template <
  typename T1,
  typename TR = typename std::enable_if<
    is_floating_point_var<typename arma::get_pod_type<T1>::result>::value,
    arma::Mat<T1> >::type>
inline TR gram_schmidt(const std::initializer_list<arma::Mat<T1> >& rho,
                       bool normalize = true) {
  return gram_schmidt(static_cast<std::vector<arma::Mat<T1> > >(rho),
                      normalize);
}

//******************************************************************************

template <typename T1, typename TR = typename std::enable_if<
                         is_floating_point_var<trait::pT<T1> >::value,
                         arma::field<arma::Col<trait::eT<T1> > > >::type>
inline TR gram_schmidt(const arma::field<T1>& rho, bool normalize = true) {
#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::gram_schmidt", Exception::type::ZERO_SIZE);

  for (auto&& ii : rho)
    if (ii.eval().n_elem == 0)
      throw Exception("qic::gram_schmidt", Exception::type::ZERO_SIZE);

  if (rho.at(0).eval().n_cols != 1)
    throw Exception("qic::gram_schmidt", Exception::type::MATRIX_NOT_CVECTOR);

  for (auto&& ii : rho)
    if (ii.eval().n_rows != rho.at(0).eval().n_rows || ii.eval().n_cols != 1)
      throw Exception("qic::gram_schmidt", Exception::type::DIMS_NOT_EQUAL);

  if (rho.at(0).eval().n_rows < rho.n_elem)
    throw Exception("qic::gram_schmidt", "Invalid number of column vectors!");
#endif

  TR ret(rho);
  TR ret2(ret.n_elem);
  arma::uword count(0);

  for (arma::uword i = 0; i < ret.n_elem; ++i) {
    trait::pT<T1> norm1 = arma::norm(ret.at(i));

    if (norm1 > _precision::eps<trait::pT<T1> >::value) {
      ret2.at(count) =
        normalize ? (ret.at(i) / norm1).eval() : ret.at(i).eval();
      ++count;
    } else
      continue;

    for (arma::uword j = i + 1; j < ret.n_elem; ++j) {
      trait::eT<T1> r =
        normalize ? cdot(ret2.at(count - 1), ret.at(j))
                  : cdot(ret2.at(count - 1), ret.at(j)) / (norm1 * norm1);
      ret.at(j) -= r * ret2.at(count - 1);
    }
  }

  return ret2.rows(0,count-1);
}

//******************************************************************************

}  // namespace qic
