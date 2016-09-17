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

namespace debug {

//******************************************************************************

template <typename T1, typename TR = typename std::enable_if<
                         is_floating_point_var<trait::pT<T1> >::value,
                         arma::Mat<trait::eT<T1> > >::type>
inline TR gram_schmidt_old(const T1& rho1, bool normalize = true) {
  const auto& rho = _internal::as_Mat(rho1);

#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::gram_schmidt", Exception::type::ZERO_SIZE);

  if (rho.n_cols > rho.n_rows)
    throw Exception("qic::gram_schmidt", "Invalid number of columns!");
#endif

  arma::Mat<trait::eT<T1> > ret(rho.n_rows, rho.n_cols);
  arma::Mat<trait::eT<T1> > prj =
    arma::eye<arma::Mat<trait::eT<T1> > >(rho.n_rows, rho.n_rows);

  arma::uword pos(0);
  for (; pos < rho.n_cols; ++pos) {
    auto norm1 = arma::norm(rho.col(pos));
    if (norm1 > _precision::eps<trait::pT<T1> >::value) {
      ret.col(0) =
        normalize ? (rho.col(pos) / norm1).eval() : rho.col(pos).eval();
      prj -=
        normalize
          ? (ret.col(0) * ret.col(0).t()).eval()
          : (ret.col(0) * ret.col(0).t() / (norm1 * norm1)).eval();  // check
      break;
    }
  }

  arma::uword count(1);
  for (arma::uword i = pos + 1; i < rho.n_cols; ++i) {

    arma::Col<trait::eT<T1> > entry = prj * rho.col(i);
    auto norm1 = arma::norm(entry);
    if (norm1 > _precision::eps<trait::pT<T1> >::value) {

      if (normalize == true) {
        ret.col(count) = entry / norm1;
        prj -= ret.col(count) * ret.col(count).t();
      }

      else {
        ret.col(count) = std::move(entry);
        prj -= ret.col(count) * ret.col(count).t() / (norm1 * norm1);
      }

      ++count;
    }
  }

  if (count != ret.n_cols)
    ret.shed_cols(count, ret.n_cols - 1);

  return ret;
}

//******************************************************************************

template <typename T1, typename TR = typename std::enable_if<
                         is_floating_point_var<trait::pT<T1> >::value,
                         arma::Mat<trait::eT<T1> > >::type>
inline TR gram_schmidt_old(const std::vector<T1>& rho) {
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

  arma::Mat<trait::eT<T1> > ret(rho[0].eval().n_rows, rho.size());
  arma::Mat<trait::eT<T1> > prj =
    arma::eye<arma::Mat<trait::eT<T1> > >(ret.n_rows, ret.n_rows);

  arma::uword pos(0);
  for (; pos < rho.size(); ++pos) {
    auto norm1 = arma::norm(rho[pos]);
    if (norm1 > _precision::eps<trait::pT<T1> >::value) {
      ret.col(0) = rho[pos] / norm1;
      break;
    }
  }

  prj -= ret.col(0) * ret.col(0).t();
  arma::uword count(1);
  for (arma::uword i = pos + 1; i < rho.size(); ++i) {
    arma::Col<trait::eT<T1> > entry = prj * rho[i];
    auto norm1 = arma::norm(entry);
    if (norm1 > _precision::eps<trait::pT<T1> >::value) {
      ret.col(count) = entry / norm1;
      prj -= ret.col(count) * ret.col(count).t();
      ++count;
    }
  }

  if (count != ret.n_cols)
    ret.shed_cols(count, ret.n_cols - 1);

  return ret;
}

//******************************************************************************

template <
  typename T1,
  typename TR = typename std::enable_if<
    is_floating_point_var<typename arma::get_pod_type<T1>::result>::value,
    arma::Mat<T1> >::type>
inline TR gram_schmidt_old(const std::initializer_list<arma::Mat<T1> >& rho) {
  return gram_schmidt(static_cast<std::vector<arma::Mat<T1> > >(rho));
}

//******************************************************************************

template <typename T1, typename TR = typename std::enable_if<
                         is_floating_point_var<trait::pT<T1> >::value,
                         arma::Mat<trait::eT<T1> > >::type>
inline TR gram_schmidt_old(const arma::field<T1>& rho) {
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

  arma::Mat<trait::eT<T1> > ret(rho.at(0).eval().n_rows, rho.n_elem);
  arma::Mat<trait::eT<T1> > prj =
    arma::eye<arma::Mat<trait::eT<T1> > >(ret.n_rows, ret.n_rows);

  arma::uword pos(0);
  for (; pos < rho.n_elem; ++pos) {
    auto norm1 = arma::norm(rho.at(pos));
    if (norm1 > _precision::eps<trait::pT<T1> >::value) {
      ret.col(0) = rho.at(pos) / norm1;
      break;
    }
  }

  prj -= ret.col(0) * ret.col(0).t();
  arma::uword count(1);
  for (arma::uword i = pos + 1; i < rho.n_elem; ++i) {
    arma::Col<trait::eT<T1> > entry = prj * rho.at(i);
    auto norm1 = arma::norm(entry);
    if (norm1 > _precision::eps<trait::pT<T1> >::value) {
      ret.col(count) = entry / norm1;
      prj -= ret.col(count) * ret.col(count).t();
      ++count;
    }
  }

  if (count != ret.n_cols)
    ret.shed_cols(count, ret.n_cols - 1);

  return ret;
}

//******************************************************************************

}  // namespace debug

}  // namespace qic
