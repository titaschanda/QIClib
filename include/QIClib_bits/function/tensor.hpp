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
            is_same_pT_var<T1, T2>::value,
            arma::Mat<typename eT_promoter_var<T1, T2>::type> >::type>

inline TR tensor(const T1& rho11, const T2& rho12) {
  const auto& rho1 = _internal::as_Mat(rho11);
  const auto& rho2 = _internal::as_Mat(rho12);

#ifndef QICLIB_NO_DEBUG
  if (rho1.n_elem == 0 || rho2.n_elem == 0)
    throw Exception("qic::tensor", Exception::type::ZERO_SIZE);
#endif

  return arma::kron(rho1, rho2);
}

//******************************************************************************

template <typename T1, typename T2, typename... T3,
          typename TR = typename std::enable_if<
            is_same_pT_var<T1, T2, T3...>::value,
            arma::Mat<typename eT_promoter_var<T1, T2, T3...>::type> >::type>

inline TR tensor(const T1& rho1, const T2& rho2, const T3&... rho3) {
  return tensor(rho1, tensor(rho2, rho3...));
}

//******************************************************************************

template <typename T1, typename TR = arma::Mat<T1> >

inline TR tensor(const arma::field<arma::Mat<T1> >& rho) {
#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::tensor", Exception::type::ZERO_SIZE);

  for (const auto& a : rho)
    if (a.n_elem == 0)
      throw Exception("qic::tensor", Exception::type::ZERO_SIZE);
#endif

  auto ret = rho.eval();

  for (arma::uword i = 1; i < rho.n_elem; ++i) ret = tensor(ret, rho.at(i));

  return ret;
}

//******************************************************************************

template <typename T1, typename TR = arma::Mat<T1> >

inline TR tensor(const std::vector<arma::Mat<T1> >& rho) {
#ifndef QICLIB_NO_DEBUG
  if (rho.size() == 0)
    throw Exception("qic::tensor", Exception::type::ZERO_SIZE);

  for (const auto& a : rho)
    if (a.n_elem == 0)
      throw Exception("qic::tensor", Exception::type::ZERO_SIZE);
#endif

  auto ret = rho[0];

  for (arma::uword i = 1; i < rho.size(); ++i) ret = tensor(ret, rho[i]);

  return ret;
}

//******************************************************************************

template <typename T1>

inline typename arma::Mat<T1>
tensor(const std::initializer_list<arma::Mat<T1> >& rho) {
  return tensor(static_cast<std::vector<arma::Mat<T1> > >(rho));
}

//******************************************************************************

template <typename T1, typename TR = arma::Mat<trait::eT<T1> > >

inline TR tensor_pow(const T1& rho1, arma::uword n) {
  const auto& rho = _internal::as_Mat(rho1);

#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::tensor_pow", Exception::type::ZERO_SIZE);

  if (n == 0)
    throw Exception("qic::tensor_pow", Exception::type::OUT_OF_RANGE);
#endif

  return _internal::TENSOR_POW(rho, n);
}

//******************************************************************************

}  //  namespace qic
