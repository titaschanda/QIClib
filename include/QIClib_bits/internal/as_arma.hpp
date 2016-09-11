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

namespace _internal {

//***************************************************************************

template <typename T1>
inline const arma::Mat<T1>& as_Mat(const arma::Mat<T1>& A) noexcept {
  return A;
}

//***************************************************************************

template <typename T1>
inline const arma::Mat<T1>& as_Mat(const arma::Col<T1>& A) noexcept {
  return A;
}

//***************************************************************************

template <typename T1>
inline const arma::Mat<T1>& as_Mat(const arma::Row<T1>& A) noexcept {
  return A;
}

//****************************************************************************

template <typename T1>
inline arma::Mat<trait::eT<T1> >
as_Mat(const arma::Base<trait::eT<T1>, T1>& A) {
  return A.eval();
}

//***************************************************************************

template <typename T1>
inline const arma::Col<T1>& as_Col(const arma::Mat<T1>& V) noexcept {
  return static_cast<const arma::Col<T1>&>(V);
}

//***************************************************************************

template <typename T1>
inline const arma::Col<T1>& as_Col(const arma::Col<T1>& V) noexcept {
  return V;
}

//****************************************************************************

template <typename T1>
inline const arma::SpMat<T1>& as_SpMat(const arma::SpMat<T1>& A) noexcept {
  return A;
}

//***************************************************************************

template <typename T1>
inline const arma::SpMat<T1>& as_SpMat(const arma::SpCol<T1>& A) noexcept {
  return A;
}

//***************************************************************************

template <typename T1>
inline const arma::SpMat<T1>& as_SpMat(const arma::SpRow<T1>& A) noexcept {
  return A;
}

//****************************************************************************

template <typename T1>
inline arma::SpMat<trait::eT<T1> >
as_SpMat(const arma::SpBase<trait::eT<T1>, T1>& A) {
  return A.eval();
}

//***************************************************************************

} // namespace _internal

}  // namespace qic
