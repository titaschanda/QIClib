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

//***************************************************************************

template <typename T1>
inline const arma::Mat<T1>& as_Mat(const arma::Mat<T1>& m) noexcept {
  return m;
}

//***************************************************************************

template <typename T1>
inline const arma::Mat<T1>& as_Mat(const arma::Col<T1>& m) noexcept {
  return m;
}

//***************************************************************************

template <typename T1>
inline const arma::Mat<T1>& as_Mat(const arma::Row<T1>& m) noexcept {
  return m;
}

//****************************************************************************

template <typename T1>
inline arma::Mat<trait::eT<T1> >
as_Mat(const arma::Base<trait::eT<T1>, T1>& X) {
  return X.eval();
}

}  // namespace qic
