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

template <typename T1,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1> >::value, trait::pT<T1> >::type>
inline TR entanglement(const T1& rho1, arma::uvec dim) {
  const auto& p = _internal::as_Mat(rho1);

#ifndef QICLIB_NO_DEBUG
  bool checkV = true;
  if (p.n_cols == 1)
    checkV = false;

  if (p.n_elem == 0)
    throw Exception("qic::entanglement", Exception::type::ZERO_SIZE);

  if (checkV)
    if (p.n_rows != p.n_cols)
      throw Exception("qic::entanglement",
                      Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);

  if (arma::any(dim) == 0)
    throw Exception("qic::entanglement", Exception::type::INVALID_DIMS);

  if (arma::prod(dim) != p.n_rows)
    throw Exception("qic::entanglement", Exception::type::DIMS_MISMATCH_MATRIX);

  if ((dim.n_elem) != 2)
    throw Exception("qic::entanglement", Exception::type::NOT_BIPARTITE);
#endif

  return entropy(TrX(p, {1}, std::move(dim)));
}

//******************************************************************************

}  // namespace qic
