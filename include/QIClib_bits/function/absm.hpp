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

#ifndef _QICLIB_ABSM_HPP_
#define _QICLIB_ABSM_HPP_

#include "../basic/type_traits.hpp"
#include "../internal/methods.hpp"
#include <armadillo>

namespace qic {

//******************************************************************************

template <typename T1, typename TR = typename std::enable_if<
                         is_floating_point_var<trait::pT<T1> >::value,
                         arma::Mat<trait::eT<T1> > >::type>

inline TR absm(const T1& rho1) {
  const auto& rho = _internal::as_Mat(rho1);
  return _internal::absm_implement(rho,
                                   typename _internal::absm_tag<T1>::type{});
}

//******************************************************************************

}  // namespace qic

#endif
