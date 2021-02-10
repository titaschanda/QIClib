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

#ifndef _QICLIB_POWM_HPP_
#define _QICLIB_POWM_HPP_

#include "../basic/type_traits.hpp"
#include "../class/exception.hpp"
#include "../internal/as_arma.hpp"
#include "../internal/methods.hpp"
#include <armadillo>

namespace qic {

//******************************************************************************

template <typename T1, typename T2,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1> >::value ||
              (is_arma_type_var<T1>::value && std::is_unsigned<T2>::value &&
               std::is_integral<T2>::value),
            typename _internal::powm_tag<T1, T2>::ret_type>::type>

inline TR powm_gen(const T1& rho1, const T2& P) {
  const auto& rho = _internal::as_Mat(rho1);
  return _internal::powm_gen_implement(
    rho, P, typename _internal::powm_tag<T1, T2>::type{});
}

//******************************************************************************

template <typename T1, typename T2,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1> >::value ||
              (is_arma_type_var<T1>::value && std::is_unsigned<T2>::value &&
               std::is_integral<T2>::value),
            typename _internal::powm_tag<T1, T2>::ret_type>::type>

inline TR powm_sym(const T1& rho1, const T2& P) {
  const auto& rho = _internal::as_Mat(rho1);
  return _internal::powm_sym_implement(
    rho, P, typename _internal::powm_tag<T1, T2>::type{});
}

//******************************************************************************

}  // namespace qic

#endif
