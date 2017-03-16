/*
 * QIClib (Quantum information and computation library)
 *
 * Copyright (c) 2015 - 2017  Titas Chanda (titas.chanda@gmail.com)
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

//****************************************************************************

namespace trait {

//****************************************************************************

template <typename T1> struct void_type {
  using type = void;
  using pod_type = void;
  using elem_type = void;
};

//****************************************************************************

}  // namespace trait

//****************************************************************************

template <typename T1> struct conditional_arma {
  using type = typename std::conditional<arma::is_arma_type<T1>::value ||
                                           arma::is_arma_sparse_type<T1>::value,
                                         T1, trait::void_type<T1> >::type;
};

//****************************************************************************

namespace trait {

//****************************************************************************

template <typename T1> using RR = typename std::remove_reference<T1>::type;

template <typename T1> using RCV = typename std::remove_cv<T1>::type;

template <typename T1>
using pT2 = typename T1::pod_type;

template <typename T1>
using eT2 = typename T1::elem_type;

template <typename T1>
using GPT = typename arma::get_pod_type<T1>::result;

template <typename T1>
using pT = typename conditional_arma<trait::RR<T1> >::type::pod_type;

template <typename T1>
using eT = typename conditional_arma<trait::RR<T1> >::type::elem_type;

//****************************************************************************

}  // namespace trait

//****************************************************************************

template <typename T1, typename... T2>
struct is_arma_type_var : std::false_type {};

template <typename T1>
struct is_arma_type_var<T1> : arma::is_arma_type<trait::RR<T1> > {};

template <typename T1, typename T2, typename... T3>
struct is_arma_type_var<T1, T2, T3...>
  : std::integral_constant<bool, is_arma_type_var<T1>::value &&
                                   is_arma_type_var<T2, T3...>::value> {};

//****************************************************************************

template <typename T1, typename... T2>
struct is_arma_sparse_type_var : std::false_type {};

template <typename T1>
struct is_arma_sparse_type_var<T1> : arma::is_arma_sparse_type<trait::RR<T1> > {
};

template <typename T1, typename T2, typename... T3>
struct is_arma_sparse_type_var<T1, T2, T3...>
  : std::integral_constant<bool, is_arma_sparse_type_var<T1>::value &&
                                   is_arma_sparse_type_var<T2, T3...>::value> {
};

//****************************************************************************

template <typename T1, typename... T2>
struct is_floating_point_var : std::false_type {};

template <typename T1>
struct is_floating_point_var<T1> : std::is_floating_point<T1> {};

template <typename T1, typename T2, typename... T3>
struct is_floating_point_var<T1, T2, T3...>
  : std::integral_constant<bool, is_floating_point_var<T1>::value &&
                                   is_floating_point_var<T2, T3...>::value> {};

//****************************************************************************

template <typename T1> struct is_complex : std::false_type {};

template <typename T1> struct is_complex<std::complex<T1> > : std::true_type {};

template <typename T1>
struct is_complex<const std::complex<T1> > : std::true_type {};

template <typename T1>
struct is_complex<volatile std::complex<T1> > : std::true_type {};

template <typename T1>
struct is_complex<const volatile std::complex<T1> > : std::true_type {};

//****************************************************************************

template <typename T1, typename... T2>
struct is_complex_var : std::false_type {};

template <typename T1> struct is_complex_var<T1> : is_complex<T1> {};

template <typename T1, typename T2, typename... T3>
struct is_complex_var<T1, T2, T3...>
  : std::integral_constant<bool, is_complex<T1>::value &&
                                   is_complex_var<T2, T3...>::value> {};

//****************************************************************************

template <typename T1> struct is_complex_fp : std::false_type {};

template <typename T1>
struct is_complex_fp<std::complex<T1> >
  : std::integral_constant<bool, is_floating_point_var<T1>::value> {};

template <typename T1>
struct is_complex_fp<const std::complex<T1> >
  : std::integral_constant<bool, is_floating_point_var<T1>::value> {};

template <typename T1>
struct is_complex_fp<volatile std::complex<T1> >
  : std::integral_constant<bool, is_floating_point_var<T1>::value> {};

template <typename T1>
struct is_complex_fp<const volatile std::complex<T1> >
  : std::integral_constant<bool, is_floating_point_var<T1>::value> {};

//****************************************************************************

template <typename T1, typename... T2>
struct is_complex_fp_var : std::false_type {};

template <typename T1> struct is_complex_fp_var<T1> : is_complex_fp<T1> {};

template <typename T1, typename T2, typename... T3>
struct is_complex_fp_var<T1, T2, T3...>
  : std::integral_constant<bool, is_complex_fp<T1>::value &&
                                   is_complex_fp_var<T2, T3...>::value> {};

//****************************************************************************

template <typename T1, typename... T2>
struct is_fp_arma_type_var : std::false_type {};

template <typename T1>
struct is_fp_arma_type_var<T1> : std::is_floating_point<trait::pT<T1> > {};

template <typename T1, typename T2, typename... T3>
struct is_fp_arma_type_var<T1, T2, T3...>
  : std::integral_constant<bool, is_fp_arma_type_var<T1>::value &&
                                   is_fp_arma_type_var<T2, T3...>::value> {};

//****************************************************************************

template <typename T1, typename... T2> struct promote_var;

template <typename T1, typename T2> struct promote_var<T1, T2> {
  using type = typename arma::is_promotable<T1, T2>::result;
};

template <typename T1, typename T2, typename... T3>
struct promote_var<T1, T2, T3...> {
  using type =
    typename arma::is_promotable<T1,
                                 typename promote_var<T2, T3...>::type>::result;
};

//***************************************************************************

template <typename T1, typename... T2> struct eT_promoter_var;

template <typename T1, typename T2> struct eT_promoter_var<T1, T2> {
  using type = typename promote_var<trait::eT<T1>, trait::eT<T2> >::type;
};

template <typename T1, typename T2, typename... T3>
struct eT_promoter_var<T1, T2, T3...> {
  using type =
    typename promote_var<trait::eT<T1>,
                         typename eT_promoter_var<T2, T3...>::type>::type;
};

//****************************************************************************

template <typename T1, typename... T2> struct is_all_same : std::true_type {};

template <typename T1, typename T2>
struct is_all_same<T1, T2> : std::is_same<trait::RR<T1>, trait::RR<T2> > {};

template <typename T1, typename T2, typename... T3>
struct is_all_same<T1, T2, T3...>
  : std::integral_constant<bool,
                           std::is_same<trait::RR<T1>, trait::RR<T2> >::value &&
                             is_all_same<T1, T3...>::value> {};

//****************************************************************************

template <typename T1, typename... T2>
struct is_same_pT_var : std::false_type {};

template <typename T1>
struct is_same_pT_var<T1>
  : std::integral_constant<bool, arma::is_arma_type<T1>::value> {};

template <typename T1, typename T2>
struct is_same_pT_var<T1, T2>
  : std::integral_constant<
      bool, arma::is_arma_type<T1>::value && arma::is_arma_type<T2>::value &&
              std::is_same<trait::pT<T1>, trait::pT<T2> >::value> {};

template <typename T1, typename T2, typename... T3>
struct is_same_pT_var<T1, T2, T3...>
  : std::integral_constant<bool, is_same_pT_var<T1, T2>::value &&
                                   is_same_pT_var<T1, T3...>::value> {};

//*****************************************************************************

}  // namespace qic
