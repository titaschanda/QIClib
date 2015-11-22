/*
 * QIC_lib (Quantum information and computation library)
 *
 * Copyright (c) 2015 - 2016  Titas Chanda (titas.chanda@gmail.com)
 *
 * This file is part of QIC_lib.
 *
 * QIC_lib is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * QIC_lib is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QIC_lib.  If not, see <http://www.gnu.org/licenses/>.
 */



namespace qic
{ 

  //***********************************************************************


  template<typename T1>
  using pT = typename std::remove_reference<T1>::type::pod_type;

  template<typename T1>
  using eT = typename std::remove_reference<T1>::type::elem_type;

  template<typename T1>
  using RR = typename std::remove_reference<T1>::type;


  //***********************************************************************


  template<typename T1, typename... T2>
  struct is_promotable_var;


  template<typename T1, typename T2>
  struct is_promotable_var<T1,T2>
  {
    typedef typename arma::is_promotable <T1,T2>::result type;
  };

  template<typename T1, typename T2, typename... T3>
  struct is_promotable_var<T1,T2,T3...>
  {
    typedef typename arma::is_promotable< 
      T1, 
      typename is_promotable_var<T2,T3... 
				 >::type 
      >::result  type;
  };
  
  
  //***************************************************************************

  template<typename T1, typename... T2>
  struct eT_promoter_var;

  template<typename T1, typename T2>
  struct eT_promoter_var<T1,T2>
  {
    typedef typename arma::is_promotable
    < eT<T1>, eT<T2> >::result type;
  };

  template<typename T1, typename T2, typename... T3>
  struct eT_promoter_var<T1,T2,T3...>
  {
    typedef typename arma::is_promotable< 
      eT<T1>, 
      typename eT_promoter_var< T2 , T3... 
				>::type 
      >::result  type;
  };
  
  //****************************************************************************


  template<typename T1, typename... T2>
  struct is_all_same : std::true_type {};

  template<typename T1, typename T2>
  struct is_all_same<T1,T2> : std::is_same< RR<T1>, RR<T2> > {};

  template<typename T1, typename T2, typename... T3>
  struct is_all_same<T1, T2, T3...> : std::integral_constant
  < bool, 
    std::is_same< RR<T1>, RR<T2> >::value 
    && is_all_same<T1,T3...>
    ::value 
    > {};


  //****************************************************************************


  template<typename T1, typename... T2>
  struct is_comparable_pT : std::true_type {};


  template<typename T1, typename T2>
  struct is_comparable_pT<T1,T2> : std::integral_constant
  < bool,
    std::is_same< eT<T1>,eT<T2> >::value 
    || std::is_same< pT<T1>,eT<T2> >::value 
    || std::is_same< eT<T1>,pT<T2> >::value
    > {};
  
  template<typename T1, typename T2, typename... T3>
  struct is_comparable_pT<T1,T2,T3...> : std::integral_constant
  < bool,
    is_comparable_pT<T1,T2>::value
    && is_comparable_pT<T1,T3...>::value
    > {};



}
