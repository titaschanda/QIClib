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

  template<typename T1, typename T2, typename TR = 
	   typename std::enable_if< is_floating_point_var< pT<T1> >::value, 
				    typename _internal::protect_subs::
				    powm_tag<T1,T2>::ret_type
				    >::type>
  inline 
  TR powm_gen(const T1& rho1 ,const T2& P)
  {
    return _internal::protect_subs::
      powm_gen_implement(rho1,P,
			 typename _internal::
			 protect_subs::powm_tag<T1,T2>::type{});
  }





  template<typename T1, typename T2, typename TR = 
	   typename std::enable_if< is_floating_point_var< pT<T1> >::value, 
				    typename _internal::protect_subs::
				    powm_tag<T1,T2>::ret_type
				    >::type>
  inline 
  TR powm_sym(const T1& rho1 ,const T2& P)
  {
    return _internal::protect_subs::
      powm_sym_implement(rho1,P,
			 typename _internal::
			 protect_subs::powm_tag<T1,T2>::type{});
  }




  template<typename T1, typename T2, typename TR = 
	   typename std::enable_if< is_arma_type_var<T1>::value
				    && std::is_unsigned<T2>::value
				    && std::is_integral<T2>::value,
				    arma::Mat< eT<T1> >
				    >::type>
  inline 
  TR powm_uword(const T1& rho1 ,const T2& P)
  {
    const auto& rho = as_Mat(rho1);
    
#ifndef QIC_LIB_NO_DEBUG  
    if(rho.n_elem == 0)
      throw Exception("qic::powm_uword",Exception::type::ZERO_SIZE);
    
    if(rho.n_rows != rho.n_cols)
      throw Exception("qic::powm_uword",Exception::type::MATRIX_NOT_SQUARE);
#endif
    return _internal::protect_subs::POWM_GEN_INT(rho,P); 
  }


}
