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

  template<typename T1, typename T2>
  inline
  typename std::enable_if< 
    std::is_arithmetic< pT<T1> >::value 
  && !std::is_integral<T2>::value,
    arma::Mat< std::complex< pT<T1> > > 
    >::type powm_gen(const T1& rho1 ,const T2& P)
    
  {
    const auto& rho = as_Mat(rho1);
    
#ifndef QIC_LIB_NO_DEBUG  
    if(rho.n_elem == 0)
      throw Exception("qic::powm_gen",Exception::type::ZERO_SIZE);
    
    if(rho.n_rows != rho.n_cols)
      throw Exception("qic::powm_gen",Exception::type::MATRIX_NOT_SQUARE);
#endif

    arma::Col<std::complex< pT<T1> > > eigval;
    arma::Mat<std::complex< pT<T1> > > eigvec;
    arma::eig_gen(eigval,eigvec,rho);

    return eigvec
      * diagmat(arma::pow(eigval,P))
      * eigvec.i();
      
  }



  template<typename T1, typename T2>
  inline
  typename std::enable_if< 
    std::is_arithmetic< pT<T1> >::value 
  && std::is_integral<T2>::value,
    arma::Mat< eT<T1> > 
    >::type powm_gen(const T1& rho1 ,const T2& P)
  
  {
    const auto& rho = as_Mat(rho1);

#ifndef QIC_LIB_NO_DEBUG  
    if(rho.n_elem == 0)
      throw Exception("qic::powm_gen",Exception::type::ZERO_SIZE);
    
    if(rho.n_rows != rho.n_cols)
      throw Exception("qic::powm_gen",Exception::type::MATRIX_NOT_SQUARE);
#endif
    
    if(P < 0)
      return _internal::protect_subs::POWM_GEN_INT(rho.i().eval(),
						   abs(P));
    else 
      return _internal::protect_subs::POWM_GEN_INT(rho,P); 
    
  }  



  template<typename T1, typename T2>
  typename std::enable_if< 
    std::is_arithmetic< pT<T1> >::value
  && !std::is_integral<T2>::value,
    arma::Mat<std::complex< pT<T1> > > 
    >::type powm_sym(const T1& rho1 ,const T2& P)
  
  {
    const auto& rho = as_Mat(rho1);


#ifndef QIC_LIB_NO_DEBUG    
    if(rho.n_elem == 0)
      throw Exception("qic::powm_sym",Exception::type::ZERO_SIZE);
    
    if(rho.n_rows!=rho.n_cols)
      throw Exception("qic::powm_sym",Exception::type::MATRIX_NOT_SQUARE);
#endif
   
    arma::Col< pT<T1> > eigval;
    arma::Mat< eT<T1> > eigvec;
    arma::eig_sym(eigval,eigvec,rho,"std");

    return eigvec 
      * arma::diagmat(arma::pow(arma::conv_to< 
				arma::Col< 
				std::complex< pT<T1> > 
				> >::from(eigval),P)) 
      * eigvec.t();
  }
  


  template<typename T1, typename T2>
  inline 
  typename std::enable_if< 
    std::is_arithmetic< pT<T1> >::value 
  && std::is_integral<T2>::value,
    arma::Mat< eT<T1> > 
    >::type powm_sym(const T1& rho1 ,const T2& P)
  {
    const auto& rho = as_Mat(rho1);

#ifndef QIC_LIB_NO_DEBUG  
    if(rho.n_elem == 0)
      throw Exception("qic::powm_sym",Exception::type::ZERO_SIZE);
    
    if(rho.n_rows != rho.n_cols)
      throw Exception("qic::powm_sym",Exception::type::MATRIX_NOT_SQUARE);
#endif
    return powm_gen(rho,P);      
  }



}
