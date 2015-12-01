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

  template<typename T1, typename functor, typename TR = 
	   typename std::enable_if< is_floating_point_var< pT<T1> >::value,
				    arma::Mat< std::complex< pT<T1> > >
				    >::type >
  inline 
  TR funcm_sym(const T1& rho1, 
	       functor P)
  {
    const auto& rho = as_Mat(rho1);

#ifndef QIC_LIB_NO_DEBUG    
    if(rho.n_elem == 0)
      throw Exception("qic::funcm_sym",Exception::type::ZERO_SIZE);
    
    if(rho.n_rows!=rho.n_cols)
      throw Exception("qic::funcm_sym",Exception::type::MATRIX_NOT_SQUARE);
#endif

       
    arma::Col< pT<T1> > eigval;
    arma::Mat< eT<T1> > eigvec;
    
    if(rho.n_rows > 20)
      arma::eig_sym(eigval,eigvec,rho,"dc");
    else
      arma::eig_sym(eigval,eigvec,rho,"std");
    

    return 
      eigvec
      * arma::diagmat(arma::conv_to< 
		      arma::Col < 
		      std::complex< pT<T1> > 
		      > >::from(eigval).transform(P))
      * eigvec.t();
  }
 


  template<typename T1, typename functor, typename TR = 
	   typename std::enable_if< is_floating_point_var< pT<T1> >::value,
				    arma::Mat<std::complex< pT<T1> > > 
				    >::type >
  inline 
  TR funcm_gen(const T1& rho1, 
	       functor P)
  {
    const auto& rho = as_Mat(rho1);

#ifndef QIC_LIB_NO_DEBUG    
    if(rho.n_elem == 0)
      throw Exception("qic::funcm_gen",Exception::type::ZERO_SIZE);
    
    if(rho.n_rows!=rho.n_cols)
      throw Exception("qic::funcm_gen",Exception::type::MATRIX_NOT_SQUARE);
#endif
   
    arma::Col< std::complex< pT<T1> > > eigval;
    arma::Mat< std::complex< pT<T1> > > eigvec;
    arma::eig_gen(eigval,eigvec,rho);
    
    return eigvec * arma::diagmat(eigval.transform(P)) * eigvec.t();
  }
 

  template<typename pT>
  struct Func
  {
    static constexpr inline std::complex<pT> sin(const std::complex<pT>& a) noexcept {return std::sin(a);}
    static constexpr inline std::complex<pT> cos(const std::complex<pT>& a) noexcept {return std::cos(a);}
    static constexpr inline std::complex<pT> tan(const std::complex<pT>& a) noexcept {return std::tan(a);}
    
    static constexpr inline std::complex<pT> asin(const std::complex<pT>& a) noexcept {return std::asin(a);}
    static constexpr inline std::complex<pT> acos(const std::complex<pT>& a) noexcept {return std::acos(a);}
    static constexpr inline std::complex<pT> atan(const std::complex<pT>& a) noexcept {return std::atan(a);}
    
    static constexpr inline std::complex<pT> sinh(const std::complex<pT>& a) noexcept {return std::sinh(a);}
    static constexpr inline std::complex<pT> cosh(const std::complex<pT>& a) noexcept {return std::cosh(a);}
    static constexpr inline std::complex<pT> tanh(const std::complex<pT>& a) noexcept {return std::tanh(a);}
    
    static constexpr inline std::complex<pT> asinh(const std::complex<pT>& a) noexcept {return std::asinh(a);}
    static constexpr inline std::complex<pT> acosh(const std::complex<pT>& a) noexcept {return std::acosh(a);}
    static constexpr inline std::complex<pT> atanh(const std::complex<pT>& a) noexcept {return std::atanh(a);}
    
    static constexpr inline std::complex<pT> sqrt(const std::complex<pT>& a) noexcept {return std::sqrt(a);}
    static constexpr inline std::complex<pT> log(const std::complex<pT>& a) noexcept {return std::log(a);}
    static constexpr inline std::complex<pT> log2(const std::complex<pT>& a) noexcept {return std::log2(a);}
    
    static constexpr inline std::complex<pT> norm(const std::complex<pT>& a) noexcept {return std::norm(a);}
    static constexpr inline std::complex<pT> real(const std::complex<pT>& a) noexcept {return std::real(a);}
    static constexpr inline std::complex<pT> imag(const std::complex<pT>& a) noexcept {return std::imag(a);}
    
   
  };

  using func = Func<double>; 
  using funcf = Func<float>;
} 
