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


  template <typename T1>
  inline
  typename std::enable_if< std::is_arithmetic< pT<T1> >::value,
			   T1&&
			   >::type tensor(T1&& rho1)
  {
    return std::forward<T1>(rho1);
  }


  template<typename T1, typename T2>
  inline 
  typename std::enable_if< is_comparable_pT<T1,T2>::value,
			   arma::Mat< typename eT_promoter_var<T1,T2>::type >
			   >::type  tensor(T1&& rho1,T2&& rho2)
  {
    return arma::kron(rho1,rho2).eval();
  }


  template<typename T1, typename... T2>
  inline 
  typename std::enable_if< is_comparable_pT<T1,T2...>::value,
			 arma::Mat< typename eT_promoter_var<T1,T2...>::type > 
			 >::type tensor(T1&& rho1,T2&&... rho2)
  {
    return arma::kron(rho1,
		      tensor(rho2...)).eval();
  }



  template<typename T1>
  inline 
  typename std::enable_if< std::is_arithmetic<T1>::value,
			   arma::Mat< eT<T1> >
			   >::type tensor(const arma::field<T1>& rho)
  {
    
#ifndef QIC_LIB_NO_DEBUG
    if (rho.n_elems == 0)
      throw Exception("qic::tensor", Exception::type::ZERO_SIZE);

    for (auto&& a : rho)
      if ( a.eval().n_elems == 0  )
	throw Exception("qic::tensor", Exception::type::ZERO_SIZE);
#endif

    auto ret = rho.at(0).eval();
    
    for(arma::uword i = 1 ; i < rho.n_elems ; ++i)
      ret = arma::kron(ret,rho.at(0));

    return ret;

  }




  template<typename T1>
  inline 
  typename std::enable_if< std::is_arithmetic<T1>::value,
			   arma::Mat< eT<T1> >
			   >::type tensor(const std::vector<T1>& rho)
  {
    
#ifndef QIC_LIB_NO_DEBUG
    if (rho.size() == 0)
      throw Exception("qic::tensor", Exception::type::ZERO_SIZE);

    for (auto&& a : rho)
      if ( a.eval().n_elems == 0  )
            throw Exception("qic::tensor", Exception::type::ZERO_SIZE);
#endif

    auto ret = rho[0].eval();
    
    for(arma::uword i = 1 ; i < rho.size() ; ++i)
      ret = arma::kron(ret,rho[i]);

    return ret;

  }


  template<typename T1>
  inline 
  typename std::enable_if< std::is_arithmetic<T1>::value,
			   arma::Mat< eT<T1> >
			   >::type tensor(const std::initializer_list<T1>& rho)
  {
    return tensor(static_cast< std::vector<T1> >(rho));
  }



  template <typename T1>
  inline 
  typename std::enable_if< std::is_arithmetic< pT<T1> >::value,
			   arma::Mat< eT<T1> >
			   >::type tensor_pow(const T1& rho1,
					      arma::uword n)
  {
    
    const auto& rho = as_Mat(rho1);

#ifndef QIC_LIB_NO_DEBUG
    if(rho.n_elem == 0)
      throw Exception("qic::tensor_pow",Exception::type::ZERO_SIZE);

    if(n == 0)
      throw Exception("qic::tensor_pow",Exception::type::OUT_OF_RANGE);
#endif

    return _internal::protect_subs::TENSOR_POW(rho,n);

  }

}
