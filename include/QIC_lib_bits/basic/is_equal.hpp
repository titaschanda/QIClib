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


  template<typename T1,typename T2, typename =   
	   typename std::enable_if< std::is_arithmetic< pT<T1> >::value 
				    && std::is_arithmetic< pT<T2> >::value
				    && is_same_pT<T1,T2>::value,
				    void 
				    >::type >
  inline 
  bool is_equal(const T1& rho11,
		const T2& rho12,
		bool typecheck = false,
		const double& atol = 1.0e-10, 
		const double& rtol = 1.0e-8)
  {
    const auto& rho1 = as_Mat(rho11);
    const auto& rho2 = as_Mat(rho12);
 

    const arma::uword n1 = rho1.n_rows;
    const arma::uword m1 = rho1.n_cols;
    const arma::uword n2 = rho2.n_rows;
    const arma::uword m2 = rho2.n_cols;
    
    if(n1!=n2 || m1!=m2 
       || (typecheck && !std::is_same< eT<T1> , eT<T2> >::value))
      return false;
    
    else
      {
	return  
	  arma::all(arma::vectorise((static_cast< pT<T1> >(atol) 
				     * arma::ones< arma::Mat< pT<T1> > >(n1,m1) 
				     + static_cast< pT<T1> >(rtol)  
				     * arma::abs(rho1)) 
				    - arma::abs(rho1-rho2)) >  0.0);
      }
  }


}


