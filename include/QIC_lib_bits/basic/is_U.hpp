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

  template<typename T1, typename =   
	   typename std::enable_if< is_floating_point_var< pT<T1> >::value, 
				    void 
				    >::type>
  inline 
  bool is_U(const T1& rho1, 
	    const double& atol = 1.0e-10, 
	    const double& rtol = 1.0e-8)
  {
    const auto& rho = as_Mat(rho1);
    
    const arma::uword n = rho.n_rows;
    const arma::uword m = rho.n_cols;
   
    if(n!=m)
      return false;
    
    else
      {
	arma::Mat< eT<T1> > eye1 = rho*rho.t();
	arma::Mat< eT<T1> > eye2 = rho.t()*rho;
	
	bool ret1 = 
	  arma::all(arma::vectorise((static_cast< pT<T1> >(atol) 
				     * arma::ones< arma::Mat< pT<T1> > >(n,m) 
				     + static_cast< pT<T1> >(rtol)  
				     * arma::abs(eye1)) 
				    - arma::abs(eye1 
						- arma::eye< arma::Mat< 
						pT<T1> > >(n,m)))>  0.0);
	
	bool ret2 = 
	  arma::all(arma::vectorise((static_cast< pT<T1> >(atol) 
				     * arma::ones< arma::Mat< pT<T1> > >(n,m) 
				     + static_cast< pT<T1> >(rtol)  
				     * arma::abs(eye2)) 
				    - arma::abs(eye2 
						- arma::eye< arma::Mat< 
						pT<T1> > >(n,m))) >  0.0);
	return (ret1 && ret2);

      }
  }


}


