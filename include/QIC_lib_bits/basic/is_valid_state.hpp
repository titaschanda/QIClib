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

  template<typename T1>
  inline 
  typename std::enable_if< std::is_arithmetic< pT<T1> >::value,
			   bool
			   >::type is_valid_state(const T1& rho1,
						  const double& tol = 
						  _precision::eps)
  {
    const auto& rho = as_Mat(rho1);
     
    if(!is_H(rho))
      return false;

    else if(is_pure(rho,true,tol))
      return true;

    else
      {
	if(std::abs(arma::trace(rho)-1.0) > static_cast< pT<T1> >(tol) 
	   || arma::any(arma::eig_sym(rho) < -static_cast< pT<T1> >(tol)) )
	  return false;
	else
	  return true;
      }

  }

}

