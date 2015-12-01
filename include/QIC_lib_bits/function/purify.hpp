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

  // ***************************************************************************
  
  template<typename T1, typename TR = 
	   typename std::enable_if< is_floating_point_var< pT<T1> >::value,
				    arma::Col< eT<T1> >
				    >::type >
  inline 
  TR purify(const T1& rho1, 
	    const pT<T1>& tol = _precision::eps< pT<T1> >::value)
  {
    const auto& rho = as_Mat(rho1);    

    bool checkV = true;
    if(rho.n_cols == 1)
      checkV = false;

#ifndef QIC_LIB_NO_DEBUG
    if(rho.n_elem == 0)
      throw Exception("qic::purify",Exception::type::ZERO_SIZE);

    if(checkV)
      if(rho.n_rows != rho.n_cols)
	throw Exception("qic::purify",
			Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);
#endif

    if(!checkV)
      return rho;

    else
      {
	arma::Col< pT<T1> > eigval;
	arma::Mat< eT<T1> > eigvec;
	
	if(rho.n_rows > 20)
	  arma::eig_sym(eigval,eigvec,rho,"dc");
	else
	  arma::eig_sym(eigval,eigvec,rho,"std");

	arma::uword dim = rho.n_rows;
	arma::uword dimE = arma::sum(eigval > tol);

	arma::Col< eT<T1> > ret = arma::zeros< arma::Col< eT<T1> > >(dim*dimE);
	
	for(arma::uword i = 0 ; i < dimE ; ++i)
	  for(arma::uword j = 0 ; j < dim ; ++j)
	    ret(i + dimE*j) = std::sqrt(eigval.at(dim-i-1)) 
	      * eigvec.at(j,dim-i-1);
	
	return ret;

      }
  }
   


}

    
