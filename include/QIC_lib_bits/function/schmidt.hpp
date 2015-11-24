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

  //****************************************************************************


  template<typename T1, typename TR = 
	   typename std::enable_if< std::is_arithmetic< pT<T1> >::value,
				    arma::Col< pT<T1> > 
				    >::type >
  inline 
  TR schmidt(const T1& rho1, 
	     const arma::uvec& dim)
  {
    const auto& rho = as_Mat(rho1);

    bool checkV = true;
    if(rho.n_cols == 1)
      checkV = false;


#ifndef QIC_LIB_NO_DEBUG
    if(rho.n_elem == 0)
      throw Exception("qic::schmidt",Exception::type::ZERO_SIZE);
    
    if(checkV)
      if(rho.n_rows!=rho.n_cols)
	throw Exception("qic::schmidt",
			Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);
      
    if(arma::any(dim == 0))
      throw Exception("qic::schmidt",
		      Exception::type::INVALID_DIMS);
    
    if(arma::prod(dim)!= rho.n_rows)
      throw Exception("qic::schmidt", 
		      Exception::type::DIMS_MISMATCH_MATRIX);
      
    if((dim.n_elem) != 2 )
      throw Exception("qic::schmidt", Exception::type::NOT_BIPARTITE);
#endif
    
    if(checkV)
      return 
	arma::svd(arma::reshape(conv_to_pure(rho),dim.at(1),dim.at(0)).st());
    else
      return arma::svd(arma::reshape(rho,dim.at(1),dim.at(0)).st());
    
  }


  //****************************************************************************

  
  template<typename T1, typename TR = 
	   typename std::enable_if< std::is_arithmetic< pT<T1> >::value,
				    void 
				    >::type >
  inline 
  bool schmidt(const T1& rho1, 
	  const arma::uvec& dim,
	  arma::Col< pT<T1> >& S,
	  arma::Mat< eT<T1> >& U, 
	  arma::Mat< eT<T1> >& V)
  {
    const auto& rho = as_Mat(rho1);

    bool checkV = true;
    if(rho.n_cols == 1)
      checkV = false;
       
#ifndef QIC_LIB_NO_DEBUG
    if(rho.n_elem == 0)
      return false;
    
    if(checkV)
      if(rho.n_rows!=rho.n_cols)
	return false;
      
    if(arma::any(dim == 0))
      return false;

    if(arma::prod(dim)!= rho.n_rows)
      return false;
      
    if((dim.n_elem) != 2 )
      return false;
#endif
    
    if(checkV)
      {
	bool ret = 
	  arma::svd_econ(U,S,V,
			 arma::reshape(conv_to_pure(rho),
				       dim.at(1),dim.at(0)).st(),"both","std");
	
	if(ret == true)
	  V = arma::conj(V);
	return (ret);
      }
      
    else
      {
	bool ret =  
	  arma::svd_econ(U,S,V, arma::reshape(rho,
					      dim.at(1),dim.at(0)).st(),
			 "both","std");
	
	if(ret == true)
	  V = arma::conj(V);
	return (ret);
      }	  
  }


  //****************************************************************************

  
  template<typename T1, typename TR = 
	   typename std::enable_if< std::is_arithmetic< pT<T1> >::value,
				    arma::Mat< eT<T1> >
				    >::type >
  inline
  TR schmidtA(const T1& rho1, 
	      const arma::uvec& dim)
  {  
    const auto& rho = as_Mat(rho1);
    
    bool checkV = true;
    if(rho.n_cols == 1)
      checkV = false;


#ifndef QIC_LIB_NO_DEBUG
    if(rho.n_elem == 0)
      throw Exception("qic::schmidt",Exception::type::ZERO_SIZE);
    
    if(checkV)
      if(rho.n_rows!=rho.n_cols)
	throw Exception("qic::schmidt",
			Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);
      
    if(arma::any(dim == 0))
      throw Exception("qic::schmidt",
		      Exception::type::INVALID_DIMS);
    
    if(arma::prod(dim)!= rho.n_rows)
      throw Exception("qic::schmidt", 
		      Exception::type::DIMS_MISMATCH_MATRIX);
      
    if((dim.n_elem) != 2 )
      throw Exception("qic::schmidt", Exception::type::NOT_BIPARTITE);
#endif
    
    arma::Mat< eT<T1> > U,V;
    arma::Col< pT<T1> > S;

    if(checkV)
      {
	arma::svd_econ(U,S,V,
		  arma::reshape(conv_to_pure(rho),
				dim.at(1),dim.at(0)).st(),"left","std");
	return U;
      }
      
    else
      {
	arma::svd_econ(U,S,V, arma::reshape(rho,
					    dim.at(1),dim.at(0)).st(),
		       "left","std");
	
	return U;
      }	  
  }


  //****************************************************************************

  
  template<typename T1, typename TR = 
	   typename std::enable_if< std::is_arithmetic< pT<T1> >::value,
				    arma::Mat< eT<T1> > 
				    >::type >
  inline 
  TR schmidtB(const T1& rho1, 
	      const arma::uvec& dim)
  {
    const auto& rho = as_Mat(rho1);
    
    bool checkV = true;
    if(rho.n_cols == 1)
      checkV = false;


#ifndef QIC_LIB_NO_DEBUG
    if(rho.n_elem == 0)
      throw Exception("qic::schmidt",Exception::type::ZERO_SIZE);
    
    if(checkV)
      if(rho.n_rows!=rho.n_cols)
	throw Exception("qic::schmidt",
			Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);
      
    if(arma::any(dim == 0))
      throw Exception("qic::schmidt",
		      Exception::type::INVALID_DIMS);
    
    if(arma::prod(dim)!= rho.n_rows)
      throw Exception("qic::schmidt", 
		      Exception::type::DIMS_MISMATCH_MATRIX);
      
    if((dim.n_elem) != 2 )
      throw Exception("qic::schmidt", Exception::type::NOT_BIPARTITE);
#endif
    
    arma::Mat< eT<T1> > U,V;
    arma::Col< pT<T1> > S;

    if(checkV)
      {
	arma::svd_econ(U,S,V,
		       arma::reshape(conv_to_pure(rho),
				     dim.at(1),dim.at(0)).st(),"right","std");
	return arma::conj(V);
      }
      
    else
      {
	arma::svd(U,S,V,arma::reshape(rho,
				      dim.at(1),dim.at(0)).st(),"right","std");
	
	return arma::conj(V);
      }	  
  }


  //****************************************************************************

  
  template<typename T1, typename TR = 
	   typename std::enable_if< std::is_arithmetic< pT<T1> >::value,
				    arma::field< arma::Mat< eT<T1> > > 
				    >::type >
  inline 
  TR schmidtAB(const T1& rho1, 
	       const arma::uvec& dim)
  {
    const auto& rho = as_Mat(rho1);
    
    bool checkV = true;
    if(rho.n_cols == 1)
      checkV = false;


#ifndef QIC_LIB_NO_DEBUG
    if(rho.n_elem == 0)
      throw Exception("qic::schmidt",Exception::type::ZERO_SIZE);
    
    if(checkV)
      if(rho.n_rows!=rho.n_cols)
	throw Exception("qic::schmidt",
			Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);
      
    if(arma::any(dim == 0))
      throw Exception("qic::schmidt",
		      Exception::type::INVALID_DIMS);
    
    if(arma::prod(dim)!= rho.n_rows)
      throw Exception("qic::schmidt", 
		      Exception::type::DIMS_MISMATCH_MATRIX);
      
    if((dim.n_elem) != 2 )
      throw Exception("qic::schmidt", Exception::type::NOT_BIPARTITE);
#endif
    
    arma::Mat< eT<T1> > U,V;
    arma::Col< pT<T1> > S;
    arma::field< arma::Mat< eT<T1> > > ret(2);


    if(checkV)
      {
	arma::svd_econ(U,S,V,
		       arma::reshape(conv_to_pure(rho),
				     dim.at(1),dim.at(0)).st(),"both","std");
	ret.at(0) = std::move(U);
	ret.at(1) = std::move(arma::conj(V));
	return ret;
      }
      
    else
      {
	arma::svd_econ(U,S,V, arma::reshape(rho,
					    dim.at(1),dim.at(0)).st(),
		       "both","std");
	ret.at(0) = std::move(U);
	ret.at(1) = std::move(arma::conj(V));
	return ret;
      }	  
  }


  //****************************************************************************

  
  template<typename T1, typename TR = 
	   typename std::enable_if< std::is_arithmetic< pT<T1> >::value,
				    arma::Mat< eT<T1> > 
				    >::type >
  inline 
  TR schmidtA_full(const T1& rho1, 
		   const arma::uvec& dim)
  {
    const auto& rho = as_Mat(rho1);
    
    bool checkV = true;
    if(rho.n_cols == 1)
      checkV = false;

#ifndef QIC_LIB_NO_DEBUG
    if(rho.n_elem == 0)
      throw Exception("qic::schmidt",Exception::type::ZERO_SIZE);
    
    if(checkV)
      if(rho.n_rows!=rho.n_cols)
	throw Exception("qic::schmidt",
			Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);
      
    if(arma::any(dim == 0))
      throw Exception("qic::schmidt",
		      Exception::type::INVALID_DIMS);
    
    if(arma::prod(dim)!= rho.n_rows)
      throw Exception("qic::schmidt", 
		      Exception::type::DIMS_MISMATCH_MATRIX);
      
    if((dim.n_elem) != 2 )
      throw Exception("qic::schmidt", Exception::type::NOT_BIPARTITE);
#endif
    
    arma::Mat< eT<T1> > U,V;
    arma::Col< pT<T1> > S;

    if(checkV)
      {
	arma::svd(U,S,V,
		  arma::reshape(conv_to_pure(rho),
				dim.at(1),dim.at(0)).st(),"std");
	return U;
      }
      
    else
      {
	arma::svd(U,S,V, arma::reshape(rho,
				       dim.at(1),dim.at(0)).st(),"std");
	
	return U;
      }	  
  }


  //****************************************************************************

  
  template<typename T1, typename TR = 
	   typename std::enable_if< std::is_arithmetic< pT<T1> >::value,
				    arma::Mat< eT<T1> >
				    >::type >
  inline 
  TR schmidtB_full(const T1& rho1, 
		   const arma::uvec& dim)
  {
    const auto& rho = as_Mat(rho1);
    
    bool checkV = true;
    if(rho.n_cols == 1)
      checkV = false;

#ifndef QIC_LIB_NO_DEBUG
    if(rho.n_elem == 0)
      throw Exception("qic::schmidt",Exception::type::ZERO_SIZE);
    
    if(checkV)
      if(rho.n_rows!=rho.n_cols)
	throw Exception("qic::schmidt",
			Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);
      
    if(arma::any(dim == 0))
      throw Exception("qic::schmidt",
		      Exception::type::INVALID_DIMS);
    
    if(arma::prod(dim)!= rho.n_rows)
      throw Exception("qic::schmidt", 
		      Exception::type::DIMS_MISMATCH_MATRIX);
      
    if((dim.n_elem) != 2 )
      throw Exception("qic::schmidt", Exception::type::NOT_BIPARTITE);
#endif
    
    arma::Mat< eT<T1> > U,V;
    arma::Col< pT<T1> > S;

    if(checkV)
      {
	arma::svd(U,S,V,
		  arma::reshape(conv_to_pure(rho),
				dim.at(1),dim.at(0)).st(),"std");
	return arma::conj(V);
      }
      
    else
      {
	arma::svd(U,S,V,arma::reshape(rho,
				      dim.at(1),dim.at(0)).st(),"std");
	
	return arma::conj(V);
      }	  
  }


  //****************************************************************************

  
  template<typename T1, typename TR = 
	   typename std::enable_if< std::is_arithmetic< pT<T1> >::value,
				    arma::field< arma::Mat< eT<T1> > > 
				    >::type >
  inline 
  TR schmidtAB_full(const T1& rho1, 
		    const arma::uvec& dim)
  {
    const auto& rho = as_Mat(rho1);
    
    bool checkV = true;
    if(rho.n_cols == 1)
      checkV = false;


#ifndef QIC_LIB_NO_DEBUG
    if(rho.n_elem == 0)
      throw Exception("qic::schmidt",Exception::type::ZERO_SIZE);
    
    if(checkV)
      if(rho.n_rows!=rho.n_cols)
	throw Exception("qic::schmidt",
			Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);
      
    if(arma::any(dim == 0))
      throw Exception("qic::schmidt",
		      Exception::type::INVALID_DIMS);
    
    if(arma::prod(dim)!= rho.n_rows)
      throw Exception("qic::schmidt", 
		      Exception::type::DIMS_MISMATCH_MATRIX);
      
    if((dim.n_elem) != 2 )
      throw Exception("qic::schmidt", Exception::type::NOT_BIPARTITE);
#endif
    
    arma::Mat< eT<T1> > U,V;
    arma::Col< pT<T1> > S;
    arma::field< arma::Mat < eT<T1> > > ret(2);


    if(checkV)
      {
	arma::svd(U,S,V,
		  arma::reshape(conv_to_pure(rho),
				dim.at(1),dim.at(0)).st(),"std");
	ret.at(0) = std::move(U);
	ret.at(1) = std::move(arma::conj(V));
	return ret;
      }
      
    else
      {
	arma::svd(U,S,V, arma::reshape(rho,
				       dim.at(1),dim.at(0)).st(),"std");
	ret.at(0) = std::move(U);
	ret.at(1) = std::move(arma::conj(V));
	return ret;
      }	  
  }



}
