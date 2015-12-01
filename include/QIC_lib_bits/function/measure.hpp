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


  template<typename T1, typename T2, typename TR = 
	   typename std::enable_if< is_floating_point_var< 
				      pT<T1>, pT<T2> 
				      >::value
				    && is_same_pT<T1,T2>::value,
				    std::tuple< arma::uword, 
						arma::Col< pT<T1> >,
						arma::field< 
						  arma::Mat< 
						    typename 
						    eT_promoter_var
						    <T1,T2>::type >
						  >
						>
				    >::type >
    inline 
    TR measure(const T1& rho1, 
	       const std::vector<T2>& Ks)
  {
    
    const auto& rho = as_Mat(rho1); 
    
    bool checkV = true;
    if(rho.n_cols == 1)
      checkV = false;


#ifndef QIC_LIB_NO_DEBUG
    if(rho.n_elem == 0)
      throw Exception("qic::measure",Exception::type::ZERO_SIZE);

    if(Ks.size() == 0)
      throw Exception("qic::measure",Exception::type::ZERO_SIZE);

    if(checkV)
      if(rho.n_rows!=rho.n_cols)
	throw Exception("qic::measure",
			Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);
    
    for(auto&& k : Ks)
      if( !((k.eval().n_rows == k.eval().n_cols)
	   || k.eval().n_cols == 1) )
	throw Exception("qic::measure",
			Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);
    
    for(auto&& k : Ks)
      if( (k.eval().n_rows != Ks[0].eval().n_rows) 
	  || (k.eval().n_cols != Ks[0].eval().n_cols) )
	throw Exception("qic::measure",Exception::type::DIMS_NOT_EQUAL);
    
    
    if(Ks[0].eval().n_rows != rho.n_rows)
      throw Exception("qic::measure",
		      Exception::type::DIMS_MISMATCH_MATRIX);
    
#endif
    
    using mattype = arma::Mat<typename eT_promoter_var<T1,T2>::type>;
    bool checkK = false;
    if(Ks[0].eval().n_cols == 1)
      checkK = true;
    
    arma::Col< pT<T1> > prob(Ks.size());
    arma::field< mattype > outstates(Ks.size());    

    if(checkV)
      outstates.fill(arma::zeros<mattype>(rho.n_rows,rho.n_cols));
    else 
      outstates.fill(arma::zeros<mattype>(1,rho.n_cols));

    if(checkV)
      {
	for (arma::uword i = 0; i < Ks.size(); ++i)
        {
	  mattype tmp;
	  if(checkK)
	    tmp = Ks[i].eval() * Ks[i].eval().t() 
	      * rho * Ks[i].eval() * Ks[i].eval().t();
	  else
	    tmp = Ks[i].eval() * rho * Ks[i].eval().t();
	  prob.at(i) = std::abs(arma::trace(tmp)); 
	  if (prob.at(i) > _precision::eps< pT<T1> >::value)
	    outstates.at(i) = tmp / prob.at(i);
        }
      }

    else
      {
	for (arma::uword i = 0; i < Ks.size(); ++i)
	  {
	    mattype tmp;
	    if(checkK)
	      tmp = Ks[i].eval() * Ks[i].eval().t() * rho;
	    else
	      tmp = Ks[i].eval() * rho;
	    prob.at(i) = std::pow(arma::norm(as_Col(tmp)),2); 
	    if (prob.at(i) > _precision::eps< pT<T1> >::value)
	      outstates.at(i) = tmp / std::sqrt(prob.at(i));
	  }
      }
    
    std::discrete_distribution<arma::uword> dd(prob.begin(),
    					       prob.end());
    arma::uword result = dd(rdevs.rng);
    return std::make_tuple(result,prob,outstates);
	

  }


  //****************************************************************************
    

  template<typename T1, typename T2, typename TR = 
	   typename std::enable_if< is_floating_point_var< 
				      pT<T1>, pT< arma::Mat<T2> > 
				      >::value
				    && is_same_pT<T1, arma::Mat<T2> >::value,
				    std::tuple< arma::uword, 
						arma::Col< pT<T1> >,
						arma::field< 
						  arma::Mat< 
						    typename 
						    eT_promoter_var
						    < T1,arma::Mat<T2> >::type >
						  >
						>
				    >::type >
    inline 
    TR measure(const T1& rho1, 
	       const std::initializer_list< arma::Mat<T2> >& Ks)
    {
      return measure(rho1, static_cast< std::vector< arma::Mat<T2> > >(Ks));
    }


  //****************************************************************************
  
  
  template<typename T1, typename T2, typename TR = 
	   typename std::enable_if< is_floating_point_var< 
				      pT<T1>, pT<T2> 
				      >::value
				    && is_same_pT<T1,T2>::value,
				    std::tuple< arma::uword, 
						arma::Col< pT<T1> >,
						arma::field< 
						  arma::Mat< 
						    typename 
						    eT_promoter_var
						    <T1,T2>::type >
						  >
						>
				    >::type >
    inline 
    TR measure(const T1& rho1, 
	       const arma::field<T2>& Ks)
  {
    
    const auto& rho = as_Mat(rho1); 
    
    bool checkV = true;
    if(rho.n_cols == 1)
      checkV = false;


#ifndef QIC_LIB_NO_DEBUG
    if(rho.n_elem == 0)
      throw Exception("qic::measure",Exception::type::ZERO_SIZE);

    if(Ks.n_elem == 0)
      throw Exception("qic::measure",Exception::type::ZERO_SIZE);

    if(checkV)
      if(rho.n_rows!=rho.n_cols)
	throw Exception("qic::measure",
			Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);
    
    for(auto&& k : Ks)
      if( !((k.eval().n_rows == k.eval().n_cols) 
	    || (k.eval().n_cols == 1)) )
	throw Exception("qic::measure",
			Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);
    
    for(auto&& k : Ks)
      if( (k.eval().n_rows != Ks.at(0).eval().n_rows)
	  || (k.eval().n_cols != Ks.at(0).eval().n_cols) )
	throw Exception("qic::measure",Exception::type::DIMS_NOT_EQUAL);
    
    
    if(Ks.at(0).eval().n_rows != rho.n_rows)
      throw Exception("qic::measure",
		      Exception::type::DIMS_MISMATCH_MATRIX);
    
#endif
    
    using mattype = arma::Mat<typename eT_promoter_var<T1,T2>::type>;
    bool checkK = false;
    if(Ks.at(0).eval().n_cols == 1)
      checkK = true;

    arma::Col< pT<T1> > prob(Ks.n_elem);
    arma::field< mattype > outstates(Ks.n_elem);

    if(checkV)
      outstates.fill(arma::zeros<mattype>(rho.n_rows,rho.n_cols));
    else 
      outstates.fill(arma::zeros<mattype>(1,rho.n_cols));

    
    if(checkV)
      {
	for (arma::uword i = 0; i < Ks.n_elem; ++i)
        {
	  mattype tmp;
	  if(checkK)
	    tmp = Ks.at(i).eval() * Ks.at(i).eval().t() 
	      * rho * Ks.at(i).eval() * Ks.at(i).eval().t();
	  else
	    tmp = Ks.at(i).eval() * rho * Ks.at(i).eval().t();
	  prob.at(i) = std::abs(arma::trace(tmp)); 
	  if (prob.at(i) > _precision::eps< pT<T1> >::value)
	    outstates.at(i) = tmp / prob.at(i);
        }
      }

    else
      {
	for (arma::uword i = 0; i < Ks.size(); ++i)
	  {
	    mattype tmp ;
	    if(checkK)
	      tmp = Ks.at(i).eval() * Ks.at(i).eval().t() * rho;
	    else
	      tmp = Ks.at(i).eval() * rho;
	    prob.at(i) = std::pow(arma::norm(as_Col(tmp)),2); 
	    if (prob.at(i) > _precision::eps< pT<T1> >::value)
	      outstates.at(i) = tmp / std::sqrt(prob.at(i));
	  }
      }
    
    std::discrete_distribution<arma::uword> dd(prob.begin(),
    					       prob.end());
    arma::uword result = dd(rdevs.rng);
    
    return std::make_tuple(result,prob,outstates);
	
  }
  

  //****************************************************************************
  

  template<typename T1, typename T2, typename TR = 
	   typename std::enable_if< is_floating_point_var< 
				      pT<T1>, pT<T2> 
				      >::value
				    && is_same_pT<T1,T2>::value,
				    std::tuple< arma::uword, 
						arma::Col< pT<T1> >,
						arma::field< 
						  arma::Mat< 
						    typename 
						    eT_promoter_var
						    <T1,T2>::type >
						  >
						>
				    >::type >
    inline 
    TR measure(const T1& rho1, 
	       const T2& U1)
  {
    
    const auto& rho = as_Mat(rho1); 
    const auto& U = as_Mat(U1);

    bool checkV = true;
    if(rho.n_cols == 1)
      checkV = false;


#ifndef QIC_LIB_NO_DEBUG
    if(rho.n_elem == 0)
      throw Exception("qic::measure",Exception::type::ZERO_SIZE);

    if(U.n_elem == 0)
      throw Exception("qic::measure",Exception::type::ZERO_SIZE);

    if(checkV)
      if(rho.n_rows!=rho.n_cols)
	throw Exception("qic::measure",
			Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);
   
    if( U.n_rows != rho.n_rows )
      throw Exception("qic::measure",
		      Exception::type::DIMS_MISMATCH_MATRIX);
    
#endif
    
    using mattype = arma::Mat<typename eT_promoter_var<T1,T2>::type>;
    
    arma::Col< pT<T1> > prob(U.n_cols);
    arma::field< mattype > outstates(U.n_cols);

    if(checkV)
      outstates.fill(arma::zeros<mattype>(rho.n_rows,rho.n_cols));
    else 
      outstates.fill(arma::zeros<mattype>(1,rho.n_cols));
    

    if(checkV)
      {
	for (arma::uword i = 0; i < U.n_cols; ++i)
        {
	  mattype tmp = U.col(i) * U.col(i).t()
	    * rho * U.col(i) * U.col(i).t();
	  prob.at(i) = std::abs(arma::trace(tmp)); 
	  if (prob.at(i) > _precision::eps< pT<T1> >::value)
	    outstates.at(i) = tmp / prob.at(i);
        }
      }

    else
      {
	for (arma::uword i = 0; i < U.n_cols; ++i)
	  {
	    mattype tmp = U.col(i) * U.col(i).t() * rho;
	    prob.at(i) = std::pow(arma::norm(as_Col(tmp)),2); 
	    if (prob.at(i) > _precision::eps< pT<T1> >::value)
	      outstates.at(i) = tmp / std::sqrt(prob.at(i));
	  }
      }
    
    std::discrete_distribution<arma::uword> dd(prob.begin(),
    					       prob.end());
    arma::uword result = dd(rdevs.rng);
    
    return std::make_tuple(result,prob,outstates);
	
  }
  

  //****************************************************************************


  template<typename T1, typename T2, typename TR = 
	   typename std::enable_if< is_floating_point_var< 
				      pT<T1>, pT<T2> 
				      >::value
				    && is_same_pT<T1,T2>::value,
				    std::tuple< arma::uword, 
						arma::Col< pT<T1> >,
						arma::field< 
						  arma::Mat< 
						    typename 
						    eT_promoter_var
						    <T1,T2>::type >
						  >
						>
				    >::type >
    inline 
    TR measure(const T1& rho1, 
	       const std::vector<T2>& Ks,
	       arma::uvec sys,
	       arma::uvec dim)
  {

    const auto& rho = as_Mat(rho1); 
    
    bool checkV = true;
    if(rho.n_cols == 1)
      checkV = false;


    arma::uword D = arma::prod(dim);
    arma::uword Dsys = arma::prod(dim(sys-1));
    arma::uword Dsysbar = D/Dsys;


#ifndef QIC_LIB_NO_DEBUG
    if(rho.n_elem == 0)
      throw Exception("qic::measure",Exception::type::ZERO_SIZE);

    if(Ks.size() == 0)
      throw Exception("qic::measure",Exception::type::ZERO_SIZE);

    if(checkV)
      if(rho.n_rows!=rho.n_cols)
	throw Exception("qic::measure",
			Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);
    
    for(auto&& k : Ks)
      if( !((k.eval().n_rows == k.eval().n_cols)
	    || k.eval().n_cols == 1) )
	throw Exception("qic::measure",
			Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);
    
    for(auto&& k : Ks)
      if( (k.eval().n_rows != Ks[0].eval().n_rows) 
	  || (k.eval().n_cols != Ks[0].eval().n_cols) )
	throw Exception("qic::measure",Exception::type::DIMS_NOT_EQUAL);
    
    if( dim.n_elem == 0 || arma::any(dim == 0))
      throw Exception("qic::measure",Exception::type::INVALID_DIMS);

    if( arma::prod(dim)!= rho.n_rows)
      throw Exception("qic::measure",Exception::type::DIMS_MISMATCH_MATRIX);

    if( Dsys != Ks[0].eval().n_rows )
      throw Exception("qic::measure",
		      Exception::type::DIMS_MISMATCH_MATRIX);

    if(sys.n_elem > dim.n_elem 
       || arma::find_unique(sys).eval().n_elem != sys.n_elem 
       || arma::any(sys > dim.n_elem) || arma::any(sys == 0) )
      throw Exception("qic::measure",Exception::type::INVALID_SUBSYS);      
#endif


    using mattype = arma::Mat<typename eT_promoter_var<T1,T2>::type>;
    bool checkK = false;
    if(Ks[0].eval().n_cols == 1)
      checkK = true;
    arma::Col< pT<T1> > prob(Ks.size());
    arma::field< mattype > outstates(Ks.size());    
    
    outstates.fill(arma::zeros<mattype>(Dsysbar,Dsysbar));

    for (arma::uword i = 0; i < Ks.size(); ++i)
      {
	mattype tmp;
	if(checkK)
	  tmp = apply(rho,(Ks[i].eval()*Ks[i].eval().t()).eval(),sys,dim);
	else 
	  tmp = apply(rho,Ks[i].eval(),sys,dim);
	tmp = TrX(tmp,sys,dim);
	prob.at(i) = std::abs(arma::trace(tmp)); 
	if (prob.at(i) > _precision::eps< pT<T1> >::value)
	  outstates.at(i) = tmp / prob.at(i);
      }
    
    std::discrete_distribution<arma::uword> dd(prob.begin(),
    					       prob.end());
    arma::uword result = dd(rdevs.rng);
    
    return std::make_tuple(result,prob,outstates);
	

  }


  //****************************************************************************

  
  template<typename T1, typename T2, typename TR = 
	   typename std::enable_if< is_floating_point_var< 
				      pT<T1>, pT< arma::Mat<T2> > 
				      >::value
				    && is_same_pT<T1,arma::Mat<T2> >::value,
				    std::tuple< arma::uword, 
						arma::Col< pT<T1> >,
						arma::field< 
						  arma::Mat< 
						    typename 
						    eT_promoter_var
						    <T1, arma::Mat<T2> >::type >
						  >
						>
				    >::type >
    inline 
    TR measure(const T1& rho1, 
	       const std::initializer_list< arma::Mat<T2> >& Ks,
	       arma::uvec sys,
	       arma::uvec dim)

  {
    return measure(rho1, static_cast< std::vector< arma::Mat<T2> > >(Ks),
		   std::move(sys),std::move(dim));
  }


  //****************************************************************************


  template<typename T1, typename T2, typename TR = 
	   typename std::enable_if< is_floating_point_var< 
				      pT<T1>, pT<T2> 
				      >::value
				    && is_same_pT<T1,T2>::value,
				    std::tuple< arma::uword, 
						arma::Col< pT<T1> >,
						arma::field< 
						  arma::Mat< 
						    typename 
						    eT_promoter_var
						    <T1,T2>::type >
						  >
						>
				    >::type >
    inline 
    TR measure(const T1& rho1, 
	       const arma::field<T2>& Ks,
	       arma::uvec sys,
	       arma::uvec dim)
  {

    const auto& rho = as_Mat(rho1); 
    
    bool checkV = true;
    if(rho.n_cols == 1)
      checkV = false;


    arma::uword D = arma::prod(dim);
    arma::uword Dsys = arma::prod(dim(sys-1));
    arma::uword Dsysbar = D/Dsys;


#ifndef QIC_LIB_NO_DEBUG
    if(rho.n_elem == 0)
      throw Exception("qic::measure",Exception::type::ZERO_SIZE);

    if(Ks.n_elem == 0)
      throw Exception("qic::measure",Exception::type::ZERO_SIZE);

    if(checkV)
      if(rho.n_rows!=rho.n_cols)
	throw Exception("qic::measure",
			Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);
    
    for(auto&& k : Ks)
      if( !((k.eval().n_rows == k.eval().n_cols)
	    || k.eval().n_cols == 1) )
	throw Exception("qic::measure",
			Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);
    
    for(auto&& k : Ks)
      if( (k.eval().n_rows != Ks.at(0).eval().n_rows) 
	  || (k.eval().n_cols != Ks.at(0).eval().n_cols) )
	throw Exception("qic::measure",Exception::type::DIMS_NOT_EQUAL);
    
    if( dim.n_elem == 0 || arma::any(dim == 0))
      throw Exception("qic::measure",Exception::type::INVALID_DIMS);

    if( arma::prod(dim)!= rho.n_rows)
      throw Exception("qic::measure",Exception::type::DIMS_MISMATCH_MATRIX);

    if( Dsys != Ks.at(0).eval().n_rows )
      throw Exception("qic::measure",
		      Exception::type::DIMS_MISMATCH_MATRIX);

    if(sys.n_elem > dim.n_elem 
       || arma::find_unique(sys).eval().n_elem != sys.n_elem 
       || arma::any(sys > dim.n_elem) || arma::any(sys == 0) )
      throw Exception("qic::measure",Exception::type::INVALID_SUBSYS);      
#endif


    using mattype = arma::Mat<typename eT_promoter_var<T1,T2>::type>;
    bool checkK = false;
    if(Ks.at(0).eval().n_cols == 1)
      checkK = true;
    arma::Col< pT<T1> > prob(Ks.n_elem);
    arma::field< mattype > outstates(Ks.n_elem);    

    outstates.fill(arma::zeros<mattype>(Dsysbar,Dsysbar));

    
    for (arma::uword i = 0; i < Ks.n_elem; ++i)
      {
	mattype tmp;
	if(checkK)
	  tmp = apply(rho,(Ks.at(i).eval()*Ks.at(i).eval().t()).eval(),sys,dim);
	else 
	  tmp = apply(rho,Ks.at(i).eval(),sys,dim);
	tmp = TrX(tmp,sys,dim);
	prob.at(i) = std::abs(arma::trace(tmp)); 
	if (prob.at(i) > _precision::eps< pT<T1> >::value)
	  outstates.at(i) = tmp / prob.at(i);
      }
    
    std::discrete_distribution<arma::uword> dd(prob.begin(),
    					       prob.end());
    arma::uword result = dd(rdevs.rng);
    
    return std::make_tuple(result,prob,outstates);
	

  }


  //****************************************************************************


  template<typename T1, typename T2, typename TR = 
	   typename std::enable_if< is_floating_point_var< 
				      pT<T1>, pT<T2> 
				      >::value
				    && is_same_pT<T1,T2>::value,
				    std::tuple< arma::uword, 
						arma::Col< pT<T1> >,
						arma::field< 
						  arma::Mat< 
						    typename 
						    eT_promoter_var
						    <T1,T2>::type >
						  >
						>
				    >::type >
    inline 
    TR measure(const T1& rho1, 
	       const std::vector<T2>& Ks,
	       arma::uvec sys,
	       arma::uword dim = 2)
  {
    const auto& rho = as_Mat(rho1); 
    
    bool checkV = true;
    if(rho.n_cols == 1)
      checkV = false;
    
#ifndef QIC_LIB_NO_DEBUG
    if(rho.n_elem == 0)
      throw Exception("qic::measure",Exception::type::ZERO_SIZE);

    if(checkV)
      if(rho.n_rows != rho.n_cols)
	throw Exception("qic::measure",
			Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);
    
    if(dim == 0)
      throw Exception("qic::measure",Exception::type::INVALID_DIMS);
#endif
    
    arma::uword n = static_cast<arma::uword>
      (std::llround(std::log(rho.n_rows) / std::log(dim)));

    arma::uvec dim2 = dim * arma::ones<arma::uvec>(n);
    
    return measure(rho,Ks,std::move(sys),std::move(dim2));
  }


//****************************************************************************

  
  template<typename T1, typename T2, typename TR = 
	   typename std::enable_if< is_floating_point_var< 
				      pT<T1>, pT< arma::Mat<T2> > 
				      >::value
				    && is_same_pT<T1,arma::Mat<T2> >::value,
				    std::tuple< arma::uword, 
						arma::Col< pT<T1> >,
						arma::field< 
						  arma::Mat< 
						    typename 
						    eT_promoter_var
						    <T1, arma::Mat<T2> >::type >
						  >
						>
				    >::type >
    inline 
    TR measure(const T1& rho1, 
	       const std::initializer_list< arma::Mat<T2> >& Ks,
	       arma::uvec sys,
	       arma::uword dim=2)

  {
    return measure(rho1, static_cast< std::vector< arma::Mat<T2> > >(Ks),
		   std::move(sys),std::move(dim));
  }


  //****************************************************************************


  template<typename T1, typename T2, typename TR = 
	   typename std::enable_if< is_floating_point_var< 
				      pT<T1>, pT<T2> 
				      >::value
				    && is_same_pT<T1,T2>::value,
				    std::tuple< arma::uword, 
						arma::Col< pT<T1> >,
						arma::field< 
						  arma::Mat< 
						    typename 
						    eT_promoter_var
						    <T1,T2>::type >
						  >
						>
				    >::type >
    inline 
    TR measure(const T1& rho1, 
	       const arma::field<T2>& Ks,
	       arma::uvec sys,
	       arma::uword dim = 2)
  {
    const auto& rho = as_Mat(rho1); 
    
    bool checkV = true;
    if(rho.n_cols == 1)
      checkV = false;
    
#ifndef QIC_LIB_NO_DEBUG
    if(rho.n_elem == 0)
      throw Exception("qic::measure",Exception::type::ZERO_SIZE);

    if(checkV)
      if(rho.n_rows != rho.n_cols)
	throw Exception("qic::measure",
			Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);
    
    if(dim == 0)
      throw Exception("qic::measure",Exception::type::INVALID_DIMS);
#endif
    
    arma::uword n = static_cast<arma::uword>
      (std::llround(std::log(rho.n_rows) / std::log(dim)));

    arma::uvec dim2 = dim * arma::ones<arma::uvec>(n);
    
    return measure(rho,Ks,std::move(sys),std::move(dim2));
  }


  //****************************************************************************


  template<typename T1, typename T2, typename TR = 
	   typename std::enable_if< is_floating_point_var< 
				      pT<T1>, pT<T2> 
				      >::value
				    && is_same_pT<T1,T2>::value,
				    std::tuple< arma::uword, 
						arma::Col< pT<T1> >,
						arma::field< 
						  arma::Mat< 
						    typename 
						    eT_promoter_var
						    <T1,T2>::type >
						  >
						>
				    >::type >
    inline 
    TR measure(const T1& rho1, 
	       const T2& U1,
	       arma::uvec sys,
	       arma::uvec dim)
  {

    const auto& rho = as_Mat(rho1); 
    const auto& U = as_Mat(U1);

    bool checkV = true;
    if(rho.n_cols == 1)
      checkV = false;


    arma::uword D = arma::prod(dim);
    arma::uword Dsys = arma::prod(dim(sys-1));
    arma::uword Dsysbar = D/Dsys;


#ifndef QIC_LIB_NO_DEBUG
    if(rho.n_elem == 0)
      throw Exception("qic::measure",Exception::type::ZERO_SIZE);

    if(U.n_elem == 0)
      throw Exception("qic::measure",Exception::type::ZERO_SIZE);

    if(checkV)
      if(rho.n_rows!=rho.n_cols)
	throw Exception("qic::measure",
			Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);
    
    if( dim.n_elem == 0 || arma::any(dim == 0))
      throw Exception("qic::measure",Exception::type::INVALID_DIMS);

    if( arma::prod(dim)!= rho.n_rows)
      throw Exception("qic::measure",Exception::type::DIMS_MISMATCH_MATRIX);

    if( Dsys != U.n_rows )
      throw Exception("qic::measure",
		      Exception::type::DIMS_MISMATCH_MATRIX);

    if(sys.n_elem > dim.n_elem 
       || arma::find_unique(sys).eval().n_elem != sys.n_elem 
       || arma::any(sys > dim.n_elem) || arma::any(sys == 0) )
      throw Exception("qic::measure",Exception::type::INVALID_SUBSYS);      
#endif


    using mattype = arma::Mat<typename eT_promoter_var<T1,T2>::type>;
    arma::Col< pT<T1> > prob(U.n_cols);
    arma::field< mattype > outstates(U.n_cols);    
    
    outstates.fill(arma::zeros<mattype>(Dsysbar,Dsysbar));


    for (arma::uword i = 0; i < U.n_cols; ++i)
      {
	mattype tmp = apply(rho,(U.col(i)*U.col(i).t()).eval(),sys,dim);
	tmp = TrX(tmp,sys,dim);
	prob.at(i) = std::abs(arma::trace(tmp)); 
	if (prob.at(i) > _precision::eps< pT<T1> >::value)
	  outstates.at(i) = tmp / prob.at(i);
      }
    
    std::discrete_distribution<arma::uword> dd(prob.begin(),
    					       prob.end());
    arma::uword result = dd(rdevs.rng);
    
    return std::make_tuple(result,prob,outstates);
	
  }


  //****************************************************************************


  template<typename T1, typename T2, typename TR = 
	   typename std::enable_if< is_floating_point_var< 
				      pT<T1>, pT<T2> 
				      >::value
				    && is_same_pT<T1,T2>::value,
				    std::tuple< arma::uword, 
						arma::Col< pT<T1> >,
						arma::field< 
						  arma::Mat< 
						    typename 
						    eT_promoter_var
						    <T1,T2>::type >
						  >
						>
				    >::type >
    inline 
    TR measure(const T1& rho1, 
	       const T2& U1,
	       arma::uvec sys,
	       arma::uword dim = 2)
  {
    const auto& rho = as_Mat(rho1); 
    const auto& U = as_Mat(U1);

    bool checkV = true;
    if(rho.n_cols == 1)
      checkV = false;
    
#ifndef QIC_LIB_NO_DEBUG
    if(rho.n_elem == 0)
      throw Exception("qic::measure",Exception::type::ZERO_SIZE);

    if(checkV)
      if(rho.n_rows != rho.n_cols)
	throw Exception("qic::measure",
			Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);
    
    if(dim == 0)
      throw Exception("qic::measure",Exception::type::INVALID_DIMS);
#endif
    
    arma::uword n = static_cast<arma::uword>
      (std::llround(std::log(rho.n_rows) / std::log(dim)));

    arma::uvec dim2 = dim * arma::ones<arma::uvec>(n);
    
    return measure(rho,U,std::move(sys),std::move(dim2));
  }


}
