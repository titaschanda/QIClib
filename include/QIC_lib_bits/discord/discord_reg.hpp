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

  template <typename T1, typename TR =
	    typename std::enable_if< is_floating_point_var< pT<T1> >::value,
				     typename arma::Col<
				       typename T1::pod_type >::
				     template fixed<3>
				     >::type >
  TR discord_reg(const T1& rho1,arma::uword nodal, arma::uvec dim)
  {
    const auto& rho = as_Mat(rho1);
    arma::uword party_no = dim.n_elem;
    arma::uword dim1 = arma::prod(dim);

#ifndef QIC_LIB_NO_DEBUG   
    if(rho.n_elems == 0)
      throw Exception("qic::discord_reg", Exception::type::ZERO_SIZE);

    if(rho.n_rows!=rho.n_cols)
      throw Exception("qic::discord_reg",Exception::type::MATRIX_NOT_SQUARE);
     
    if(any(dim == 0))
      throw Exception("qic::discord_reg",Exception::type::INVALID_DIMS);
      
    if( dim1 != rho.n_rows)
      throw Exception("qic::discord_reg", 
		      Exception::type::DIMS_MISMATCH_MATRIX);

    if(nodal<=0 || nodal > party_no)
      throw Exception("qic::discord_reg", "Invalid measured party index");

    if( dim(nodal-1) != 2 )
      throw Exception("qic::discord_reg", "Measured party is not qubit");
#endif

    arma::uvec party = arma::zeros<arma::uvec>(party_no); 
    for(arma::uword i=0; i<party_no ; i++)
      party.at(i)=i+1;
    
    arma::uvec rest = party;
    rest.shed_row(nodal-1);


    auto rho_A = TrX(rho,rest,dim);
    auto rho_B = TrX(rho,{nodal},dim); 

    auto S_A = entropy(rho_A); 
    auto S_B = entropy(rho_B); 
    auto S_A_B = entropy(rho); 
    auto I1 = S_A + S_B - S_A_B;


    typename arma::Col< pT<T1> >::template fixed<3> ret; 

    dim1 /= 2;
    arma::uword dim2 (1);
    for(arma::uword i = 0 ;i < nodal-1 ; ++i)
      dim2 *= dim.at(i);
    arma::uword dim3 (1);
    for(arma::uword i = nodal ; i < party_no ; ++i)
      dim3 *= dim.at(i);
	  
    arma::Mat< pT<T1> > eye2 = arma::eye< arma::Mat< pT<T1> > >(dim1,dim1);
    arma::Mat< pT<T1> > eye3 = arma::eye< arma::Mat< pT<T1> > >(dim2,dim2);
    arma::Mat< pT<T1> > eye4 = arma::eye< arma::Mat< pT<T1> > >(dim3,dim3);

	  
	

    for(arma::uword i=0 ; i<3 ; ++i)
      {

	arma::Mat< std::complex< pT<T1> > > proj1 = 
	  STATES< pT<T1> >::get_instance().proj2.at(0,i+1);
	
	arma::Mat< std::complex< pT<T1> > > proj2 = 
	  STATES< pT<T1> >::get_instance().proj2.at(1,i+1);

	if(nodal ==1)
	  {
	    proj1 = kron(proj1,eye2 );
	    proj2 = kron(proj2,eye2 );
	  }
	else if (party_no==nodal)
	  {
	    proj1 = kron(eye2,proj1);
	    proj2 = kron(eye2,proj2);
	  }
	else
	  {
	    proj1 = kron(kron(eye3,proj1),eye4);
	    proj2 = kron(kron(eye3,proj2),eye4);
	  }
  
  

	arma::Mat< std::complex< pT<T1> > > rho_1 = (proj1*rho*proj1);
	arma::Mat< std::complex< pT<T1> > > rho_2 = (proj2*rho*proj2);
   

	pT<T1> p1 = std::real(arma::trace(rho_1));
	pT<T1> p2 = std::real(arma::trace(rho_2));
 
	pT<T1> S_max = 0.0;
	if( p1 > _precision::eps< pT<T1> >::value)
	  {
	    rho_1 /= p1;
	    S_max += p1*entropy(rho_1);
	  }
	if( p2 > _precision::eps< pT<T1> >::value )
	  {
	    rho_2 /= p2;
	    S_max += p2*entropy(rho_2);
	  }
	ret.at(i) = I1 - S_B + S_max;
      }
	
    return(ret);
  }
}

  

