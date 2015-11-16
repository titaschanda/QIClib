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
			   pT<T1> 
			   >::type concurrence(const T1& rho1)
  {
    const auto& p = as_Mat(rho1);

#ifndef QIC_LIB_NO_DEBUG
    if(p.n_elem == 0)
      throw Exception("qic::concurrence",Exception::type::ZERO_SIZE);

    if(p.n_rows!=p.n_cols)
      throw Exception("qic::concurrence",
		      Exception::type::MATRIX_NOT_SQUARE);
    
    if (p.n_rows != 4)
      throw Exception("qic::concurrence", 
		      Exception::type::NOT_QUBIT_SUBSYS);
#endif

    auto& S2 = STATES< pT<T1> >::get_instance().S.at(2);
      
    typename arma::Mat< std::complex< pT<T1> > >::template fixed<4,4> pbar = 
      p*arma::kron(S2,S2)*arma::conj(p)*arma::kron(S2,S2);

    typename arma::Col< pT<T1> >::template fixed<4> eig = 
      arma::sort(arma::real(arma::eig_gen(pbar)));

    for(auto& i : eig)
      {
	if( i  <  static_cast< pT<T1> >(_precision::eps) )
	  i = 0.0;
      }

    return std::max(static_cast< pT<T1> >(0.0), 
		    std::sqrt(eig.at(3)) - std::sqrt(eig.at(2))
		    - std::sqrt(eig.at(1)) - std::sqrt(eig.at(0)));
  }
}
  



