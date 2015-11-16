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
			   >::type entanglement(const T1& rho1, 
						arma::uvec dim)
  {
    const auto& p = as_Mat(rho1); 

    bool checkV = true;
    if(p.n_cols == 1)
      checkV = false;

#ifndef QIC_LIB_NO_DEBUG
    if(p.n_elem == 0)
      throw Exception("qic::entanglement",Exception::type::ZERO_SIZE);

    if(checkV)
      if(p.n_rows!=p.n_cols)
	throw Exception("qic::entanglement",
			Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);
    
    if( arma::any(dim) == 0)
      throw Exception("qic::entanglement",Exception::type::INVALID_DIMS);

    if(arma::prod(dim)!= p.n_rows)
      throw Exception("qic::entanglement", 
		      Exception::type::DIMS_MISMATCH_MATRIX);
      
    if((dim.n_elem) != 2 )
      throw Exception("qic::entanglement", 
		      Exception::type::NOT_BIPARTITE);
#endif

    return entropy(TrX(p,{1},std::move(dim)));



  }
}
  



