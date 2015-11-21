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
			   pT<T1> 
			   >::type EoF(const T1& rho1)
  {
    const auto& p = as_Mat(rho1); 

    bool checkV = true;
    if(p.n_cols == 1)
      checkV = false;

#ifndef QIC_LIB_NO_DEBUG
    if(p.n_elem == 0)
      throw Exception("qic::EoF",Exception::type::ZERO_SIZE);

    if(checkV)
      if(p.n_rows!=p.n_cols)
	throw Exception("qic::EoF",
			Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);
    
    if (p.n_rows != 4)
      throw Exception("qic::EoF",Exception::type::NOT_QUBIT_SUBSYS);
#endif

    if(!checkV)
      return entanglement(p,{2,2});

    else
      {
	pT<T1> ret = 0.5*(1.0 + std::sqrt(1.0 - std::pow(concurrence(p),2.0))) ;
	pT<T1> ret2(0.0);
	if(ret > static_cast< pT<T1> >(_precision::eps) )
	  ret2 -= ret*std::log2(ret);
	if(1.0-ret > static_cast< pT<T1> >(_precision::eps) )
	  ret2 -= (1.0-ret)*std::log2(1.0-ret);
	return ret2;
      }
  }
    
  
 
}


