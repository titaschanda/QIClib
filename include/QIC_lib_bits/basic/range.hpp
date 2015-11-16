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

  //***************************************************************************


  template <typename T1,typename T2, typename T3>
  inline 
  typename std::enable_if< 
    std::is_arithmetic<T1>::value 
  && std::is_arithmetic<T2>::value
  && std::is_arithmetic<T3>::value,
    std::vector< typename  is_promotable_var<T1,T2,T3>::type > 
    >::type range(T1 start, T2 stop, T3 step)
  {
    using pTr = typename is_promotable_var<T1,T2,T3>::type;
    
    if (static_cast<pTr>(step) == static_cast<pTr>(0))
      throw Exception("qic::range","Step must be non-zero");
   
    bool check = (static_cast<pTr>(step) > static_cast<pTr>(0)) ? 
      (static_cast<pTr>(start) < static_cast<pTr>(stop)) : 
      (static_cast<pTr>(start) > static_cast<pTr>(stop));
    
    if(!check)
      throw Exception("qic::range","Invalid start, stop, step");

    std::vector<pTr> result;
    pTr i = static_cast<pTr>(start);
    while ((static_cast<pTr>(step) > static_cast<pTr>(0)) ? 
	   (i < static_cast<pTr>(stop)) : (i > static_cast<pTr>(stop)))
      {
	result.push_back(i);
	i += static_cast<pTr>(step);
      }

    return result;
  }


  //****************************************************************************


  template <typename T1, typename T2>
  inline 
  typename std::enable_if< 
    std::is_arithmetic<T1>::value 
  && std::is_arithmetic<T2>::value,
    std::vector< typename  is_promotable_var<T1,T2>::type > 
    >::type range(T1 start, T2 stop)
  {
    using pTr = typename  is_promotable_var<T1,T2>::type;
    return range(static_cast<pTr>(start), 
		 static_cast<pTr>(stop), 
		 static_cast<pTr>(1));
  }
  

  //****************************************************************************


  template <typename T1>
  inline 
  typename std::enable_if< std::is_arithmetic<T1>::value,
			   std::vector<T1> 
			   >::type range(T1 stop)
  {
    return range(static_cast<T1>(0), stop, static_cast<T1>(1));
  }


}
