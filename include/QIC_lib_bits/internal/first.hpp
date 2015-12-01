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

  namespace _internal
  {

    namespace protect_subs
    {
     
      template<typename T1>
      struct cond_I1
      {
	static const T1 value;
      };
      template<typename T1> const T1 cond_I1<T1>::value = {0,1};
      
      template<typename T1>
      struct cond_I0
      {
	static const T1 value;
      };
      template<typename T1> const T1 cond_I0<T1>::value = 0;
      

    }

  }

}


