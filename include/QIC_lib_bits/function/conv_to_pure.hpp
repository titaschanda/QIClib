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



namespace qic {

template< typename T1, typename TR =
          typename std::enable_if< is_floating_point_var< pT<T1> >::value,
                                   arma::Col< eT<T1> >
                                   >::type >
inline
TR conv_to_pure(const T1& rho1
                ) {
  const auto& rho = as_Mat(rho1);

#ifndef QIC_LIB_DEBUG
  if ( rho.n_elem == 0 )
    throw Exception("qic::conv_to_pure", Exception::type::ZERO_SIZE);
#endif

  if ( rho.n_cols == 1 )
    return rho;


#ifndef QIC_LIB_NO_DEBUG
  else if ( rho.n_rows != rho.n_cols )
    throw Exception("qic::conv_to_pure",
                    Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);
#endif

  arma::Mat< eT<T1> > eig_vec;
  arma::Col< pT<T1> > eig_val;

  if ( rho.n_rows > 20 )
    arma::eig_sym(eig_val, eig_vec, rho, "dc");
  else
    arma::eig_sym(eig_val, eig_vec, rho, "std");

  return eig_vec.col(eig_vec.n_cols  - 1);
}

}  // namespace qic
