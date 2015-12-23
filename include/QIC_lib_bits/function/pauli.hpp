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
                                   arma::Mat< pT<T1> >
                                   >::type >
inline
TR conv_to_HS(const T1& rho1
              ) {
  const auto& p = as_Mat(rho1);

#ifndef QIC_LIB_NO_DEBUG
  if ( p.n_elem == 0 )
    throw Exception("qic::conv_to_HS", Exception::type::ZERO_SIZE);

  if ( p.n_rows != p.n_cols )
    throw Exception("qic::conv_to_HS",
                    Exception::type::MATRIX_NOT_SQUARE);

  if ( p.n_rows != 4 )
    throw Exception("qic::conv_to_HS", Exception::type::NOT_QUBIT_SUBSYS);
#endif

  auto& S = STATES< pT<T1> >::get_instance().S;

  arma::Mat< pT<T1> > ret = arma::zeros< arma::Mat< pT<T1> > >(4, 4);

  for ( arma:: uword j = 0 ; j < 4 ; ++j ) {
    for ( arma::uword i = 0 ; i < 4 ; ++i )
      ret.at(i, j) = std::real(arma::trace(arma::kron(S.at(i), S.at(j)) * p));
  }
  return ret;
}



template< typename T1, typename TR =
          typename std::enable_if< is_floating_point_var< pT<T1> >::value,
                                   arma::Mat< std::complex < pT<T1> > >
                                   >::type>
inline
TR conv_to_std(const T1& rho1
               ) {
  const auto& p = as_Mat(rho1);

#ifndef QIC_LIB_NO_DEBUG
  if ( p.n_elem == 0 )
    throw Exception("qic::conv_to_std", Exception::type::ZERO_SIZE);

  if ( !std::is_same< eT<T1>, pT<T1> >::value )
    throw Exception("qic::conv_to_std", "Matrix is not real");


  if ( p.n_rows != p.n_cols )
    throw Exception("qic::conv_to_std",
                    Exception::type::MATRIX_NOT_SQUARE);

  if ( p.n_rows != 4 )
    throw Exception("qic::conv_to_std", Exception::type::NOT_QUBIT_SUBSYS);
#endif


  auto& S = STATES< pT<T1> >::get_instance().S;

  arma::Mat< std::complex < pT<T1> > > ret =
      arma::zeros< arma::Mat< std::complex< pT<T1> > > >(4, 4);

  for ( arma:: uword j = 0 ; j < 4 ; ++j ) {
    for ( arma::uword i = 0 ; i < 4 ; ++i )
      ret += p.at(i, j) * arma::kron(S.at(i), S.at(j)) *0.25;
  }
  return ret;
}

}  // namespace qic


