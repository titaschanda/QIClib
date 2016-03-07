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

template< typename T1, typename T2, typename TR =
          typename std::enable_if< is_arma_type_var<T1, T2>::value
                                   && is_same_pT<T1, T2>::value,
                                   arma::Mat<
                                     typename eT_promoter_var<T1, T2>::type
                                     >
                                   >::type>
    inline
    TR dsum(const T1& rho11,
            const T2& rho12
            ) {
  const auto& rho1 = as_Mat(rho11);
  const auto& rho2 = as_Mat(rho12);

  using mattype = arma::Mat< typename eT_promoter_var<T1, T2>::type >;

#ifndef QIC_LIB_NO_DEBUG
  if ( rho1.n_elem == 0 || rho2.n_elem == 0 )
    throw Exception("qic::dsum", Exception::type::ZERO_SIZE);
#endif

  const arma::uword n1 = rho1.n_rows;
  const arma::uword n2 = rho2.n_rows;

  const arma::uword m1 = rho1.n_cols;
  const arma::uword m2 = rho2.n_cols;

  mattype ret = arma::zeros< mattype >(n1+n2, m1+m2);

  ret.submat(0, 0, n1-1, m1-1) = rho1;
  ret.submat(n1, m1, n1+n2-1, m1+m2-1) = rho2;

  return ret;
}



template< typename T1, typename T2, typename... T3,
          typename TR =
          typename std::enable_if< is_arma_type_var<T1, T2, T3...>::value
                                   && is_same_pT<T1, T2, T3...>::value,
                                   arma::Mat<
                                     typename eT_promoter_var<
                                       T1, T2, T3...>::type
                                     >
                                   >::type>
    inline
    TR dsum(const T1& rho1,
            const T2& rho2,
            const T3&... rho3
            ) {
  return dsum(rho1,
              dsum(rho2, rho3...));
}



template< typename T1, typename TR =
          typename std::enable_if< is_arma_type_var<T1>::value,
                                   arma::Mat< eT<T1> >
                                   >::type>
inline
TR dsum(const arma::field<T1>& rho
        ) {
#ifndef QIC_LIB_NO_DEBUG
  if ( rho.n_elem == 0 )
    throw Exception("qic::dsum", Exception::type::ZERO_SIZE);

  for ( auto&& a : rho )
    if ( a.eval().n_elem == 0  )
      throw Exception("qic::dsum", Exception::type::ZERO_SIZE);
#endif

  arma::uword N(0), M(0);
  for ( arma::uword i = 0 ; i < rho.n_elem ; ++i ) {
    N += rho.at(i).eval().n_rows;
    M += rho.at(i).eval().n_cols;
  }

  arma::Mat< eT<T1> > ret = arma::zeros< arma::Mat< eT<T1> >
                                         >(N, M);
  arma::uword n(0), m(0);
  for ( arma::uword i = 0 ; i < rho.n_elem ; ++i ) {
    ret.submat(n, m, n + rho.at(i).eval().n_rows - 1,
               m + rho.at(i).eval().n_cols -1) = rho.at(i).eval();
    n += rho.at(i).eval().n_rows;
    m += rho.at(i).eval().n_cols;
  }


  return ret;
}




template< typename T1, typename TR =
          typename std::enable_if< is_arma_type_var<T1>::value,
                                   arma::Mat< eT<T1> >
                                   >::type>
inline
TR dsum(const std::vector<T1>& rho
        ) {
#ifndef QIC_LIB_NO_DEBUG
  if ( rho.size() == 0 )
    throw Exception("qic::dsum", Exception::type::ZERO_SIZE);

  for ( auto&& a : rho )
    if ( a.eval().n_elem == 0  )
      throw Exception("qic::dsum", Exception::type::ZERO_SIZE);
#endif


  arma::uword N(0), M(0);
  for ( arma::uword i = 0 ; i < rho.size() ; ++i ) {
    N += rho[i].eval().n_rows;
    M += rho[i].eval().n_cols;
  }

  arma::Mat< eT<T1> > ret = arma::zeros< arma::Mat< eT<T1> >
                                         >(N, M);
  arma::uword n(0), m(0);
  for ( arma::uword i = 0 ; i < rho.size() ; ++i ) {
    ret.submat(n, m, n + rho[i].eval().n_rows - 1,
               m + rho[i].eval().n_cols -1) = rho[i].eval();
    n += rho[i].eval().n_rows;
    m += rho[i].eval().n_cols;
  }

  return ret;
}




template< typename T1 >
inline
arma::Mat<T1> dsum(const std::initializer_list< arma::Mat<T1> >& rho
                   ) {
  return dsum(static_cast< std::vector< arma::Mat<T1> > >(rho));
}





template< typename T1, typename TR =
          typename std::enable_if< is_arma_type_var<T1>::value,
                                   arma::Mat< eT<T1> >
                                   >::type>
inline
TR dsum_pow(const T1& rho1,
            arma::uword n ) {
  const auto& rho = as_Mat(rho1);

#ifndef QIC_LIB_NO_DEBUG
  if ( rho.n_elem == 0 )
    throw Exception("qic::dsum_pow", Exception::type::ZERO_SIZE);

  if ( n == 0 )
    throw Exception("qic::dsum_pow", Exception::type::OUT_OF_RANGE);
#endif

  std::vector< arma::Mat< eT<T1> > > ret(n, rho);
  return dsum(ret);
}


}  // namespace qic
