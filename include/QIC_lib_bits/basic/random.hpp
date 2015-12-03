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


  template<typename T1, typename TR =
	   typename std::enable_if< is_floating_point_var< pT<T1> >::value
				    && arma::is_Col<T1>::value,
				    arma::Col< eT<T1> >
				    >::type > 
    inline 
    TR randU(const arma::uword& N,
	     const arma::Col< pT<T1> >& range = {0,1})
  {
    
#ifndef QIC_LIB_NO_DEBUG
    if( range.n_elem != 2)
      throw Exception("qic::randU","Not proper range");
#endif
    
    std::uniform_real_distribution< pT<T1> > dis(range.at(0), range.at(1));
    arma::Col< eT<T1> > ret(N);

    if( std::is_same< eT<T1> ,pT<T1> >::value)
      ret.imbue( [&](){return dis(rdevs.rng);} );
    else
      {
	auto& I = _internal::protect_subs::cond_I< eT<T1> >::value;
	ret.imbue( [&](){return dis(rdevs.rng) + I * dis(rdevs.rng);} );
      }
    
    return ret;
  }


  //****************************************************************************


  template<typename T1, typename TR =
	   typename std::enable_if< is_floating_point_var< pT<T1> >::value
				    && arma::is_Mat_only<T1>::value,
				    arma::Mat< eT<T1> >
				    >::type >
    inline 
    TR randU(const arma::uword& m,
	     const arma::uword& n,
	     const arma::Col< pT<T1> >& range = {0,1})
  {

#ifndef QIC_LIB_NO_DEBUG
    if( range.n_elem != 2)
      throw Exception("qic::randU","Not proper range");
#endif
    
    std::uniform_real_distribution< pT<T1> > dis(range.at(0), range.at(1));
    arma::Mat< eT<T1> > ret(m,n);
    
    if( std::is_same< eT<T1> ,pT<T1> >::value)
      ret.imbue( [&](){return dis(rdevs.rng);} );
    else
      {
	auto& I = _internal::protect_subs::cond_I< eT<T1> >::value;
	ret.imbue( [&](){return dis(rdevs.rng) + I * dis(rdevs.rng);} );
      }


    return ret;
  }


  //****************************************************************************


  template<typename T1, typename TR =
	   typename std::enable_if< is_floating_point_var< pT<T1> >::value
				    && arma::is_Col<T1>::value,
				    arma::Col< eT<T1> >
				    >::type > 
    inline 
    TR randN(const arma::uword& N,
	     const arma::Col< pT<T1> >& meansd = {0,1})
  {

#ifndef QIC_LIB_NO_DEBUG
    if( meansd.n_elem != 2)
      throw Exception("qic::randN","Not proper mean and standard deviation");
#endif
    
    std::normal_distribution< pT<T1> > dis(meansd.at(0), meansd.at(1));
    arma::Col< eT<T1> > ret(N);
    
    if( std::is_same< eT<T1> ,pT<T1> >::value)
      ret.imbue( [&](){return dis(rdevs.rng);} );
    else
      {
	auto& I = _internal::protect_subs::cond_I< eT<T1> >::value;
	ret.imbue( [&](){return dis(rdevs.rng) + I * dis(rdevs.rng);} );
      }
    
    return ret;
  }


  //****************************************************************************


  template<typename T1, typename TR =
	   typename std::enable_if< is_floating_point_var< pT<T1> >::value
				    && arma::is_Mat_only<T1>::value,
				    arma::Mat< eT<T1> >
				    >::type > 
    inline 
    TR randN(const arma::uword& m,
	     const arma::uword& n,
	     const arma::Col< pT<T1> >& meansd = {0,1})
  {


#ifndef QIC_LIB_NO_DEBUG
    if( meansd.n_elem != 2)
      throw Exception("qic::randN","Not proper mean and standard deviation");
#endif
    
    std::normal_distribution< pT<T1> > dis(meansd.at(0), meansd.at(1));
    arma::Mat< eT<T1> > ret(m,n);
    
    if( std::is_same< eT<T1> ,pT<T1> >::value)
      ret.imbue( [&](){return dis(rdevs.rng);} );
    else
      {
	auto& I = _internal::protect_subs::cond_I< eT<T1> >::value;
	ret.imbue( [&](){return dis(rdevs.rng) + I * dis(rdevs.rng);} );
      }

    return ret;
  }


  //****************************************************************************


  template<typename T1, typename TR =
	   typename std::enable_if< is_arma_type_var<T1>::value
				    && arma::is_Col<T1>::value,
				    arma::Col< eT<T1> >
				    >::type > 
    inline 
    TR randI(const arma::uword& N,
	     const arma::Col< arma::sword >& range = 
	       { std::is_unsigned< pT<T1> >::value ? 
		   0 : std::numeric_limits< arma::sword >::min(),
		   std::numeric_limits< arma::sword >::max()})
  {
    
#ifndef QIC_LIB_NO_DEBUG
    if( range.n_elem != 2)
      throw Exception("qic::randI","Not proper range");
    
    if( std::is_unsigned< pT<T1> >::value && arma::any(range < 0) )
      throw Exception("qic::randI","Negative range for unsigned type");
#endif
    
    std::uniform_int_distribution< arma::sword > dis(range.at(0), range.at(1));
    arma::Col< eT<T1> > ret(N);

    if( std::is_same< eT<T1> ,pT<T1> >::value)
      ret.imbue( [&](){return static_cast< pT<T1> >(dis(rdevs.rng));} );
    else
      {
	auto& I = _internal::protect_subs::cond_I< eT<T1> >::value;
	ret.imbue( [&](){return static_cast< pT<T1> >(dis(rdevs.rng)) 
	      + I * static_cast< pT<T1> >(dis(rdevs.rng));} );
      }
    
    return ret;
  }


  //****************************************************************************


  template<typename T1, typename TR =
	   typename std::enable_if< is_arma_type_var<T1>::value
				    && arma::is_Mat_only<T1>::value,
				    arma::Mat< eT<T1> >
				    >::type >
    inline 
    TR randI(const arma::uword& m,
	     const arma::uword& n,
	     const arma::Col< arma::sword >& range = 
	       { std::is_unsigned< pT<T1> >::value ? 
		   0 : std::numeric_limits< arma::sword >::min(),
		   std::numeric_limits< arma::sword >::max()})
  {

#ifndef QIC_LIB_NO_DEBUG
    if( range.n_elem != 2)
      throw Exception("qic::randI","Not proper range");
    
    if( std::is_unsigned< pT<T1> >::value && arma::any(range < 0) )
      throw Exception("qic::randI","Negative range for unsigned type");
#endif
    
    std::uniform_int_distribution< arma::sword > dis(range.at(0), range.at(1));
    arma::Mat< eT<T1> > ret(m,n);
    
    if( std::is_same< eT<T1> ,pT<T1> >::value)
      ret.imbue( [&](){return static_cast< pT<T1> >(dis(rdevs.rng));} );
    else
      {
	auto& I = _internal::protect_subs::cond_I< eT<T1> >::value;
	ret.imbue( [&](){return static_cast< pT<T1> >(dis(rdevs.rng)) 
	      + I * static_cast< pT<T1> >(dis(rdevs.rng));} );
      }

    return ret;
  }


  //****************************************************************************





}
