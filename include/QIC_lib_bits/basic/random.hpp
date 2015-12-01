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
	     const pT<T1>& a = 0, 
	     const pT<T1>& b = 1)
  {
    std::uniform_real_distribution< pT<T1> > dis(a, b);
    arma::Col< eT<T1> > ret(N);
    
    eT<T1>* mem = ret.memptr();
    arma::uword ii, jj;

    if( std::is_same< eT<T1> ,pT<T1> >::value)
      {
	for(ii=0, jj=1; jj < N; ii+=2, jj+=2)
	  {
	    mem[ii] = dis(rdevs.rng);
	    mem[jj] = dis(rdevs.rng);
	  }
      }

    else
      {
	auto& I = std::conditional< std::is_same< eT<T1> ,pT<T1> >::value, 
				    _internal::protect_subs::cond_I0< eT<T1> >, 
				    _internal::protect_subs::cond_I1< eT<T1> > 
				    >::type::value ;
	for(ii=0, jj=1; jj < N; ii+=2, jj+=2)
	  {
	    mem[ii] = dis(rdevs.rng) + I * dis(rdevs.rng);
	    mem[jj] = dis(rdevs.rng) + I * dis(rdevs.rng);
	  }
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
	     const pT<T1>& a = 0, 
	     const pT<T1>& b = 1)
  {
    std::uniform_real_distribution< pT<T1> > dis(a, b);
    arma::Mat< eT<T1> > ret(m,n);
    
    eT<T1>* mem = ret.memptr();
    arma::uword ii, jj;
    arma::uword N = ret.n_elem;

    if( std::is_same< eT<T1> ,pT<T1> >::value)
      {
	for(ii=0, jj=1; jj < N; ii+=2, jj+=2)
	  {
	    mem[ii] = dis(rdevs.rng);
	    mem[jj] = dis(rdevs.rng);
	  }
      }

    else
      {
	auto& I = std::conditional< std::is_same< eT<T1> ,pT<T1> >::value, 
				    _internal::protect_subs::cond_I0< eT<T1> >, 
				    _internal::protect_subs::cond_I1< eT<T1> > 
				    >::type::value ;
	for(ii=0, jj=1; jj < N; ii+=2, jj+=2)
	  {
	    mem[ii] = dis(rdevs.rng) + I * dis(rdevs.rng);
	    mem[jj] = dis(rdevs.rng) + I * dis(rdevs.rng);
	  }
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
	     const pT<T1>& mean = 0, 
	     const pT<T1>& sd = 1)
  {
    std::normal_distribution< pT<T1> > dis(mean, sd);
    arma::Col< eT<T1> > ret(N);
    
    eT<T1>* mem = ret.memptr();
    arma::uword ii, jj;

    if( std::is_same< eT<T1> ,pT<T1> >::value)
      {
	for(ii=0, jj=1; jj < N; ii+=2, jj+=2)
	  {
	    mem[ii] = dis(rdevs.rng);
	    mem[jj] = dis(rdevs.rng);
	  }
      }

    else
      {
	auto& I = std::conditional< std::is_same< eT<T1> ,pT<T1> >::value, 
				    _internal::protect_subs::cond_I0< eT<T1> >, 
				    _internal::protect_subs::cond_I1< eT<T1> > 
				    >::type::value ;
	for(ii=0, jj=1; jj < N; ii+=2, jj+=2)
	  {
	    mem[ii] = dis(rdevs.rng) + I * dis(rdevs.rng);
	    mem[jj] = dis(rdevs.rng) + I * dis(rdevs.rng);
	  }
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
	     const pT<T1>& mean = 0, 
	     const pT<T1>& sd = 1)
  {
    std::normal_distribution< pT<T1> > dis(mean, sd);
    arma::Mat< eT<T1> > ret(m,n);
    
    eT<T1>* mem = ret.memptr();
    arma::uword ii, jj;
    arma::uword N = ret.n_elem;

    if( std::is_same< eT<T1> ,pT<T1> >::value)
      {
	for(ii=0, jj=1; jj < N; ii+=2, jj+=2)
	  {
	    mem[ii] = dis(rdevs.rng);
	    mem[jj] = dis(rdevs.rng);
	  }
      }

    else
      {
	auto& I = std::conditional< std::is_same< eT<T1> ,pT<T1> >::value, 
				    _internal::protect_subs::cond_I0< eT<T1> >, 
				    _internal::protect_subs::cond_I1< eT<T1> > 
				    >::type::value ;
	for(ii=0, jj=1; jj < N; ii+=2, jj+=2)
	  {
	    mem[ii] = dis(rdevs.rng) + I * dis(rdevs.rng);
	    mem[jj] = dis(rdevs.rng) + I * dis(rdevs.rng);
	  }
      }

    return ret;
  }


  //****************************************************************************

}
