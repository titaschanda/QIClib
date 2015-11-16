
namespace qic
{

  template <typename T1>
  arma::Mat<typename T1::elem_type> BF_channel(const T1& rho1,arma::uword party_no, typename T1::pod_type p)
  {
    typedef typename T1::elem_type eT;
    typedef typename T1::pod_type pT;

    auto rho = as_Mat(rho1);

#ifndef QIC_LIB_NO_DEBUG
    if(rho.n_rows!=pow(2,party_no))
      throw std::invalid_argument("channel(): Party no. does not matches");
#endif

    arma::Mat<pT> E0 = arma::eye< arma::Mat<pT> >(2,2);
    E0 *= std::sqrt(1.0 - 0.5*p);

    arma::Mat<pT> E1;
    E1<<0.0<<1.0<<arma::endr
      <<1.0<<0.0;
    E1 *= std::sqrt(pT(p*0.5) );

    if(party_no == 1)
      {
	rho = E0 * rho * E0.t() + E1 * rho * E1.t();
	return rho;
      }



    for(arma::uword i=0 ; i<party_no ; ++i)
      {
	if(i==0)
	  {
	    arma::Mat<eT> E0_1 = arma::kron(E0,arma::eye< arma::Mat<eT> >(std::pow(2,party_no-1),std::pow(2,party_no-1)));
	    arma::Mat<eT> E1_1 = arma::kron(E1,arma::eye< arma::Mat<eT> >(std::pow(2,party_no-1),std::pow(2,party_no-1)));
	    rho = E0_1 * rho * E0_1.t() + E1_1 * rho * E1_1.t();
	  }
	else if(i>0 && i<party_no-1)
	  {
	    arma::Mat<eT> E0_1 = arma::kron(arma::eye< arma::Mat<eT> >(std::pow(2,i),std::pow(2,1)),
					    arma::kron(E0,arma::eye< arma::Mat<eT> >(std::pow(2,party_no-i-1),std::pow(2,party_no-i-1))));
	    
	    arma::Mat<eT> E1_1 = arma::kron(arma::eye<arma::Mat<eT> >(std::pow(2,i),std::pow(2,1)),
					    arma::kron(E1,arma::eye< arma::Mat<eT> >(std::pow(2,party_no-i-1),std::pow(2,party_no-i-1))));

	    rho = E0_1 * rho * E0_1.t() + E1_1 * rho * E1_1.t();
	  }
	else if(i==party_no-1)
	  {
	    arma::Mat<eT> E0_1 = arma::kron(arma::eye< arma::Mat<eT> >(std::pow(2,party_no-1),std::pow(2,party_no-1)),E0);
	    arma::Mat<eT> E1_1 = arma::kron(arma::eye< arma::Mat<eT> >(std::pow(2,party_no-1),std::pow(2,party_no-1)),E1);
	    rho = E0_1 * rho * E0_1.t() + E1_1 * rho * E1_1.t();
	  }
      }
    return rho;
  }



  template <typename T1>
  arma::Mat<typename T1::elem_type> PF_channel(const T1& rho1,arma::uword party_no, typename T1::pod_type p)
  {
    typedef typename T1::elem_type eT;
    typedef typename T1::pod_type pT;

    auto rho = as_Mat(rho1);

#ifndef QIC_LIB_NO_DEBUG
    if(rho.n_rows!=pow(2,party_no))
      throw std::invalid_argument("channel(): Party no. does not matches");
#endif

    arma::Mat<pT> E0 = arma::eye< arma::Mat<pT> >(2,2);
    E0 *= std::sqrt(1.0 - 0.5*p);

    arma::Mat<pT> E1;
    E1<<1.0<<0.0<<arma::endr
      <<0.0<<-1.0;
    E1 *= std::sqrt(pT(p*0.5) );

    if(party_no == 1)
      {
	rho = E0 * rho * E0.t() + E1 * rho * E1.t();
	return rho;
      }



    for(arma::uword i=0 ; i<party_no ; ++i)
      {
	if(i==0)
	  {
	    arma::Mat<eT> E0_1 = arma::kron(E0,arma::eye< arma::Mat<eT> >(std::pow(2,party_no-1),std::pow(2,party_no-1)));
	    arma::Mat<eT> E1_1 = arma::kron(E1,arma::eye< arma::Mat<eT> >(std::pow(2,party_no-1),std::pow(2,party_no-1)));
	    rho = E0_1 * rho * E0_1.t() + E1_1 * rho * E1_1.t();
	  }
	else if(i>0 && i<party_no-1)
	  {
	    arma::Mat<eT> E0_1 = arma::kron(arma::eye< arma::Mat<eT> >(std::pow(2,i),std::pow(2,1)),
					    arma::kron(E0,arma::eye< arma::Mat<eT> >(std::pow(2,party_no-i-1),std::pow(2,party_no-i-1))));

	    arma::Mat<eT> E1_1 = arma::kron(arma::eye< arma::Mat<eT> >(std::pow(2,i),std::pow(2,1)),
					    arma::kron(E1,arma::eye< arma::Mat<eT> >(std::pow(2,party_no-i-1),std::pow(2,party_no-i-1))));

	    rho = E0_1 * rho * E0_1.t() + E1_1 * rho * E1_1.t();
	  }
	else if(i==party_no-1)
	  {
	    arma::Mat<eT> E0_1 = arma::kron(arma::eye< arma::Mat<eT> >(std::pow(2,party_no-1),std::pow(2,party_no-1)),E0);
	    arma::Mat<eT> E1_1 = arma::kron(arma::eye< arma::Mat<eT> >(std::pow(2,party_no-1),std::pow(2,party_no-1)),E1);
	    rho = E0_1 * rho * E0_1.t() + E1_1 * rho * E1_1.t();
	  }
      }
    return rho;
  }




  template <typename T1>
  arma::Mat< std::complex<typename T1::pod_type> > BPF_channel(const T1& rho1,arma::uword party_no, typename T1::pod_type p)
  {
    typedef typename T1::elem_type eT;
    typedef typename T1::pod_type pT;

    const auto& rho2 = as_Mat(rho1);
    arma::Mat< std::complex<pT> > rho = arma::conv_to<arma::Mat<std::complex<pT>>>::from(rho2);

#ifndef QIC_LIB_NO_DEBUG
    if(rho.n_rows!=pow(2,party_no))
      throw std::invalid_argument("channel(): Party no. does not matches");
#endif

    arma::Mat<pT> E0 = arma::eye< arma::Mat<pT> >(2,2);
    E0 *= std::sqrt(1.0 - 0.5*p);

    std::complex<pT> I(0.0,1.0);

    arma::Mat<std::complex<pT>> E1;
    E1<<0.0<<-I<<arma::endr
      <<I<<0.0;
    E1 *= std::sqrt(pT(p*0.5) );


    if(party_no == 1)
      {
	rho = E0 * rho * E0.t() + E1 * rho * E1.t();
	return rho;
      }



    for(arma::uword i=0 ; i<party_no ; ++i)
      {
	if(i==0)
	  {
	    arma::Mat<eT> E0_1 = arma::kron(E0,arma::eye< arma::Mat<eT> >(std::pow(2,party_no-1),std::pow(2,party_no-1)));

	    arma::Mat<std::complex<pT>> E1_1 = arma::kron(E1,arma::eye< arma::Mat< std::complex<pT> > >(std::pow(2,party_no-1),
													std::pow(2,party_no-1)));
	    
	    rho = E0_1 * rho * E0_1.t() + E1_1 * rho * E1_1.t();
	  }
	else if(i>0 && i<party_no-1)
	  {
	    arma::Mat<eT> E0_1 = arma::kron(arma::eye< arma::Mat<eT> >(std::pow(2,i),std::pow(2,1)),
					    arma::kron(E0,arma::eye< arma::Mat<eT> >(std::pow(2,party_no-i-1),std::pow(2,party_no-i-1))));

	    arma::Mat<std::complex<pT>> E1_1 = arma::kron(arma::eye< arma::Mat< std::complex<pT> > >(std::pow(2,i),std::pow(2,1)),
							  arma::kron(E1,arma::eye< arma::Mat< std::complex<pT> > >(std::pow(2,party_no-i-1),
														   std::pow(2,party_no-i-1))));

	    rho = E0_1 * rho * E0_1.t() + E1_1 * rho * E1_1.t();
	  }
	else if(i==party_no-1)
	  {
	    arma::Mat<eT> E0_1 = arma::kron(arma::eye< arma::Mat<eT> >(std::pow(2,party_no-1),std::pow(2,party_no-1)),E0);
	    arma::Mat<std::complex<pT>> E1_1 = arma::kron(arma::eye< arma::Mat< std::complex<pT> > >(std::pow(2,party_no-1),
												     std::pow(2,party_no-1)),E1);
	    rho = E0_1 * rho * E0_1.t() + E1_1 * rho * E1_1.t();
	  }
      }
    return rho;
  }



}
