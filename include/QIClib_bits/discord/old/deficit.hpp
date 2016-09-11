/*
 * QIClib (Quantum information and computation library)
 *
 * Copyright (c) 2015 - 2016  Titas Chanda (titas.chanda@gmail.com)
 *
 * This file is part of QIClib.
 *
 * QIClib is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * QIClib is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QIClib.  If not, see <http://www.gnu.org/licenses/>.
 */

namespace qic {

//******************************************************************************

namespace {

//******************************************************************************

inline void set_deficit_global_opt(nlopt::algorithm a) noexcept {
  protect::_deficit_global_opt = a;
}

inline void set_deficit_global_xtol(double a) noexcept {
  protect::_deficit_global_xtol = a;
}

inline void set_deficit_global_ftol(double a) noexcept {
  protect::_deficit_global_ftol = a;
}

inline void set_deficit_global(bool a) noexcept {
  protect::_deficit_global = a;
}

inline void set_deficit_local_opt(nlopt::algorithm a) noexcept {
  protect::_deficit_local_opt = a;
}

inline void set_deficit_local_xtol(double a) noexcept {
  protect::_deficit_local_xtol = a;
}

inline void set_deficit_local_ftol(double a) noexcept {
  protect::_deficit_local_ftol = a;
}

inline void set_deficit_theta_phi_range(double a, double b) noexcept {
  protect::_deficit_theta_range = a;
  protect::_deficit_phi_range = b;
}

inline void set_deficit_theta_phi_initial(double a, double b) noexcept {
  protect::_deficit_theta_ini = a;
  protect::_deficit_phi_ini = b;
}

//******************************************************************************

namespace protect {

//******************************************************************************

template <typename T1> struct TO_PASS_def {
  const T1& rho;
  const arma::Mat<trait::pT<T1> >& eye2;
  const arma::Mat<trait::pT<T1> >& eye3;
  const arma::Mat<trait::pT<T1> >& eye4;
  arma::uword nodal;
  arma::uword party_no;

  TO_PASS_def(const T1& a, const arma::Mat<trait::pT<T1> >& c,
              const arma::Mat<trait::pT<T1> >& d,
              const arma::Mat<trait::pT<T1> >& e, arma::uword f, arma::uword g)
      : rho(a), eye2(c), eye3(d), eye4(e), nodal(f), party_no(g) {}
  ~TO_PASS_def() {}
};

//******************************************************************************

template <typename T1>
double def_def(const std::vector<double>& x, std::vector<double>& grad,
               void* my_func_data) {
  (void)grad;
  std::complex<trait::pT<T1> > I(0.0, 1.0);
  trait::pT<T1> theta = static_cast<trait::pT<T1> >(x[0]);
  trait::pT<T1> phi = static_cast<trait::pT<T1> >(x[1]);

  TO_PASS_def<arma::Mat<trait::eT<T1> > >* pB =
    static_cast<TO_PASS_def<arma::Mat<trait::eT<T1> > >*>(my_func_data);

  auto& u = SPM<trait::pT<T1> >::get_instance().basis2.at(0, 0);
  auto& d = SPM<trait::pT<T1> >::get_instance().basis2.at(1, 0);

  arma::Mat<std::complex<trait::pT<T1> > > proj1 =
    std::cos(static_cast<trait::pT<T1> >(0.5) * theta) * u +
    std::exp(I * phi) * std::sin(static_cast<trait::pT<T1> >(0.5) * theta) * d;

  arma::Mat<std::complex<trait::pT<T1> > > proj2 =
    std::sin(static_cast<trait::pT<T1> >(0.5) * theta) * u -
    std::exp(I * phi) * std::cos(static_cast<trait::pT<T1> >(0.5) * theta) * d;

  proj1 *= proj1.t();
  proj2 *= proj2.t();

  if ((*pB).nodal == 1) {
    proj1 = kron(proj1, (*pB).eye2);
    proj2 = kron(proj2, (*pB).eye2);

  } else if ((*pB).party_no == (*pB).nodal) {
    proj1 = kron((*pB).eye2, proj1);
    proj2 = kron((*pB).eye2, proj2);

  } else {
    proj1 = kron(kron((*pB).eye3, proj1), (*pB).eye4);
    proj2 = kron(kron((*pB).eye3, proj2), (*pB).eye4);
  }

  arma::Mat<std::complex<trait::pT<T1> > > rho_1 =
    (proj1 * ((*pB).rho) * proj1);
  arma::Mat<std::complex<trait::pT<T1> > > rho_2 =
    (proj2 * ((*pB).rho) * proj2);

  rho_1 += rho_2;
  trait::pT<T1> S_max = entropy(rho_1);
  return static_cast<double>(S_max);
}

//******************************************************************************

}  // namespace protect

//******************************************************************************

template <typename T1,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1> >::value,
            typename arma::Col<trait::pT<T1> >::template fixed<3> >::type>
inline TR deficit(const T1& rho1, arma::uword nodal, arma::uvec dim) {
  const auto& rho = _internal::as_Mat(rho1);
  arma::uword party_no = dim.n_elem;
  arma::uword dim1 = arma::prod(dim);

#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::deficit", Exception::type::ZERO_SIZE);

  if (rho.n_rows != rho.n_cols)
    throw Exception("qic::deficit", Exception::type::MATRIX_NOT_SQUARE);

  if (arma::any(dim == 0))
    throw Exception("qic::deficit", Exception::type::INVALID_DIMS);

  if (dim1 != rho.n_rows)
    throw Exception("qic::deficit", Exception::type::DIMS_MISMATCH_MATRIX);

  if (nodal <= 0 || nodal > party_no)
    throw Exception("qic::deficit", "Invalid measured party index");

  if (dim(nodal - 1) != 2)
    throw Exception("qic::deficit", "Measured party is not qubit");
#endif

  auto S_A_B = entropy(rho);

  dim1 /= 2;
  arma::uword dim2(1);
  for (arma::uword i = 0; i < nodal - 1; ++i) dim2 *= dim.at(i);

  arma::uword dim3(1);
  for (arma::uword i = nodal; i < party_no; ++i) dim3 *= dim.at(i);

  arma::Mat<trait::pT<T1> > eye2 =
    arma::eye<arma::Mat<trait::pT<T1> > >(dim1, dim1);
  arma::Mat<trait::pT<T1> > eye3 =
    arma::eye<arma::Mat<trait::pT<T1> > >(dim2, dim2);
  arma::Mat<trait::pT<T1> > eye4 =
    arma::eye<arma::Mat<trait::pT<T1> > >(dim3, dim3);

  protect::TO_PASS_def<arma::Mat<trait::eT<T1> > > pass(rho, eye2, eye3, eye4,
                                                        nodal, party_no);

  std::vector<double> lb(2);
  std::vector<double> ub(2);

  lb[0] = 0.0;
  lb[1] = 0.0;
  ub[0] = protect::_deficit_theta_range * arma::datum::pi;
  ub[1] = protect::_deficit_phi_range * arma::datum::pi;

  std::vector<double> x(2);
  x[0] = protect::_deficit_theta_ini * arma::datum::pi;
  x[1] = protect::_deficit_phi_ini * arma::datum::pi;

  double minf;

  if (protect::_deficit_global == true) {
    double minf1;
    nlopt::opt opt1(protect::_deficit_global_opt, 2);
    opt1.set_lower_bounds(lb);
    opt1.set_upper_bounds(ub);
    opt1.set_min_objective(protect::def_def<T1>, static_cast<void*>(&pass));
    opt1.set_xtol_rel(protect::_deficit_global_xtol);
    opt1.set_ftol_rel(protect::_deficit_global_ftol);
    opt1.optimize(x, minf1);
  }

  nlopt::opt opt(protect::_deficit_local_opt, 2);
  opt.set_lower_bounds(lb);
  opt.set_upper_bounds(ub);
  opt.set_min_objective(protect::def_def<T1>, static_cast<void*>(&pass));
  opt.set_xtol_rel(protect::_deficit_local_xtol);
  opt.set_ftol_rel(protect::_deficit_local_ftol);
  opt.optimize(x, minf);

  trait::pT<T1> D = -S_A_B + static_cast<trait::pT<T1> >(minf);

  typename arma::Col<trait::pT<T1> >::template fixed<3> ret{
    D, static_cast<trait::pT<T1> >(x[0]), static_cast<trait::pT<T1> >(x[1])};

  return ret;
}

//******************************************************************************

}  // namespace

//******************************************************************************

}  // namespace qic
