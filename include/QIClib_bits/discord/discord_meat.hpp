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

template <typename T1>
inline discord_space<T1>::discord_space(T1* rho1, arma::uword nodal,
                                        arma::uvec dim)
    : _rho(rho1), _nodal(nodal), _n_cols(rho1->n_cols), _n_rows(rho1->n_rows),
      _is_minfo_computed(false), _is_computed(false), _is_reg_computed(false) {
#ifndef QICLIB_NO_DEBUG
  if (_rho->n_elem == 0)
    throw Exception("qic::discord_space", Exception::type::ZERO_SIZE);

  if (_rho->n_rows != _rho->n_cols)
    throw Exception("qic::discord_space", Exception::type::MATRIX_NOT_SQUARE);
#endif

  init(std::move(dim));
}

//****************************************************************************

template <typename T1>
inline discord_space<T1>::discord_space(T1* rho1, arma::uword nodal,
                                        arma::uword dim)
    : _rho(rho1), _nodal(nodal), _n_cols(rho1->n_cols), _n_rows(rho1->n_rows),
      _is_minfo_computed(false), _is_computed(false), _is_reg_computed(false) {
#ifndef QICLIB_NO_DEBUG
  if (_rho->n_elem == 0)
    throw Exception("qic::discord_space", Exception::type::ZERO_SIZE);

  if (_rho->n_rows != _rho->n_cols)
    throw Exception("qic::discord_space", Exception::type::MATRIX_NOT_SQUARE);

  if (dim == 0)
    throw Exception("qic::discord_space", Exception::type::INVALID_DIMS);
#endif

  arma::uword n = static_cast<arma::uword>(
    QICLIB_ROUND_OFF(std::log(_rho->n_rows) / std::log(dim)));

  arma::uvec dim2(n);
  dim2.fill(dim);

  init(std::move(dim2));
}

//****************************************************************************

template <typename T1> inline void discord_space<T1>::init(arma::uvec dim) {
  _dim = std::move(dim);
  _party_no = _dim.n_elem;

#ifndef QICLIB_NO_DEBUG
  if (arma::any(_dim == 0))
    throw Exception("qic::discord_space", Exception::type::INVALID_DIMS);

  if (arma::prod(_dim) != _rho->n_rows)
    throw Exception("qic::discord_space",
                    Exception::type::DIMS_MISMATCH_MATRIX);

  if (_nodal <= 0 || _nodal > _party_no)
    throw Exception("qic::discord_space", "Invalid measured party index!");
#endif

  _discord2 = (_dim.at(_nodal - 1) == 2);
  _discord3 = (_dim.at(_nodal - 1) == 3);

#ifndef QICLIB_NO_DEBUG
  if (!_discord2 && !_discord3)
    throw Exception("qic::discord_space",
                    "Measured party is not qubit or qutrit!");
#endif

  default_setting();
}

//****************************************************************************

template <typename T1> inline void discord_space<T1>::check_size_change() {
  if (_rho->n_cols != _n_cols || _rho->n_rows != _n_rows)
    throw std::runtime_error(
      "qic::discord_space(): Matrix size changed! Use reset()!");
}

//****************************************************************************

template <typename T1> inline void discord_space<T1>::default_setting() {

  if (_discord2) {
    _discord_global_opt = nlopt::GN_DIRECT_L;
    _discord_global_xtol = 4.0e-2;
    _discord_global_ftol = 0;
    _discord_global = true;
    _discord_local_opt = nlopt::LN_COBYLA;
    _discord_local_xtol = 10 * _precision::eps<double>::value;
    _discord_local_ftol = 0.0;

    _discord_angle_range = {1.0, 2.0};
    _discord_angle_ini = {0.1, 0.1};
  }

  if (_discord3) {
    _discord_global_opt = nlopt::GN_DIRECT_L;
    _discord_global_xtol = 0.25;
    _discord_global_ftol = 0;
    _discord_global = true;
    _discord_local_opt = nlopt::LN_COBYLA;
    _discord_local_xtol = 10 * _precision::eps<double>::value;
    _discord_local_ftol = 0.0;

    _discord_angle_range.set_size(5);
    _discord_angle_range.fill(2.0);
    _discord_angle_ini.set_size(5);
    _discord_angle_ini.fill(0.1);
  }
}

//****************************************************************************

template <typename T1>
inline discord_space<T1>&
discord_space<T1>::global_algorithm(nlopt::algorithm a) noexcept {
  _discord_global_opt = a;
  _is_computed = false;
  return *this;
}

//****************************************************************************

template <typename T1>
inline discord_space<T1>& discord_space<T1>::global_xtol(double a) noexcept {
  _discord_global_xtol = a;
  _is_computed = false;
  return *this;
}

//****************************************************************************

template <typename T1>
inline discord_space<T1>& discord_space<T1>::global_ftol(double a) noexcept {
  _discord_global_ftol = a;
  _is_computed = false;
  return *this;
}

//****************************************************************************

template <typename T1>
inline discord_space<T1>& discord_space<T1>::global_opt(bool a) noexcept {
  _discord_global = a;
  _is_computed = false;
  return *this;
}

//****************************************************************************

template <typename T1>
inline discord_space<T1>&
discord_space<T1>::local_algorithm(nlopt::algorithm a) noexcept {
  _discord_local_opt = a;
  _is_computed = false;
  return *this;
}

//****************************************************************************

template <typename T1>
inline discord_space<T1>& discord_space<T1>::local_xtol(double a) noexcept {
  _discord_local_xtol = a;
  _is_computed = false;
  return *this;
}

//****************************************************************************

template <typename T1>
inline discord_space<T1>& discord_space<T1>::local_ftol(double a) noexcept {
  _discord_local_ftol = a;
  _is_computed = false;
  return *this;
}

//****************************************************************************

template <typename T1>
inline discord_space<T1>& discord_space<T1>::angle_range(const arma::vec& a) {
  check_size_change();

#ifndef QICLIB_NO_DEBUG
  if (_discord2 && a.n_elem != 2)
    throw Exception(
      "qic::discord_space::angle_range",
      "Number of elements has to be 2, when measured party is a qubit!");

  if (_discord3 && a.n_elem != 5)
    throw Exception(
      "qic::discord_space::angle_range",
      "Number of elements has to be 5, when measured party is a qutrit!");
#endif

  _discord_angle_range = a;
  _is_computed = false;
  return *this;
}

//****************************************************************************

template <typename T1>
inline discord_space<T1>& discord_space<T1>::initial_angle(const arma::vec& a) {
  check_size_change();

#ifndef QICLIB_NO_DEBUG
  if (_discord2 && a.n_elem != 2)
    throw Exception(
      "qic::discord_space::initial_angle",
      "Number of elements has to be 2, when measured party is a qubit!");

  if (_discord3 && a.n_elem != 5)
    throw Exception(
      "qic::discord_space::initial_angle",
      "Number of elements has to be 5, when measured party is a qutrit!");
#endif

  _discord_angle_ini = a;
  _is_computed = false;
  return *this;
}

//****************************************************************************

template <typename T1> inline void discord_space<T1>::minfo_p() {
  if (!_is_minfo_computed) {
    arma::uvec party = arma::zeros<arma::uvec>(_party_no);
    for (arma::uword i = 0; i < _party_no; i++) party.at(i) = i + 1;

    arma::uvec rest = party;
    rest.shed_row(_nodal - 1);

    auto rho_A = TrX(*_rho, rest, _dim);
    auto S_A = entropy(rho_A);
    auto S_A_B = entropy(*_rho);
    _mutual_info = S_A - S_A_B;
    _is_minfo_computed = true;
  }
}

//****************************************************************************

template <typename T1> inline discord_space<T1>& discord_space<T1>::compute() {
  check_size_change();
  minfo_p();

  if (_discord2) {
    arma::uword dim1 = arma::prod(_dim);
    dim1 /= 2;
    arma::uword dim2(1);
    for (arma::uword i = 0; i < _nodal - 1; ++i) dim2 *= _dim.at(i);
    arma::uword dim3(1);
    for (arma::uword i = _nodal; i < _party_no; ++i) dim3 *= _dim.at(i);

    arma::Mat<trait::pT<T1> > eye2 =
      arma::eye<arma::Mat<trait::pT<T1> > >(dim1, dim1);
    arma::Mat<trait::pT<T1> > eye3 =
      arma::eye<arma::Mat<trait::pT<T1> > >(dim2, dim2);
    arma::Mat<trait::pT<T1> > eye4 =
      arma::eye<arma::Mat<trait::pT<T1> > >(dim3, dim3);

    _internal::TO_PASS<arma::Mat<trait::eT<T1> > > pass(*_rho, eye2, eye3, eye4,
                                                        _nodal, _party_no);

    std::vector<double> lb(2);
    std::vector<double> ub(2);

    lb[0] = 0.0;
    lb[1] = 0.0;
    ub[0] = _discord_angle_range.at(0) * arma::datum::pi;
    ub[1] = _discord_angle_range.at(1) * arma::datum::pi;

    std::vector<double> x(2);
    x[0] = _discord_angle_ini.at(0) * arma::datum::pi;
    x[1] = _discord_angle_ini.at(1) * arma::datum::pi;

    double minf;

    if (_discord_global == true) {
      double minf1;
      nlopt::opt opt1(_discord_global_opt, 2);
      opt1.set_lower_bounds(lb);
      opt1.set_upper_bounds(ub);
      opt1.set_min_objective(_internal::disc_nlopt2<T1>,
                             static_cast<void*>(&pass));
      opt1.set_ftol_rel(_discord_global_ftol);
      opt1.set_xtol_rel(_discord_global_xtol);
      opt1.optimize(x, minf1);
    }

    nlopt::opt opt(_discord_local_opt, 2);
    opt.set_lower_bounds(lb);
    opt.set_upper_bounds(ub);
    opt.set_min_objective(_internal::disc_nlopt2<T1>,
                          static_cast<void*>(&pass));
    opt.set_xtol_rel(_discord_local_xtol);
    opt.set_ftol_rel(_discord_local_ftol);
    opt.optimize(x, minf);

    _result = _mutual_info + static_cast<trait::pT<T1> >(minf);
    _tp = {static_cast<trait::pT<T1> >(x[0]),
           static_cast<trait::pT<T1> >(x[1])};
    _is_computed = true;
  }

  if (_discord3) {
    arma::uword dim1 = arma::prod(_dim);
    dim1 /= 3;
    arma::uword dim2(1);
    for (arma::uword i = 0; i < _nodal - 1; ++i) dim2 *= _dim.at(i);
    arma::uword dim3(1);
    for (arma::uword i = _nodal; i < _party_no; ++i) dim3 *= _dim.at(i);

    arma::Mat<trait::pT<T1> > eye2 =
      arma::eye<arma::Mat<trait::pT<T1> > >(dim1, dim1);
    arma::Mat<trait::pT<T1> > eye3 =
      arma::eye<arma::Mat<trait::pT<T1> > >(dim2, dim2);
    arma::Mat<trait::pT<T1> > eye4 =
      arma::eye<arma::Mat<trait::pT<T1> > >(dim3, dim3);

    _internal::TO_PASS<arma::Mat<trait::eT<T1> > > pass(*_rho, eye2, eye3, eye4,
                                                        _nodal, _party_no);

    std::vector<double> lb(5);
    std::vector<double> ub(5);

    for (arma::uword i = 0; i < 5; i++) {
      lb[i] = 0.0;
      ub[i] = _discord_angle_range.at(i) * arma::datum::pi;
    }

    std::vector<double> x(5);
    for (arma::uword i = 0; i < 5; i++) {
      x[i] = _discord_angle_ini.at(i) * arma::datum::pi;
    }
    double minf;

    if (_discord_global == true) {
      double minf1;
      nlopt::opt opt1(_discord_global_opt, 5);
      opt1.set_lower_bounds(lb);
      opt1.set_upper_bounds(ub);
      opt1.set_min_objective(_internal::disc_nlopt3<T1>,
                             static_cast<void*>(&pass));
      opt1.set_xtol_rel(_discord_global_xtol);
      opt1.set_ftol_rel(_discord_global_ftol);
      opt1.optimize(x, minf1);
    }

    nlopt::opt opt(_discord_local_opt, 5);
    opt.set_lower_bounds(lb);
    opt.set_upper_bounds(ub);
    opt.set_min_objective(_internal::disc_nlopt3<T1>,
                          static_cast<void*>(&pass));
    opt.set_xtol_rel(_discord_local_xtol);
    opt.set_ftol_rel(_discord_local_ftol);
    opt.optimize(x, minf);

    _result = _mutual_info + static_cast<trait::pT<T1> >(minf);
    _tp = _internal::as_type<arma::Col<trait::pT<T1> > >::from(x);
    _is_computed = true;
  }
  return *this;
}

//****************************************************************************

template <typename T1>
inline discord_space<T1>& discord_space<T1>::compute_reg() {
  check_size_change();
  minfo_p();

  if (_discord2) {
    arma::Col<trait::pT<T1> > ret(3);

    arma::uword dim1 = arma::prod(_dim);
    dim1 /= 2;
    arma::uword dim2(1);
    for (arma::uword i = 0; i < _nodal - 1; ++i) dim2 *= _dim.at(i);
    arma::uword dim3(1);
    for (arma::uword i = _nodal; i < _party_no; ++i) dim3 *= _dim.at(i);

    arma::Mat<trait::pT<T1> > eye2 =
      arma::eye<arma::Mat<trait::pT<T1> > >(dim1, dim1);
    arma::Mat<trait::pT<T1> > eye3 =
      arma::eye<arma::Mat<trait::pT<T1> > >(dim2, dim2);
    arma::Mat<trait::pT<T1> > eye4 =
      arma::eye<arma::Mat<trait::pT<T1> > >(dim3, dim3);

    for (arma::uword i = 0; i < 3; ++i) {
      arma::Mat<std::complex<trait::pT<T1> > > proj1 =
        SPM<trait::pT<T1> >::get_instance().proj2.at(0, i + 1);

      arma::Mat<std::complex<trait::pT<T1> > > proj2 =
        SPM<trait::pT<T1> >::get_instance().proj2.at(1, i + 1);

      if (_nodal == 1) {
        proj1 = kron(proj1, eye2);
        proj2 = kron(proj2, eye2);
      } else if (_party_no == _nodal) {
        proj1 = kron(eye2, proj1);
        proj2 = kron(eye2, proj2);
      } else {
        proj1 = kron(kron(eye3, proj1), eye4);
        proj2 = kron(kron(eye3, proj2), eye4);
      }

      arma::Mat<std::complex<trait::pT<T1> > > rho_1 =
        (proj1 * (*_rho) * proj1);
      arma::Mat<std::complex<trait::pT<T1> > > rho_2 =
        (proj2 * (*_rho) * proj2);

      trait::pT<T1> p1 = std::real(arma::trace(rho_1));
      trait::pT<T1> p2 = std::real(arma::trace(rho_2));

      trait::pT<T1> S_max = 0.0;
      if (p1 > _precision::eps<trait::pT<T1> >::value) {
        rho_1 /= p1;
        S_max += p1 * entropy(rho_1);
      }
      if (p2 > _precision::eps<trait::pT<T1> >::value) {
        rho_2 /= p2;
        S_max += p2 * entropy(rho_2);
      }
      ret.at(i) = _mutual_info + S_max;
    }

    _result_reg = arma::min(ret);
    _result_reg_all = std::move(ret);
    _is_reg_computed = true;
  }

  if (_discord3) {
    arma::uword dim1 = arma::prod(_dim);
    dim1 /= 3;
    arma::uword dim2(1);
    for (arma::uword i = 0; i < _nodal - 1; ++i) dim2 *= _dim.at(i);
    arma::uword dim3(1);
    for (arma::uword i = _nodal; i < _party_no; ++i) dim3 *= _dim.at(i);

    arma::Mat<trait::pT<T1> > eye2 =
      arma::eye<arma::Mat<trait::pT<T1> > >(dim1, dim1);
    arma::Mat<trait::pT<T1> > eye3 =
      arma::eye<arma::Mat<trait::pT<T1> > >(dim2, dim2);
    arma::Mat<trait::pT<T1> > eye4 =
      arma::eye<arma::Mat<trait::pT<T1> > >(dim3, dim3);

    arma::Col<trait::pT<T1> > ret(3);

    for (arma::uword i = 0; i < 3; ++i) {
      arma::Mat<std::complex<trait::pT<T1> > > proj1 =
        SPM<trait::pT<T1> >::get_instance().proj3.at(0, i + 1);

      arma::Mat<std::complex<trait::pT<T1> > > proj2 =
        SPM<trait::pT<T1> >::get_instance().proj3.at(1, i + 1);

      arma::Mat<std::complex<trait::pT<T1> > > proj3 =
        SPM<trait::pT<T1> >::get_instance().proj3.at(2, i + 1);

      if (_nodal == 1) {
        proj1 = kron(proj1, eye2);
        proj2 = kron(proj2, eye2);
        proj3 = kron(proj3, eye2);
      } else if (_party_no == _nodal) {
        proj1 = kron(eye2, proj1);
        proj2 = kron(eye2, proj2);
        proj3 = kron(eye2, proj3);
      } else {
        proj1 = kron(kron(eye3, proj1), eye4);
        proj2 = kron(kron(eye3, proj2), eye4);
        proj3 = kron(kron(eye3, proj3), eye4);
      }

      arma::Mat<std::complex<trait::pT<T1> > > rho_1 =
        (proj1 * (*_rho) * proj1);
      arma::Mat<std::complex<trait::pT<T1> > > rho_2 =
        (proj2 * (*_rho) * proj2);
      arma::Mat<std::complex<trait::pT<T1> > > rho_3 =
        (proj3 * (*_rho) * proj3);

      trait::pT<T1> p1 = std::real(arma::trace(rho_1));
      trait::pT<T1> p2 = std::real(arma::trace(rho_2));
      trait::pT<T1> p3 = std::real(arma::trace(rho_3));

      trait::pT<T1> S_max = 0.0;
      if (p1 > _precision::eps<trait::pT<T1> >::value) {
        rho_1 /= p1;
        S_max += p1 * entropy(rho_1);
      }
      if (p2 > _precision::eps<trait::pT<T1> >::value) {
        rho_2 /= p2;
        S_max += p2 * entropy(rho_2);
      }
      if (p3 > _precision::eps<trait::pT<T1> >::value) {
        rho_3 /= p3;
        S_max += p3 * entropy(rho_3);
      }
      ret.at(i) = _mutual_info + S_max;
    }

    _result_reg = arma::min(ret);
    _result_reg_all = std::move(ret);
    _is_reg_computed = true;
  }
  return *this;
}

//****************************************************************************

template <typename T1>
inline const arma::Col<trait::pT<T1> >& discord_space<T1>::opt_angles() {
  check_size_change();
  if (!_is_computed)
    compute();
  return _tp;
}

//******************************************************************************

template <typename T1> inline const trait::pT<T1>& discord_space<T1>::result() {
  check_size_change();
  if (!_is_computed)
    compute();
  return _result;
}

//******************************************************************************

template <typename T1>
inline const trait::pT<T1>& discord_space<T1>::result_reg() {
  check_size_change();
  if (!_is_reg_computed)
    compute_reg();
  return _result_reg;
}

//******************************************************************************

template <typename T1>
inline const arma::Col<trait::pT<T1> >& discord_space<T1>::result_reg_all() {
  check_size_change();
  if (!_is_reg_computed)
    compute_reg();
  return _result_reg_all;
}

//******************************************************************************

template <typename T1> inline discord_space<T1>& discord_space<T1>::refresh() {
  check_size_change();
  _is_computed = false;
  _is_reg_computed = false;
  _is_minfo_computed = false;
  default_setting();
  return *this;
}

//******************************************************************************

template <typename T1>
inline discord_space<T1>& discord_space<T1>::reset(arma::uword nodal) {
  _is_computed = false;
  _is_reg_computed = false;
  _is_minfo_computed = false;
  _nodal = nodal;
  _n_cols = _rho->n_cols;
  _n_rows = _rho->n_rows;

#ifndef QICLIB_NO_DEBUG
  if (_rho->n_elem == 0)
    throw Exception("qic::discord_space::reset", Exception::type::ZERO_SIZE);

  if (_rho->n_rows != _rho->n_cols)
    throw Exception("qic::discord_space::reset",
                    Exception::type::MATRIX_NOT_SQUARE);

  if (arma::prod(_dim) != _rho->n_rows)
    throw Exception("qic::discord_space::reset",
                    Exception::type::DIMS_MISMATCH_MATRIX);

  if (_nodal <= 0 || _nodal > _party_no)
    throw Exception("qic::discord_space::reset",
                    "Invalid measured party index!");
#endif

  _discord2 = (_dim(_nodal - 1) == 2);
  _discord3 = (_dim(_nodal - 1) == 3);

#ifndef QICLIB_NO_DEBUG
  if (!_discord2 && !_discord3)
    throw Exception("qic::discord_space::reset",
                    "Measured party is not qubit or qutrit!");
#endif

  default_setting();

  return *this;
}

//******************************************************************************

template <typename T1>
inline discord_space<T1>& discord_space<T1>::reset(arma::uword nodal,
                                                   arma::uvec dim) {
  _dim = std::move(dim);
  _party_no = _dim.n_elem;
  return reset(nodal);
}

//******************************************************************************

template <typename T1>
inline discord_space<T1>& discord_space<T1>::reset(arma::uword nodal,
                                                   arma::uword dim) {

#ifndef QICLIB_NO_DEBUG
  if (dim == 0)
    throw Exception("qic::discord_space::reset", Exception::type::INVALID_DIMS);
#endif

  arma::uword n = static_cast<arma::uword>(
    QICLIB_ROUND_OFF(std::log(_rho->n_rows) / std::log(dim)));

  arma::uvec dim2(n);
  dim2.fill(dim);
  return reset(nodal, std::move(dim2));
}

//******************************************************************************

template <typename T1>
inline discord_space<T1>& discord_space<T1>::reset(T1* rho, arma::uword nodal,
                                                   arma::uvec dim) {
  _rho = rho;
  return reset(nodal, std::move(dim));
}

//******************************************************************************

template <typename T1>
inline discord_space<T1>& discord_space<T1>::reset(T1* rho, arma::uword nodal,
                                                   arma::uword dim) {
  _rho = rho;
  return reset(nodal, dim);
}

//******************************************************************************

}  // namespace qic
