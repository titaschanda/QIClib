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

template <typename T1 = arma::cx_mat> class deficit_space {
 private:
  const arma::Mat<trait::eT<T1> >& _rho;
  arma::uword _nodal;
  arma::uword _party_no;
  arma::uvec _dim;

  trait::pT<T1> _S_A_B;
  trait::pT<T1> _result;
  arma::Col<trait::pT<T1> > _tp;
  trait::pT<T1> _result_reg;
  arma::Col<trait::pT<T1> > _result_reg_all;

  bool _is_computed;
  bool _is_reg_computed;
  bool _is_sab_computed;
  bool _deficit2;
  bool _deficit3;
  
  nlopt::algorithm _deficit_global_opt;
  double _deficit_global_xtol;
  double _deficit_global_ftol;
  bool _deficit_global;
  nlopt::algorithm _deficit_local_opt;
  double _deficit_local_xtol;
  double _deficit_local_ftol;
  arma::vec _deficit_angle_range;
  arma::vec _deficit_angle_ini;

  inline void init(arma::uvec dim);
  inline void s_a_b();
  
  //****************************************************************************

 public:
  //****************************************************************************

  deficit_space() = delete;
  deficit_space(const deficit_space&) = delete;
  deficit_space& operator=(const deficit_space&) = delete;
  ~deficit_space() = default;
  
  //****************************************************************************

  inline deficit_space(const T1& rho1, arma::uword nodal, arma::uvec dim);
  inline deficit_space(const T1& rho1, arma::uword nodal, arma::uword dim = 2);

  //****************************************************************************

  inline deficit_space& global_algorithm(nlopt::algorithm a) noexcept;
  inline deficit_space& global_xtol(double a) noexcept;
  inline deficit_space& global_ftol(double a) noexcept;
  inline deficit_space& global_opt(bool a) noexcept;
  inline deficit_space& local_algorithm(nlopt::algorithm a) noexcept;
  inline deficit_space& local_xtol(double a) noexcept;
  inline deficit_space& local_ftol(double a) noexcept;
  inline deficit_space& angle_range(const arma::vec& a);
  inline deficit_space& initial_angle(const arma::vec& a);
  
  //****************************************************************************

  inline deficit_space& compute();
  inline deficit_space& compute_reg();
  inline const arma::Col<trait::pT<T1> >& opt_angles() noexcept;
  inline const trait::pT<T1>& result() noexcept;
  inline const trait::pT<T1>& result_reg() noexcept;
  inline const arma::Col<trait::pT<T1> >& result_reg_all() noexcept;
  inline deficit_space& refresh() noexcept;
  inline deficit_space& reset_party(arma::uword);
};

//******************************************************************************

}  // namespace qic
