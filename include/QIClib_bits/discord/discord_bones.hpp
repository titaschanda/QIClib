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

template <typename T1> class discord_space {

  static_assert(
    std::is_same<T1, arma::Mat<trait::eT<T1> > >::value,
    "discord_space requires Armadillo Mat object as template argument!");

 private :
  T1* _rho;
  arma::uword _nodal;
  arma::uword _n_cols;
  arma::uword _n_rows;

  arma::uword _party_no;
  arma::uvec _dim;
  
  trait::pT<T1> _mutual_info;
  trait::pT<T1> _result;
  arma::Col<trait::pT<T1> > _tp;
  trait::pT<T1> _result_reg;
  arma::Col<trait::pT<T1> > _result_reg_all;

  bool _is_minfo_computed;
  bool _is_computed;
  bool _is_reg_computed;
  bool _discord2;
  bool _discord3;
  
  nlopt::algorithm _discord_global_opt;
  double _discord_global_xtol;
  double _discord_global_ftol;
  bool _discord_global;
  nlopt::algorithm _discord_local_opt;
  double _discord_local_xtol;
  double _discord_local_ftol;
  arma::vec _discord_angle_range;
  arma::vec _discord_angle_ini;

  inline void init(arma::uvec dim);
  inline void check_size_change();
  inline void minfo_p();
  inline void default_setting();
  
  //****************************************************************************

 public:
  //****************************************************************************

  discord_space() = delete;
  discord_space(const discord_space&) = delete;
  discord_space& operator=(const discord_space&) = delete;
  ~discord_space() = default;
  
  //****************************************************************************

  inline discord_space(T1* rho1, arma::uword nodal, arma::uvec dim);
  inline discord_space(T1* rho1, arma::uword nodal, arma::uword dim = 2);

  //****************************************************************************

  inline discord_space& global_algorithm(nlopt::algorithm a) noexcept;
  inline discord_space& global_xtol(double a) noexcept;
  inline discord_space& global_ftol(double a) noexcept;
  inline discord_space& global_opt(bool a) noexcept;
  inline discord_space& local_algorithm(nlopt::algorithm a) noexcept;
  inline discord_space& local_xtol(double a) noexcept;
  inline discord_space& local_ftol(double a) noexcept;
  inline discord_space& angle_range(const arma::vec& a);
  inline discord_space& initial_angle(const arma::vec& a);
  
  //****************************************************************************

  inline discord_space& compute();
  inline discord_space& compute_reg();
  inline const arma::Col<trait::pT<T1> >& opt_angles();
  inline const trait::pT<T1>& result();
  inline const trait::pT<T1>& result_reg();
  inline const arma::Col<trait::pT<T1> >& result_reg_all();
  inline discord_space& refresh();
  inline discord_space& reset(arma::uword);
  inline discord_space& reset(arma::uword, arma::uvec);
  inline discord_space& reset(arma::uword, arma::uword);
  inline discord_space& reset(T1*, arma::uword, arma::uvec);
  inline discord_space& reset(T1*, arma::uword, arma::uword);
};

//******************************************************************************

}  // namespace qic
