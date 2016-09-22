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

template <typename T1,
          typename Enable = typename std::enable_if<
            std::is_same<T1, arma::Mat<trait::eT<T1> > >::value, void>::type>
class discord_space;

template <typename T1> class discord_space<T1> {

  // static_assert(
  // std::is_same<T1, arma::Mat<trait::eT<T1> > >::value,
  // "discord_space requires Armadillo Mat object as template argument!");

 private:
  T1* _rho{nullptr};
  arma::uword _nodal{0};
  arma::uword _n_cols{0};
  arma::uword _n_rows{0};

  arma::uword _party_no{0};
  arma::uvec _dim{0};

  trait::pT<T1> _mutual_info{0};
  trait::pT<T1> _result{0};
  arma::Col<trait::pT<T1> > _tp{0};
  trait::pT<T1> _result_reg{0};
  arma::Col<trait::pT<T1> > _result_reg_all{0};

  bool _is_minfo_computed{0};
  bool _is_computed{0};
  bool _is_reg_computed{0};
  bool _discord2{0};
  bool _discord3{0};

  nlopt::algorithm _discord_global_opt{};
  double _discord_global_xtol{0};
  double _discord_global_ftol{0};
  bool _discord_global{0};
  nlopt::algorithm _discord_local_opt{};
  double _discord_local_xtol{0};
  double _discord_local_ftol{0};
  arma::vec _discord_angle_range{0};
  arma::vec _discord_angle_ini{0};

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
