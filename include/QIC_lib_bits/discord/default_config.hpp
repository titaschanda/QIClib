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

  namespace
  {

    namespace protect
    {

  

      double _discord_reg_prob_tol = _precision::eps;
      double _discord3_reg_prob_tol = _precision::eps;


      //nlopt dependent configs

      nlopt::algorithm _discord3_global_opt = nlopt::GN_DIRECT_L;
      double _discord3_global_xtol = 0.25;
      double _discord3_global_ftol = 0;
      bool _discord3_global = true;
      nlopt::algorithm _discord3_local_opt = nlopt::LN_COBYLA;
      double _discord3_local_xtol = _precision::eps;
      double _discord3_local_ftol = 0;
      arma::vec  _discord3_angle_range = 2.0*arma::ones<arma::vec>(5);
      arma::vec _discord3_angle_ini = 0.1*arma::ones<arma::vec>(5);
      double _discord3_prob_tol = _precision::eps;

      nlopt::algorithm _discord_global_opt = nlopt::GN_DIRECT_L;
      double _discord_global_xtol = 4.0e-2;
      double _discord_global_ftol = 0;
      bool _discord_global = true;
      nlopt::algorithm _discord_local_opt = nlopt::LN_COBYLA;
      double _discord_local_xtol = _precision::eps;
      double _discord_local_ftol = 0;
      double _discord_theta_range = 1.0;
      double _discord_phi_range = 2.0;
      double _discord_theta_ini = 0.1;
      double _discord_phi_ini = 0.1;
      double _discord_prob_tol = _precision::eps;

      nlopt::algorithm _deficit3_global_opt = nlopt::GN_DIRECT_L;
      double _deficit3_global_xtol = 0.25;
      double _deficit3_global_ftol = 0;
      bool _deficit3_global = true;
      nlopt::algorithm _deficit3_local_opt = nlopt::LN_COBYLA;
      double _deficit3_local_xtol = _precision::eps;
      double _deficit3_local_ftol = 0;
      arma::vec _deficit3_angle_range = 2.0*arma::ones<arma::vec>(5);
      arma::vec _deficit3_angle_ini = 0.1*arma::ones<arma::vec>(5);

      nlopt::algorithm _deficit_global_opt = nlopt::GN_DIRECT_L;
      double _deficit_global_xtol = 4.0e-2;
      double _deficit_global_ftol = 0;
      bool _deficit_global = true;
      nlopt::algorithm _deficit_local_opt = nlopt::LN_COBYLA;
      double _deficit_local_xtol = _precision::eps;
      double _deficit_local_ftol = 0;
      double _deficit_theta_range = 1.0;
      double _deficit_phi_range = 2.0;
      double _deficit_theta_ini = 0.1;
      double _deficit_phi_ini = 0.1;

      //nlopt dependent configs


    }
  }
}

