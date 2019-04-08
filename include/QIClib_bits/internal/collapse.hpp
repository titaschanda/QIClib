/*
 * QIClib (Quantum information and computation library)
 *
 * Copyright (c) 2015 - 2017  Titas Chanda (titas.chanda@gmail.com)
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

//************************************************************************

namespace _internal {

//******************************************************************************

inline void dim_collapse_sys(arma::uvec& dim, arma::uvec& sys) {
  if (arma::any(dim == 1)) {
    arma::uvec onedim = arma::find(dim == 1);
    arma::uvec onesys = arma::find(dim(sys - 1) == 1);

    for (arma::uword i = 0; i < onesys.n_elem; ++i)
      sys.shed_row(onesys.at(i) - i);

    arma::uword syscount(0);
    for (arma::uword i = 0; i < onedim.n_elem; ++i) {
      dim.shed_row(onedim.at(i) - i);
      arma::uvec sysf = arma::find(sys > onedim.at(i) - syscount);
      if (sysf.n_elem != 0) {
        ++syscount;
        sys(sysf) = sys(sysf) - 1;
      }
    }
  }

  arma::uword a(1), b(0);
  arma::uword index(0);
  arma::uvec dim2(dim.n_elem);
  arma::uvec sys2(sys);

  for (arma::uword i = 0; i != dim.n_elem; ++i) {
    if (arma::all(sys != i + 1)) {
      a *= dim.at(i);
      ++b;

    } else {
      if (a == 1) {
        dim2.at(index) = dim.at(i);
        ++index;

      } else {
        dim2.at(index) = a;
        ++index;
        dim2.at(index) = dim.at(i);
        ++index;

        if (b > 1) {
          arma::uvec index2 = arma::find(sys > i);
          sys2(index2) += -b + 1;
        }

        a = 1;
        b = 0;
      }
    }

    if (i == dim.n_elem - 1 && a != 1) {
      dim2.at(index) = a;
      ++index;
    }
  }

  if (index < dim.n_elem)
    dim2.shed_rows(index, dim.n_elem - 1);
  sys = std::move(sys2);
  dim = std::move(dim2);
}

//************************************************************************

inline void dim_collapse_sys_ctrl(arma::uvec& dim, arma::uvec& sys,
                                  arma::uvec& ctrl) {
  if (arma::any(dim == 1)) {
    arma::uvec onedim = arma::find(dim == 1);
    arma::uvec onesys = arma::find(dim(sys - 1) == 1);
    arma::uvec onectrl = arma::find(dim(ctrl - 1) == 1);

    for (arma::uword i = 0; i < onesys.n_elem; ++i)
      sys.shed_row(onesys.at(i) - i);

    for (arma::uword i = 0; i < onectrl.n_elem; ++i)
      ctrl.shed_row(onectrl.at(i) - i);

    arma::uword syscount(0), ctrlcount(0);
    for (arma::uword i = 0; i < onedim.n_elem; ++i) {
      dim.shed_row(onedim.at(i) - i);
      arma::uvec sysf = arma::find(sys > onedim.at(i) - syscount);
      if (sysf.n_elem != 0) {
        ++syscount;
        sys(sysf) = sys(sysf) - 1;
      }
      arma::uvec ctrlf = arma::find(ctrl > onedim.at(i) - ctrlcount);
      if (ctrlf.n_elem != 0) {
        ++ctrlcount;
        ctrl(ctrlf) = ctrl(ctrlf) - 1;
      }
    }
  }

  arma::uword a(1), b(0);
  arma::uword index(0);
  arma::uvec dim2(dim.n_elem);
  arma::uvec sys2(sys);
  arma::uvec ctrl2(ctrl);

  for (arma::uword i = 0; i != dim.n_elem; ++i) {
    if (arma::all(sys != i + 1) && arma::all(ctrl != i + 1)) {
      a *= dim.at(i);
      ++b;

    } else {
      if (a == 1) {
        dim2.at(index) = dim(i);
        ++index;

      } else {
        dim2.at(index) = a;
        ++index;
        dim2.at(index) = dim.at(i);
        ++index;

        if (b > 1) {
          arma::uvec index2 = arma::find(sys > i);
          sys2(index2) += -b + 1;
          arma::uvec index3 = arma::find(ctrl > i);
          ctrl2(index3) += -b + 1;
        }
        a = 1;
        b = 0;
      }
    }

    if (i == dim.n_elem - 1 && a != 1) {
      dim2.at(index) = a;
      ++index;
    }
  }

  if (index < dim.n_elem)
    dim2.shed_rows(index, dim.n_elem - 1);

  sys = std::move(sys2);
  ctrl = std::move(ctrl2);
  dim = std::move(dim2);
}

//************************************************************************

}  // namespace _internal

//************************************************************************

}  // namespace qic
