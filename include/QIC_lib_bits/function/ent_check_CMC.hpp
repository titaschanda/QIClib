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

namespace qic {

//******************************************************************************

template <typename T1,
          typename = typename std::enable_if<
            is_floating_point_var<trait::pT<T1> >::value, void>::type>
inline bool ent_check_CMC(const T1& rho1, arma::uword dim) {
  const auto& rho = as_Mat(rho1);

#ifndef QIC_LIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::ent_check_CMC", Exception::type::ZERO_SIZE);

  if (rho.n_rows != rho.n_cols)
    throw Exception("qic::ent_check_CMC", Exception::type::MATRIX_NOT_SQUARE);

  if (rho.n_rows != dim * dim)
    throw Exception("qic::ent_check_CMC",
                    Exception::type::DIMS_MISMATCH_MATRIX);
#endif

  std::complex<trait::pT<T1> > I(0.0, 1.0);
  arma::Mat<std::complex<trait::pT<T1> > > Eye(dim, dim, arma::fill::eye);

  arma::Cube<std::complex<trait::pT<T1> > > M(dim, dim, dim * dim);

  for (arma::uword i = 0; i < dim; ++i)
    M.slice(i) = Eye.col(i) * (Eye.col(i)).t();

  arma::uword count_1(0);
  for (arma::uword i = 0; i < dim; ++i) {
    for (arma::uword j = 0; j < dim; ++j) {
      if (i < j) {
        ++count_1;
        M.slice(dim - 1 + count_1) =
          (Eye.col(i) * Eye.col(j).t() + Eye.col(j) * Eye.col(i).t()) *
          static_cast<trait::pT<T1> >(std::sqrt(0.5));
      }
    }
  }

  arma::uword count_2(0);
  for (arma::uword i = 0; i < dim; ++i) {
    for (arma::uword j = 0; j < dim; ++j) {
      if (i < j) {
        ++count_2;
        M.slice(((dim - 1) * (dim + 2)) / 2 + count_2) =
          static_cast<trait::pT<T1> >(std::sqrt(0.5)) * I *
          (Eye.col(i) * Eye.col(j).t() - Eye.col(j) * Eye.col(i).t());
      }
    }
  }

  arma::Mat<std::complex<trait::pT<T1> > > C(dim * dim, dim * dim);

  for (arma::uword j = 0; j < dim * dim; ++j) {
    for (arma::uword i = 0; i < dim * dim; ++i) {
      C.at(i, j) = arma::trace(rho * arma::kron(M.slice(i), Eye) *
                               arma::kron(Eye, M.slice(j))) -
                   arma::trace(rho * arma::kron(M.slice(i), Eye)) *
                     arma::trace(rho * arma::kron(Eye, M.slice(j)));
    }
  }

  C *= C.t();
  auto Ctr = arma::eig_sym(C);
  auto rhoA = TrX(rho, {2}, {dim, dim});
  auto rhoB = TrX(rho, {1}, {dim, dim});

  if (std::pow(arma::sum(arma::sqrt(arma::abs(Ctr))), 2) >
      (1.0 - std::real(arma::trace(rhoA * rhoA))) *
        (1.0 - std::real(arma::trace(rhoB * rhoB))))
    return true;
  else
    return false;
}

//******************************************************************************

template <typename T1,
          typename = typename std::enable_if<
            is_floating_point_var<trait::pT<T1> >::value, void>::type>
inline bool ent_check_CMC(const T1& rho1, arma::uword dim1, arma::uword dim2) {
  const auto& rho = as_Mat(rho1);

#ifndef QIC_LIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::ent_check_CMC", Exception::type::ZERO_SIZE);

  if (rho.n_rows != rho.n_cols)
    throw Exception("qic::ent_check_CMC", Exception::type::MATRIX_NOT_SQUARE);

  if (rho.n_rows != dim1 * dim2)
    throw Exception("qic::ent_check_CMC",
                    Exception::type::DIMS_MISMATCH_MATRIX);
#endif

  std::complex<trait::pT<T1> > I(0.0, 1.0);
  arma::Mat<std::complex<trait::pT<T1> > > Eye1(dim1, dim1, arma::fill::eye);
  arma::Mat<std::complex<trait::pT<T1> > > Eye2(dim2, dim2, arma::fill::eye);

  arma::Cube<std::complex<trait::pT<T1> > > M1(dim1, dim1, dim1 * dim1);
  arma::Cube<std::complex<trait::pT<T1> > > M2(dim2, dim2, dim2 * dim2);

  for (arma::uword i = 0; i < dim1; ++i)
    M1.slice(i) = Eye1.col(i) * (Eye1.col(i)).t();

  arma::uword count_1(0);
  for (arma::uword i = 0; i < dim1; ++i) {
    for (arma::uword j = 0; j < dim1; ++j) {
      if (i < j) {
        ++count_1;
        M1.slice(dim1 - 1 + count_1) =
          (Eye1.col(i) * Eye1.col(j).t() + Eye1.col(j) * Eye1.col(i).t()) *
          static_cast<trait::pT<T1> >(std::sqrt(0.5));
      }
    }
  }

  arma::uword count_2(0);
  for (arma::uword i = 0; i < dim1; ++i) {
    for (arma::uword j = 0; j < dim1; ++j) {
      if (i < j) {
        ++count_2;
        M1.slice(((dim1 - 1) * (dim1 + 2)) / 2 + count_2) =
          static_cast<trait::pT<T1> >(std::sqrt(0.5)) * I *
          (Eye1.col(i) * Eye1.col(j).t() - Eye1.col(j) * Eye1.col(i).t());
      }
    }
  }

  for (arma::uword i = 0; i < dim2; ++i)
    M2.slice(i) = Eye2.col(i) * (Eye2.col(i)).t();

  arma::uword count_3(0);
  for (arma::uword i = 0; i < dim2; ++i) {
    for (arma::uword j = 0; j < dim2; ++j) {
      if (i < j) {
        ++count_3;
        M2.slice(dim2 - 1 + count_3) =
          (Eye2.col(i) * Eye2.col(j).t() + Eye2.col(j) * Eye2.col(i).t()) *
          static_cast<trait::pT<T1> >(std::sqrt(0.5));
      }
    }
  }

  arma::uword count_4(0);
  for (arma::uword i = 0; i < dim2; ++i) {
    for (arma::uword j = 0; j < dim2; ++j) {
      if (i < j) {
        ++count_4;
        M2.slice(((dim2 - 1) * (dim2 + 2)) / 2 + count_4) =
          static_cast<trait::pT<T1> >(std::sqrt(0.5)) * I *
          (Eye2.col(i) * Eye2.col(j).t() - Eye2.col(j) * Eye2.col(i).t());
      }
    }
  }

  arma::Mat<std::complex<trait::pT<T1> > > C(dim1 * dim1, dim2 * dim2);

  for (arma::uword j = 0; j < dim2 * dim2; ++j) {
    for (arma::uword i = 0; i < dim1 * dim1; ++i) {
      C.at(i, j) = arma::trace(rho * arma::kron(M1.slice(i), Eye2) *
                               arma::kron(Eye1, M2.slice(j))) -
                   arma::trace(rho * arma::kron(M1.slice(i), Eye2)) *
                     arma::trace(rho * arma::kron(Eye1, M2.slice(j)));
    }
  }

  C *= C.t();
  auto Ctr = arma::eig_sym(C);
  auto rhoA = TrX(rho, {2}, {dim1, dim2});
  auto rhoB = TrX(rho, {1}, {dim1, dim2});

  if (std::pow(arma::sum(arma::sqrt(abs(Ctr))), 2) >
      (1.0 - std::real(arma::trace(rhoA * rhoA))) *
        (1.0 - std::real(arma::trace(rhoB * rhoB))))
    return true;
  else
    return false;
}

//******************************************************************************

}  // namespace qic
