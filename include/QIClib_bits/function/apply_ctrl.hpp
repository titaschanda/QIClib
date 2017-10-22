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

//******************************************************************************

template <typename T1, typename T2,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1>, trait::pT<T2> >::value &&
              is_same_pT_var<T1, T2>::value,
            arma::Mat<typename eT_promoter_var<T1, T2>::type> >::type>

inline TR apply_ctrl(const T1& rho1, const T2& A, arma::uvec ctrl,
                     arma::uvec subsys, arma::uvec dim) {
  using eTR = typename eT_promoter_var<T1, T2>::type;

  const auto& rho = _internal::as_Mat(rho1);
  const auto& A1 = _internal::as_Mat(A);

  const arma::uvec ctrlsubsys = arma::join_cols(subsys, ctrl);

  const bool checkV = (rho.n_cols != 1);
  const arma::uword d = ctrl.n_elem > 0 ? dim.at(ctrl.at(0) - 1) : 1;

  const arma::uword DT = arma::prod(dim);
  const arma::uvec dimS = dim(subsys - 1);
  const arma::uword DS = arma::prod(dimS);

#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::apply_ctrl", Exception::type::ZERO_SIZE);

  if (A1.n_elem == 0)
    throw Exception("qic::apply_ctrl", Exception::type::ZERO_SIZE);

  if (checkV)
    if (rho.n_rows != rho.n_cols)
      throw Exception("qic::apply_ctrl",
                      Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);

  if (A1.n_rows != A1.n_cols)
    throw Exception("qic::apply_ctrl", Exception::type::MATRIX_NOT_SQUARE);

  for (arma::uword i = 1; i < ctrl.n_elem; ++i)
    if (dim.at(ctrl.at(i) - 1) != d)
      throw Exception("qic::apply_ctrl", Exception::type::DIMS_NOT_EQUAL);

  if (dim.n_elem == 0 || arma::any(dim == 0))
    throw Exception("qic::apply_ctrl", Exception::type::INVALID_DIMS);

  if (DT != rho.n_rows)
    throw Exception("qic::apply_ctrl", Exception::type::DIMS_MISMATCH_MATRIX);

  if (DS != A1.n_rows)
    throw Exception("qic::apply_ctrl", Exception::type::DIMS_MISMATCH_MATRIX);

  if (ctrlsubsys.n_elem > dim.n_elem ||
      arma::unique(ctrlsubsys).eval().n_elem != ctrlsubsys.n_elem ||
      arma::any(ctrlsubsys > dim.n_elem) || arma::any(ctrlsubsys == 0))
    throw Exception("qic::apply_ctrl", Exception::type::INVALID_SUBSYS);
#endif

  _internal::dim_collapse_sys_ctrl(dim, subsys, ctrl);

  const arma::uword sizeT = dim.n_elem;
  const arma::uword sizeS = subsys.n_elem;
  const arma::uword sizeC = ctrl.n_elem;

  arma::uvec keep(sizeT - sizeS - sizeC);
  arma::uword keep_count(0);
  for (arma::uword run = 0; run < sizeT; ++run) {
    if (!arma::any(ctrlsubsys == run + 1)) {
      keep.at(keep_count) = run + 1;
      ++keep_count;
    }
  }

  const arma::uvec dimK = dim(keep - 1);
  const arma::uword DK = arma::prod(dimK);

  const arma::uword p_num = std::max(static_cast<arma::uword>(1), d - 1);

  arma::field<arma::Mat<trait::eT<T2> > > Ap(p_num + 1);
  for (arma::uword i = 0; i <= p_num; ++i)
    Ap.at(i) = _internal::POWM_GEN_INT(A1, i);

  if (!checkV) {
    auto worker_pure = [sizeS, sizeC, DS, &ctrl, &subsys, &dim, &keep, &dimS,
                        &dimK, &Ap,
                        &rho](arma::uword _p, arma::uword _M, arma::uword _R)
      noexcept -> std::pair<eTR, arma::uword> {

      arma::uword indexT[_internal::MAXQDIT];
      arma::uword indexS[_internal::MAXQDIT];
      arma::uword indexK[_internal::MAXQDIT];

      for (arma::uword i = 0; i < sizeC; ++i) {
        indexT[ctrl.at(i) - 1] = _p;
      }

      _internal::num_to_lexi(_R, dimK, indexK);
      for (arma::uword i = 0; i < keep.n_elem; ++i) {
        indexT[keep.at(i) - 1] = indexK[i];
      }

      _internal::num_to_lexi(_M, dimS, indexS);
      for (arma::uword i = 0; i < sizeS; ++i) {
        indexT[subsys.at(i) - 1] = indexS[i];
      }

      arma::uword _I = _internal::lexi_to_num(indexT, dim);
      eTR ret(0);

      for (arma::uword N = 0; N < DS; ++N) {
        _internal::num_to_lexi(N, dimS, indexS);
        for (arma::uword j = 0; j < sizeS; ++j) {
          indexT[subsys.at(j) - 1] = indexS[j];
        }
        ret +=
          Ap.at(_p).at(_M, N) * rho.at(_internal::lexi_to_num(indexT, dim));
      }

      return std::make_pair(ret, _I);
    };

    arma::Col<eTR> rho_ret(rho);

#if (defined(QICLIB_USE_OPENMP) || defined(QICLIB_USE_OPENMP_APPLY)) &&        \
  defined(_OPENMP)
#pragma omp parallel for collapse(2)
#endif
    for (arma::uword R = 0; R < DK; ++R) {
      for (arma::uword M = 0; M < DS; ++M) {
        if (sizeC == 0) {
          auto W = worker_pure(1, M, R);
          rho_ret(W.second) = W.first;
        } else
          for (arma::uword p = 0; p < d; ++p) {
            auto W = worker_pure(p, M, R);
            rho_ret(W.second) = W.first;
          }
      }
    }

    return rho_ret;

  } else {

    auto worker_mix =
      [sizeS, sizeC, DS, &ctrl, &subsys, &dim, &keep, &dimS, &dimK,
       &Ap](arma::uword _p, arma::uword _M, arma::uword _N, arma::uword _R)
        noexcept -> std::tuple<trait::eT<T2>, arma::uword, arma::uword> {

      arma::uword indexT[_internal::MAXQDIT];
      arma::uword indexS[_internal::MAXQDIT];
      arma::uword indexK[_internal::MAXQDIT];

      for (arma::uword i = 0; i < sizeC; ++i) {
        indexT[ctrl.at(i) - 1] = _p;
      }

      _internal::num_to_lexi(_R, dimK, indexK);
      for (arma::uword i = 0; i < keep.n_elem; ++i) {
        indexT[keep.at(i) - 1] = indexK[i];
      }

      _internal::num_to_lexi(_M, dimS, indexS);
      for (arma::uword i = 0; i < sizeS; ++i) {
        indexT[subsys.at(i) - 1] = indexS[i];
      }

      arma::uword _I = _internal::lexi_to_num(indexT, dim);
      _internal::num_to_lexi(_N, dimS, indexS);
      for (arma::uword j = 0; j < sizeS; ++j) {
        indexT[subsys.at(j) - 1] = indexS[j];
      }
      arma::uword _J = _internal::lexi_to_num(indexT, dim);

      return std::make_tuple(Ap.at(_p).at(_M, _N), _I, _J);
    };

    arma::Mat<trait::eT<T2> > U(DT, DT, arma::fill::eye);

#if (defined(QICLIB_USE_OPENMP) || defined(QICLIB_USE_OPENMP_APPLY)) &&        \
  defined(_OPENMP)
#pragma omp parallel for collapse(2)
#endif
    for (arma::uword R = 0; R < DK; ++R) {
      for (arma::uword N = 0; N < DS; ++N) {
        for (arma::uword M = 0; M < DS; ++M) {
          if (sizeC == 0) {
            auto W = worker_mix(1, M, N, R);
            U(std::get<1>(W), std::get<2>(W)) = std::get<0>(W);
          } else
            for (arma::uword p = 0; p < d; ++p) {
              auto W = worker_mix(p, M, N, R);
              U(std::get<1>(W), std::get<2>(W)) = std::get<0>(W);
            }
        }
      }
    }

    return U * rho * U.t();
  }
}

//******************************************************************************

template <typename T1, typename T2,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1>, trait::pT<T2> >::value &&
              is_same_pT_var<T1, T2>::value,
            arma::Mat<typename eT_promoter_var<T1, T2>::type> >::type>

inline TR apply_ctrl(const T1& rho1, const T2& A, arma::uvec ctrl,
                     arma::uvec subsys, arma::uword dim = 2) {
  const auto& rho = _internal::as_Mat(rho1);

#ifndef QICLIB_NO_DEBUG
  bool checkV = (rho.n_cols != 1);
  if (rho.n_elem == 0)
    throw Exception("qic::apply_ctrl", Exception::type::ZERO_SIZE);

  if (checkV)
    if (rho.n_rows != rho.n_cols)
      throw Exception("qic::apply_ctrl",
                      Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);

  if (dim == 0)
    throw Exception("qic::apply_ctrl", Exception::type::INVALID_DIMS);
#endif

  const arma::uword n = static_cast<arma::uword>(
    QICLIB_ROUND_OFF(std::log(rho.n_rows) / std::log(dim)));

  arma::uvec dim2(n);
  dim2.fill(dim);
  return apply_ctrl(rho, A, std::move(ctrl), std::move(subsys),
                    std::move(dim2));
}

//******************************************************************************

}  // namespace qic
