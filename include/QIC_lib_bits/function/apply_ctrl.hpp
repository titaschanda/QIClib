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

template <typename T1, typename T2,
          typename TR = typename std::enable_if<
            is_floating_point_var<pT<T1>, pT<T2> >::value &&
              is_same_pT_var<T1, T2>::value,
            arma::Mat<typename eT_promoter_var<T1, T2>::type> >::type>
inline TR apply_ctrl(const T1& rho1, const T2& A, arma::uvec ctrl,
                     arma::uvec sys, arma::uvec dim) {
  using eTR = typename eT_promoter_var<T1, T2>::type;

  const auto& p = as_Mat(rho1);
  const auto& A1 = as_Mat(A);

  bool checkV = true;
  if (p.n_cols == 1)
    checkV = false;

  arma::uword d = ctrl.n_elem > 0 ? dim.at(ctrl.at(0) - 1) : 1;

#ifndef QIC_LIB_NO_DEBUG
  if (p.n_elem == 0)
    throw Exception("qic::apply_ctrl", Exception::type::ZERO_SIZE);

  if (A1.n_elem == 0)
    throw Exception("qic::apply_ctrl", Exception::type::ZERO_SIZE);

  if (checkV)
    if (p.n_rows != p.n_cols)
      throw Exception("qic::apply_ctrl",
                      Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);

  if (A1.n_rows != A1.n_cols)
    throw Exception("qic::apply_ctrl", Exception::type::MATRIX_NOT_SQUARE);

  for (arma::uword i = 1; i < ctrl.n_elem; ++i)
    if (dim.at(ctrl.at(i) - 1) != d)
      throw Exception("qic::apply_ctrl", Exception::type::DIMS_NOT_EQUAL);

  if (dim.n_elem == 0 || arma::any(dim == 0))
    throw Exception("qic::apply_ctrl", Exception::type::INVALID_DIMS);

  if (arma::prod(dim) != p.n_rows)
    throw Exception("qic::apply_ctrl", Exception::type::DIMS_MISMATCH_MATRIX);

  if (arma::prod(dim(sys - 1)) != A1.n_rows)
    throw Exception("qic::apply_ctrl", Exception::type::DIMS_MISMATCH_MATRIX);

  const arma::uvec ctrlsys = arma::join_cols(sys, ctrl);

  if (ctrlsys.n_elem > dim.n_elem ||
      arma::find_unique(ctrlsys, false).eval().n_elem != ctrlsys.n_elem ||
      arma::any(ctrlsys > dim.n_elem) || arma::any(ctrlsys == 0))
    throw Exception("qic::apply_ctrl", Exception::type::INVALID_SUBSYS);
#endif

  _internal::protect_subs::dim_collapse_sys_ctrl(dim, sys, ctrl);

  const arma::uword n = dim.n_elem;
  const arma::uword m = sys.n_elem;
  const arma::uword o = ctrl.n_elem;

  arma::uvec keep(n - m);
  arma::uword keep_count(0);
  for (arma::uword run = 0; run < n; ++run) {
    if (!arma::any(sys == run + 1)) {
      keep.at(keep_count) = run + 1;
      ++keep_count;
    }
  }

  arma::uvec product(n, arma::fill::ones);
  for (arma::sword i = n - 2; i >= 0; --i)
    product.at(i) = product.at(i + 1) * dim.at(i + 1);

  arma::uvec productr(m, arma::fill::ones);
  for (arma::sword i = m - 2; i >= 0; --i)
    productr.at(i) = productr.at(i + 1) * dim.at(sys(i) - 1);

  arma::uword p_num = std::max(static_cast<arma::uword>(1), d - 1);

  arma::field<arma::Mat<eT<T2> > > Ap(p_num + 1);
  for (arma::uword i = 0; i <= p_num; ++i) Ap.at(i) = powm_gen(A1, i);

  if (!checkV) {
    arma::Col<eTR> rho(p.n_rows, arma::fill::zeros);

    const arma::uword loop_no = 2 * n;
    arma::uword* loop_counter = new arma::uword[loop_no + 1];
    arma::uword* MAX = new arma::uword[loop_no + 1];

    for (arma::uword i = 0; i < n; ++i) {
      MAX[i] = dim.at(i);
      if (arma::any(keep == i + 1))
        MAX[i + n] = 1;
      else
        MAX[i + n] = dim.at(i);
    }
    MAX[loop_no] = 2;

    for (arma::uword i = 0; i < loop_no + 1; ++i) loop_counter[i] = 0;

    arma::uword p1 = 0;

    while (loop_counter[loop_no] == 0) {
      arma::uword count1(0), count2(0);

      for (arma::uword i = 0; i < n; ++i) {
        count1 += (arma::any(ctrl == i + 1) && loop_counter[i] != 0) ? 1 : 0;
        count2 += loop_counter[i + n] == 0 ? 1 : 0;
      }

      if ((count1 != o) && (count2 == n)) {
        arma::uword I(0);
        for (arma::uword i = 0; i < n; ++i)
          I += product.at(i) * loop_counter[i];
        rho.at(I) = static_cast<eTR>(p.at(I));

      } else if (count1 == o) {
        arma::uword I(0), J(0), K(0), L(0);
        arma::uword power = o == 0 ? 1 : 0;

        for (arma::uword i = 0; i < n; ++i) {
          if (arma::any(keep == i + 1)) {
            I += product.at(i) * loop_counter[i];
            J += product.at(i) * loop_counter[i];

          } else {
            I += product.at(i) * loop_counter[i];
            J += product.at(i) * loop_counter[i + n];
          }

          if (o != 0) {
            arma::uword counter_1(1);
            for (arma::uword j = 1; j < o; ++j)
              counter_1 +=
                loop_counter[ctrl.at(0) - 1] == loop_counter[ctrl.at(j) - 1]
                  ? 1
                  : 0;

            power = counter_1 == o ? loop_counter[ctrl.at(0) - 1] : 0;
          }

          arma::uword counter(0);
          while (any(sys == i + 1)) {
            if (sys.at(counter) != i + 1) {
              ++counter;
            } else {
              K += productr.at(counter) * loop_counter[i];
              L += productr.at(counter) * loop_counter[i + n];
              break;
            }
          }
        }
        rho.at(I) += Ap.at(power).at(K, L) * p.at(J);
      }

      ++loop_counter[0];
      while (loop_counter[p1] == MAX[p1]) {
        loop_counter[p1] = 0;
        loop_counter[++p1]++;
        if (loop_counter[p1] != MAX[p1])
          p1 = 0;
      }
    }
    delete[] loop_counter;
    delete[] MAX;
    return rho;

  } else {
    arma::Mat<eT<T2> > U(p.n_rows, p.n_rows, arma::fill::zeros);

    const arma::uword loop_no = 2 * n;
    arma::uword* loop_counter = new arma::uword[loop_no + 1];
    arma::uword* MAX = new arma::uword[loop_no + 1];

    for (arma::uword i = 0; i < n; ++i) {
      MAX[i] = dim.at(i);
      if (arma::any(keep == i + 1))
        MAX[i + n] = 1;
      else
        MAX[i + n] = dim.at(i);
    }
    MAX[loop_no] = 2;

    for (arma::uword i = 0; i < loop_no + 1; ++i) loop_counter[i] = 0;

    arma::uword p1 = 0;

    while (loop_counter[loop_no] == 0) {
      arma::uword count1(0), count2(0);
      for (arma::uword i = 0; i < n; ++i) {
        count1 += (arma::any(ctrl == i + 1) && loop_counter[i] != 0) ? 1 : 0;
        count2 += loop_counter[i + n] == 0 ? 1 : 0;
      }

      if ((count1 != o) && (count2 == n)) {
        arma::uword I(0);
        for (arma::uword i = 0; i < n; ++i)
          I += product.at(i) * loop_counter[i];
        U.at(I, I) = static_cast<eT<T2> >(1.0);

      } else if (count1 == o) {
        arma::uword I(0), J(0), K(0), L(0);
        int power = o == 0 ? 1 : 0;

        for (arma::uword i = 0; i < n; ++i) {
          if (arma::any(keep == i + 1)) {
            I += product.at(i) * loop_counter[i];
            J += product.at(i) * loop_counter[i];

          } else {
            I += product.at(i) * loop_counter[i];
            J += product.at(i) * loop_counter[i + n];
          }

          if (o != 0) {
            arma::uword counter_1(1);
            for (arma::uword j = 1; j < o; ++j)
              counter_1 +=
                loop_counter[ctrl.at(0) - 1] == loop_counter[ctrl.at(j) - 1]
                  ? 1
                  : 0;

            power = counter_1 == o ? loop_counter[ctrl.at(0) - 1] : 0;
          }

          arma::uword counter(0);
          while (any(sys == i + 1)) {
            if (sys.at(counter) != i + 1) {
              ++counter;
            } else {
              K += productr.at(counter) * loop_counter[i];
              L += productr.at(counter) * loop_counter[i + n];
              break;
            }
          }
        }
        U.at(I, J) = Ap.at(power).at(K, L);
      }

      ++loop_counter[0];
      while (loop_counter[p1] == MAX[p1]) {
        loop_counter[p1] = 0;
        loop_counter[++p1]++;
        if (loop_counter[p1] != MAX[p1])
          p1 = 0;
      }
    }
    delete[] loop_counter;
    delete[] MAX;
    return U * p * U.t();
  }
}

//******************************************************************************

template <typename T1, typename T2,
          typename TR = typename std::enable_if<
            is_floating_point_var<pT<T1>, pT<T2> >::value &&
              is_same_pT_var<T1, T2>::value,
            arma::Mat<typename eT_promoter_var<T1, T2>::type> >::type>
inline TR apply_ctrl(const T1& rho1, const T2& A, arma::uvec ctrl,
                     arma::uvec sys, arma::uword dim = 2) {
  const auto& rho = as_Mat(rho1);

#ifndef QIC_LIB_NO_DEBUG
  bool checkV = true;
  if (rho.n_cols == 1)
    checkV = false;

  if (rho.n_elem == 0)
    throw Exception("qic::apply_ctrl", Exception::type::ZERO_SIZE);

  if (checkV)
    if (rho.n_rows != rho.n_cols)
      throw Exception("qic::apply_ctrl",
                      Exception::type::MATRIX_NOT_SQUARE_OR_CVECTOR);

  if (dim == 0)
    throw Exception("qic::apply_ctrl", Exception::type::INVALID_DIMS);
#endif

  arma::uword n = static_cast<arma::uword>(
    std::llround(std::log(rho.n_rows) / std::log(dim)));

  arma::uvec dim2(n);
  dim2.fill(dim);
  return apply_ctrl(rho, A, std::move(ctrl), std::move(sys), std::move(dim2));
}

//******************************************************************************

}  // namespace qic
