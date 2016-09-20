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

//************************************************************************

namespace _internal {

//************************************************************************

// More efficient than iterative counterpart, as iterative one needs
// extra copy constructor
template <typename T1, typename T2,
          typename TR = typename std::enable_if<
            is_arma_type_var<T1>::value && std::is_integral<T2>::value,
            arma::Mat<trait::eT<T1> > >::type>
inline TR POWM_GEN_INT(const T1& rho, const T2& P) {
  if (P == 0) {
    arma::uword n = rho.eval().n_rows;
    return arma::eye<arma::Mat<trait::eT<T1> > >(n, n);

  } else if (P == 1) {
    return rho;

  } else if (P == 2) {
    return rho * rho;

  } else {
    if (P % 2 == 0)
      return POWM_GEN_INT(POWM_GEN_INT(rho, 2), P / 2);
    else
      return POWM_GEN_INT(POWM_GEN_INT(rho, 2), (P - 1) / 2) * rho;
  }
}

//************************************************************************

template <typename T1> struct int_tag {
  typedef int_tag type;
  typedef arma::Mat<trait::eT<T1> > ret_type;
};

template <typename T1> struct uint_tag {
  typedef uint_tag type;
  typedef arma::Mat<trait::eT<T1> > ret_type;
};

template <typename T1> struct nonint_tag {
  typedef nonint_tag type;
  typedef arma::Mat<std::complex<trait::pT<T1> > > ret_type;
};

template <typename T1, typename T2>
struct powm_tag : std::conditional<
                    std::is_integral<T2>::value,
                    typename std::conditional<std::is_unsigned<T2>::value,
                                              uint_tag<T1>, int_tag<T1> >::type,
                    nonint_tag<T1> >::type {};

//************************************************************************

template <typename T1, typename T2,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1> >::value,
            arma::Mat<std::complex<trait::pT<T1> > > >::type>
inline TR powm_gen_implement(const T1& rho1, const T2& P, nonint_tag<T1>) {
  const auto& rho = as_Mat(rho1);

#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::powm_gen", Exception::type::ZERO_SIZE);

  if (rho.n_rows != rho.n_cols)
    throw Exception("qic::powm_gen", Exception::type::MATRIX_NOT_SQUARE);
#endif

  arma::Col<std::complex<trait::pT<T1> > > eigval;
  arma::Mat<std::complex<trait::pT<T1> > > eigvec;
  bool check = arma::eig_gen(eigval, eigvec, rho);
  if (!check)
    throw std::runtime_error("qic::powm_gen(): Decomposition failed!");

  return eigvec * diagmat(arma::pow(eigval, P)) * eigvec.i();
}

//************************************************************************

template <typename T1, typename T2,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1> >::value,
            arma::Mat<trait::eT<T1> > >::type>
inline TR powm_gen_implement(const T1& rho1, const T2& P, int_tag<T1>) {
  const auto& rho = as_Mat(rho1);

#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::powm_gen", Exception::type::ZERO_SIZE);

  if (rho.n_rows != rho.n_cols)
    throw Exception("qic::powm_gen", Exception::type::MATRIX_NOT_SQUARE);
#endif

  if (P < 0)
    return POWM_GEN_INT(rho.i().eval(), -P);
  else
    return POWM_GEN_INT(rho, P);
}

//************************************************************************

template <typename T1, typename T2,
          typename TR = typename std::enable_if<
            is_arma_type_var<T1>::value, arma::Mat<trait::eT<T1> > >::type>
inline TR powm_gen_implement(const T1& rho1, const T2& P, uint_tag<T1>) {
  const auto& rho = as_Mat(rho1);

#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::powm_gen", Exception::type::ZERO_SIZE);

  if (rho.n_rows != rho.n_cols)
    throw Exception("qic::powm_gen", Exception::type::MATRIX_NOT_SQUARE);
#endif

  return POWM_GEN_INT(rho, P);
}

//************************************************************************

template <typename T1, typename T2,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1> >::value,
            arma::Mat<std::complex<trait::pT<T1> > > >::type>
inline TR powm_sym_implement(const T1& rho1, const T2& P, nonint_tag<T1>) {
  const auto& rho = as_Mat(rho1);

#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::powm_sym", Exception::type::ZERO_SIZE);

  if (rho.n_rows != rho.n_cols)
    throw Exception("qic::powm_sym", Exception::type::MATRIX_NOT_SQUARE);
#endif

  arma::Col<trait::pT<T1> > eigval;
  arma::Mat<trait::eT<T1> > eigvec;

  if (rho.n_rows > 20) {
    bool check = arma::eig_sym(eigval, eigvec, rho, "dc");
    if (!check)
      throw std::runtime_error("qic::powm_sym(): Decomposition failed!");

  } else {
    bool check = arma::eig_sym(eigval, eigvec, rho, "std");
    if (!check)
      throw std::runtime_error("qic::powm_sym(): Decomposition failed!");
  }

  return eigvec *
         arma::diagmat(arma::pow(
           arma::conv_to<arma::Col<std::complex<trait::pT<T1> > > >::from(
             eigval),
           P)) *
         eigvec.t();
}

//************************************************************************

template <typename T1, typename T2,
          typename TR = typename std::enable_if<
            is_floating_point_var<trait::pT<T1> >::value,
            arma::Mat<trait::eT<T1> > >::type>
inline TR powm_sym_implement(const T1& rho1, const T2& P, int_tag<T1>) {
  const auto& rho = as_Mat(rho1);

#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::powm_sym", Exception::type::ZERO_SIZE);

  if (rho.n_rows != rho.n_cols)
    throw Exception("qic::powm_sym", Exception::type::MATRIX_NOT_SQUARE);
#endif
  if (P < 0)
    return POWM_GEN_INT(rho.i().eval(), -P);
  else
    return POWM_GEN_INT(rho, P);
}

//************************************************************************

template <typename T1, typename T2,
          typename TR = typename std::enable_if<
            is_arma_type_var<T1>::value, arma::Mat<trait::eT<T1> > >::type>
inline TR powm_sym_implement(const T1& rho1, const T2& P, uint_tag<T1>) {
  const auto& rho = as_Mat(rho1);

#ifndef QICLIB_NO_DEBUG
  if (rho.n_elem == 0)
    throw Exception("qic::powm_sym", Exception::type::ZERO_SIZE);

  if (rho.n_rows != rho.n_cols)
    throw Exception("qic::powm_sym", Exception::type::MATRIX_NOT_SQUARE);
#endif

  return POWM_GEN_INT(rho, P);
}

//************************************************************************

template <typename T1,
          typename TR = typename std::enable_if<
            is_arma_type_var<T1>::value, arma::Mat<trait::eT<T1> > >::type>
inline TR TENSOR_POW(const T1& rho, arma::uword n) {
  if (n == 1) {
    return rho;

  } else if (n == 2) {
    return arma::kron(rho, rho);

  } else {
    if (n % 2 == 0)
      return TENSOR_POW(TENSOR_POW(rho, 2), n / 2);
    else
      return arma::kron(TENSOR_POW(TENSOR_POW(rho, 2), (n - 1) / 2), rho);
  }
}

//************************************************************************

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
