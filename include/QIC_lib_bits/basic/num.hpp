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

inline arma::uword gcd(arma::uword m, arma::uword n) {
#ifndef QIC_LIB_NO_DEBUG
  if (m == 0 && n == 0)
    throw Exception("qic::gcd", Exception::type::OUT_OF_RANGE);
#endif

  if (m == 0 || n == 0)
    return std::max(m, n);

  arma::uword result = 1;
  while (n) {
    result = n;
    n = m % result;
    m = result;
  }

  return result;
}

//******************************************************************************

inline arma::uword gcd(const arma::uvec& M) {
#ifndef QIC_LIB_NO_DEBUG
  if (M.n_elem == 0)
    throw Exception("qic::gcd", Exception::type::ZERO_SIZE);
#endif

  arma::uword result = M.at(0);
  for (arma::uword i = 1; i < M.n_elem; ++i) result = gcd(result, M.at(i));

  return result;
}

//******************************************************************************

inline arma::uword lcm(arma::uword m, arma::uword n) {
#ifndef QIC_LIB_NO_DEBUG
  if (m == 0 || n == 0)
    throw Exception("qic::lcm", Exception::type::OUT_OF_RANGE);
#endif

  return m * n / gcd(m, n);
}

//******************************************************************************

inline arma::uword lcm(const arma::uvec& M) {
#ifndef QIC_LIB_NO_DEBUG
  if (M.n_elem == 0)
    throw Exception("qic::lcm", Exception::type::ZERO_SIZE);

  if (arma::any(M) == 0)
    throw Exception("qpp::lcm", Exception::type::OUT_OF_RANGE);
#endif

  if (M.n_elem == 1)
    return M.at(0);

  arma::uword prod1 = arma::prod(M);

  return prod1 / gcd(M);
}

//******************************************************************************

inline arma::ivec real_to_contfrac(double x, arma::uword n,
                                   arma::uword tol = 1e5) {
#ifndef QIC_LIB_NO_DEBUG
  if (n == 0)
    throw Exception("qic::real_to_contfrac", Exception::type::OUT_OF_RANGE);
#endif

  arma::ivec result = arma::zeros<arma::ivec>(n);
  arma::uword count(0);

  for (arma::uword i = 0; i < n; ++i) {
    result.at(i) = std::llround(std::floor(x));
    x = 1 / (x - std::floor(x));
    count++;
    if (!std::isfinite(x) || x > tol) {
      result.shed_rows(count, n - 1);
      break;
    }
  }
  return result;
}

//******************************************************************************

inline double contfrac_to_real(const arma::ivec& frac, arma::sword n = -1) {
#ifndef QIC_LIB_NO_DEBUG
  if (frac.n_elem == 0)
    throw Exception("qic::contfrac_to_real", Exception::type::ZERO_SIZE);

  if (n == 0 || n < -1)
    throw Exception("qic::contfrac_to_real", Exception::type::OUT_OF_RANGE);
#endif

  if (n > static_cast<arma::sword>(frac.n_elem) || n == -1)
    n = frac.n_elem;

  if (n == 1)
    return frac.at(0);

  double tmp = 1.0 / static_cast<double>(frac.at(n - 1));
  for (arma::uword i = n - 2; i != 0; --i) {
    tmp = 1.0 / (tmp + static_cast<double>(frac.at(i)));
  }

  return frac.at(0) + tmp;
}

//******************************************************************************

inline arma::uword denominator(double c, arma::uword qmax) {
#ifndef QIC_LIB_NO_DEBUG
  if (qmax == 0)
    throw Exception("qic::denominator", Exception::type::OUT_OF_RANGE);
#endif

  double y = c;
  arma::uword q0 = 0;
  arma::uword q1 = 1;
  arma::uword q2 = 0;
  while (true) {
    double z = y - std::floor(y);
    if (z < 0.5 / std::pow(qmax, 2)) {
      return (q1);
    }
    if (z != 0) {
      y = 1 / z;
    } else {
      return (q1);
    }
    q2 = std::llround(floor(y)) * q1 + q0;
    if (q2 >= qmax) {
      return (q1);
    }
    q0 = q1;
    q1 = q2;
  }
}

//******************************************************************************

inline arma::sword numerator(double c, arma::uword qmax) {
#ifndef QIC_LIB_NO_DEBUG
  if (qmax == 0)
    throw Exception("qic::numerator", Exception::type::OUT_OF_RANGE);
#endif

  arma::uword den = denominator(c, qmax);
  return std::llround(std::floor(den * c + 0.5));
}

//******************************************************************************

inline arma::uword modexp(arma::uword x, arma::uword a, arma::uword n) {
#ifndef QIC_LIB_NO_DEBUG
  if (n == 0 || (a == 0 && n == 0))
    throw Exception("qic::modexp", Exception::type::OUT_OF_RANGE);
#endif

  arma::uword value = 1;
  arma::uword tmp;
  tmp = x % n;
  while (a > 0) {
    if (a & 1) {
      value = (value * tmp) % n;
    }
    tmp = tmp * tmp % n;
    a = a >> 1;
  }
  return value;
}

//******************************************************************************

inline bool is_prime(arma::uword n) {
#ifndef QIC_LIB_NO_DEBUG
  if (n == 0 || n == 1)
    throw Exception("qic::is_prime", Exception::type::OUT_OF_RANGE);
#endif

  for (arma::uword i = 2;
       i <= static_cast<arma::uword>(std::llround(std::floor(std::sqrt(n))));
       i++) {
    if (n % i == 0) {
      return false;
    }
  }
  return true;
}

//******************************************************************************

inline arma::uvec factors(arma::uword n) {
#ifndef QIC_LIB_NO_DEBUG
  if (n == 0 || n == 1)
    throw Exception("qic::is_prime", Exception::type::OUT_OF_RANGE);
#endif

  std::vector<arma::uword> ret;

  for (arma::uword i = 2;
       i <= static_cast<arma::uword>(std::llround(std::floor(std::sqrt(n))));
       i++) {
    if (n % i == 0) {
      ret.push_back(i);
      ret.push_back(n / i);
    }
  }
  return arma::conv_to<arma::uvec>::from(ret);
}

//******************************************************************************

}  // namespace qic
