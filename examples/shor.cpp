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

#include <QIClib>

bool shor(arma::uword N, arma::uvec &fact) {
  if (N % 2 == 0) {
    std::cout << "ERROR... Number is even..." << std::endl;
    return false;
  }

  if (qic::is_prime(N)) {
    std::cout << "ERROR... Number is prime..." << std::endl;
    return false;
  }

  double pi = arma::datum::pi;
  arma::cx_double I(0.0, 1.0);
  bool ret = false;
  std::uniform_int_distribution<arma::uword> dis(1, N - 1);
  arma::uword count(0);

  arma::uword l = std::ceil(std::log2(static_cast<double>(N)));
  arma::uword Q = std::pow(2, l);
  arma::uword l2 = std::ceil(std::log2(static_cast<double>(N)));
  arma::uword Q2 = std::pow(2, l2);

  std::cout << "Created register 1 of size " << l << " qubits" << std::endl;
  std::cout << "Created register 2 of size " << l2 << " qubits" << std::endl;
  std::cout << std::endl << std::endl;

  while (!ret) {
    arma::uword a = dis(qic::rdevs.rng);
    arma::uword gcd1 = qic::gcd(a, N);

    if (gcd1 != 1) {
      std::cout << "GCD != 1... trying again..." << std::endl;
      continue;
    }

    std::cout << std::endl << std::endl;
    std::cout << "Trial --->   # " << count + 1 << std::endl << std::endl;

    std::cout << "a = " << a << std::endl;
    std::cout << "gcd = " << gcd1 << std::endl << std::endl;

    arma::vec reg1 = arma::zeros<arma::vec>(Q);
    arma::vec reg2 = arma::zeros<arma::vec>(Q2);
    arma::uvec modex = arma::zeros<arma::uvec>(Q);

    for (arma::uword i = 0; i < Q; ++i) {
      modex(i) = qic::modexp(a, i, N);
      reg2.at(modex(i)) += 1.0;
    }
    reg2 = arma::normalise(reg2);

    auto measure1 = qic::measure_comp(reg2);
    arma::uword result1 = std::get<0>(measure1);

    for (arma::uword i = 0; i < Q; ++i) {
      if (result1 == modex(i))
        reg1.at(i)++;
    }
    reg1 = arma::normalise(reg1);

    arma::cx_vec state2 = arma::zeros<arma::cx_vec>(Q);

    std::cout << "QFT begins..." << std::endl;

    QICLIB_OPENMP_FOR
    for (arma::uword j = 0; j < Q; ++j) {
      for (arma::uword i = 0; i < Q; ++i) {
        state2.at(j) += std::exp(2.0 * pi * I * static_cast<double>(i * j) /
                                 static_cast<double>(Q)) *
                        reg1.at(i);
      }
    }
    state2 /= std::sqrt(Q);

    std::cout << "QFT done." << std::endl << std::endl;

    arma::uword count2(0);

    while (true) {
      auto measure3 = qic::measure_comp(state2);
      auto C = std::get<0>(measure3);

      auto q = static_cast<double>(C) / (static_cast<double>(Q));
      auto r = qic::denominator(q, Q);

      if (r % 2 != 0 && 2 * r < Q && C != 0)
        r *= 2;

      if (r % 2 == 0 && C != 0) {
        arma::uword e = qic::modexp(a, r / 2, N);
        arma::uword a = (e + 1) % N;
        arma::uword b = (e + N - 1) % N;
        arma::uword factor = std::max(qic::gcd(N, a), qic::gcd(N, b));

        if (factor == 1 || factor == N || factor == 0) {
          if (factor == 1 || factor == N)
            std::cout << "Found trivial factors... trying again..."
                      << std::endl;
          else
            std::cout << "Found factor to be 0... trying again..." << std::endl;

        } else {
          fact.resize(2);
          fact.at(0) = factor;
          fact.at(1) = N / factor;
          ret = true;
          std::cout << std::endl;
          std::cout << "Measurement result = " << C << std::endl;
          std::cout << "Possible period = " << r << std::endl << std::endl;

          std::cout << "Found factors..." << std::endl << fact << std::endl;
          break;
        }
      }
      count2++;
      if (count2 == 20)
        break;
    }

    if (count == 9) {
      break;
    }
    if (!ret)
      std::cout << "FAIL... trying again..." << std::endl << std::endl;
    count++;
  }
  return ret;
}

int main() {
  arma::uvec f;
  shor(821 * 163, f);

  std::cout << f << std::endl;
}
