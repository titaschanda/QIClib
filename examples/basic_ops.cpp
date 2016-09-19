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

int main(int argc, char** argv) {
  using namespace std;
  using namespace arma;
  using namespace qic;

  mat A = randU<mat>(5, 5);        // 5x5 real matrix
  cx_mat B = randU<cx_mat>(4, 4);  // 4x4 complex matrix

  // Hermiticity check
  cout << is_Hermitian(A) << endl << is_Hermitian(B * B.t()) << endl;

  cx_mat C = B;

  // check for equality
  cout << is_equal(A, B) << endl << is_equal(B, C) << endl;

  B *= B.t();     // B is now Hermitian
  B /= trace(B);  // normalise B

  cx_mat D = randRho(12);  // 12x12 random density matrix

  // check if the matrix is a valid state
  cout << is_valid_state(A) << endl
       << is_valid_state(B) << endl
       << is_valid_state(D) << endl;

  // trace out first party of B
  cx_mat B2 = TrX(B, {1});  // B2 is 2x2 matrix
  // cx_mat B2 = TrX(B, {1}, 2); // same as above
  // cx_mat B2 = TrX(B, {1}, {2,2}); // same as above

  // trace out 2nd and 3rd party of D,
  // where 2nd party is qutrit and 3rd party is qudit
  cx_mat D2 = TrX(D, {2, 3}, {2, 3, 2});  // D2 is now 2x2 matrix

  return 0;
}
