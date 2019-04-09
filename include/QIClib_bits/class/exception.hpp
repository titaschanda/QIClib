/*
 * This file contains modified version of Exception class
 * released as a part of Quantum++-v0.8.7 by Vlad Gheorghiu
 * (vgheorgh@gmail.com) under GPLv3, see <https://github.com/vsoftco/qpp>.
 *
 * QIClib (Quantum information and computation library)
 *
 * Copyright (c) 2015 - 2019  Titas Chanda (titas.chanda@gmail.com)
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

#ifndef _QICLIB_EXCEPTION_HPP_
#define _QICLIB_EXCEPTION_HPP_

#include <armadillo>

namespace qic {

//******************************************************************************

class Exception : public std::exception {
 public:
  enum class type {
    UNKNOWN_EXCEPTION = 1,
    ZERO_SIZE,
    MATRIX_NOT_SQUARE,  //
    MATRIX_NOT_CVECTOR,
    MATRIX_NOT_RVECTOR,
    MATRIX_NOT_VECTOR,
    MATRIX_NOT_SQUARE_OR_CVECTOR,
    MATRIX_NOT_SQUARE_OR_RVECTOR,
    MATRIX_NOT_SQUARE_OR_VECTOR,
    MATRIX_MISMATCH_SUBSYS,  //
    SUBSYS_MISMATCH_DIMS,
    INVALID_SUBSYS,        //
    INVALID_DIMS,          //
    DIMS_NOT_EQUAL,        //
    DIMS_MISMATCH_MATRIX,  //
    DIMS_MISMATCH_CVECTOR,
    DIMS_MISMATCH_RVECTOR,
    DIMS_MISMATCH_VECTOR,
    INVALID_PERM,  //
    NOT_QUBIT_MATRIX,
    NOT_QUBIT_CVECTOR,
    NOT_QUBIT_RVECTOR,
    NOT_QUBIT_VECTOR,
    NOT_TWO_QUBIT_MATRIX,
    NOT_TWO_QUBIT_CVECTOR,
    NOT_TWO_QUBIT_RVECTOR,
    NOT_TWO_QUBIT_VECTOR,
    NOT_QUBIT_SUBSYS,
    NOT_BIPARTITE,  //
    NO_CODEWORD,
    OUT_OF_RANGE,
    TYPE_MISMATCH,  //
    MATRIX_SIZE_MISMATCH,
    SIZE_MISMATCH,  //
    UNDEFINED_type,
    CUSTOM_EXCEPTION
  };

  inline Exception(const std::string& where, const type& Type)
      : _where{where}, _msg{}, _type{Type}, _custom{} {
    _construct_exception_msg();
  }

  inline Exception(const std::string& where, const std::string& custom)
      : _where{where}, _msg{}, _type{type::CUSTOM_EXCEPTION}, _custom{custom} {
    _construct_exception_msg();
    _msg += custom;
  }

  inline const char* what() const noexcept override { return _msg.c_str(); }

 private:
  std::string _where, _msg;
  type _type;
  std::string _custom;

  inline void _construct_exception_msg() {
    _msg += "In ";
    _msg += _where;
    _msg += "(): ";

    switch (_type) {
    case type::UNKNOWN_EXCEPTION:
      _msg += "UNKNOWN EXCEPTION!";
      break;

    case type::ZERO_SIZE:
      _msg += "Object has zero size!";
      break;

    case type::MATRIX_NOT_SQUARE:
      _msg += "Matrix is not square!";
      break;

    case type::MATRIX_NOT_CVECTOR:
      _msg += "Matrix is not column vector!";
      break;

    case type::MATRIX_NOT_RVECTOR:
      _msg += "Matrix is not row vector!";
      break;

    case type::MATRIX_NOT_VECTOR:
      _msg += "Matrix is not vector!";
      break;

    case type::MATRIX_NOT_SQUARE_OR_CVECTOR:
      _msg += "Matrix is not square nor column vector!";
      break;

    case type::MATRIX_NOT_SQUARE_OR_RVECTOR:
      _msg += "Matrix is not square nor row vector!";
      break;

    case type::MATRIX_NOT_SQUARE_OR_VECTOR:
      _msg += "Matrix is not square nor vector!";
      break;

    case type::MATRIX_MISMATCH_SUBSYS:
      _msg += "Matrix mismatch subsystems!";
      break;

    case type::INVALID_SUBSYS:
      _msg += "Invalid subsystem index!";
      break;

    case type::INVALID_DIMS:
      _msg += "Invalid dimension(s)!";
      break;

    case type::DIMS_NOT_EQUAL:
      _msg += "Dimensions not equal!";
      break;

    case type::DIMS_MISMATCH_MATRIX:
      _msg += "Dimension(s) mismatch matrix size!";
      break;

    case type::SUBSYS_MISMATCH_DIMS:
      _msg += "Subsystem(s) mismatch dimensions!";
      break;

    case type::DIMS_MISMATCH_CVECTOR:
      _msg += "Dimension(s) mismatch column vector!";
      break;

    case type::DIMS_MISMATCH_RVECTOR:
      _msg += "Dimension(s) mismatch row vector!";
      break;

    case type::DIMS_MISMATCH_VECTOR:
      _msg += "Dimension(s) mismatch vector!";
      break;

    case type::INVALID_PERM:
      _msg += "Invalid permutation!";
      break;

    case type::NOT_QUBIT_MATRIX:
      _msg += "Matrix is not 2 x 2!";
      break;

    case type::NOT_QUBIT_CVECTOR:
      _msg += "Column vector is not 2 x 1!";
      break;

    case type::NOT_QUBIT_RVECTOR:
      _msg += "Row vector is not 1 x 2!";
      break;

    case type::NOT_QUBIT_VECTOR:
      _msg += "Vector is not 2 x 1 nor 1 x 2!";
      break;

    case type::NOT_QUBIT_SUBSYS:
      _msg += "Subsystems are not qubits!";
      break;

    case type::NOT_TWO_QUBIT_MATRIX:
      _msg += "Matrix is not 4 x 4!";
      break;

    case type::NOT_TWO_QUBIT_CVECTOR:
      _msg += "Column vector is not 4 x 1!";
      break;

    case type::NOT_TWO_QUBIT_RVECTOR:
      _msg += "Row vector is not 1 x 4!";
      break;

    case type::NOT_TWO_QUBIT_VECTOR:
      _msg += "Vector is not 4 x 1 nor 1 x 4!";
      break;

    case type::NOT_BIPARTITE:
      _msg += "Not bi-partite!";
      break;

    case type::NO_CODEWORD:
      _msg += "Codeword does not exist!";
      break;

    case type::OUT_OF_RANGE:
      _msg += "Parameter out of range!";
      break;

    case type::TYPE_MISMATCH:
      _msg += "Type mismatch!";
      break;

    case type::MATRIX_SIZE_MISMATCH:
      _msg += "Matrix size mismatch!";
      break;

    case type::SIZE_MISMATCH:
      _msg += "Size mismatch!";
      break;

    case type::UNDEFINED_type:
      _msg += "Not defined for this type!";
      break;

    case type::CUSTOM_EXCEPTION:
      _msg += "";
      break;
    }
  }
};

//******************************************************************************

}  // namespace qic

#endif
