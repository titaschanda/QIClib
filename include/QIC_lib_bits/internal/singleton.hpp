/*
 * This file contains modified version of Singleton class
 * released as a part of Quantum++-v0.8.6 by Vlad Gheorghiu
 * (vgheorgh@gmail.com) under GPLv3, see <https://github.com/vsoftco/qpp>.
 *
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

namespace _internal {

template <typename T> class Singleton {
 protected:
  Singleton() noexcept = default;
  Singleton(const Singleton&) = delete;
  Singleton& operator=(const Singleton&) = delete;
  virtual ~Singleton() = default;

 public:
  static T& get_instance() noexcept(std::is_nothrow_constructible<T>::value) {
    static T instance;
    return instance;
  }

#ifndef _NO_THREAD_LOCAL

  static T& get_thread_local_instance()
    noexcept(std::is_nothrow_constructible<T>::value) {
    thread_local static T instance;
    return instance;
  }

#endif
};

}  // namespace _internal

}  // namespace qic
