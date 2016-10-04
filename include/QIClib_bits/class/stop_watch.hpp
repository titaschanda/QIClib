/*
 * This file contains modified version of Exception class
 * released as a part of Quantum++-v0.8.7 by Vlad Gheorghiu
 * (vgheorgh@gmail.com) under GPLv3, see <https://github.com/vsoftco/qpp>.
 *
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

//******************************************************************************

class stop_watch {
  private:
   std::chrono::steady_clock::time_point start{}, end{};

  public:
  stop_watch() noexcept : start{std::chrono::steady_clock::now()},
                           end{start} {}

  stop_watch(const stop_watch&) = default;
  stop_watch(stop_watch&&) = default;

  stop_watch& operator=(const stop_watch&) = default;
  stop_watch& operator=(stop_watch&&) = default;
  
  inline void tic() noexcept {
    start = end = std::chrono::steady_clock::now();
  }

  inline const stop_watch& toc() noexcept {
    end = std::chrono::steady_clock::now();
    return *this;
  }

  inline double tics() const noexcept {
    return std::chrono::duration_cast<std::chrono::duration<double> >(end -
                                                                      start)
      .count();
  }

  friend inline std::ostream& operator<<(std::ostream&, const stop_watch&); 
}; 

//******************************************************************************

inline std::ostream& operator<<(std::ostream& os, const stop_watch& clock) {
  os << clock.tics() << " seconds";
  return os;
}

}  // namespace qic
