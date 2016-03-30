/*
 * This file contains modified version of random_devices class
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

//******************************************************************************

class RandomDevices final
  : public _internal::protect_subs::Singleton<RandomDevices> {
  friend class _internal::protect_subs::Singleton<RandomDevices>;

  std::random_device rd;

 public:
#ifdef ARMA_64BIT_WORD
  std::mt19937_64 rng;
  using seed_type = std::mt19937_64::result_type;
#else
  std::mt19937 rng;
  using seed_type = std::mt19937::result_type;
#endif

  inline void set_seed(seed_type a) { rng.seed(a); }

  inline void set_seed_random() { rng.seed(rd()); }

 private:
  RandomDevices() : rd{}, rng{rd()} {}

  ~RandomDevices() = default;
};

//******************************************************************************

#ifdef _NO_THREAD_LOCAL
static RandomDevices& rdevs _QIC_UNUSED_ = RandomDevices::get_instance();
#else
thread_local static RandomDevices& rdevs _QIC_UNUSED_ =
  RandomDevices::get_thread_local_instance();
#endif

//******************************************************************************

}  // namespace qic
