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


#ifndef _QIC_LIB_HPP
#define _QIC_LIB_HPP


#include<armadillo>
#include<iostream>
#include<iomanip>
#include<cmath>
#include<complex>
#include<stdexcept>
#include<type_traits>
#include<tuple>
#include<random>
#include<limits>

#define QIC_LIB_NLOPT
// Comment out the previous line, if you don't want to use
// NLopt dependent features




#if (__GNUC__ && !__clang__)
#define _QIC_UNUSED_  __attribute__ ((unused))
#else
#define _QIC_UNUSED_
#endif

#include "QIC_lib_bits/internal/first.hpp"
#include "QIC_lib_bits/internal/singleton.hpp"

#include "QIC_lib_bits/class/exception.hpp"
#include "QIC_lib_bits/class/constants.hpp"
#include "QIC_lib_bits/class/init.hpp"
#include "QIC_lib_bits/class/random_devices.hpp"


#include "QIC_lib_bits/basic/type_traits.hpp"
#include "QIC_lib_bits/basic/as_Mat.hpp"
#include "QIC_lib_bits/basic/as_Col.hpp"
#include "QIC_lib_bits/basic/is_equal.hpp"
#include "QIC_lib_bits/basic/is_H.hpp"
#include "QIC_lib_bits/basic/is_U.hpp"
#include "QIC_lib_bits/basic/is_pure.hpp"
#include "QIC_lib_bits/basic/is_valid_state.hpp"
#include "QIC_lib_bits/basic/range.hpp"
#include "QIC_lib_bits/basic/random.hpp"

#include "QIC_lib_bits/internal/second.hpp"

#include "QIC_lib_bits/function/Tx.hpp"
#include "QIC_lib_bits/function/TrX.hpp"
#include "QIC_lib_bits/function/sysperm.hpp"
#include "QIC_lib_bits/function/powm.hpp"
#include "QIC_lib_bits/function/expm.hpp"
#include "QIC_lib_bits/function/sqrtm.hpp"
#include "QIC_lib_bits/function/funcm.hpp"
#include "QIC_lib_bits/function/tensor.hpp"
#include "QIC_lib_bits/function/dsum.hpp"
#include "QIC_lib_bits/function/absm.hpp"
#include "QIC_lib_bits/function/conv_to_pure.hpp"
#include "QIC_lib_bits/function/pauli.hpp"
#include "QIC_lib_bits/function/purify.hpp"
#include "QIC_lib_bits/function/generator.hpp"


#include "QIC_lib_bits/function/apply_ctrl.hpp"
#include "QIC_lib_bits/function/apply.hpp"
#include "QIC_lib_bits/function/measure.hpp"
#include "QIC_lib_bits/function/entropy.hpp"
#include "QIC_lib_bits/function/entanglement.hpp"
#include "QIC_lib_bits/function/neg.hpp"
#include "QIC_lib_bits/function/mutual_info.hpp"
#include "QIC_lib_bits/function/ent_check_CMC.hpp"
#include "QIC_lib_bits/function/concurrence.hpp"
#include "QIC_lib_bits/function/EoF.hpp"
#include "QIC_lib_bits/function/distance.hpp"
#include "QIC_lib_bits/function/channel.hpp"
#include "QIC_lib_bits/function/schmidt.hpp"
#include "QIC_lib_bits/function/schatten.hpp"


#ifdef QIC_LIB_NLOPT

// NLopt dependent features

#include<nlopt.hpp>
#include "QIC_lib_bits/discord/default_config.hpp"
#include "QIC_lib_bits/discord/discord_reg.hpp"
#include "QIC_lib_bits/discord/discord3_reg.hpp"
#include "QIC_lib_bits/discord/deficit_reg.hpp"
#include "QIC_lib_bits/discord/deficit3_reg.hpp"
#include "QIC_lib_bits/discord/discord.hpp"
#include "QIC_lib_bits/discord/discord3.hpp"
#include "QIC_lib_bits/discord/deficit.hpp"
#include "QIC_lib_bits/discord/deficit3.hpp"

#endif

#endif
