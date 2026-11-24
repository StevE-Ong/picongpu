/* Copyright 2016-2026 Heiko Burau, Jian Fuh Ong
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/defines.hpp"

#include <pmacc/memory/buffers/GridBuffer.hpp>

#include <cstdint>

namespace picongpu::particles::bremsstrahlung
{
    /** device buffer type of all Bremsstrahlung lookup tables */
    using TableBuffer = pmacc::GridBuffer<float_X, 2>;

    /** Bilinear interpolation in a 2D lookup table with values on the nodes
     *
     * The position is given in index units of the two table axes. It is clamped to the table, so a lookup
     * never reads outside of the table. The callers detect out-of-range positions and report them.
     * Inside the table this is the same rule as the cuSTL LinearInterp cursor of PIConGPU 0.6.0.
     */
    struct LookupTable2D
    {
        TableBuffer::DataBoxType box;
        uint32_t size0;
        uint32_t size1;

        /** linear interpolation
         *
         * @param pos0 position along axis 0 in index units
         * @param pos1 position along axis 1 in index units
         * @return interpolated table value
         */
        HDINLINE float_X operator()(float_X pos0, float_X pos1) const
        {
            pos0 = math::max(float_X(0.0), math::min(pos0, static_cast<float_X>(size0 - 1u)));
            pos1 = math::max(float_X(0.0), math::min(pos1, static_cast<float_X>(size1 - 1u)));

            int const i0 = math::min(static_cast<int>(pos0), static_cast<int>(size0) - 2);
            int const i1 = math::min(static_cast<int>(pos1), static_cast<int>(size1) - 2);
            float_X const f0 = pos0 - static_cast<float_X>(i0);
            float_X const f1 = pos1 - static_cast<float_X>(i1);

            float_X const v00 = box(DataSpace<2>{i0, i1});
            float_X const v10 = box(DataSpace<2>{i0 + 1, i1});
            float_X const v01 = box(DataSpace<2>{i0, i1 + 1});
            float_X const v11 = box(DataSpace<2>{i0 + 1, i1 + 1});

            return (float_X(1.0) - f0) * ((float_X(1.0) - f1) * v00 + f1 * v01)
                   + f0 * ((float_X(1.0) - f1) * v10 + f1 * v11);
        }
    };
} // namespace picongpu::particles::bremsstrahlung
