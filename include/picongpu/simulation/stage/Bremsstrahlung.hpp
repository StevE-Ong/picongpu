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
#include <map>
#include <memory>

namespace picongpu::particles::bremsstrahlung
{
    struct ScaledSpectrum;
    struct PhotonEmissionAngle;
} // namespace picongpu::particles::bremsstrahlung

namespace picongpu::simulation::stage
{
    /** Functor for the stage of the PIC loop computing Bremsstrahlung
     *
     * Only affects electron species with the bremsstrahlungIons<> and bremsstrahlungPhotons<> flags,
     * see param/bremsstrahlung.param.
     */
    class Bremsstrahlung
    {
    public:
        //! atomic number -> lookup tables of the scaled spectrum and stopping power
        using ScaledSpectrumMap = std::map<float_X, std::shared_ptr<particles::bremsstrahlung::ScaledSpectrum>>;

        /** Create the Bremsstrahlung stage; generates the lookup tables if a species uses Bremsstrahlung
         *
         * @param cellDescription mapping for kernels
         */
        Bremsstrahlung(MappingDesc const cellDescription);

        /** Bremsstrahlung of all electron species for one time step
         *
         * @param step index of time iteration
         */
        void operator()(uint32_t const step);

    private:
        //! creates the lookup tables and the requirement flag
        void init();

        //! writes the lookup tables as text files (MPI rank 0 only)
        void dumpTables() const;

        //! reads the requirement flag from the device, prints a warning once per kind and resets the flag
        void checkRequirements();

        //! Mapping for kernels
        MappingDesc cellDescription;

        ScaledSpectrumMap scaledSpectrumMap;
        std::shared_ptr<particles::bremsstrahlung::PhotonEmissionAngle> photonAngle;

        //! bit mask of failed requirements, see particles::bremsstrahlung::requirement
        std::shared_ptr<GridBuffer<int32_t, 1>> failedRequirement;
        int32_t printedRequirements = 0;
    };
} // namespace picongpu::simulation::stage
