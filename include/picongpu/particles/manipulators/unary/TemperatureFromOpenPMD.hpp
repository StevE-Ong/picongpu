/* Copyright 2026 StevE-Ong
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

//! @file implements a manipulator setting the particle temperature from an openPMD file

#if (ENABLE_OPENPMD == 1)

#    pragma once

#    include "picongpu/defines.hpp"
#    include "picongpu/particles/functor/misc/OpenPMDScalarField.hpp"
#    include "picongpu/particles/manipulators/unary/Temperature.hpp"
#    include "picongpu/particles/manipulators/unary/TemperatureFromOpenPMD.def"

#    include <cstdint>
#    include <string>

namespace picongpu::particles::manipulators::unary::acc
{
    //! see TemperatureFromOpenPMD.def for documentation
    template<typename T_ParamClass, typename T_SpeciesType, typename T_ValueFunctor>
    struct TemperatureFromOpenPMD : public detail::TemperatureImpl<T_ValueFunctor>
    {
        //! Base implementation class
        using Base = detail::TemperatureImpl<T_ValueFunctor>;

        //! Parameters type
        using ParamClass = T_ParamClass;

        //! Species type
        using SpeciesType = T_SpeciesType;

        //! Name of the functor
        static constexpr char const* name = "temperatureFromOpenPMD";

        /** Initialize the functor on the host side, includes loading the file
         *
         * This is an MPI collective operation, must be called by all ranks.
         *
         * @param currentStep current time iteration
         */
        HINLINE TemperatureFromOpenPMD(uint32_t currentStep)
            : field(
                  functor::misc::getOpenPMDFilename<functor::misc::RuntimeTemperatureFileTag, SpeciesType>(
                      ParamClass::filename),
                  ParamClass::datasetName,
                  ParamClass::iteration,
                  ParamClass::offset,
                  ParamClass::defaultTemperature,
                  std::string("temperature [keV] for species \"") + SpeciesType::FrameType::getName() + "\"")
        {
        }

        /** Manipulate the momentum of the given macroparticle
         *
         * @tparam T_StandardNormalRng functor::misc::RngWrapper, standard normal random number generator type
         * @tparam T_Particle particle type
         *
         * @param totalCellOffset total offset including all slides [in cells]
         * @param standardNormalRng standard normal random number generator
         * @param particle particle to be manipulated
         */
        template<typename T_StandardNormalRng, typename T_Particle>
        HDINLINE void operator()(
            DataSpace<simDim> const& totalCellOffset,
            T_StandardNormalRng& standardNormalRng,
            T_Particle& particle)
        {
            float_X const temperatureKeV = math::max(field(totalCellOffset), 0.0_X);
            Base::operator()(standardNormalRng, particle, temperatureKeV);
        }

    private:
        //! Temperature values in keV loaded from the file
        functor::misc::OpenPMDScalarField field;
    };
} // namespace picongpu::particles::manipulators::unary::acc

#endif
