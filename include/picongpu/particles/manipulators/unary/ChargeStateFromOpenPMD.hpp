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

//! @file implements a manipulator setting the ion charge state from an openPMD file

#if (ENABLE_OPENPMD == 1)

#    pragma once

#    include "picongpu/defines.hpp"
#    include "picongpu/particles/atomicPhysics/SetChargeState.hpp"
#    include "picongpu/particles/functor/misc/OpenPMDScalarField.hpp"
#    include "picongpu/particles/manipulators/unary/ChargeStateFromOpenPMD.def"
#    include "picongpu/particles/traits/GetAtomicNumbers.hpp"

#    include <cstdint>
#    include <string>

namespace picongpu::particles::manipulators::unary::acc
{
    //! see ChargeStateFromOpenPMD.def for documentation
    template<typename T_ParamClass, typename T_SpeciesType>
    struct ChargeStateFromOpenPMD
    {
        //! Parameters type
        using ParamClass = T_ParamClass;

        //! Species type
        using SpeciesType = T_SpeciesType;

        //! Name of the functor
        static constexpr char const* name = "chargeStateFromOpenPMD";

        /** Initialize the functor on the host side, includes loading the file
         *
         * This is an MPI collective operation, must be called by all ranks.
         *
         * @param currentStep current time iteration
         */
        HINLINE ChargeStateFromOpenPMD(uint32_t currentStep)
            : field(
                  functor::misc::getOpenPMDFilename<functor::misc::RuntimeChargeStateFileTag, SpeciesType>(
                      ParamClass::filename),
                  ParamClass::datasetName,
                  ParamClass::iteration,
                  ParamClass::offset,
                  ParamClass::defaultChargeState,
                  std::string("charge state for species \"") + SpeciesType::FrameType::getName() + "\"")
        {
        }

        /** Set the charge state of the given macro-ion
         *
         * @tparam T_UniformRng functor::misc::RngWrapper, uniform random number generator type in [0, 1)
         * @tparam T_Ion ion particle type
         *
         * @param totalCellOffset total offset including all slides [in cells]
         * @param uniformRng uniform random number generator
         * @param ion macro-ion to be manipulated
         */
        template<typename T_UniformRng, typename T_Ion>
        HDINLINE void operator()(DataSpace<simDim> const& totalCellOffset, T_UniformRng& uniformRng, T_Ion& ion)
        {
            constexpr float_X numberOfProtons = picongpu::traits::GetAtomicNumbers<T_Ion>::type::numberOfProtons;

            float_X const chargeState = math::min(math::max(field(totalCellOffset), 0.0_X), numberOfProtons);
            // Stochastic rounding: the expected value of the result equals chargeState
            float_X const lowerChargeState = math::floor(chargeState);
            float_X const fraction = chargeState - lowerChargeState;
            float_X const macroIonChargeState
                = math::min(lowerChargeState + ((uniformRng() < fraction) ? 1.0_X : 0.0_X), numberOfProtons);

            picongpu::particles::atomicPhysics::SetChargeState{}(ion, numberOfProtons - macroIonChargeState);
        }

    private:
        //! Charge state values loaded from the file
        functor::misc::OpenPMDScalarField field;
    };
} // namespace picongpu::particles::manipulators::unary::acc

#endif
