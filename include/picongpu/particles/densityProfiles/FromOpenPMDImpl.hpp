/* Copyright 2013-2024 Felix Schmitt, Rene Widera, Sergei Bastrakov
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

#if (ENABLE_OPENPMD == 1)

#    pragma once

#    include "picongpu/defines.hpp"
#    include "picongpu/particles/functor/misc/OpenPMDScalarField.hpp"

#    include <cstdint>
#    include <string>

namespace picongpu
{
    namespace densityProfiles
    {
        /** Access to the runtime density file name for a species
         *
         * Used to propagate runtime values to FromOpenPMDImpl.
         * Same runtime value is used for all FromOpenPMDImpl<> invocations for the same species with no compile-time
         * value set.
         * The value is stored by species name, see particles::functor::misc::RuntimeFile.
         *
         * @tparam T_SpeciesType species type
         */
        template<typename T_SpeciesType>
        struct RuntimeDensityFile
        {
            //! Access the statically stored value
            static std::string& get()
            {
                return particles::functor::misc::RuntimeFile<particles::functor::misc::RuntimeDensityFileTag>::get(
                    T_SpeciesType::FrameType::getName());
            }
        };

        namespace detail
        {
            /** Implementation of loading density from a file
             *
             * @tparam T_ParamClass parameter type
             * @tparam T_SpeciesType species type
             */
            template<typename T_ParamClass, typename T_SpeciesType>
            struct FromOpenPMDImpl : public T_ParamClass
            {
                //! Parameters type
                using ParamClass = T_ParamClass;

                //! Species type
                using SpeciesType = T_SpeciesType;

                /** Initialize the functor on the host side, includes loading the file
                 *
                 * This is MPI collective operation, must be called by all ranks.
                 *
                 * @param currentStep current time iteration
                 */
                HINLINE FromOpenPMDImpl(uint32_t currentStep)
                    : field(
                          particles::functor::misc::
                              getOpenPMDFilename<particles::functor::misc::RuntimeDensityFileTag, SpeciesType>(
                                  ParamClass::filename),
                          ParamClass::datasetName,
                          ParamClass::iteration,
                          ParamClass::offset,
                          ParamClass::defaultDensity,
                          std::string("density for species \"") + SpeciesType::FrameType::getName() + "\"")
                {
                }

                /** Calculate the normalized density based on the file contents
                 *
                 * @param totalCellOffset total offset including all slides [in cells]
                 */
                HDINLINE float_X operator()(DataSpace<simDim> const& totalCellOffset)
                {
                    return field(totalCellOffset);
                }

            private:
                //! Density values loaded from the file
                PMACC_ALIGN(field, particles::functor::misc::OpenPMDScalarField);
            };
        } // namespace detail

        /** Wrapper to be used in density.param, compatible with other density definitions
         *
         * Hooks internal implementation in detail:: to boost::mpl::apply
         *
         * @tparam T_ParamClass parameter type
         */
        template<typename T_ParamClass>
        struct FromOpenPMDImpl : public T_ParamClass
        {
            template<typename T_SpeciesType>
            struct apply
            {
                using type = detail::FromOpenPMDImpl<T_ParamClass, T_SpeciesType>;
            };
        };

    } // namespace densityProfiles
} // namespace picongpu

#endif
