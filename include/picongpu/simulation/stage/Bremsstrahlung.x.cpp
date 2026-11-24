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

#include "picongpu/simulation/stage/Bremsstrahlung.hpp"

#include "picongpu/defines.hpp"
#include "picongpu/param/bremsstrahlung.param"
#include "picongpu/particles/bremsstrahlung/AlgorithmBremsstrahlung.hpp"
#include "picongpu/particles/bremsstrahlung/PhotonEmissionAngle.hpp"
#include "picongpu/particles/bremsstrahlung/ScaledSpectrum.hpp"
#include "picongpu/particles/creation/creation.hpp"
#include "picongpu/particles/filter/filter.hpp"
#include "picongpu/particles/param.hpp"
#include "picongpu/particles/particleToGrid/ComputeGridValuePerFrame.hpp"
#include "picongpu/particles/particleToGrid/derivedAttributes/DerivedAttributes.hpp"
#include "picongpu/particles/traits/GetAtomicNumbers.hpp"

#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/meta/ForEach.hpp>
#include <pmacc/particles/meta/FindByNameOrType.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>
#include <pmacc/traits/GetFlagType.hpp>
#include <pmacc/traits/Resolve.hpp>

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace picongpu::simulation::stage
{
    namespace
    {
        //! electron species with Bremsstrahlung
        using SpeciesWithBremsstrahlung =
            typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, bremsstrahlungIons<>>::type;

        /** ion and photon species of an electron species with Bremsstrahlung
         *
         * @tparam T_ElectronSpecies type or name as PMACC_CSTRING of the electron species
         */
        template<typename T_ElectronSpecies>
        struct BremsstrahlungSpecies
        {
            using ElectronSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_ElectronSpecies>;
            using FrameType = typename ElectronSpecies::FrameType;

            using IonSpecies = pmacc::particles::meta::FindByNameOrType_t<
                VectorAllSpecies,
                typename pmacc::traits::Resolve<
                    typename pmacc::traits::GetFlagType<FrameType, bremsstrahlungIons<>>::type>::type>;
            using PhotonSpecies = pmacc::particles::meta::FindByNameOrType_t<
                VectorAllSpecies,
                typename pmacc::traits::Resolve<
                    typename pmacc::traits::GetFlagType<FrameType, bremsstrahlungPhotons<>>::type>::type>;

            static constexpr float_X targetZ = picongpu::traits::GetAtomicNumbers<IonSpecies>::type::numberOfProtons;
        };

        /** create the lookup tables of the ion species of an electron species (once per atomic number)
         *
         * @tparam T_ElectronSpecies type or name as PMACC_CSTRING of the electron species
         */
        template<typename T_ElectronSpecies>
        struct FillScaledSpectrumMap
        {
            void operator()(Bremsstrahlung::ScaledSpectrumMap& map) const
            {
                constexpr float_X targetZ = BremsstrahlungSpecies<T_ElectronSpecies>::targetZ;
                if(map.count(targetZ) == 0)
                {
                    auto scaledSpectrum = std::make_shared<particles::bremsstrahlung::ScaledSpectrum>();
                    scaledSpectrum->init(static_cast<float_64>(targetZ));
                    map[targetZ] = scaledSpectrum;
                }
            }
        };

        /** Bremsstrahlung of one electron species
         *
         * @tparam T_ElectronSpecies type or name as PMACC_CSTRING of the electron species
         */
        template<typename T_ElectronSpecies>
        struct CallBremsstrahlung
        {
            using Species = BremsstrahlungSpecies<T_ElectronSpecies>;
            using ElectronSpecies = typename Species::ElectronSpecies;
            using IonSpecies = typename Species::IonSpecies;
            using PhotonSpecies = typename Species::PhotonSpecies;

            void operator()(
                MappingDesc const& cellDesc,
                uint32_t const currentStep,
                Bremsstrahlung::ScaledSpectrumMap const& scaledSpectrumMap,
                particles::bremsstrahlung::PhotonEmissionAngle const& photonAngle,
                GridBuffer<int32_t, 1>& failedRequirement) const
            {
                DataConnector& dc = Environment<>::get().DataConnector();

                auto electronSpeciesPtr = dc.get<ElectronSpecies>(ElectronSpecies::FrameType::getName());
                auto photonSpeciesPtr = dc.get<PhotonSpecies>(PhotonSpecies::FrameType::getName());

                auto const& scaledSpectrum = *scaledSpectrumMap.at(Species::targetZ);

                using Functor
                    = particles::bremsstrahlung::AlgorithmBremsstrahlung<IonSpecies, ElectronSpecies, PhotonSpecies>;
                Functor bremsstrahlungFunctor(
                    currentStep,
                    scaledSpectrum.getScaledSpectrumFunctor(),
                    scaledSpectrum.getStoppingPowerFunctor(),
                    photonAngle.getPhotonAngleFunctor(),
                    failedRequirement.getDeviceBuffer().getDataBox());

                particles::creation::createParticlesFromSpecies(
                    *electronSpeciesPtr,
                    *photonSpeciesPtr,
                    bremsstrahlungFunctor,
                    cellDesc);
            }
        };
    } // namespace

    Bremsstrahlung::Bremsstrahlung(MappingDesc const cellDescription) : cellDescription(cellDescription)
    {
        if constexpr(pmacc::mp_size<SpeciesWithBremsstrahlung>::value > 0)
        {
            init();
        }
    }

    void Bremsstrahlung::init()
    {
        pmacc::meta::ForEach<SpeciesWithBremsstrahlung, FillScaledSpectrumMap<boost::mpl::_1>> fillScaledSpectrumMap;
        fillScaledSpectrumMap(scaledSpectrumMap);

        photonAngle = std::make_shared<particles::bremsstrahlung::PhotonEmissionAngle>();
        photonAngle->init();

        failedRequirement = std::make_shared<GridBuffer<int32_t, 1>>(DataSpace<1>{1});
        failedRequirement->getHostBuffer().getDataBox()(DataSpace<1>{0}) = 0;
        failedRequirement->hostToDevice();

        if constexpr(particles::bremsstrahlung::dumpTables)
            dumpTables();
    }

    void Bremsstrahlung::operator()(uint32_t const step)
    {
        if constexpr(pmacc::mp_size<SpeciesWithBremsstrahlung>::value > 0)
        {
            pmacc::meta::ForEach<SpeciesWithBremsstrahlung, CallBremsstrahlung<boost::mpl::_1>> bremsstrahlung;
            bremsstrahlung(cellDescription, step, scaledSpectrumMap, *photonAngle, *failedRequirement);

            if constexpr(particles::bremsstrahlung::supressRequirementWarning == false)
                checkRequirements();
        }
    }

    void Bremsstrahlung::checkRequirements()
    {
        namespace req = particles::bremsstrahlung::requirement;
        namespace el = particles::bremsstrahlung::electron;
        namespace ph = particles::bremsstrahlung::photon;

        failedRequirement->deviceToHost();
        int32_t const flags = failedRequirement->getHostBuffer().getDataBox()(DataSpace<1>{0});
        if(flags == 0)
            return;

        int32_t const newFlags = flags & ~printedRequirements;
        if(newFlags & req::ekinAboveTable)
            std::cerr << "[Bremsstrahlung] warning: electron kinetic energy above MAX_ENERGY_MeV = "
                      << el::MAX_ENERGY_MeV << ", the table values at MAX_ENERGY_MeV are used." << std::endl;
        if(newFlags & req::gammaAboveTable)
            std::cerr << "[Bremsstrahlung] warning: electron gamma above MAX_GAMMA = " << ph::MAX_GAMMA
                      << ", the photon angle table at MAX_GAMMA is used." << std::endl;
        if(newFlags & req::emissionProbability)
            std::cerr << "[Bremsstrahlung] warning: photon emission probability per time step above "
                         "SINGLE_EMISSION_PROB_LIMIT = "
                      << ph::SINGLE_EMISSION_PROB_LIMIT
                      << ", reduce the time step or WEIGHTING_RATIO (emission is under-counted if p > 1)."
                      << std::endl;
        if(newFlags & req::deflectionProbability)
            std::cerr << "[Bremsstrahlung] warning: deflection probability per time step above "
                         "SINGLE_EMISSION_PROB_LIMIT = "
                      << ph::SINGLE_EMISSION_PROB_LIMIT << ", reduce the time step or increase MIN_THETA."
                      << std::endl;
        if(newFlags != 0)
            std::cerr << "[Bremsstrahlung] each warning is printed once per MPI rank." << std::endl;
        printedRequirements |= flags;

        failedRequirement->getHostBuffer().getDataBox()(DataSpace<1>{0}) = 0;
        failedRequirement->hostToDevice();
    }

    void Bremsstrahlung::dumpTables() const
    {
        namespace el = particles::bremsstrahlung::electron;
        namespace ph = particles::bremsstrahlung::photon;
        using particles::bremsstrahlung::PhotonEmissionAngle;
        using particles::bremsstrahlung::ScaledSpectrum;

        if(Environment<simDim>::get().GridController().getGlobalRank() != 0)
            return;

        float_64 const unitArea = sim.unit.length() * sim.unit.length();
        float_64 const unitEnergy = sim.unit.energy();
        float_64 const MeV = sim.si.conv().eV2Joule(1.0e6);

        for(auto const& [targetZ, scaledSpectrum] : scaledSpectrumMap)
        {
            auto hSpectrum = scaledSpectrum->scaledSpectrum->getHostBuffer().getDataBox();
            auto hStopping = scaledSpectrum->stoppingPower->getHostBuffer().getDataBox();

            std::stringstream fileName;
            fileName << "bremsstrahlung_spectrum_Z" << static_cast<int>(targetZ) << ".dat";
            std::ofstream file(fileName.str());
            file << "# PIConGPU Bremsstrahlung lookup tables, Z = " << targetZ << "\n"
                 << "# Ekin_MeV kappa scaledSpectrum_m2 stoppingPowerTable_Jm2\n"
                 << "# scaledSpectrum = kappa dsigma/dkappa; stoppingPowerTable = int_{MIN_KAPPA Ekin}^{kappa Ekin}"
                    " W dsigma/dW dW\n"
                 << "# float_X table values, written from the host copy\n";
            file << std::setprecision(10);
            for(uint32_t i = 0u; i < el::NUM_SAMPLES_EKIN; ++i)
                for(uint32_t j = 0u; j < el::NUM_SAMPLES_KAPPA; ++j)
                {
                    DataSpace<2> const idx{static_cast<int>(i), static_cast<int>(j)};
                    file << ScaledSpectrum::ekinOfIndex(i) * unitEnergy / MeV << " " << ScaledSpectrum::kappaOfIndex(j)
                         << " " << static_cast<float_64>(hSpectrum(idx)) * unitArea << " "
                         << static_cast<float_64>(hStopping(idx)) * unitArea * unitEnergy << "\n";
                }
        }

        auto hTheta = photonAngle->thetaTable->getHostBuffer().getDataBox();
        std::ofstream file("bremsstrahlung_photonAngle.dat");
        file << "# PIConGPU Bremsstrahlung photon emission angle table\n"
             << "# delta gamma theta_rad\n";
        file << std::setprecision(10);
        for(uint32_t j = 0u; j < ph::NUM_SAMPLES_GAMMA; ++j)
            for(uint32_t i = 0u; i < ph::NUM_SAMPLES_DELTA; ++i)
                file << PhotonEmissionAngle::deltaOfIndex(i) << " " << PhotonEmissionAngle::gammaOfIndex(j) << " "
                     << static_cast<float_64>(hTheta(DataSpace<2>{static_cast<int>(i), static_cast<int>(j)})) << "\n";
    }
} // namespace picongpu::simulation::stage
