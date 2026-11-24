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

#include "picongpu/algorithms/Gamma.hpp"
#include "picongpu/defines.hpp"
#include "picongpu/fields/FieldTmp.hpp"
#include "picongpu/fields/FieldTmpOperations.hpp"
#include "picongpu/fields/YeeCell.hpp"
#include "picongpu/param/bremsstrahlung.param"
#include "picongpu/particles/bremsstrahlung/PhotonEmissionAngle.hpp"
#include "picongpu/particles/bremsstrahlung/ScaledSpectrum.hpp"
#include "picongpu/particles/particleToGrid/ComputeGridValuePerFrame.def"
#include "picongpu/particles/particleToGrid/derivedAttributes/DerivedAttributes.def"
#include "picongpu/particles/traits/GetAtomicNumbers.hpp"
#include "picongpu/traits/FieldPosition.hpp"
#include "picongpu/traits/GetMargin.hpp"
#include "picongpu/traits/frame/GetMass.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/lockstep/lockstep.hpp>
#include <pmacc/mappings/threads/ThreadCollective.hpp>
#include <pmacc/math/operation.hpp>
#include <pmacc/memory/boxes/CachedBox.hpp>
#include <pmacc/memory/boxes/DataBox.hpp>
#include <pmacc/memory/boxes/SharedBox.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>
#include <pmacc/particles/meta/FindByNameOrType.hpp>
#include <pmacc/particles/operations/Deselect.hpp>
#include <pmacc/random/RNGProvider.hpp>
#include <pmacc/random/distributions/Uniform.hpp>
#include <pmacc/random/methods/methods.hpp>
#include <pmacc/traits/GetFlagType.hpp>
#include <pmacc/traits/Resolve.hpp>

#include <cstdint>

namespace picongpu::particles::bremsstrahlung
{
    /** bits of the requirement flag, set on the device and reported once by the stage */
    namespace requirement
    {
        //! kinetic energy above electron::MAX_ENERGY_MeV, table value at the maximum used
        constexpr int32_t ekinAboveTable = 1;
        //! gamma above photon::MAX_GAMMA, angle table value at the maximum used
        constexpr int32_t gammaAboveTable = 2;
        //! emission probability per step above photon::SINGLE_EMISSION_PROB_LIMIT
        constexpr int32_t emissionProbability = 4;
        //! deflection probability per step above photon::SINGLE_EMISSION_PROB_LIMIT
        constexpr int32_t deflectionProbability = 8;
    } // namespace requirement

    /** Handling of the Bremsstrahlung effect of one electron species on one ion species.
     *
     * Port of the PIConGPU 0.6.0 extension (Heiko Burau) to the creation-kernel interface of the
     * synchrotron extension.
     *
     * Here the screened Bethe-Heitler cross section is used. See e.g.:
     * Salvat, F., et al. "Monte Carlo simulation of bremsstrahlung emission by electrons."
     * Radiation Physics and Chemistry 75.10 (2006): 1201-1219.
     *
     * The numerics separates the energy spectrum into two parts. In the low-energy part
     * photon emission is neglected and a drag force is applied to the electrons. In the high-energy part
     * photons are created in addition to the drag force.
     *
     * Electron deflection is treated as small-angle Rutherford scattering, see e.g. Jackson, chap. 13.5
     *
     * The photon emission angle is taken from the Lorentz-boosted dipole radiation formula,
     * see e.g. Jackson, chap. 15.2
     *
     * Differences to 0.6.0: the ion density includes the guard contributions of the neighbouring devices;
     * the photon emission angle uses the gamma of the electron (0.6.0 used the momentum of the electron
     * divided by the photon weighting); table lookups are clamped to the table; rates use the electron
     * speed instead of c; out-of-range lookups and probabilities above the limit set a requirement flag.
     *
     * @tparam T_IonSpecies type or name of the ion species (needs atomicNumbers<>)
     * @tparam T_ElectronSpecies type or name of the electron species
     * @tparam T_PhotonSpecies type or name of the photon species
     */
    template<typename T_IonSpecies, typename T_ElectronSpecies, typename T_PhotonSpecies>
    struct AlgorithmBremsstrahlung
    {
        using IonSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_IonSpecies>;
        using ElectronSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_ElectronSpecies>;
        using PhotonSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_PhotonSpecies>;

        using FrameType = typename ElectronSpecies::FrameType;

        /** specify field to particle interpolation scheme */
        using Field2ParticleInterpolation = typename pmacc::traits::Resolve<
            typename pmacc::traits::GetFlagType<FrameType, interpolation<>>::type>::type;

        /** margins around the supercell for the interpolation of the field on the cells */
        using LowerMargin = typename picongpu::traits::GetMargin<Field2ParticleInterpolation>::LowerMargin;
        using UpperMargin = typename picongpu::traits::GetMargin<Field2ParticleInterpolation>::UpperMargin;

        /** relevant area of a block */
        using BlockArea = SuperCellDescription<typename MappingDesc::SuperCellSize, LowerMargin, UpperMargin>;

        BlockArea BlockDescription;

    private:
        using SuperCellSize = MappingDesc::SuperCellSize;
        using ValueTypeIonDensity = FieldTmp::ValueType;

        /** random number generator */
        using RNGFactory = pmacc::random::RNGProvider<simDim, random::Generator>;
        using Distribution = pmacc::random::distributions::Uniform<float_X>;
        using RandomGen = typename RNGFactory::GetRandomType<Distribution>::type;
        RandomGen randomGen;

        /** global memory ion density device databox */
        PMACC_ALIGN(ionDensityBox, FieldTmp::DataBoxType);
        /** shared memory ion density device databox */
        PMACC_ALIGN(
            cachedIonDensity,
            pmacc::DataBox<pmacc::SharedBox<ValueTypeIonDensity, typename BlockArea::FullSuperCellSize, 0>>);

        SpectrumLookup scaledSpectrumFunctor;
        SpectrumLookup stoppingPowerFunctor;
        PhotonAngleLookup photonAngleFunctor;
        pmacc::GridBuffer<int32_t, 1>::DataBoxType failedRequirementBox;

        //! momentum of the photon to be created (macro-particle momentum, before the emission angle)
        float3_X photonMom;

        template<typename T_Worker>
        DINLINE void setRequirementFlag(T_Worker const& worker, int32_t const bit)
        {
            if constexpr(supressRequirementWarning == false)
                alpaka::atomicOr(worker.getAcc(), &(failedRequirementBox(DataSpace<1>{0})), bit);
        }

    public:
        /** host constructor: computes the ion density of the current step into the FieldTmp slot
         *
         * @param currentStep current time step
         * @param scaledSpectrum device functor of the scaled differential cross section of this ion species
         * @param stoppingPower device functor of the stopping power table of this ion species
         * @param photonAngle device functor of the photon emission angle
         * @param failedRequirement device databox of the requirement flag
         */
        HINLINE AlgorithmBremsstrahlung(
            uint32_t const currentStep,
            SpectrumLookup const& scaledSpectrum,
            SpectrumLookup const& stoppingPower,
            PhotonAngleLookup const& photonAngle,
            pmacc::GridBuffer<int32_t, 1>::DataBoxType failedRequirement)
            : randomGen(RNGFactory::createRandom<Distribution>())
            , scaledSpectrumFunctor(scaledSpectrum)
            , stoppingPowerFunctor(stoppingPower)
            , photonAngleFunctor(photonAngle)
            , failedRequirementBox(failedRequirement)
            , photonMom(float3_X::create(0.0_X))
        {
            /* The compiler is allowed to evaluate an expression that does not depend on a template parameter
             * even if the class is never instantiated, so the condition carries a template dependency.
             */
            PMACC_CASSERT_MSG(
                _please_allocate_enough_FieldTmp_slots_in_memory_param_for_bremsstrahlung_fieldTmpSlot,
                (fieldTmpSlot < fieldTmpNumSlots) && (sizeof(T_IonSpecies) != 0));

            DataConnector& dc = Environment<>::get().DataConnector();

            auto fieldIonDensity = dc.get<FieldTmp>(FieldTmp::getUniqueId(fieldTmpSlot));
            /* reset values to zero */
            fieldIonDensity->getGridBuffer().getDeviceBuffer().setValue(FieldTmp::ValueType(0.0));

            /* load species without copying the particle data to the host */
            auto ionSpecies = dc.get<IonSpecies>(IonSpecies::FrameType::getName());

            /* compute ion density */
            using DensitySolver = typename particleToGrid::
                CreateFieldTmpOperation_t<IonSpecies, particleToGrid::derivedAttributes::Density>::Solver;
            computeFieldTmpValue<CORE + BORDER, DensitySolver>(*fieldIonDensity, *ionSpecies, currentStep);

            /* contributions from neighboring devices to our border area */
            EventTask densityEvent = fieldIonDensity->asyncCommunication(eventSystem::getTransactionEvent());
            densityEvent += fieldIonDensity->asyncCommunicationGather(densityEvent);
            eventSystem::setTransactionEvent(densityEvent);

            /* initialize device-side databox */
            ionDensityBox = fieldIonDensity->getDeviceDataBox();
        }

        /** cache fields used by this functor
         *
         * @warning this is a collective method and calls synchronize
         *
         * @tparam T_Worker lockstep worker type
         *
         * @param worker lockstep worker
         * @param blockCell relative offset (in cells) to the local domain plus the guarding cells
         */
        template<typename T_Worker>
        DINLINE void collectiveInit(T_Worker const& worker, DataSpace<simDim> const& blockCell)
        {
            cachedIonDensity = CachedBox::create<0, ValueTypeIonDensity>(worker, BlockArea());

            pmacc::math::operation::Assign assign;
            auto fieldIonDensityBlock = ionDensityBox.shift(blockCell);
            auto collective = makeThreadCollective<BlockArea>();
            collective(worker, assign, cachedIonDensity, fieldIonDensityBlock);

            /* wait for shared memory to be initialized */
            worker.sync();
        }

        /** Initialization function on device: random number generator
         *
         * @param localSuperCellOffset offset (in superCells, without any guards) relative
         *                             to the origin of the local domain
         * @param rngIdx linear index rng number index within the supercell, valid range[0;numFrameSlots)
         */
        template<typename T_Worker>
        DINLINE void init(
            [[maybe_unused]] T_Worker const& worker,
            DataSpace<simDim> const& localSuperCellOffset,
            uint32_t const rngIdx)
        {
            auto rngOffset = DataSpace<simDim>::create(0);
            rngOffset.x() = rngIdx;
            auto numRNGsPerSuperCell = DataSpace<simDim>::create(1);
            numRNGsPerSuperCell.x() = FrameType::frameSize;
            this->randomGen.init(localSuperCellOffset * numRNGsPerSuperCell + rngOffset);
        }

        /** Rotates a vector to a given polar angle and a random azimuthal angle.
         *
         * @param vec vector to be rotated
         * @param theta polar angle
         * @return rotated vector
         */
        template<typename T_Worker>
        DINLINE float3_X scatterByTheta(T_Worker const& worker, float3_X const vec, float_X const theta)
        {
            float_X sinTheta, cosTheta;
            pmacc::math::sincos(theta, sinTheta, cosTheta);

            float_X const phi
                = -pmacc::math::Pi<float_X>::value + pmacc::math::Pi<float_X>::doubleValue * this->randomGen(worker);
            float_X sinPhi, cosPhi;
            pmacc::math::sincos(phi, sinPhi, cosPhi);

            float3_X const vecUp(0.0, 0.0, 1.0);
            float3_X vecOrtho1 = pmacc::math::cross(vecUp, vec);
            float_X const vecOrtho1Abs = pmacc::math::l2norm(vecOrtho1);

            float3_X vecOrtho1_norm;
            if(vecOrtho1Abs == float_X(0.0))
                vecOrtho1_norm = float3_X(1.0, 0.0, 0.0);
            else
                vecOrtho1_norm = vecOrtho1 / vecOrtho1Abs;
            float3_X const vecOrtho2 = pmacc::math::cross(vecOrtho1_norm, vec);
            vecOrtho1 = vecOrtho1_norm * pmacc::math::l2norm(vec);

            return vec * cosTheta + vecOrtho1 * (sinTheta * cosPhi) + vecOrtho2 * (sinTheta * sinPhi);
        }

        /** Deflection and drag of the electron; returns the number of macro-photons to be created (0 or 1).
         *
         * Called for each particle of the source species.
         *
         * @param electronFrame frame of the electron species
         * @param localIdx index of the electron within the frame
         * @return number of macro-photons to be created from this electron
         */
        template<typename T_Worker>
        DINLINE uint32_t numNewParticles(T_Worker const& worker, FrameType& electronFrame, int localIdx)
        {
            auto particle = electronFrame[localIdx];

            /* particle position, used for field-to-particle interpolation */
            floatD_X const pos = particle[position_];
            int const particleCellIdx = particle[localCellIdx_];
            /* multi-dim coordinate of the local cell inside the super cell */
            DataSpace<SuperCellSize::dim> const localCell
                = pmacc::math::mapToND(SuperCellSize::toRT(), particleCellIdx);
            picongpu::traits::FieldPosition<fields::YeeCell, FieldTmp> const fieldTmpPos;
            ValueTypeIonDensity const ionDensity_norm
                = Field2ParticleInterpolation()(cachedIonDensity.shift(localCell), pos, fieldTmpPos());

            /* number density in PIC units (FieldTmp density is per typicalNumParticlesPerMacroParticle) */
            float_X const ionDensity
                = ionDensity_norm[0] * static_cast<float_X>(sim.unit.typicalNumParticlesPerMacroParticle());
            if(ionDensity <= float_X(0.0))
                return 0u;

            float_X const weighting = particle[weighting_];
            constexpr float_X c = sim.pic.getSpeedOfLight();
            constexpr float_X dt = sim.pic.getDt();

            float3_X mom = particle[momentum_] / weighting;
            float_X const momAbs = pmacc::math::l2norm(mom);
            if(momAbs == float_X(0.0))
                return 0u;
            float3_X mom_norm = mom / momAbs;

            float_X const mass = picongpu::traits::frame::getMass<FrameType>();
            float_X const gamma = Gamma()(mom, mass);
            float_X const Ekin = (gamma - float_X(1.0)) * mass * c * c;
            if(Ekin < electron::MIN_ENERGY)
                return 0u;
            if(Ekin > electron::MAX_ENERGY)
                setRequirementFlag(worker, requirement::ekinAboveTable);

            /* path of the electron in one time step */
            float_X const speed = momAbs / (gamma * mass);
            float_X const pathLength = speed * dt;

            /* electron deflection due to Rutherford scattering without modifying the electron
               energy based on radiation emission */
            if constexpr(electron::Deflection)
            {
                constexpr float_X pi = pmacc::math::Pi<float_X>::value;
                constexpr float_X zMin = float_X(1.0) / (pi * pi);
                constexpr float_X zMax = static_cast<float_X>(1.0 / (electron::MIN_THETA * electron::MIN_THETA));
                float_X const z = zMin + this->randomGen(worker) * (zMax - zMin);
                float_X const theta = math::rsqrt(z);
                constexpr float_X targetZ = picongpu::traits::GetAtomicNumbers<IonSpecies>::type::numberOfProtons;
                constexpr float_X e2_4piEps0 = sim.pic.getElectronCharge() * sim.pic.getElectronCharge()
                                               / (float_X(4.0) * pi * sim.pic.getEps0());
                float_X const rutherfordCoeff = float_X(2.0) * e2_4piEps0 * targetZ / Ekin;
                float_X const scaledDeflectionDCS = pi * (zMax - zMin) * rutherfordCoeff * rutherfordCoeff;
                float_X const deflectionProb = ionDensity * pathLength * scaledDeflectionDCS;

                if(deflectionProb > static_cast<float_X>(photon::SINGLE_EMISSION_PROB_LIMIT))
                    setRequirementFlag(worker, requirement::deflectionProbability);

                if(this->randomGen(worker) < deflectionProb)
                {
                    mom = this->scatterByTheta(worker, mom, theta);
                    mom_norm = mom / momAbs;
                }
            }

            /* non-radiative Bremsstrahlung: drag force of the soft photons */
            float_X const kappaCutoff = math::min(photon::SOFT_PHOTONS_CUTOFF / Ekin, float_X(1.0));
            float_X newMomAbs = momAbs;
            if constexpr(electron::EnergyLoss)
            {
                float_X const energyLoss = ionDensity * pathLength * this->stoppingPowerFunctor(Ekin, kappaCutoff);
                float_X const newEkin = math::max(Ekin - energyLoss, float_X(0.0));
                float_X const newEkin_norm = newEkin / (mass * c * c);
                /* This is based on: (p / mc)^2 = (E_kin / mc^2)^2 + 2 * (E_kin / mc^2) */
                newMomAbs = mass * c * math::sqrt(newEkin_norm * newEkin_norm + float_X(2.0) * newEkin_norm);
            }
            particle[momentum_] = mom_norm * (newMomAbs * weighting);

            /* photon emission */
            if(kappaCutoff >= float_X(1.0))
                return 0u;
            float_X const delta = this->randomGen(worker);
            float_X const kappa = math::pow(kappaCutoff, delta);
            float_X const scalingFactor = -math::log(kappaCutoff);
            float_X const emissionProb = static_cast<float_X>(photon::WEIGHTING_RATIO) * scalingFactor * ionDensity
                                         * pathLength * this->scaledSpectrumFunctor(Ekin, kappa);

            if(emissionProb > static_cast<float_X>(photon::SINGLE_EMISSION_PROB_LIMIT))
                setRequirementFlag(worker, requirement::emissionProbability);

            if(this->randomGen(worker) < emissionProb)
            {
                float_X const photonEnergy = kappa * Ekin;
                this->photonMom
                    = mom_norm * (weighting / static_cast<float_X>(photon::WEIGHTING_RATIO) * photonEnergy / c);
                return 1u;
            }

            return 0u;
        }

        /** Creation of one macro-photon
         *
         * @param worker lockstep worker
         * @param idGen particle id generator
         * @param parentElectron electron that emits the photon
         * @param childPhoton photon that is created
         */
        template<typename T_Electron, typename T_Photon, typename T_Worker>
        DINLINE void operator()(
            T_Worker const& worker,
            IdGenerator& idGen,
            T_Electron& parentElectron,
            T_Photon& childPhoton)
        {
            namespace partOp = pmacc::particles::operations;

            childPhoton[multiMask_] = 1u;

            /* clone the electron attributes except multiMask, momentum and weighting */
            auto targetPhotonClone = partOp::deselect<pmacc::mp_list<multiMask, momentum, weighting>>(childPhoton);
            targetPhotonClone.derive(worker, idGen, parentElectron);

            float_X const electronWeighting = parentElectron[weighting_];
            float3_X const elMom = parentElectron[momentum_];

            /* photon emission angle from the gamma of the emitting electron */
            float_X const mass = picongpu::traits::frame::getMass<FrameType>();
            float_X const gamma = Gamma()(elMom / electronWeighting, mass);
            if(gamma > static_cast<float_X>(photon::MAX_GAMMA))
                setRequirementFlag(worker, requirement::gammaAboveTable);

            float_X const theta = this->photonAngleFunctor(this->randomGen(worker), gamma);

            childPhoton[momentum_] = this->scatterByTheta(worker, this->photonMom, theta);
            childPhoton[weighting_] = electronWeighting / static_cast<float_X>(photon::WEIGHTING_RATIO);

            /* recoil, ultra-relativistic limit in terms of energy */
            if constexpr(electron::ElectronRecoil)
                parentElectron[momentum_] = elMom - this->photonMom;
        }
    };
} // namespace picongpu::particles::bremsstrahlung
