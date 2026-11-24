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
#include "picongpu/param/bremsstrahlung.param"
#include "picongpu/particles/bremsstrahlung/LookupTable.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace picongpu::particles::bremsstrahlung
{
    /** Device functor for a table over (Ekin, kappa)
     *
     * Ekin axis: logarithmic in [MIN_ENERGY, MAX_ENERGY], kappa axis: linear in [0, 1].
     * Below MIN_ENERGY the functor returns zero. Above MAX_ENERGY it returns the value at MAX_ENERGY
     * (the caller reports this case).
     */
    struct SpectrumLookup
    {
        LookupTable2D table;
        float_X lnEMin;
        float_X lnEMax;

        /** table value
         *
         * @param Ekin kinetic energy of the incident electron in PIC units
         * @param kappa energy loss normalized to Ekin
         */
        HDINLINE float_X operator()(float_X const Ekin, float_X const kappa) const
        {
            float_X const lnE = math::log(Ekin);
            float_X const binE
                = (lnE - lnEMin) / (lnEMax - lnEMin) * static_cast<float_X>(electron::NUM_SAMPLES_EKIN - 1u);
            // in the low-energy limit Bremsstrahlung is not taken into account
            if(binE < float_X(0.0))
                return float_X(0.0);
            float_X const binKappa = kappa * static_cast<float_X>(electron::NUM_SAMPLES_KAPPA - 1u);
            return table(binE, binKappa);
        }
    };

    /** Generates and holds the lookup tables for the scaled differential cross section
     * and the stopping power of one target atomic number.
     *
     * scaled differential cross section = kappa * d sigma / d kappa = W * d sigma / d W
     *   (W: energy of the emitted photon, kappa = W / Ekin)
     *
     * stopping power table = integral of W * d sigma / d W from MIN_KAPPA * Ekin to kappa * Ekin
     *   (energy loss per unit length per unit ion density)
     *
     * The lookup tables are generated from the screened Bethe-Heitler cross section, see
     * Salvat, F., et al. "Monte Carlo simulation of bremsstrahlung emission by electrons."
     * Radiation Physics and Chemistry 75.10 (2006): 1201-1219.
     */
    struct ScaledSpectrum
    {
        /** differential cross section: cross section per unit photon energy
         *
         * Screened Bethe-Heitler cross section (PIConGPU 0.6.0 form), all quantities in PIC units.
         *
         * @param Ekin kinetic electron energy
         * @param kappa energy loss normalized to Ekin
         * @param targetZ atomic number of the target material
         * @return d sigma / d W
         */
        static float_64 dcs(float_64 const Ekin, float_64 const kappa, float_64 const targetZ)
        {
            constexpr float_64 pi = pmacc::math::Pi<float_64>::value;
            constexpr float_64 eps0 = sim.pic.getEps0<float_64>();
            constexpr float_64 hbar = sim.pic.getHbar<float_64>();
            constexpr float_64 m_e = sim.pic.getElectronMass<float_64>();
            constexpr float_64 q_e = sim.pic.getElectronCharge<float_64>();
            constexpr float_64 c = sim.pic.getSpeedOfLight<float_64>();
            constexpr float_64 c2 = c * c;

            constexpr float_64 bohrRadius = pi * 4.0 * eps0 * hbar * hbar / (m_e * q_e * q_e);
            constexpr float_64 r_e = q_e * q_e / (pi * 4.0 * eps0 * m_e * c2);
            constexpr float_64 alpha = q_e * q_e / (pi * 4.0 * eps0 * hbar * c);

            float_64 const W = kappa * Ekin;
            float_64 const eps = W / (Ekin + m_e * c2);
            float_64 const R = std::pow(targetZ, -1.0 / 3.0) * bohrRadius;
            float_64 const gamma = Ekin / (m_e * c2) + 1.0;
            float_64 const b = R * m_e * c / hbar / (2.0 * gamma) * eps / (1.0 - eps);

            float_64 const phi_1 = 4.0 * std::log(R * m_e * c / hbar) + 2.0 - 2.0 * std::log(1.0 + b * b)
                                   - 4.0 * b * std::atan(1.0 / b);
            float_64 const phi_2
                = 4.0 * std::log(R * m_e * c / hbar) + 7.0 / 3.0 - 2.0 * std::log(1.0 + b * b)
                  - 6.0 * b * std::atan(1.0 / b)
                  - b * b * (4.0 - 4.0 * b * std::atan(1.0 / b) - 3.0 * std::log(1.0 + 1.0 / (b * b)));

            return r_e * r_e * alpha * targetZ * targetZ / W * (eps * eps * phi_1 + 4.0 / 3.0 * (1.0 - eps) * phi_2);
        }

        /** kinetic energy of the Ekin node with index EkinIdx, PIC units */
        static float_64 ekinOfIndex(uint32_t const EkinIdx)
        {
            float_64 const lnEMin = std::log(static_cast<float_64>(electron::MIN_ENERGY));
            float_64 const lnEMax = std::log(static_cast<float_64>(electron::MAX_ENERGY));
            float_64 const lnE_norm
                = static_cast<float_64>(EkinIdx) / static_cast<float_64>(electron::NUM_SAMPLES_EKIN - 1u);
            return std::exp(lnEMin + (lnEMax - lnEMin) * lnE_norm);
        }

        /** kappa of the node with index kappaIdx (MIN_KAPPA instead of 0) */
        static float_64 kappaOfIndex(uint32_t const kappaIdx)
        {
            float_64 kappa = static_cast<float_64>(kappaIdx) / static_cast<float_64>(electron::NUM_SAMPLES_KAPPA - 1u);
            if(kappa == 0.0)
                kappa = electron::MIN_KAPPA;
            return kappa;
        }

        /** integral of W * dcs(W) dW from MIN_KAPPA * Ekin to kappa * Ekin (composite Simpson rule) */
        static float_64 stoppingPowerIntegral(float_64 const Ekin, float_64 const kappa, float_64 const targetZ)
        {
            float_64 const lower = electron::MIN_KAPPA * Ekin;
            float_64 const upper = kappa * Ekin;
            if(upper <= lower)
                return 0.0;
            // even number of intervals for Simpson's rule
            uint32_t const n
                = electron::NUM_STEPS_STOPPING_POWER_INTEGRAL + (electron::NUM_STEPS_STOPPING_POWER_INTEGRAL % 2u);
            float_64 const h = (upper - lower) / static_cast<float_64>(n);
            auto integrand = [&](float_64 const W) { return W * dcs(Ekin, W / Ekin, targetZ); };
            float_64 sum = integrand(lower) + integrand(upper);
            for(uint32_t i = 1u; i < n; ++i)
                sum += (i % 2u == 1u ? 4.0 : 2.0) * integrand(lower + h * static_cast<float_64>(i));
            return sum * h / 3.0;
        }

        std::shared_ptr<TableBuffer> scaledSpectrum;
        std::shared_ptr<TableBuffer> stoppingPower;

        /** Generate the lookup tables on the host and copy them to the device
         *
         * @param targetZ atomic number of the target material
         */
        void init(float_64 const targetZ)
        {
            DataSpace<2> const size{
                static_cast<int>(electron::NUM_SAMPLES_EKIN),
                static_cast<int>(electron::NUM_SAMPLES_KAPPA)};
            scaledSpectrum = std::make_shared<TableBuffer>(size);
            stoppingPower = std::make_shared<TableBuffer>(size);

            auto hSpectrum = scaledSpectrum->getHostBuffer().getDataBox();
            auto hStopping = stoppingPower->getHostBuffer().getDataBox();

            for(uint32_t EkinIdx = 0u; EkinIdx < electron::NUM_SAMPLES_EKIN; EkinIdx++)
            {
                for(uint32_t kappaIdx = 0u; kappaIdx < electron::NUM_SAMPLES_KAPPA; kappaIdx++)
                {
                    float_64 const kappa = kappaOfIndex(kappaIdx);
                    float_64 const Ekin = ekinOfIndex(EkinIdx);

                    float_64 const valueSpectrum = Ekin * kappa * dcs(Ekin, kappa, targetZ);
                    float_64 const valueStopping = stoppingPowerIntegral(Ekin, kappa, targetZ);

                    if(!std::isfinite(valueSpectrum) || !std::isfinite(valueStopping))
                    {
                        float_64 const Ekin_MeV = Ekin * sim.unit.energy() / sim.si.conv().eV2Joule(1.0e6);
                        std::stringstream errMsg;
                        errMsg << "[Bremsstrahlung] lookup table has a non-finite entry at Ekin = " << Ekin_MeV
                               << " MeV, kappa = " << kappa << ", Z = " << targetZ;
                        throw std::runtime_error(errMsg.str());
                    }

                    DataSpace<2> const idx{static_cast<int>(EkinIdx), static_cast<int>(kappaIdx)};
                    hSpectrum(idx) = static_cast<float_X>(valueSpectrum);
                    hStopping(idx) = static_cast<float_X>(valueStopping);
                }
            }

            scaledSpectrum->hostToDevice();
            stoppingPower->hostToDevice();
        }

        //! device functor of the scaled differential cross section
        SpectrumLookup getScaledSpectrumFunctor() const
        {
            return makeLookup(*scaledSpectrum);
        }

        //! device functor of the stopping power table
        SpectrumLookup getStoppingPowerFunctor() const
        {
            return makeLookup(*stoppingPower);
        }

    private:
        static SpectrumLookup makeLookup(TableBuffer& buffer)
        {
            return SpectrumLookup{
                LookupTable2D{
                    buffer.getDeviceBuffer().getDataBox(),
                    electron::NUM_SAMPLES_EKIN,
                    electron::NUM_SAMPLES_KAPPA},
                static_cast<float_X>(std::log(static_cast<float_64>(electron::MIN_ENERGY))),
                static_cast<float_X>(std::log(static_cast<float_64>(electron::MAX_ENERGY)))};
        }
    };
} // namespace picongpu::particles::bremsstrahlung
