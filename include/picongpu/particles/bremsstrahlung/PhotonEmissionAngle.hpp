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

namespace picongpu::particles::bremsstrahlung
{
    /** Device functor mapping `delta` to the photon emission polar angle `theta`,
     * where delta is a uniformly distributed random number in [0, 1).
     *
     * Table axes: delta (linear in [0, MAX_DELTA]), gamma (logarithmic in [MIN_GAMMA, MAX_GAMMA]).
     * Above MAX_GAMMA the value at MAX_GAMMA is returned (the caller reports this case).
     */
    struct PhotonAngleLookup
    {
        LookupTable2D table;
        float_X lnMinGamma;
        float_X lnMaxGamma;

        /** polar emission angle of the photon relative to the electron momentum
         *
         * @param delta uniformly distributed random number in [0, 1)
         * @param gamma relativistic factor of the incident electron
         */
        HDINLINE float_X operator()(float_X const delta, float_X const gamma) const
        {
            float_X const deltaLookupPos = delta * static_cast<float_X>(photon::NUM_SAMPLES_DELTA - 1u);
            float_X const lnGamma = math::log(gamma);
            float_X const gammaLookupPos = (lnGamma - lnMinGamma) / (lnMaxGamma - lnMinGamma)
                                           * static_cast<float_X>(photon::NUM_SAMPLES_GAMMA - 1u);
            return table(deltaLookupPos, gammaLookupPos);
        }
    };

    /** Creates and holds the lookup table for the photon emission angle.
     *
     * The angular distribution is the ultra-relativistic limit of the Lorentz-boosted dipole formula
     * (PIConGPU 0.6.0, see Jackson chap. 15.2):
     *
     *   dP/dtheta = 3 theta gamma^2 (1 + gamma^4 theta^4) / (1 + gamma^2 theta^2)^4
     *
     * Its cumulative distribution from 0 to theta has the closed form (t = 1 + gamma^2 theta^2)
     *
     *   delta(theta) = 1 - 3 / (2 t) + 3 / (2 t^2) - 1 / t^3
     *
     * which replaces the numerical integration and minimisation of PIConGPU 0.6.0.
     * The polar angle is limited to [0, pi]; if delta(pi) < MAX_DELTA all larger deltas map to pi, as in 0.6.0.
     */
    struct PhotonEmissionAngle
    {
        //! cumulative distribution of the polar emission angle
        static float_64 cdf(float_64 const theta, float_64 const gamma)
        {
            float_64 const t = 1.0 + gamma * gamma * theta * theta;
            return 1.0 - 1.5 / t + 1.5 / (t * t) - 1.0 / (t * t * t);
        }

        /** inverse of the cumulative distribution in [0, pi] by bisection
         *
         * @param delta target value of the cumulative distribution
         * @param gamma relativistic factor
         * @return theta with cdf(theta) = delta, or pi if delta > cdf(pi)
         */
        static float_64 theta(float_64 const delta, float_64 const gamma)
        {
            constexpr float_64 pi = pmacc::math::Pi<float_64>::value;
            if(delta <= 0.0)
                return 0.0;
            if(cdf(pi, gamma) <= delta)
                return pi;
            float_64 lo = 0.0;
            float_64 hi = pi;
            for(int i = 0; i < 200 && hi - lo > 1.0e-15 * hi; ++i)
            {
                float_64 const mid = 0.5 * (lo + hi);
                if(cdf(mid, gamma) < delta)
                    lo = mid;
                else
                    hi = mid;
            }
            return 0.5 * (lo + hi);
        }

        //! gamma of the node with index gammaIdx
        static float_64 gammaOfIndex(uint32_t const gammaIdx)
        {
            float_64 const lnMinGamma = std::log(photon::MIN_GAMMA);
            float_64 const lnMaxGamma = std::log(photon::MAX_GAMMA);
            float_64 const lnGamma_norm
                = static_cast<float_64>(gammaIdx) / static_cast<float_64>(photon::NUM_SAMPLES_GAMMA - 1u);
            return std::exp(lnMinGamma + (lnMaxGamma - lnMinGamma) * lnGamma_norm);
        }

        //! delta of the node with index deltaIdx
        static float_64 deltaOfIndex(uint32_t const deltaIdx)
        {
            return photon::MAX_DELTA * static_cast<float_64>(deltaIdx)
                   / static_cast<float_64>(photon::NUM_SAMPLES_DELTA - 1u);
        }

        std::shared_ptr<TableBuffer> thetaTable;

        //! Generate the lookup table on the host and copy it to the device
        void init()
        {
            thetaTable = std::make_shared<TableBuffer>(DataSpace<2>{
                static_cast<int>(photon::NUM_SAMPLES_DELTA),
                static_cast<int>(photon::NUM_SAMPLES_GAMMA)});
            auto hTheta = thetaTable->getHostBuffer().getDataBox();

            for(uint32_t gammaIdx = 0u; gammaIdx < photon::NUM_SAMPLES_GAMMA; gammaIdx++)
            {
                float_64 const gamma = gammaOfIndex(gammaIdx);
                for(uint32_t deltaIdx = 0u; deltaIdx < photon::NUM_SAMPLES_DELTA; deltaIdx++)
                {
                    hTheta(DataSpace<2>{static_cast<int>(deltaIdx), static_cast<int>(gammaIdx)})
                        = static_cast<float_X>(theta(deltaOfIndex(deltaIdx), gamma));
                }
            }
            thetaTable->hostToDevice();
        }

        //! device functor mapping delta to the photon emission polar angle theta
        PhotonAngleLookup getPhotonAngleFunctor() const
        {
            return PhotonAngleLookup{
                LookupTable2D{
                    thetaTable->getDeviceBuffer().getDataBox(),
                    photon::NUM_SAMPLES_DELTA,
                    photon::NUM_SAMPLES_GAMMA},
                static_cast<float_X>(std::log(photon::MIN_GAMMA)),
                static_cast<float_X>(std::log(photon::MAX_GAMMA))};
        }
    };
} // namespace picongpu::particles::bremsstrahlung
