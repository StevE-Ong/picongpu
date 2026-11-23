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

/** @file
 *
 * Helpers to load a cell-wise scalar field from an openPMD file for particle initialization.
 *
 * Used by densityProfiles::FromOpenPMDImpl and by the manipulators
 * unary::TemperatureFromOpenPMD and unary::ChargeStateFromOpenPMD.
 */

#pragma once

#include "picongpu/defines.hpp"

#include <map>
#include <string>

namespace picongpu::particles::functor::misc
{
    /** Tags to distinguish runtime file names for the different initialization quantities
     *
     * @{
     */
    struct RuntimeDensityFileTag
    {
    };

    struct RuntimeTemperatureFileTag
    {
    };

    struct RuntimeChargeStateFileTag
    {
    };

    /** @} */

    /** Static storage for runtime file names of all species
     *
     * Used to propagate command-line values to the openPMD initialization functors.
     * The same runtime value is used for all invocations of the functors of the same kind (tag) for the same
     * species with no compile-time file name set.
     *
     * The storage is keyed by the species name and not by the species type on purpose:
     * species types may contain types generated with __COUNTER__ (e.g. by PMACC_CONST_VECTOR for ionization
     * energies), so the same species can have a different C++ type in different translation units.
     * A static variable keyed by the type would then not be shared between the command-line parsing and the
     * initialization functors.
     *
     * @tparam T_Tag kind of the initialized quantity, one of Runtime*FileTag
     */
    template<typename T_Tag>
    struct RuntimeFile
    {
        /** Access the statically stored value for the given species
         *
         * The returned reference stays valid for the whole program run.
         *
         * @param speciesName species name, as given by FrameType::getName()
         */
        static std::string& get(std::string const& speciesName)
        {
            static auto values = std::map<std::string, std::string>{};
            return values[speciesName];
        }
    };

    /** Get the file name to load from
     *
     * @tparam T_Tag kind of the initialized quantity, one of Runtime*FileTag
     * @tparam T_SpeciesType species type
     *
     * @param compileTimeFilename file name given in a .param file, empty means using the runtime value
     */
    template<typename T_Tag, typename T_SpeciesType>
    HINLINE std::string getOpenPMDFilename(char const* compileTimeFilename)
    {
        auto const filename = std::string{compileTimeFilename};
        if(!filename.empty())
            return filename;
        return RuntimeFile<T_Tag>::get(T_SpeciesType::FrameType::getName());
    }
} // namespace picongpu::particles::functor::misc

#if (ENABLE_OPENPMD == 1)

#    include "picongpu/fields/FieldTmp.hpp"

#    include <pmacc/dataManagement/DataConnector.hpp>
#    include <pmacc/memory/buffers/GridBuffer.hpp>
#    include <pmacc/static_assert.hpp>

#    include <algorithm>
#    include <array>
#    include <cstdint>
#    include <functional>
#    include <memory>
#    include <numeric>
#    include <stdexcept>
#    include <vector>

#    include <openPMD/openPMD.hpp>

namespace picongpu::particles::functor::misc
{
    /** Cell-wise scalar field loaded from an openPMD file
     *
     * The dataset must be a scalar mesh record of type float_X, type mismatch would cause errors.
     * All openPMD metadata but axisLabels and dataOrder is ignored, only C dataOrder is supported.
     * Each value in the dataset defines the value in the cell with the corresponding total coordinate minus the
     * given offset. On construction, the part matching the current local domain is loaded.
     * Cells not present in the file get the given default value.
     * Dimensionality of the file indexing must match the simulation dimensionality.
     *
     * The values are stored in FieldTmp slot 0.
     * So an instance is only valid until FieldTmp slot 0 is used for something else.
     * This is fine for particle initialization functors which are created and used right away.
     *
     * The object is trivially copyable and can be used on the device.
     */
    class OpenPMDScalarField
    {
    public:
        /** Load the field on the host side
         *
         * This is an MPI collective operation, must be called by all ranks.
         *
         * @param filename openPMD series name
         * @param datasetName name of the mesh record inside the iteration
         * @param iteration iteration inside the file
         * @param offset offset in the file in cells, in (x, y, z) coordinates
         * @param defaultValue value for cells with no corresponding file value
         * @param description human-readable description used for logging, e.g. "density for species e"
         */
        HINLINE OpenPMDScalarField(
            std::string const& filename,
            std::string const& datasetName,
            uint32_t iteration,
            DataSpace<simDim> const& offset,
            float_X defaultValue,
            std::string const& description)
        {
            auto const& subGrid = Environment<simDim>::get().SubGrid();
            totalLocalDomainOffset = subGrid.getGlobalDomain().offset + subGrid.getLocalDomain().offset;
            loadFile(filename, datasetName, iteration, offset, defaultValue, description);
        }

        /** Get the value at the given cell
         *
         * @param totalCellOffset total offset including all slides [in cells]
         */
        HDINLINE float_X operator()(DataSpace<simDim> const& totalCellOffset) const
        {
            auto const idx = totalCellOffset - totalLocalDomainOffset;
            return precisionCast<float_X>(deviceDataBox(idx).x());
        }

    private:
        //! Load the field from the file to a device buffer
        HINLINE void loadFile(
            std::string const& filename,
            std::string const& datasetName,
            uint32_t iteration,
            DataSpace<simDim> const& offset,
            float_X defaultValue,
            std::string const& description)
        {
            auto& dc = Environment<>::get().DataConnector();
            PMACC_CASSERT_MSG(_please_allocate_at_least_one_FieldTmp_in_memory_param, fieldTmpNumSlots > 0);
            auto fieldTmp = dc.get<FieldTmp>(FieldTmp::getUniqueId(0));
            auto& fieldBuffer = fieldTmp->getGridBuffer();
            // Set all values to the default values, the values present in the file will be overwritten
            fieldBuffer.getHostBuffer().setValue(defaultValue);
            auto const guards = fieldBuffer.getGridLayout().guardSizeND();
            deviceDataBox = fieldBuffer.getDeviceBuffer().getDataBox().shift(guards);

            /* Open a series (this does not read the dataset itself).
             * This is MPI collective and so has to be done by all ranks.
             */
            auto& gc = Environment<simDim>::get().GridController();
            log<picLog::PHYSICS>("Loading %1% from file \"%2%\", dataset \"%3%\"") % description % filename
                % datasetName;
            auto series = ::openPMD::Series{filename, ::openPMD::Access::READ_ONLY, gc.getCommunicator().getMPIComm()};
            auto mesh = series.iterations[iteration].open().meshes[datasetName];
            auto const indexConverter = IndexConverter{mesh};
            auto const datasetExtent = indexConverter.openPMDToXyz(mesh.getExtent());

            // Offset of the local domain in file coordinates: global coordinates, no guards, no moving window
            auto const& localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();
            bool readFromFile = true;
            // All indices are in PIConGPU x-y-z coordinates, unless explicitly stated otherwise
            // Where the fieldBuffer data is starting from (no guards), so inside the local domain
            auto localDataBoxStart = pmacc::DataSpace<simDim>::create(0);
            // Start and extend of the file for the local domain
            auto chunkOffset = ::openPMD::Offset(simDim, 0);
            auto chunkExtent = ::openPMD::Extent(simDim, 0);
            for(uint32_t d = 0; d < simDim; ++d)
            {
                localDataBoxStart[d] = std::max(offset[d] - totalLocalDomainOffset[d], 0);
                chunkOffset[d] = std::max(totalLocalDomainOffset[d] - offset[d], 0);
                // Here we take care, as chunkExtent is unsigned
                int32_t extent = std::min(
                    static_cast<int32_t>(localDomain.size[d] - localDataBoxStart[d]),
                    static_cast<int32_t>(datasetExtent[d] - chunkOffset[d]));
                if(extent <= 0)
                    readFromFile = false;
                else
                    chunkExtent[d] = extent;
            }

            using ValueType = FieldTmp::ValueType::type;
            auto data = std::shared_ptr<ValueType>{nullptr};
            if(readFromFile)
            {
                data = mesh.template loadChunk<ValueType>(
                    indexConverter.xyzToOpenPMD(chunkOffset),
                    indexConverter.xyzToOpenPMD(chunkExtent));
            }
            // This is MPI collective and so has to be done by all ranks
            mesh.seriesFlush();

            if(readFromFile)
            {
                auto const* rawData = data.get();
                auto const numElements
                    = std::accumulate(std::begin(chunkExtent), std::end(chunkExtent), 1u, std::multiplies<uint32_t>());
                auto hostDataBox = fieldBuffer.getHostBuffer().getDataBox().shift(guards + localDataBoxStart);
                for(uint32_t linearIdx = 0u; linearIdx < numElements; linearIdx++)
                {
                    auto const idx = indexConverter.linearToXyz(linearIdx, chunkExtent);
                    hostDataBox(idx) = rawData[linearIdx];
                }
            }

            // Copy host data to the device
            fieldBuffer.hostToDevice();
            eventSystem::getTransactionEvent().waitForFinished();
        }

        /** Helper class to convert indices between x-y-z coordinates and a system defined by openPMD API mesh
         *
         * The latter is defined by a combination of axisLabels and dataOrder of the given mesh object.
         * Thus, the transformation is bound to the mesh object given in the constructor, not to openPMD API in
         * general. However, for brevity in this class it is called "openPMD coordinates"
         */
        class IndexConverter
        {
        public:
            /** Create an index converter for the given mesh
             *
             * @param mesh openPMD API mesh
             */
            IndexConverter(::openPMD::Mesh const& mesh)
            {
                if(mesh.dataOrder() != ::openPMD::Mesh::DataOrder::C)
                    throw std::runtime_error("Unsupported dataOrder in openPMD input dataset, only C is supported");
                auto axisLabels = std::vector<std::string>{mesh.axisLabels()};
                // When the attribute is not set, openPMD API currently makes it a vector of single "x"
                if(axisLabels.size() <= 1)
                    axisLabels = std::vector<std::string>{"x", "y", "z"};
                std::array<std::string, 3> supportedAxes = {"x", "y", "z"};
                for(auto d = 0; d < simDim; d++)
                {
                    auto it = std::find(begin(supportedAxes), begin(supportedAxes) + simDim, axisLabels[d]);
                    if(it != std::end(supportedAxes))
                    {
                        openPMDAxisIndex[d] = std::distance(begin(supportedAxes), it);
                        xyzAxisIndex[openPMDAxisIndex[d]] = d;
                    }
                    else
                        throw std::runtime_error(
                            "Unsupported axis label " + axisLabels[d] + " in openPMD input dataset");
                }
            }

            /** Convert a multidimentional index from x-y-z to the openPMD coordinates
             *
             * @tparam T_Vector vector type, compatible to std::vector
             *
             * @param vector input vector
             */
            template<typename T_Vector>
            T_Vector xyzToOpenPMD(T_Vector const& vector) const
            {
                auto result = vector;
                for(auto d = 0; d < simDim; d++)
                    result[openPMDAxisIndex[d]] = vector[d];
                return result;
            }

            /** Convert a multidimentional index from openPMD to the x-y-z coordinates
             *
             * @tparam T_Vector vector type, compatible to std::vector
             *
             * @param vector input vector
             */
            template<typename T_Vector>
            T_Vector openPMDToXyz(T_Vector const& vector) const
            {
                auto result = vector;
                for(int32_t d = 0; d < simDim; d++)
                    result[xyzAxisIndex[d]] = vector[d];
                return result;
            }

            /** Convert a linear index in openPMD chunk to a multidimentional x-y-z index.
             *
             * @param openPMDLinearIndex linear index inside openPMD chunk
             * @param xyzChunkExtent multidimentional chunk extent in xyz
             */
            pmacc::DataSpace<simDim> linearToXyz(uint32_t openPMDLinearIndex, ::openPMD::Extent xyzChunkExtent) const
            {
                // Convert xyz extent to openPMD one
                auto const openPMDChunkExtent = xyzToOpenPMD(xyzChunkExtent);
                // This is index in the openPMD coordinate system, the calculation relies on the C data order
                pmacc::DataSpace<simDim> openPMDIdx;
                auto tmpIndex = openPMDLinearIndex;
                for(int32_t d = simDim - 1; d >= 0; d--)
                {
                    openPMDIdx[d] = tmpIndex % openPMDChunkExtent[d];
                    tmpIndex /= openPMDChunkExtent[d];
                }
                // Now we convert it to the xyz coordinates
                return openPMDToXyz(openPMDIdx);
            }

        private:
            // openPMDAxisIndex[0] is openPMD axis index for x, [1] - for y, [2] - for z
            pmacc::DataSpace<simDim> openPMDAxisIndex;

            // xyzAxisIndex[0] is x axis index in openPMD, [1] - y, [2] - z
            pmacc::DataSpace<simDim> xyzAxisIndex;
        };

        /** Device data box with the field values
         *
         * Value at given totalCellIdx is given by element (totalCellIdx - totalLocalDomainOffset).
         */
        PMACC_ALIGN(deviceDataBox, FieldTmp::DataBoxType);

        //! Total offset of the local domain in cells
        PMACC_ALIGN(totalLocalDomainOffset, DataSpace<simDim>);
    };

} // namespace picongpu::particles::functor::misc

#endif
