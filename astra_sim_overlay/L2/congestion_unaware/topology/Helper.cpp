/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_unaware/Helper.h"
#include "congestion_unaware/BasicTopology.h"
#include "congestion_unaware/FlattenedButterfly.h"
#include "congestion_unaware/FullyConnected.h"
#include "congestion_unaware/MultiDimTopology.h"
#include "congestion_unaware/Ring.h"
#include "congestion_unaware/Switch.h"
#include <cstdlib>
#include <iostream>

using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionUnaware;

static std::unique_ptr<BasicTopology> make_basic_topology(
    const TopologyBuildingBlock topology_type,
    const int npus_count,
    const int fb_rows,
    const Bandwidth bandwidth,
    const Latency latency,
    const Bandwidth elec_bandwidth,
    const Latency elec_latency) noexcept {

    switch (topology_type) {
    case TopologyBuildingBlock::Ring:
        return std::make_unique<Ring>(npus_count, bandwidth, latency);
    case TopologyBuildingBlock::Switch:
        return std::make_unique<Switch>(npus_count, bandwidth, latency);
    case TopologyBuildingBlock::FullyConnected:
        return std::make_unique<FullyConnected>(npus_count, bandwidth, latency);
    case TopologyBuildingBlock::FlattenedButterfly: {
        // fb_rows == 0: auto-factorize from npus_count
        // fb_rows > 0: use explicitly provided row count, cols = npus_count / fb_rows
        const int rows = fb_rows;
        const int cols = (rows > 0) ? (npus_count / rows) : 0;
        // ``bandwidth``/``latency`` are the optical (far same-row/col) link;
        // ``elec_bandwidth``/``elec_latency`` are the electrical RDL (adjacent
        // tiles). When the elec_* values equal the optical ones (the parser's
        // default when no elec fields are present), this reproduces the legacy
        // uniform FlattenedButterfly behaviour.
        return std::make_unique<FlattenedButterfly>(rows, cols, elec_bandwidth, elec_latency, bandwidth, latency);
    }
    default:
        std::cerr << "[Error] (network/analytical/congestion_unaware) Not supported basic-topology" << std::endl;
        std::exit(-1);
    }
}

std::shared_ptr<Topology> NetworkAnalyticalCongestionUnaware::construct_topology(
    const NetworkParser& network_parser) noexcept {
    // get network_parser info
    const auto dims_count = network_parser.get_dims_count();
    const auto topologies_per_dim = network_parser.get_topologies_per_dim();
    const auto npus_counts_per_dim = network_parser.get_npus_counts_per_dim();
    const auto bandwidths_per_dim = network_parser.get_bandwidths_per_dim();
    const auto latencies_per_dim = network_parser.get_latencies_per_dim();
    const auto fb_rows_per_dim = network_parser.get_fb_rows_per_dim();
    const auto elec_bandwidths_per_dim = network_parser.get_elec_bandwidths_per_dim();
    const auto elec_latencies_per_dim = network_parser.get_elec_latencies_per_dim();

    // if dims_count is 1, just create basic topology
    if (dims_count == 1) {
        const auto topology_type = topologies_per_dim[0];
        const auto npus_count    = npus_counts_per_dim[0];
        const auto bandwidth     = bandwidths_per_dim[0];
        const auto latency       = latencies_per_dim[0];
        const auto fb_rows       = fb_rows_per_dim[0];
        const auto elec_bw       = elec_bandwidths_per_dim[0];
        const auto elec_lat      = elec_latencies_per_dim[0];

        auto topo = make_basic_topology(topology_type, npus_count, fb_rows, bandwidth, latency, elec_bw, elec_lat);
        // unique_ptr → shared_ptr
        return std::shared_ptr<BasicTopology>(std::move(topo));
    }

    // multi-dim topology
    const auto multi_dim_topology = std::make_shared<MultiDimTopology>();

    for (auto dim = 0; dim < dims_count; dim++) {
        const auto topology_type = topologies_per_dim[dim];
        const auto npus_count    = npus_counts_per_dim[dim];
        const auto bandwidth     = bandwidths_per_dim[dim];
        const auto latency       = latencies_per_dim[dim];
        const auto fb_rows       = fb_rows_per_dim[dim];
        const auto elec_bw       = elec_bandwidths_per_dim[dim];
        const auto elec_lat      = elec_latencies_per_dim[dim];

        auto dim_topology =
            make_basic_topology(topology_type, npus_count, fb_rows, bandwidth, latency, elec_bw, elec_lat);
        multi_dim_topology->append_dimension(std::move(dim_topology));
    }

    return multi_dim_topology;
}
