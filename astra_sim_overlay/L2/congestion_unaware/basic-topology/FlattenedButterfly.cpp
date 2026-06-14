/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_unaware/FlattenedButterfly.h"
#include "common/NetworkFunction.h"
#include <cassert>
#include <cmath>

using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionUnaware;

static void auto_factorize(int n, int& rows, int& cols) noexcept {
    // Find the largest factor of n that is <= sqrt(n).
    // This gives the most-square (rows, cols) factorization with rows <= cols.
    rows = 1;
    for (int i = 1; static_cast<long long>(i) * i <= n; ++i) {
        if (n % i == 0) {
            rows = i;
        }
    }
    cols = n / rows;
}

// Legacy uniform-link constructor: delegate with electrical == optical so
// every hop behaves exactly as before for configs that don't split links.
FlattenedButterfly::FlattenedButterfly(
    const int num_rows_in,
    const int num_cols_in,
    const Bandwidth bandwidth,
    const Latency latency) noexcept
    : FlattenedButterfly(num_rows_in, num_cols_in, bandwidth, latency, bandwidth, latency) {}

// Heterogeneous-link constructor: separate electrical (adjacent tiles) and
// optical (far same-row/col tiles) bandwidth/latency.
FlattenedButterfly::FlattenedButterfly(
    const int num_rows_in,
    const int num_cols_in,
    const Bandwidth elec_bw,
    const Latency elec_lat_in,
    const Bandwidth opt_bw,
    const Latency opt_lat_in) noexcept
    // Hand the optical params to BasicTopology as the fallback link; the
    // overridden send() below applies per-hop electrical/optical costs.
    : BasicTopology(num_rows_in * num_cols_in, opt_bw, opt_lat_in) {

    if (num_rows_in == 0) {
        // auto-factorize: caller passed 0 to request square-ish layout
        auto_factorize(npus_count, num_rows, num_cols);
    } else {
        assert(num_rows_in > 0);
        assert(num_cols_in > 0);
        assert(num_rows_in * num_cols_in == npus_count);
        num_rows = num_rows_in;
        num_cols = num_cols_in;
    }

    assert(num_rows > 0);
    assert(num_cols > 0);
    assert(num_rows * num_cols == npus_count);
    assert(elec_bw > 0);
    assert(opt_bw > 0);
    assert(elec_lat_in >= 0);
    assert(opt_lat_in >= 0);

    elec_bw_Bpns = bw_GBps_to_Bpns(elec_bw);
    elec_lat = elec_lat_in;
    opt_bw_Bpns = bw_GBps_to_Bpns(opt_bw);
    opt_lat = opt_lat_in;

    basic_topology_type = TopologyBuildingBlock::FlattenedButterfly;
}

int FlattenedButterfly::get_num_rows() const noexcept {
    return num_rows;
}

int FlattenedButterfly::get_num_cols() const noexcept {
    return num_cols;
}

int FlattenedButterfly::compute_hops_count(const DeviceId src, const DeviceId dest) const noexcept {
    assert(0 <= src  && src  < npus_count);
    assert(0 <= dest && dest < npus_count);
    assert(src != dest);

    const int src_row  = src  / num_cols;
    const int src_col  = src  % num_cols;
    const int dst_row  = dest / num_cols;
    const int dst_col  = dest % num_cols;

    // Same row or same column: direct 1-hop link
    if (src_row == dst_row || src_col == dst_col) {
        return 1;
    }

    // Different row AND different column: 2-hop via (src_row * num_cols + dst_col)
    return 2;
}

EventTime FlattenedButterfly::send(const DeviceId src, const DeviceId dest, const ChunkSize chunk_size) const noexcept {
    assert(0 <= src  && src  < npus_count);
    assert(0 <= dest && dest < npus_count);
    assert(src != dest);
    assert(chunk_size > 0);

    const int src_row = src  / num_cols;
    const int src_col = src  % num_cols;
    const int dst_row = dest / num_cols;
    const int dst_col = dest % num_cols;

    // A same-row/col hop uses the electrical RDL when the two tiles are
    // grid-adjacent (distance == 1), else the optical waveguide.
    // ``col_dist`` measures a hop along a row (columns differ);
    // ``row_dist`` measures a hop along a column (rows differ).
    const int col_dist = (src_col > dst_col) ? (src_col - dst_col) : (dst_col - src_col);
    const int row_dist = (src_row > dst_row) ? (src_row - dst_row) : (dst_row - src_row);

    double total_latency = 0.0;     // ns, summed across hops
    double min_bw_Bpns = 0.0;       // serialization bottleneck = slowest hop

    auto add_hop = [&](int distance) noexcept {
        const bool adjacent = (distance == 1);
        const double bw = adjacent ? elec_bw_Bpns : opt_bw_Bpns;
        const double lat = adjacent ? elec_lat : opt_lat;
        total_latency += lat;
        if (min_bw_Bpns == 0.0 || bw < min_bw_Bpns) {
            min_bw_Bpns = bw;
        }
    };

    if (src_row == dst_row) {
        // single hop along a row
        add_hop(col_dist);
    } else if (src_col == dst_col) {
        // single hop along a column
        add_hop(row_dist);
    } else {
        // 2 hops: row hop (to dst column) + column hop (to dst row)
        add_hop(col_dist);
        add_hop(row_dist);
    }

    const double serialization_delay = static_cast<double>(chunk_size) / min_bw_Bpns;
    const double comms_delay = total_latency + serialization_delay;

    return static_cast<EventTime>(comms_delay);
}
