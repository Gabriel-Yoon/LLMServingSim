/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_unaware/Mesh2D.h"
#include <cassert>

using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionUnaware;

// Mesh2D is always constructed with explicit (rows, cols) by the parser/Helper
// (Helper passes rows = fb_rows, cols = npus_count / fb_rows), so npus_count is
// known up front and no auto-factorization path is needed.
Mesh2D::Mesh2D(const int num_rows_in,
               const int num_cols_in,
               const Bandwidth bandwidth,
               const Latency latency) noexcept
    : BasicTopology(num_rows_in * num_cols_in, bandwidth, latency),
      num_rows(num_rows_in),
      num_cols(num_cols_in) {
    assert(num_rows_in > 0);
    assert(num_cols_in > 0);
    assert(num_rows_in * num_cols_in == npus_count);

    basic_topology_type = TopologyBuildingBlock::Mesh2D;
}

int Mesh2D::get_num_rows() const noexcept {
    return num_rows;
}

int Mesh2D::get_num_cols() const noexcept {
    return num_cols;
}

int Mesh2D::compute_hops_count(const DeviceId src, const DeviceId dest) const noexcept {
    assert(0 <= src && src < npus_count);
    assert(0 <= dest && dest < npus_count);
    assert(src != dest);

    const int src_row = src / num_cols;
    const int src_col = src % num_cols;
    const int dst_row = dest / num_cols;
    const int dst_col = dest % num_cols;

    const int row_dist = (src_row > dst_row) ? (src_row - dst_row) : (dst_row - src_row);
    const int col_dist = (src_col > dst_col) ? (src_col - dst_col) : (dst_col - src_col);

    // 2-D Mesh: Manhattan distance, no wrap-around
    return row_dist + col_dist;
}
