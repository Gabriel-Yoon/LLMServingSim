/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/Type.h"
#include "congestion_unaware/BasicTopology.h"

using namespace NetworkAnalytical;

namespace NetworkAnalyticalCongestionUnaware {

/**
 * Implements a 2-D Mesh topology — a rows x cols grid where each node links
 * only to its grid-adjacent neighbours (up/down/left/right), NO wrap-around.
 *
 * Mesh2D(rows=4, cols=8) example (32 total nodes):
 *   node_id = row * cols + col
 *   hops(src, dest) = |row_src - row_dst| + |col_src - col_dst|   (Manhattan)
 *
 * Compared to FlattenedButterfly (<=2 hops via rich long links) and Torus
 * (wrap-around halves the worst-case per dimension), the Mesh is the cheapest
 * (degree <= 4, only short wires) but has the largest diameter
 * (rows-1)+(cols-1). It is the low-cost / high-hop end of the FB Pareto study.
 *
 * Delay model is BasicTopology's uniform alpha-beta:
 *   delay = hops * latency + chunk_size / bandwidth
 * (every link is identical, so only hop count differs across topologies).
 */
class Mesh2D final : public BasicTopology {
  public:
    /**
     * @param num_rows  number of rows (0 = auto-factorize from npus_count)
     * @param num_cols  number of columns (ignored when num_rows == 0)
     * @param bandwidth bandwidth of every link
     * @param latency   latency of every link
     */
    Mesh2D(int num_rows, int num_cols, Bandwidth bandwidth, Latency latency) noexcept;

    [[nodiscard]] int get_num_rows() const noexcept;
    [[nodiscard]] int get_num_cols() const noexcept;

  private:
    [[nodiscard]] int compute_hops_count(DeviceId src, DeviceId dest) const noexcept override;

    int num_rows;
    int num_cols;
};

}  // namespace NetworkAnalyticalCongestionUnaware
