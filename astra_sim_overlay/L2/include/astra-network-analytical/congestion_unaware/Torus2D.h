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
 * Implements a 2-D Torus topology — a rows x cols grid like Mesh2D but with
 * wrap-around links on each row and each column (each dimension is a Ring).
 *
 * Torus2D(rows=4, cols=8) example (32 total nodes):
 *   node_id = row * cols + col
 *   row_hops = min(|dr|, rows - |dr|)     (ring distance along columns)
 *   col_hops = min(|dc|, cols - |dc|)     (ring distance along a row)
 *   hops(src, dest) = row_hops + col_hops
 *
 * Same degree as Mesh2D (<=4) but the wrap-around halves the worst-case hops
 * per dimension, so diameter is floor(rows/2)+floor(cols/2). It sits between
 * Mesh (cheap, high-hop) and FlattenedButterfly (rich long links, <=2 hops)
 * on the cost/diameter Pareto curve.
 *
 * Delay model is BasicTopology's uniform alpha-beta:
 *   delay = hops * latency + chunk_size / bandwidth
 */
class Torus2D final : public BasicTopology {
  public:
    /**
     * @param num_rows  number of rows (must be > 0)
     * @param num_cols  number of columns (must be > 0)
     * @param bandwidth bandwidth of every link
     * @param latency   latency of every link
     */
    Torus2D(int num_rows, int num_cols, Bandwidth bandwidth, Latency latency) noexcept;

    [[nodiscard]] int get_num_rows() const noexcept;
    [[nodiscard]] int get_num_cols() const noexcept;

  private:
    [[nodiscard]] int compute_hops_count(DeviceId src, DeviceId dest) const noexcept override;

    int num_rows;
    int num_cols;
};

}  // namespace NetworkAnalyticalCongestionUnaware
