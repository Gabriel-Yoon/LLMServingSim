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
 * Implements a FlattenedButterfly topology — a 2D grid where every node
 * in the same row and every node in the same column share a direct link.
 *
 * FlattenedButterfly(rows=4, cols=8) example (32 total nodes):
 *   node_id = row * cols + col
 *   Same row  → 1 hop  (direct link)
 *   Same col  → 1 hop  (direct link)
 *   Different row AND different col → 2 hops
 *       (route via  row_src * cols + col_dst  as the intermediate)
 *
 * Compared to a Ring of N=rows*cols nodes:
 *   Ring: up to N/2 hops for arbitrary pairs
 *   FlattenedButterfly: at most 2 hops for any pair
 *
 * When num_rows == 0 the constructor auto-factorises npus_count into the
 * most-square (rows, cols) pair such that rows <= cols.
 *
 * Heterogeneous link model (glass-panel DSE)
 * ------------------------------------------
 * A same-row / same-column hop may use one of two physical link types:
 *   - electrical RDL    for *adjacent* tiles (grid distance == 1)
 *   - optical waveguide for *far* tiles      (grid distance  > 1)
 * Each type carries its own bandwidth and latency. The 6-argument
 * constructor takes both; the legacy 4-argument constructor applies a
 * single (bandwidth, latency) to every hop (electrical == optical), which
 * preserves the original uniform behaviour for configs that don't request
 * the split.
 *
 * Delay model (consistent with BasicTopology's cut-through assumption):
 *   delay = sum(per-hop latency) + chunk_size / min(per-hop bandwidth)
 * i.e. latencies add across hops, while the slowest link on the path sets
 * the serialization bottleneck.
 */
class FlattenedButterfly final : public BasicTopology {
  public:
    /**
     * Legacy constructor — uniform link (electrical == optical).
     *
     * @param num_rows  number of rows (0 = auto-factorize from npus_count)
     * @param num_cols  number of columns (ignored when num_rows == 0)
     * @param bandwidth bandwidth of every link
     * @param latency   latency of every link
     */
    FlattenedButterfly(int num_rows, int num_cols, Bandwidth bandwidth, Latency latency) noexcept;

    /**
     * Heterogeneous constructor — separate electrical (adjacent tiles) and
     * optical (far same-row/col tiles) link parameters.
     *
     * @param num_rows   number of rows (0 = auto-factorize from npus_count)
     * @param num_cols   number of columns (ignored when num_rows == 0)
     * @param elec_bw    electrical RDL bandwidth (GB/s), adjacent tiles
     * @param elec_lat   electrical RDL latency (ns), adjacent tiles
     * @param opt_bw     optical waveguide bandwidth (GB/s), far tiles
     * @param opt_lat    optical waveguide latency (ns), far tiles
     */
    FlattenedButterfly(int num_rows,
                       int num_cols,
                       Bandwidth elec_bw,
                       Latency elec_lat,
                       Bandwidth opt_bw,
                       Latency opt_lat) noexcept;

    /** Number of rows in the 2-D grid. */
    [[nodiscard]] int get_num_rows() const noexcept;

    /** Number of columns in the 2-D grid. */
    [[nodiscard]] int get_num_cols() const noexcept;

    /**
     * Override BasicTopology::send so that each hop picks electrical or
     * optical link parameters by tile adjacency.
     */
    [[nodiscard]] EventTime send(DeviceId src, DeviceId dest, ChunkSize chunk_size) const noexcept override;

  private:
    /**
     * Compute hops between src and dest.
     *   1 hop  if same row or same column
     *   2 hops otherwise
     */
    [[nodiscard]] int compute_hops_count(DeviceId src, DeviceId dest) const noexcept override;

    int num_rows;
    int num_cols;

    /// electrical RDL link (adjacent tiles): bandwidth in B/ns, latency in ns
    Bandwidth elec_bw_Bpns;
    Latency elec_lat;

    /// optical waveguide link (far tiles): bandwidth in B/ns, latency in ns
    Bandwidth opt_bw_Bpns;
    Latency opt_lat;
};

}  // namespace NetworkAnalyticalCongestionUnaware
