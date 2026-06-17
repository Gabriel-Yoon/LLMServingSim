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
 * Implements a (balanced) Dragonfly topology — the HPC gold-standard low-diameter
 * network (Kim et al., ISCA 2008). GPUs are partitioned into groups; every GPU in
 * a group is directly connected (intra-group full mesh), and groups are connected
 * by global links such that any group reaches any other in one global hop.
 *
 * Dragonfly(num_per_group=8, num_groups=4) example (32 GPUs):
 *   node_id = group * num_per_group + local
 *   Same group        -> 1 hop  (intra-group direct link)
 *   Different group    -> 3 hops (local -> global -> local), the Dragonfly diameter
 *
 * Like FlattenedButterfly this is a near-constant-diameter topology (3 vs FB's 2),
 * in contrast to Mesh/Torus/Ring whose diameter grows with N. We model the
 * minimal-routing diameter (cross-group = 3) as the per-pair hop count, matching
 * how the all-to-all "direct" collective is charged its worst-case path.
 *
 * Delay model is BasicTopology's uniform alpha-beta:
 *   delay = hops * latency + chunk_size / bandwidth
 */
class Dragonfly final : public BasicTopology {
  public:
    /**
     * @param num_per_group  GPUs (routers) per group (must be > 0)
     * @param num_groups      number of groups (must be > 0)
     * @param bandwidth       bandwidth of every link
     * @param latency         latency of every link
     */
    Dragonfly(int num_per_group, int num_groups, Bandwidth bandwidth, Latency latency) noexcept;

    [[nodiscard]] int get_num_per_group() const noexcept;
    [[nodiscard]] int get_num_groups() const noexcept;

  private:
    [[nodiscard]] int compute_hops_count(DeviceId src, DeviceId dest) const noexcept override;

    int num_per_group;
    int num_groups;
};

}  // namespace NetworkAnalyticalCongestionUnaware
