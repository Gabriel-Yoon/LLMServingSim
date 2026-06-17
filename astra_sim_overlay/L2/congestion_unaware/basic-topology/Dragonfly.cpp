/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_unaware/Dragonfly.h"
#include <cassert>

using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionUnaware;

// Dragonfly is constructed with explicit (num_per_group, num_groups) by the
// parser/Helper (Helper passes num_per_group = fb_rows, num_groups = npus / fb_rows).
Dragonfly::Dragonfly(const int num_per_group_in,
                     const int num_groups_in,
                     const Bandwidth bandwidth,
                     const Latency latency) noexcept
    : BasicTopology(num_per_group_in * num_groups_in, bandwidth, latency),
      num_per_group(num_per_group_in),
      num_groups(num_groups_in) {
    assert(num_per_group_in > 0);
    assert(num_groups_in > 0);
    assert(num_per_group_in * num_groups_in == npus_count);

    basic_topology_type = TopologyBuildingBlock::Dragonfly;
}

int Dragonfly::get_num_per_group() const noexcept {
    return num_per_group;
}

int Dragonfly::get_num_groups() const noexcept {
    return num_groups;
}

int Dragonfly::compute_hops_count(const DeviceId src, const DeviceId dest) const noexcept {
    assert(0 <= src && src < npus_count);
    assert(0 <= dest && dest < npus_count);
    assert(src != dest);

    const int src_group = src / num_per_group;
    const int dst_group = dest / num_per_group;

    // Same group: one intra-group direct hop.
    if (src_group == dst_group) {
        return 1;
    }
    // Different group: local -> global -> local = Dragonfly diameter of 3.
    return 3;
}
