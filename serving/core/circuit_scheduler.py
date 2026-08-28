"""Optical circuit plane model and circuit schedulers.

The hybrid fabric's optical plane (dim 1 of the hybrid_fabric topology)
is modeled at the copper-domain granularity: each domain has one optical
transceiver pair (1 in-port, 1 out-port), and a circuit connects one
domain's out-port to another domain's in-port. Retargeting a port takes
``reconfig_ns``. Because the ASTRA-Sim analytical backend is
congestion-unaware, the circuit plane's time multiplexing implemented
here IS the contention model for bulk optical transfers: a KV stream may
only start once its (src, dst) circuit is active, and the resulting
stall is injected into the prefill trace (see trace_generator kv_send).

Policies:
  IDEAL     circuit always ready (stall 0) — upper bound.
  ROTOR     traffic-oblivious rotation over the N-1 domain offsets
            (RotorNet-style); a transfer waits for its pair's slot.
  REACTIVE  on-demand circuit setup when the data is ready
            (QPS-like reactive baseline: pays reconfiguration on the
            critical path unless the port already points at the same
            destination).
  PQPS      predictive reservation: demand is announced at request
            admission with the (deterministically known) size and an
            estimated ready time; ports are booked on an interval
            calendar so reconfiguration completes ``resv_guard_ns``
            before the estimated ready time.

This is the "PQPS-lite" system integration: with few domains the
propose/accept sampling of SW-QPS degenerates to the greedy calendar
booking implemented here. The full parallel propose-accept matching
(for large domain counts) is planned for the standalone evaluation
(docs/sbqps-notes.md).

Units: time ns, sizes bytes, bandwidth bytes/ns (numerically equal to
GB/s when GB = 1e9).
"""

try:
    from .logger import get_logger
    logger = get_logger("CircuitScheduler")
except ImportError:  # standalone use (unit tests) without sim deps
    import logging
    logger = logging.getLogger("CircuitScheduler")

IDEAL = "IDEAL"
ROTOR = "ROTOR"
REACTIVE = "REACTIVE"
PQPS = "PQPS"
CORD = "CORD"   # Coalesced Reservation with Deadlines (ours, non-QPS lineage)
STEER = "STEER" # CORD + hybrid plane steering: bulk transfers may take the
                # electrical (packet) plane when its predicted completion
                # beats the optical option (announced-size breakeven)
STEER_NA = "STEER_NA"  # ablation: STEER without announcements — ignores
                       # admission-time demand entirely (no reservations,
                       # no pair bookings); steers per transfer from
                       # observed state only, optical path is REACTIVE.
                       # Isolates the value of prediction inside STEER.
ELECTRICAL = "ELECTRICAL"  # all-electrical status quo: models a deployed
                           # packet-switched fabric (InfiniBand/NVLink/RoCE)
                           # with NO optical plane and NO reconfiguration
                           # delay -- every KV transfer rides the electrical
                           # plane at C_e with per-source serialization.
                           # The baseline optical must beat to justify itself.
# Slotted/epoch-based baselines (QPS-Fit's own benchmark set; see
# docs/evaluation-baselines.md and refs/papers/QPS-Fit.pdf):
NEGOTIATOR = "NEGOTIATOR"   # NegotiaToR (SIGCOMM'24): per-slot binary
                            # request/RR-grant, non-iterative
BFF = "BFF"                 # Best-First-Fit (CLOUD'18): centralized epoch
                            # batch, largest-first best-fit hole packing
QPSFIT = "QPSFIT"           # QPS-Fit (CLOUD'25): epoch batch, QPS-sampled
                            # request + First-Fit / Largest-Fit grant

POLICIES = (IDEAL, ROTOR, REACTIVE, PQPS, CORD, STEER, STEER_NA, ELECTRICAL)
SLOTTED_POLICIES = (NEGOTIATOR, BFF, QPSFIT)
ALL_POLICIES = POLICIES + SLOTTED_POLICIES


class PrefillEstimator:
    """Running-average prefill throughput estimator per instance.

    Fed by the main loop on every completed prefill batch; used at
    admission to predict when a request's KV stream will start.
    """

    def __init__(self, init_tokens_per_ns=20e3 / 1e9):
        # default 20k tokens/s until first observation
        self._init_rate = init_tokens_per_ns
        self._rate = {}            # instance_id -> tokens/ns (EWMA)
        self._backlog_tokens = {}  # instance_id -> queued prefill tokens

    def observe_batch(self, instance_id, tokens, duration_ns):
        if duration_ns <= 0 or tokens <= 0:
            return
        rate = tokens / duration_ns
        prev = self._rate.get(instance_id)
        self._rate[instance_id] = rate if prev is None else 0.7 * prev + 0.3 * rate

    def add_backlog(self, instance_id, tokens):
        self._backlog_tokens[instance_id] = self._backlog_tokens.get(instance_id, 0) + tokens

    def remove_backlog(self, instance_id, tokens):
        self._backlog_tokens[instance_id] = max(
            0, self._backlog_tokens.get(instance_id, 0) - tokens)

    def rate(self, instance_id):
        return self._rate.get(instance_id, self._init_rate)

    def estimate_ready(self, instance_id, prompt_tokens, now_ns):
        """Predicted start of this request's KV streaming (its prefill
        begins after the instance's current backlog drains)."""
        backlog = self._backlog_tokens.get(instance_id, 0)
        rate = self.rate(instance_id)
        return int(now_ns + backlog / rate)


class _Calendar:
    """Sorted list of booked half-open intervals [start, end) per port."""

    def __init__(self):
        self.iv = []  # list of [start, end, tag], sorted by start

    def remove(self, tag):
        self.iv = [x for x in self.iv if x[2] != tag]

    def busy_at(self, t):
        for s, e, _ in self.iv:
            if s <= t < e:
                return True
            if s > t:
                break
        return False


def _earliest_free(cals, t0, dur, ignore_resv_after=None):
    """Earliest t >= t0 such that [t, t+dur) is free in ALL calendars.

    ``ignore_resv_after``: when set, reservation intervals (non-"xfer"
    tags) whose setup has not started by that time are treated as soft
    and skipped — an actually-ready transfer outranks estimated future
    reservations (which get displaced and re-booked by the caller).
    """
    events = sorted(iv for cal in cals for iv in cal.iv)
    t = t0
    for s, e, tag in events:
        if ignore_resv_after is not None and not _is_xfer(tag) \
                and s > ignore_resv_after:
            continue
        if e <= t:
            continue
        if s >= t + dur:
            break
        t = max(t, e)
    return t


def _is_xfer(tag):
    return isinstance(tag, tuple) and len(tag) == 2 and tag[0] == "xfer"


def _occupy(cals, start, end, tag):
    for cal in cals:
        cal.iv.append([start, end, tag])
        cal.iv.sort()


class _Reservation:
    __slots__ = ("req_id", "src", "dst", "nbytes", "est_ready", "deadline", "active_from")

    def __init__(self, req_id, src, dst, nbytes, est_ready, deadline):
        self.req_id = req_id
        self.src = src
        self.dst = dst
        self.nbytes = nbytes
        self.est_ready = est_ready
        self.deadline = deadline
        self.active_from = None


class CircuitManager:
    """Domain-level optical circuit bookkeeping on the simulation
    timeline. Queried synchronously by the serving main loop."""

    def __init__(self, num_domains, bw_bytes_per_ns, reconfig_ns,
                 policy=REACTIVE, rotor_slot_ns=None, resv_guard_ns=None,
                 cord_slack_ns=None, electrical_bw=None, afd_load=0.0,
                 electrical_spine_bw=None, adaptive_guard=False):
        if policy not in POLICIES:
            raise ValueError(f"Unknown circuit policy '{policy}' (choose from {POLICIES})")
        self.n = num_domains
        self.bw = float(bw_bytes_per_ns)
        self.reconfig_ns = int(reconfig_ns)
        self.policy = policy
        # rotor slot: must fit reconfiguration plus useful serving time
        self.rotor_slot_ns = int(rotor_slot_ns) if rotor_slot_ns else max(
            10 * self.reconfig_ns, 100_000)
        # PQPS books the circuit to be active resv_guard_ns before the
        # estimated ready time, absorbing estimator error.
        self.resv_guard_ns = int(resv_guard_ns) if resv_guard_ns is not None \
            else self.reconfig_ns
        # learning-augmented guard: an online newsvendor-quantile estimate of the
        # estimator error that replaces the fixed resv_guard_ns when enabled.
        self.adaptive_guard = bool(adaptive_guard)
        if self.adaptive_guard:
            from .adaptive_guard import AdaptiveGuard
            # g_max must be generous: with bias correction the learned lead can
            # exceed many reconfig delays (a badly biased estimator needs it).
            self._ag = AdaptiveGuard(g_safe=self.resv_guard_ns,
                                     g_max=50 * self.resv_guard_ns)
        else:
            self._ag = None
        # CORD: per-transfer latency budget available for coalescing holds
        self.cord_slack_ns = int(cord_slack_ns) if cord_slack_ns is not None \
            else 2 * self.reconfig_ns
        self._pair_pending = {}  # (src, dst) -> list of [req_id, est_ready, nbytes]
        # STEER: electrical (packet) plane per-flow bandwidth and a simple
        # per-source-domain egress serialization. The simulated wire still
        # runs at optical rate, so the electrical route's slower service is
        # folded into the returned stall (completion-time equivalence).
        self.electrical_bw = float(electrical_bw) if electrical_bw else 25.0
        # AFD background contention: attention-FFN disaggregation runs
        # per-layer A2F/F2A traffic on the same electrical/scale-up plane
        # STEER steers onto. Modeled as a link-sharing utilization rho in
        # [0,1): AFD occupies rho of the plane, so STEER's electrical
        # transfer sees effective bandwidth C_e*(1-rho) (first-order
        # link-sharing; burst-level queuing not modeled).
        self.afd_load = min(max(float(afd_load), 0.0), 0.95)
        self.electrical_bw_eff = self.electrical_bw * (1.0 - self.afd_load)
        self.elec_free = [0] * num_domains
        # Hyperscale spine bisection: a real electrical fat-tree routes all
        # cross-domain packet traffic through a shared (oversubscribed) spine
        # of aggregate bandwidth C_spine GB/s. Modeled as one shared FIFO
        # resource every packet transfer serializes through, on top of the
        # per-source NIC egress. Disabled (None) = full-bisection / NIC-bound.
        # The OCS optical plane bypasses this (dedicated lightpaths), which is
        # exactly why optical scales -- so it applies ONLY to the packet plane
        # (ELECTRICAL policy and STEER's steered-electrical transfers).
        self.electrical_spine_bw = (float(electrical_spine_bw)
                                    if electrical_spine_bw else None)
        self.elec_spine_free = 0

        self.out_cal = [_Calendar() for _ in range(num_domains)]
        self.in_cal = [_Calendar() for _ in range(num_domains)]
        self.out_target = [None] * num_domains  # current circuit pointing
        self.in_source = [None] * num_domains
        self._reservations = {}  # req_id -> _Reservation
        # admission-time start estimate per req_id, kept for the whole run so
        # the transfer log can report the estimator error eta = ready - est.
        self._announced_est = {}
        self.stats = []

    def _guard(self):
        """Reservation guard band: learned (adaptive) or fixed resv_guard_ns."""
        return int(self._ag.guard()) if self.adaptive_guard else self.resv_guard_ns

    # ---------------- predictive API ----------------

    def announce(self, req_id, src, dst, nbytes, est_ready_ns, deadline_ns=None):
        """Admission-time demand announcement (PQPS only acts on it)."""
        if src == dst or self.policy == STEER_NA:
            return
        # record the admission-time estimate for every announced transfer so
        # the estimator error can be characterized (learning-augmented guard).
        self._announced_est[req_id] = int(est_ready_ns)
        r = _Reservation(req_id, src, dst, int(nbytes), int(est_ready_ns), deadline_ns)
        self._reservations[req_id] = r
        if self.policy == PQPS:
            self._book(r)
        elif self.policy in (CORD, STEER):
            self._pair_pending.setdefault((src, dst), []).append(
                [req_id, int(est_ready_ns), int(nbytes)])
            self._cord_book_pair(src, dst)

    def _book(self, r):
        """Book both ports on their calendars so the circuit becomes
        active ``resv_guard_ns`` before the estimated ready time."""
        serve_ns = int(r.nbytes / self.bw)
        dur = self.reconfig_ns + serve_ns
        desired_setup = r.est_ready - self._guard() - self.reconfig_ns
        cals = (self.out_cal[r.src], self.in_cal[r.dst])
        t = _earliest_free(cals, desired_setup, dur)
        _occupy(cals, t, t + dur, r.req_id)
        r.active_from = t + self.reconfig_ns

    def reannounce(self, req_id, est_ready_ns):
        """Refine a reservation's ready estimate (re-books the ports)."""
        r = self._reservations.get(req_id)
        if r is None or self.policy != PQPS:
            return
        r.est_ready = int(est_ready_ns)
        self.out_cal[r.src].remove(req_id)
        self.in_cal[r.dst].remove(req_id)
        self._book(r)

    def cancel(self, req_id):
        r = self._reservations.pop(req_id, None)
        if r is not None:
            self.out_cal[r.src].remove(req_id)
            self.in_cal[r.dst].remove(req_id)

    # ---------------- transfer request ----------------

    def request_transfer(self, src, dst, nbytes, now_ns, req_id=None):
        """The data is ready at ``now_ns``; returns stall_ns before the
        KV stream may start."""
        nbytes = int(nbytes)
        if src == dst or nbytes <= 0 or self.policy == IDEAL:
            self._log(req_id, src, dst, nbytes, now_ns, now_ns, 0)
            return 0

        serve_ns = int(nbytes / self.bw)

        if self.policy == ROTOR:
            start = self._rotor_start(src, dst, now_ns)
            self._log(req_id, src, dst, nbytes, now_ns, start, self.reconfig_ns)
            return int(start - now_ns)

        r = self._reservations.pop(req_id, None) if req_id is not None else None
        cals = (self.out_cal[src], self.in_cal[dst])
        if self.policy == PQPS and r is not None and r.active_from is not None:
            # replace the reservation with the actual transfer; future
            # (not-yet-started) reservations of other requests are soft
            # and get displaced.
            self.out_cal[src].remove(req_id)
            self.in_cal[dst].remove(req_id)
            if now_ns >= r.active_from:
                start = _earliest_free(cals, now_ns, serve_ns,
                                       ignore_resv_after=now_ns)
            else:
                # booked activation still in the future (late booking or
                # early data): waiting for it competes with starting a
                # fresh reconfiguration right now — take the earlier. If
                # the circuit already points at this pair (e.g. sticky
                # routing), no fresh reconfiguration is needed at all.
                redo_setup = 0 if (self.out_target[src] == dst
                                   and self.in_source[dst] == src) \
                    else self.reconfig_ns
                wait_start = _earliest_free(cals, r.active_from, serve_ns,
                                            ignore_resv_after=now_ns)
                redo_t = _earliest_free(cals, now_ns,
                                        redo_setup + serve_ns,
                                        ignore_resv_after=now_ns)
                start = min(wait_start, redo_t + redo_setup)
            hit = start == now_ns
            _occupy(cals, start, start + serve_ns, ("xfer", req_id))
            self._displace_overlapping(src, dst, start, start + serve_ns)
            self.out_target[src] = dst
            self.in_source[dst] = src
            self._log(req_id, src, dst, nbytes, now_ns, start, 0, hit=hit)
            return int(start - now_ns)

        if self.policy == CORD:
            return self._cord_transfer(src, dst, nbytes, serve_ns, now_ns, req_id)
        if self.policy in (STEER, STEER_NA):
            return self._steer_transfer(src, dst, nbytes, serve_ns, now_ns, req_id)
        if self.policy == ELECTRICAL:
            return self._electrical_transfer(src, dst, nbytes, now_ns, req_id)

        # REACTIVE (and PQPS without a reservation)
        return self._reactive_transfer(src, dst, nbytes, serve_ns, now_ns, req_id)

    def _electrical_peek(self, src, nbytes, now_ns):
        """Predicted packet-plane COMPLETION (start + serve) without
        committing, for STEER's load-aware plane comparison. Completion
        accounts for both queuing and the (slower) serve rate so a
        congested/slow plane is penalized in the choice."""
        serve_nic = int(nbytes / self.electrical_bw_eff)
        nic_start = max(now_ns, self.elec_free[src])
        if self.electrical_spine_bw is None:
            return nic_start + serve_nic
        serve_spine = int(nbytes / self.electrical_spine_bw)
        start = max(nic_start, max(now_ns, self.elec_spine_free))
        return start + max(serve_nic, serve_spine)

    def _electrical_commit(self, src, nbytes, now_ns):
        """Commit a packet-plane transfer and return its START time. The
        transfer starts after queuing on the NIC egress and shared spine;
        its serve then STREAMS behind prefill compute (hidden, exactly as
        the optical serve is hidden in _reactive/_cord_transfer -- both
        planes stream layer-wise above the ~10 GB/s hiding threshold,
        expD). Serve occupies the resources for subsequent transfers only;
        the critical-path stall is the queuing (start - now), not the serve."""
        serve_nic = int(nbytes / self.electrical_bw_eff)
        nic_start = max(now_ns, self.elec_free[src])
        if self.electrical_spine_bw is None:
            self.elec_free[src] = nic_start + serve_nic
            return nic_start
        serve_spine = int(nbytes / self.electrical_spine_bw)
        start = max(nic_start, max(now_ns, self.elec_spine_free))
        self.elec_free[src] = start + serve_nic
        self.elec_spine_free = start + serve_spine
        return start

    def _electrical_transfer(self, src, dst, nbytes, now_ns, req_id):
        """All-electrical status quo: no optical circuit, no reconfiguration
        delay -- the transfer rides the packet plane (per-source NIC + shared
        spine). Stall is the queuing before it can start; its serve streams
        behind compute (hidden, like the optical serve)."""
        start = self._electrical_commit(src, nbytes, now_ns)
        self._log(req_id, src, dst, nbytes, now_ns, start, 0, hit=None, plane="E")
        return int(start - now_ns)

    def _reactive_transfer(self, src, dst, nbytes, serve_ns, now_ns, req_id):
        """Cold-path transfer: reuse a standing circuit or pay a fresh
        setup at the ports' earliest free slot."""
        cals = (self.out_cal[src], self.in_cal[dst])
        setup = 0 if (self.out_target[src] == dst and self.in_source[dst] == src) \
            else self.reconfig_ns
        t = _earliest_free(cals, now_ns, setup + serve_ns,
                           ignore_resv_after=now_ns)
        start = t + setup
        _occupy(cals, t, start + serve_ns, ("xfer", req_id))
        self._displace_overlapping(src, dst, t, start + serve_ns)
        self.out_target[src] = dst
        self.in_source[dst] = src
        self._log(req_id, src, dst, nbytes, now_ns, start, setup)
        return int(start - now_ns)

    def _displace_overlapping(self, src, dst, start, end):
        """Re-book soft reservations that now collide with an actual
        transfer occupation on either port."""
        displaced = set()
        for cal in (self.out_cal[src], self.in_cal[dst]):
            for s, e, tag in cal.iv:
                if _is_xfer(tag):
                    continue
                if s < end and e > start:
                    displaced.add(tag)
        for tag in displaced:
            r = self._reservations.get(tag)
            if r is None:
                continue
            self.out_cal[r.src].remove(tag)
            self.in_cal[r.dst].remove(tag)
            self._book(r)

    # ---------------- CORD: Coalesced Reservation with Deadlines ----------
    #
    # Regime: optical bandwidth makes service time << reconfiguration
    # delta, so scheduling degenerates to choosing WHEN to pay delta
    # toward each destination. CORD (i) pre-books ONE pair-level
    # activation per (src, dst) at the pair's earliest announced ready
    # time (inheriting the delta-hiding property without QPS sampling),
    # and (ii) on a pair flip, first drains the CURRENT pair's imminent
    # announced demand if the new transfer's slack budget allows —
    # sequencing flips into runs (A A A | B B) instead of interleaving
    # (A B A B), which minimizes reconfigurations paid per byte.

    def _cord_book_pair(self, src, dst):
        """(Re)book the single pair-level activation for (src, dst) at
        its earliest pending announced ready time."""
        pend = self._pair_pending.get((src, dst))
        if not pend:
            return
        tag = ("pair", src, dst)
        self.out_cal[src].remove(tag)
        self.in_cal[dst].remove(tag)
        # book only the activation window for the EARLIEST pending
        # transfer — the standing circuit then serves followers with no
        # further setup; booking the full pending sum would clog the
        # port calendar for demand that arrives spread out in time.
        head = min(pend, key=lambda p: p[1])
        est0 = head[1]
        dur = self.reconfig_ns + int(head[2] / self.bw)
        cals = (self.out_cal[src], self.in_cal[dst])
        t = _earliest_free(cals, est0 - self._guard() - self.reconfig_ns, dur)
        _occupy(cals, t, t + dur, tag)

    def _cord_transfer(self, src, dst, nbytes, serve_ns, now_ns, req_id):
        pend = self._pair_pending.get((src, dst), [])
        self._pair_pending[(src, dst)] = [p for p in pend if p[0] != req_id]
        self._reservations.pop(req_id, None)
        cals = (self.out_cal[src], self.in_cal[dst])

        if self.out_target[src] == dst and self.in_source[dst] == src:
            # same pair: ride the standing circuit
            start = _earliest_free(cals, now_ns, serve_ns,
                                   ignore_resv_after=now_ns)
            _occupy(cals, start, start + serve_ns, ("xfer", req_id))
            self._displace_overlapping(src, dst, start, start + serve_ns)
            self._log(req_id, src, dst, nbytes, now_ns, start, 0,
                      hit=(start == now_ns))
            return int(start - now_ns)

        # pair flip needed: hold to drain the current pair's imminent
        # announced demand if this transfer's slack budget allows
        base = now_ns
        cur = self.out_target[src]
        if cur is not None:
            imminent = [p for p in self._pair_pending.get((src, cur), [])
                        if p[1] <= now_ns + self.reconfig_ns]
            if imminent:
                drain_ns = int(sum(p[2] for p in imminent) / self.bw)
                if drain_ns + self.reconfig_ns <= self.cord_slack_ns:
                    base = now_ns + drain_ns

        # anticipatory pair booking may already have the circuit up
        tag = ("pair", src, dst)
        booked_from = None
        for s_, e_, t_ in self.out_cal[src].iv:
            if t_ == tag:
                booked_from = s_ + self.reconfig_ns
                break
        self.out_cal[src].remove(tag)
        self.in_cal[dst].remove(tag)

        redo_t = _earliest_free(cals, base, self.reconfig_ns + serve_ns,
                                ignore_resv_after=now_ns)
        start = redo_t + self.reconfig_ns
        hit = False
        if booked_from is not None:
            # the pair booking's reconfiguration either completed already
            # (ride it now) or completes sooner than a fresh one would
            wait_start = _earliest_free(cals, max(now_ns, booked_from),
                                        serve_ns, ignore_resv_after=now_ns)
            if wait_start < start:
                start = wait_start
                hit = start == now_ns
        _occupy(cals, start, start + serve_ns, ("xfer", req_id))
        self._displace_overlapping(src, dst, start, start + serve_ns)
        self.out_target[src] = dst
        self.in_source[dst] = src
        # re-book the pair we just left if it still has pending demand
        if cur is not None and self._pair_pending.get((src, cur)):
            self._cord_book_pair(src, cur)
        self._log(req_id, src, dst, nbytes, now_ns, start,
                  0 if hit else self.reconfig_ns, hit=hit)
        return int(start - now_ns)

    def estimate_optical_start(self, src, dst, serve_ns, now_ns):
        """Conservative, commitment-free estimate of when an optical
        transfer (src->dst) could start: rides a standing circuit, else
        uses the pair booking's activation, else pays a fresh setup.
        Used by STEER's plane comparison and by MultiPlaneManager's
        transceiver selection."""
        if self.out_target[src] == dst and self.in_source[dst] == src:
            return _earliest_free((self.out_cal[src], self.in_cal[dst]),
                                  now_ns, serve_ns, ignore_resv_after=now_ns)
        booked_from = None
        tag = ("pair", src, dst)
        for s_, e_, t_ in self.out_cal[src].iv:
            if t_ == tag:
                booked_from = s_ + self.reconfig_ns
                break
        base = max(now_ns, booked_from) if booked_from is not None \
            else now_ns + self.reconfig_ns
        return _earliest_free((self.out_cal[src], self.in_cal[dst]),
                              base, serve_ns, ignore_resv_after=now_ns)

    def probe_booking_activation(self, src, dst, nbytes, est_ready_ns):
        """Where would _cord_book_pair place this pair's activation?
        Mirrors its placement math without committing — used by
        MultiPlaneManager to pick the least-loaded transceiver."""
        serve_ns = int(int(nbytes) / self.bw)
        dur = self.reconfig_ns + serve_ns
        t = _earliest_free((self.out_cal[src], self.in_cal[dst]),
                           int(est_ready_ns) - self._guard() - self.reconfig_ns,
                           dur)
        return t + self.reconfig_ns

    def _steer_transfer(self, src, dst, nbytes, serve_ns, now_ns, req_id):
        """Hybrid plane steering: pick the plane with the earlier
        predicted completion (Eq. steer in the theory draft). Electrical
        completion folds the slower packet-plane service into the stall
        so the ASTRA-Sim wire (which runs at optical rate) lands at the
        equivalent instant."""
        # predicted packet-plane completion (NIC + shared spine), no commit
        e_completion = self._electrical_peek(src, nbytes, now_ns)

        # conservative optical estimate (no commitment yet)
        o_start = self.estimate_optical_start(src, dst, serve_ns, now_ns)
        o_completion = o_start + serve_ns

        if e_completion < o_completion:
            # take the electrical plane; release optical bookkeeping.
            # decision uses completion (load-aware), but the stall is the
            # queuing before start -- serve streams behind compute (hidden).
            self.cancel(req_id)
            pend = self._pair_pending.get((src, dst), [])
            self._pair_pending[(src, dst)] = [p for p in pend if p[0] != req_id]
            start = self._electrical_commit(src, nbytes, now_ns)
            self._log(req_id, src, dst, nbytes, now_ns, start, 0,
                      hit=None, plane="E")
            return int(start - now_ns)
        if self.policy == STEER_NA:
            # no-announcement ablation: the optical side has no bookings
            # to lean on — behave exactly like a reactive circuit.
            return self._reactive_transfer(src, dst, nbytes, serve_ns,
                                           now_ns, req_id)
        return self._cord_transfer(src, dst, nbytes, serve_ns, now_ns, req_id)

    def _rotor_start(self, src, dst, now_ns):
        """First instant the rotating circuit serves (src, dst)."""
        offset = (dst - src) % self.n  # 1 .. n-1
        period = self.rotor_slot_ns * (self.n - 1)
        slot_in_period = (offset - 1) * self.rotor_slot_ns
        cycle = now_ns // period
        for c in (cycle, cycle + 1):
            slot_start = c * period + slot_in_period
            usable_from = slot_start + self.reconfig_ns
            slot_end = slot_start + self.rotor_slot_ns
            if now_ns < usable_from:
                return usable_from
            if now_ns < slot_end:
                return now_ns
        return (cycle + 2) * period + slot_in_period + self.reconfig_ns

    # ---------------- bookkeeping ----------------

    def _log(self, req_id, src, dst, nbytes, ready, start, setup_ns, hit=None,
             plane="O"):
        est = self._announced_est.get(req_id)
        if self.adaptive_guard and est is not None:
            self._ag.update(est, ready)   # online-learn the guard from realized eta
        self.stats.append({
            "req_id": req_id, "policy": self.policy, "src": src, "dst": dst,
            "bytes": nbytes, "ready_ns": ready, "start_ns": start,
            "stall_ns": start - ready, "setup_ns": setup_ns,
            "resv_hit": hit, "plane": plane,
            # estimator error: actual data-ready minus admission-time estimate.
            "est_ready_ns": est,
            "est_err_ns": (ready - est) if est is not None else None,
        })

    def dump_stats(self, path):
        import csv
        if not self.stats:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.stats[0].keys()))
            w.writeheader()
            w.writerows(self.stats)

    def summary(self):
        if not self.stats:
            return {}
        stalls = sorted(s["stall_ns"] for s in self.stats)
        n = len(stalls)
        hits = [s for s in self.stats if s["resv_hit"] is True]
        return {
            "transfers": n,
            "stall_p50_us": stalls[n // 2] / 1e3,
            "stall_p99_us": stalls[min(n - 1, int(n * 0.99))] / 1e3,
            "stall_mean_us": sum(stalls) / n / 1e3,
            "resv_hit_rate": (len(hits) / n) if self.policy == PQPS else None,
        }


class SlottedCircuitManager:
    """Epoch/slot-quantized circuit plane for the reactive baselines that
    QPS-Fit itself benchmarks against (NegotiaToR, BFF) plus QPS-Fit.

    Time is divided into slots of ``slot_ns``; BFF/QPSFIT batch demand
    per epoch of ``slots_per_epoch`` slots (QPS-Fit defaults: 3 ms epoch,
    T=200), while NEGOTIATOR grants slot-by-slot without batching but
    pays a per-slot guardband and a scheduling-pipeline delay.

    Faithfulness notes (documented approximations at domain granularity):
    - BFF vs QPSFIT differ mainly in packing rule (best-fit, largest
      demand first vs QPS-sampled first-fit with largest-fit fallback);
      the original paper reports near-identical schedule quality, which
      this abstraction reproduces. Their computational-cost difference
      is irrelevant in simulation.
    - Demand becomes schedulable at the first epoch boundary after it
      arrives (the epoch-batching latency PQPS eliminates).
    - NEGOTIATOR: non-iterative request/grant with RR arbitration is
      approximated as FCFS port booking at slot granularity plus a
      2-epoch scheduling-pipeline delay (paper section 5.4); each slot
      pays the reconfiguration guardband, shrinking its usable share.
    """

    def __init__(self, num_domains, bw_bytes_per_ns, reconfig_ns,
                 policy=QPSFIT, slot_ns=None, slots_per_epoch=200):
        if policy not in SLOTTED_POLICIES:
            raise ValueError(f"Unknown slotted policy '{policy}'")
        self.n = num_domains
        self.bw = float(bw_bytes_per_ns)
        self.reconfig_ns = int(reconfig_ns)
        self.policy = policy
        # QPS-Fit quantizes a 3 ms epoch into 200 x 15 us slots and folds
        # the reconfiguration delay into the slots an allocation needs.
        # NegotiaToR instead needs the guardband to be a small fraction
        # of the slot (paper: 10 ns guard per 60 ns slot), so its slot
        # scales with the reconfiguration delay.
        if slot_ns:
            self.slot_ns = int(slot_ns)
        elif policy == NEGOTIATOR:
            self.slot_ns = max(15_000, 6 * self.reconfig_ns)
        else:
            self.slot_ns = 15_000
        self.T = int(slots_per_epoch)
        self.epoch_ns = self.slot_ns * self.T
        self.out_slots = [set() for _ in range(num_domains)]  # occupied abs slot idx
        self.in_slots = [set() for _ in range(num_domains)]
        self._reservations = {}  # always empty: these baselines are reactive
        self.out_target = [None] * num_domains  # last granted circuit target
        self.stats = []

    # predictive API: no-ops (these baselines are reactive by design)
    def announce(self, *a, **k):
        return

    def reannounce(self, *a, **k):
        return

    def cancel(self, *a, **k):
        return

    def _free_runs(self, src, dst, first_slot, horizon_slots):
        """Return (start, length) of maximal free runs in both ports'
        slot maps within [first_slot, first_slot + horizon)."""
        occ = self.out_slots[src] | self.in_slots[dst]
        runs = []
        run_start, run_len = None, 0
        for s in range(first_slot, first_slot + horizon_slots):
            if s in occ:
                if run_start is not None:
                    runs.append((run_start, run_len))
                    run_start, run_len = None, 0
            else:
                if run_start is None:
                    run_start = s
                run_len += 1
        if run_start is not None:
            runs.append((run_start, run_len))
        return runs

    def _occupy_run(self, src, dst, start, length):
        for s in range(start, start + length):
            self.out_slots[src].add(s)
            self.in_slots[dst].add(s)

    def request_transfer(self, src, dst, nbytes, now_ns, req_id=None):
        nbytes = int(nbytes)
        if src == dst or nbytes <= 0:
            self._log(req_id, src, dst, nbytes, now_ns, now_ns, 0)
            return 0

        if self.policy == NEGOTIATOR:
            return self._negotiator(src, dst, nbytes, now_ns, req_id)
        return self._epoch_fit(src, dst, nbytes, now_ns, req_id)

    def _epoch_fit(self, src, dst, nbytes, now_ns, req_id):
        """BFF / QPSFIT: demand joins the next epoch's batch; the grant
        fits it into one contiguous hole of ceil((D/bw + delta)/slot)
        slots (delta paid once), falling back to the largest hole with
        the residual recursing into later epochs."""
        need_ns = int(nbytes / self.bw) + self.reconfig_ns
        need_slots = max(1, -(-need_ns // self.slot_ns))
        # epoch batching: schedulable from the next epoch boundary
        epoch_idx = now_ns // self.epoch_ns + 1
        first_slot = epoch_idx * self.T
        remaining = need_slots
        first_alloc = None
        guard = 0
        while remaining > 0 and guard < 64:
            guard += 1
            runs = self._free_runs(src, dst, first_slot, self.T)
            if not runs:
                first_slot += self.T
                continue
            fitting = [r for r in runs if r[1] >= remaining]
            if fitting:
                if self.policy == BFF:
                    # best-fit: tightest hole
                    start, length = min(fitting, key=lambda r: r[1])
                else:
                    # QPS-Fit: first-fit
                    start, length = fitting[0]
                self._occupy_run(src, dst, start, remaining)
                if first_alloc is None:
                    first_alloc = start
                remaining = 0
            else:
                # largest-fit: grant the biggest hole, residual re-requested
                start, length = max(runs, key=lambda r: r[1])
                self._occupy_run(src, dst, start, length)
                if first_alloc is None:
                    first_alloc = start
                remaining -= length
                first_slot += self.T
        self.out_target[src] = dst
        start_ns = first_alloc * self.slot_ns + self.reconfig_ns
        self._log(req_id, src, dst, nbytes, now_ns, start_ns, self.reconfig_ns)
        return int(start_ns - now_ns)

    def _negotiator(self, src, dst, nbytes, now_ns, req_id):
        """Slot-by-slot on-demand granting: per-slot guardband shrinks
        the usable share; scheduling pipeline adds 2 slots (paper 5.4,
        epochs there are slot-scale here)."""
        usable_per_slot = max(1, self.slot_ns - self.reconfig_ns)
        need_slots = max(1, -(-int(nbytes / self.bw) // usable_per_slot))
        pipeline_ns = 2 * self.slot_ns
        first_slot = (now_ns + pipeline_ns) // self.slot_ns + 1
        occ = self.out_slots[src] | self.in_slots[dst]
        got, s, first_alloc, guard = 0, first_slot, None, 0
        while got < need_slots and guard < 1_000_000:
            guard += 1
            if s not in occ:
                self.out_slots[src].add(s)
                self.in_slots[dst].add(s)
                if first_alloc is None:
                    first_alloc = s
                got += 1
            s += 1
        self.out_target[src] = dst
        start_ns = first_alloc * self.slot_ns + self.reconfig_ns
        self._log(req_id, src, dst, nbytes, now_ns, start_ns, self.reconfig_ns)
        return int(start_ns - now_ns)

    def _log(self, req_id, src, dst, nbytes, ready, start, setup_ns):
        self.stats.append({
            "req_id": req_id, "policy": self.policy, "src": src, "dst": dst,
            "bytes": nbytes, "ready_ns": ready, "start_ns": start,
            "stall_ns": start - ready, "setup_ns": setup_ns,
            "resv_hit": None,
        })

    dump_stats = CircuitManager.dump_stats
    summary = CircuitManager.summary


def make_circuit_manager(policy, num_domains, bw_bytes_per_ns, reconfig_ns,
                         rotor_slot_ns=None, resv_guard_ns=None,
                         slot_ns=None, slots_per_epoch=200, cord_slack_ns=None,
                         electrical_bw=None, transceivers=1, afd_load=0.0,
                         electrical_spine_bw=None, adaptive_guard=False):
    """Factory covering both the continuous-time policies (IDEAL, ROTOR,
    REACTIVE, PQPS) and the slotted baselines (NEGOTIATOR, BFF, QPSFIT)."""
    if policy in SLOTTED_POLICIES:
        return SlottedCircuitManager(num_domains, bw_bytes_per_ns, reconfig_ns,
                                     policy=policy, slot_ns=slot_ns,
                                     slots_per_epoch=slots_per_epoch)
    if int(transceivers) > 1:
        return MultiPlaneManager(policy, transceivers, num_domains,
                                 bw_bytes_per_ns, reconfig_ns,
                                 resv_guard_ns=resv_guard_ns,
                                 cord_slack_ns=cord_slack_ns,
                                 electrical_bw=electrical_bw,
                                 adaptive_guard=adaptive_guard)
    return CircuitManager(num_domains, bw_bytes_per_ns, reconfig_ns,
                          policy=policy, rotor_slot_ns=rotor_slot_ns,
                          resv_guard_ns=resv_guard_ns, cord_slack_ns=cord_slack_ns,
                          electrical_bw=electrical_bw, afd_load=afd_load,
                          electrical_spine_bw=electrical_spine_bw,
                          adaptive_guard=adaptive_guard)


class MultiPlaneManager:
    """CORD-k / STEER-k: k transceiver pairs per domain, modeled as k
    parallel optical planes (SPECTRA-style: plane h's out-ports connect
    only to plane h's in-ports, which matches CPO port structure).

    Composition of k single-plane managers, untouched internally.
    Selection is the calendar-time analog of LPT (docs/keslassy-review):
      - announce: assign the reservation to the plane with the earliest
        estimated optical start at the announced ready time;
      - transfer: transfers with a reservation go to their plane;
        unannounced ones probe all planes and take the earliest.
    STEER-k keeps ONE shared electrical plane in this wrapper (the inner
    managers run CORD) and applies the plane-steering comparison here.
    Striping one transfer across planes is future work (EQUALIZE analog).
    """

    def __init__(self, policy, k, num_domains, bw_bytes_per_ns, reconfig_ns,
                 resv_guard_ns=None, cord_slack_ns=None, electrical_bw=None,
                 adaptive_guard=False):
        if policy not in (CORD, STEER, REACTIVE):
            raise ValueError(f"MultiPlaneManager supports CORD/STEER/REACTIVE, got {policy}")
        self.policy = policy
        self.k = int(k)
        self.n = num_domains
        self.bw = float(bw_bytes_per_ns)
        self.reconfig_ns = int(reconfig_ns)
        inner_policy = CORD if policy == STEER else policy
        self.planes = [
            CircuitManager(num_domains, bw_bytes_per_ns, reconfig_ns,
                           policy=inner_policy, resv_guard_ns=resv_guard_ns,
                           cord_slack_ns=cord_slack_ns,
                           adaptive_guard=adaptive_guard)
            for _ in range(self.k)
        ]
        self._req_plane = {}
        # router API compat: `req_id in manager._reservations` membership
        # checks — reservations live per plane, keyed here by assignment
        self._reservations = self._req_plane
        self.electrical_bw = float(electrical_bw) if electrical_bw else 25.0
        self.elec_free = [0] * num_domains
        self.stats = []

    # ---------------- predictive API ----------------

    def announce(self, req_id, src, dst, nbytes, est_ready_ns, deadline_ns=None):
        if src == dst:
            return
        h = min(range(self.k),
                key=lambda i: self.planes[i].probe_booking_activation(
                    src, dst, nbytes, est_ready_ns))
        self._req_plane[req_id] = h
        self.planes[h].announce(req_id, src, dst, nbytes, est_ready_ns, deadline_ns)

    def reannounce(self, req_id, est_ready_ns):
        h = self._req_plane.get(req_id)
        if h is not None:
            self.planes[h].reannounce(req_id, est_ready_ns)

    def cancel(self, req_id):
        h = self._req_plane.pop(req_id, None)
        if h is not None:
            self.planes[h].cancel(req_id)

    # ---------------- transfer ----------------

    def request_transfer(self, src, dst, nbytes, now_ns, req_id=None):
        nbytes = int(nbytes)
        if src == dst or nbytes <= 0:
            self.stats.append({"req_id": req_id, "policy": self.policy,
                               "src": src, "dst": dst, "bytes": nbytes,
                               "ready_ns": now_ns, "start_ns": now_ns,
                               "stall_ns": 0, "setup_ns": 0,
                               "resv_hit": None, "plane": "O0"})
            return 0
        serve_ns = int(nbytes / self.bw)

        h = self._req_plane.pop(req_id, None) if req_id is not None else None
        if h is None:
            h = min(range(self.k),
                    key=lambda i: self.planes[i].estimate_optical_start(
                        src, dst, serve_ns, now_ns))

        if self.policy == STEER:
            serve_e_ns = int(nbytes / self.electrical_bw)
            e_start = max(now_ns, self.elec_free[src])
            e_completion = e_start + serve_e_ns  # for the load-aware decision
            o_start = self.planes[h].estimate_optical_start(src, dst, serve_ns, now_ns)
            if e_completion < o_start + serve_ns:
                self.planes[h].cancel(req_id)
                # start-based stall: serve streams behind compute (hidden),
                # occupies the NIC for subsequent transfers only (mirrors the
                # single-plane fix; the completion drove the choice, not the stall)
                start = e_start
                self.elec_free[src] = e_start + serve_e_ns
                self.stats.append({"req_id": req_id, "policy": self.policy,
                                   "src": src, "dst": dst, "bytes": nbytes,
                                   "ready_ns": now_ns, "start_ns": start,
                                   "stall_ns": start - now_ns, "setup_ns": 0,
                                   "resv_hit": None, "plane": "E"})
                return int(start - now_ns)

        stall = self.planes[h].request_transfer(src, dst, nbytes, now_ns, req_id=req_id)
        rec = dict(self.planes[h].stats[-1])
        rec["plane"] = f"O{h}"
        self.stats.append(rec)
        return stall

    dump_stats = CircuitManager.dump_stats
    summary = CircuitManager.summary
