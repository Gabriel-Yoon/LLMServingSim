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

POLICIES = (IDEAL, ROTOR, REACTIVE, PQPS)


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
                 policy=REACTIVE, rotor_slot_ns=None, resv_guard_ns=None):
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

        self.out_cal = [_Calendar() for _ in range(num_domains)]
        self.in_cal = [_Calendar() for _ in range(num_domains)]
        self.out_target = [None] * num_domains  # current circuit pointing
        self.in_source = [None] * num_domains
        self._reservations = {}  # req_id -> _Reservation
        self.stats = []

    # ---------------- predictive API ----------------

    def announce(self, req_id, src, dst, nbytes, est_ready_ns, deadline_ns=None):
        """Admission-time demand announcement (PQPS only acts on it)."""
        if src == dst:
            return
        r = _Reservation(req_id, src, dst, int(nbytes), int(est_ready_ns), deadline_ns)
        self._reservations[req_id] = r
        if self.policy == PQPS:
            self._book(r)

    def _book(self, r):
        """Book both ports on their calendars so the circuit becomes
        active ``resv_guard_ns`` before the estimated ready time."""
        serve_ns = int(r.nbytes / self.bw)
        dur = self.reconfig_ns + serve_ns
        desired_setup = r.est_ready - self.resv_guard_ns - self.reconfig_ns
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
                # fresh reconfiguration right now — take the earlier.
                wait_start = _earliest_free(cals, r.active_from, serve_ns,
                                            ignore_resv_after=now_ns)
                redo_t = _earliest_free(cals, now_ns,
                                        self.reconfig_ns + serve_ns,
                                        ignore_resv_after=now_ns)
                start = min(wait_start, redo_t + self.reconfig_ns)
            hit = start == now_ns
            _occupy(cals, start, start + serve_ns, ("xfer", req_id))
            self._displace_overlapping(src, dst, start, start + serve_ns)
            self.out_target[src] = dst
            self.in_source[dst] = src
            self._log(req_id, src, dst, nbytes, now_ns, start, 0, hit=hit)
            return int(start - now_ns)

        # REACTIVE (and PQPS without a reservation)
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

    def _log(self, req_id, src, dst, nbytes, ready, start, setup_ns, hit=None):
        self.stats.append({
            "req_id": req_id, "policy": self.policy, "src": src, "dst": dst,
            "bytes": nbytes, "ready_ns": ready, "start_ns": start,
            "stall_ns": start - ready, "setup_ns": setup_ns,
            "resv_hit": hit,
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
