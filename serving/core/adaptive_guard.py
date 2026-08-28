"""Learning-augmented reservation lead: online bias correction + residual guard.

A naive prefill-completion estimator can be systematically biased (it over-
predicts the ready time by ~85-95 ms at high load: eta = actual - est is far
below 0). A single guard band then has to be huge to cover it. We decompose the
needed lead ``-eta`` into a bias term and a residual quantile:

    -eta = (-b) + (-(eta - b)),   b = E[eta]  (learned online, EWMA)

The bias b converges in a handful of samples; the residual guard
g_r = Quantile_tau(-(eta-b)) is a small correction. The scheduler books the
circuit ready at ``est - lead``, ``lead = -b + g_r``. Self-contained; mirrors
analysis/adaptive_guard.py.
"""


class AdaptiveGuard:
    def __init__(self, c_stall=50.0, c_idle=1.0, g_safe=1_000_000.0,
                 g_max=None, lam=0.25, lr0=None, warmup=20,
                 bias_correct=True, bias_beta=0.1):
        self.tau = c_stall / (c_stall + c_idle)
        self.g_safe = float(g_safe)
        self.g_max = float(g_max) if g_max is not None else 50.0 * self.g_safe
        self.lam = min(max(lam, 0.0), 1.0)
        self.lr0 = float(lr0) if lr0 is not None else max(self.g_safe, 1.0)
        self.warmup = int(warmup)
        self.bias_correct = bool(bias_correct)
        self.bias_beta = float(bias_beta)
        self._b = 0.0          # learned bias ~ E[eta]
        self._g = self.g_safe  # residual guard estimate
        self.n = 0

    def guard(self):
        """Total reservation lead = bias correction (-b) + residual guard."""
        g_res = self.lam * self.g_safe + (1.0 - self.lam) * self._g
        lead = (-self._b if self.bias_correct else 0.0) + g_res
        return max(0.0, min(self.g_max, lead))

    def update(self, est_ready_ns, actual_ready_ns):
        eta = float(actual_ready_ns) - float(est_ready_ns)
        self.n += 1
        if self.bias_correct:
            self._b = (1.0 - self.bias_beta) * self._b + self.bias_beta * eta
        resid = eta - (self._b if self.bias_correct else 0.0)
        L = -resid
        lr = self.lr0 / (1.0 + self.n / max(1, self.warmup))
        indicator = 1.0 if L <= self._g else 0.0
        self._g += lr * (self.tau - indicator)
        self._g = max(0.0, min(self.g_max, self._g))
        return self.guard()

    @property
    def learned(self):
        return self.guard()

    @property
    def bias(self):
        return self._b
