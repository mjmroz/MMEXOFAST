import math

from scipy.optimize import brentq


class KHParameterization:
    """Convert between (k, h) and (s, q).

    Based on Hall et al. (2024), PASP, 136, 094401.

    Uses log10 throughout.
    """

    LOG_S_REF = 0.03
    M = 35.0

    def __init__(self):
        self.s_ref = 10.0**self.LOG_S_REF
        self.C = self.M * self.LOG_S_REF

        # Exact form of the offset approximated as 2.3 in the paper.
        self.k_offset = math.log10(4.0 / (self.s_ref - 1.0 / self.s_ref) ** 2)

        self.log_diff_ref = self._log_abs_s_minus_inv_s(self.LOG_S_REF)

    @staticmethod
    def _log_abs_s_minus_inv_s(abs_log_s):
        """Calculate log10(|s - 1/s|), with abs_log_s=|log10(s)|."""
        if abs_log_s <= 0.0:
            raise ValueError("The expression is undefined at s = 1.")

        return abs_log_s + math.log1p(
            -(10.0 ** (-2.0 * abs_log_s))
        ) / math.log(10.0)

    def from_kh_to_sq(self, k, h):
        """Return (s, q) for given scalar values of k and h."""

        k = float(k)
        h = float(h)

        if not (math.isfinite(k) and math.isfinite(h)):
            raise ValueError("k and h must be finite.")

        abs_h = abs(h)

        if abs_h <= self.C:
            # Central region:
            # h = M * log10(s)
            # k = log10(q) + k_offset
            log_s = h / self.M
            log_q = k - self.k_offset

        else:
            # Close or wide branch:
            # Find |log10(s)| along a constant-k contour.
            radius = abs_h - self.C
            x_ref = self.LOG_S_REF

            def residual(x):
                log_diff = self._log_abs_s_minus_inv_s(x)

                # Difference in log10(q) from its value at s_ref.
                delta_log_q = 2.0 * (log_diff - self.log_diff_ref)

                return (x - x_ref) ** 2 + delta_log_q**2 - radius**2

            lower = x_ref
            upper = x_ref + radius

            if upper == lower:
                abs_log_s = lower
            else:
                abs_log_s = brentq(
                    residual,
                    lower,
                    upper,
                    xtol=1e-12,
                )

            # Positive h gives the wide solution (s > 1);
            # negative h gives the close solution (s < 1).
            log_s = math.copysign(abs_log_s, h)

            # From k = log10[4q / (s - 1/s)^2].
            log_diff = self._log_abs_s_minus_inv_s(abs_log_s)
            log_q = k - math.log10(4.0) + 2.0 * log_diff

        try:
            s = 10.0**log_s
            q = 10.0**log_q
        except OverflowError as exc:
            raise ValueError(
                "The requested k and h exceed the numerical range."
            ) from exc

        if not (math.isfinite(s) and math.isfinite(q) and s > 0.0 and q > 0.0):
            raise ValueError(
                "The resulting s or q is outside the numerical range."
            )

        return s, q

    def from_sq_to_kh(self, s, q):
        """Return (k, h) for given positive s and q."""

        s = float(s)
        q = float(q)

        if not (math.isfinite(s) and math.isfinite(q) and s > 0.0 and q > 0.0):
            raise ValueError("s and q must be finite and positive.")

        log_s = math.log10(s)
        log_q = math.log10(q)
        abs_log_s = abs(log_s)

        if abs_log_s <= self.LOG_S_REF:
            k = log_q + self.k_offset
            h = self.M * log_s

        else:
            log_diff = self._log_abs_s_minus_inv_s(abs_log_s)

            k = math.log10(4.0) + log_q - 2.0 * log_diff

            log_q_ref = k - self.k_offset

            radius = math.sqrt(
                (abs_log_s - self.LOG_S_REF) ** 2 + (log_q - log_q_ref) ** 2
            )

            h = math.copysign(radius + self.C, log_s)

        return k, h
