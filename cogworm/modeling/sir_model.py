"""SIR-inspired propagation model for multi-agent AI systems.

Adapted from epidemiological SIR (Susceptible–Infected–Recovered)
to model adversarial signal propagation in agent networks.

Agent states map to SIR as:
    S = susceptible (never seen signal)
    E = exposed (received but not yet activated)
    I = activated (propagating the signal)
    R = recovered (signal died out) OR contained (defense blocked)

This is a CANDIDATE model tested against simpler baselines — it is
not assumed to be the correct model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit


@dataclass
class SIRParams:
    """Parameters of the fitted SIR model."""
    beta: float    # transmission rate (S → I)
    gamma: float   # recovery rate (I → R)
    sigma: float   # incubation rate (E → I), for SEIR extension
    r0: float      # basic reproduction number (beta / gamma)
    r_squared: float = 0.0


def sir_ode(y: list[float], t: float, beta: float, gamma: float) -> list[float]:
    """Standard SIR differential equations."""
    S, I, R = y
    N = S + I + R
    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I
    return [dS, dI, dR]


def seir_ode(
    y: list[float], t: float, beta: float, sigma: float, gamma: float,
) -> list[float]:
    """SEIR model — adds an Exposed compartment."""
    S, E, I, R = y
    N = S + E + I + R
    dS = -beta * S * I / N
    dE = beta * S * I / N - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I
    return [dS, dE, dI, dR]


class PropagationModel:
    """Fit and predict signal propagation using SIR/SEIR models."""

    def __init__(self) -> None:
        self.params: SIRParams | None = None
        self._model_type: str = "sir"

    def fit_sir(
        self,
        time_steps: list[int],
        activated_counts: list[int],
        total_agents: int,
    ) -> SIRParams:
        """Fit SIR model parameters to observed propagation data."""
        t = np.array(time_steps, dtype=float)
        I_obs = np.array(activated_counts, dtype=float)

        N = float(total_agents)
        I0 = I_obs[0] if I_obs[0] > 0 else 1.0
        S0 = N - I0
        R0_init = 0.0

        def sir_for_fit(t_arr: np.ndarray, beta: float, gamma: float) -> np.ndarray:
            y0 = [S0, I0, R0_init]
            sol = odeint(sir_ode, y0, t_arr, args=(beta, gamma))
            return sol[:, 1]  # return I(t)

        try:
            popt, _ = curve_fit(
                sir_for_fit, t, I_obs,
                p0=[0.5, 0.1],
                bounds=([0.001, 0.001], [5.0, 5.0]),
                maxfev=10000,
            )
            beta, gamma = popt

            # R² goodness of fit
            I_pred = sir_for_fit(t, beta, gamma)
            ss_res = np.sum((I_obs - I_pred) ** 2)
            ss_tot = np.sum((I_obs - np.mean(I_obs)) ** 2)
            r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            self.params = SIRParams(
                beta=beta,
                gamma=gamma,
                sigma=0.0,
                r0=beta / gamma if gamma > 0 else float("inf"),
                r_squared=r_sq,
            )
            self._model_type = "sir"
            return self.params

        except (RuntimeError, ValueError):
            # Fit failed — return degenerate params
            self.params = SIRParams(
                beta=0.0, gamma=0.0, sigma=0.0, r0=0.0, r_squared=0.0,
            )
            return self.params

    def predict(
        self,
        total_agents: int,
        time_steps: int,
        initial_infected: int = 1,
    ) -> dict[str, list[float]]:
        """Predict propagation using fitted parameters."""
        if not self.params or self.params.beta == 0:
            return {"S": [], "I": [], "R": []}

        N = float(total_agents)
        I0 = float(initial_infected)
        S0 = N - I0
        R0 = 0.0

        t = np.linspace(0, time_steps, time_steps + 1)
        y0 = [S0, I0, R0]
        sol = odeint(sir_ode, y0, t, args=(self.params.beta, self.params.gamma))

        return {
            "S": sol[:, 0].tolist(),
            "I": sol[:, 1].tolist(),
            "R": sol[:, 2].tolist(),
        }


# ── Baseline models ──────────────────────────────────────────────────

def linear_baseline(
    time_steps: list[int],
    activated_counts: list[int],
) -> dict:
    """Simple linear regression baseline."""
    t = np.array(time_steps, dtype=float)
    y = np.array(activated_counts, dtype=float)
    if len(t) < 2:
        return {"slope": 0.0, "intercept": 0.0, "r_squared": 0.0}

    coeffs = np.polyfit(t, y, 1)
    y_pred = np.polyval(coeffs, t)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {"slope": coeffs[0], "intercept": coeffs[1], "r_squared": r_sq}


def exponential_baseline(
    time_steps: list[int],
    activated_counts: list[int],
) -> dict:
    """Exponential growth baseline: I(t) = a * exp(b * t)."""
    t = np.array(time_steps, dtype=float)
    y = np.array(activated_counts, dtype=float)
    y_safe = np.maximum(y, 0.1)

    try:
        log_y = np.log(y_safe)
        coeffs = np.polyfit(t, log_y, 1)
        b, log_a = coeffs
        a = np.exp(log_a)
        y_pred = a * np.exp(b * t)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {"a": a, "b": b, "r_squared": r_sq}
    except (ValueError, RuntimeError):
        return {"a": 0.0, "b": 0.0, "r_squared": 0.0}
