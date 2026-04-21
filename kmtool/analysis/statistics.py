import math

import pandas as pd

from kmtool.models import PairwiseResult

try:
    from lifelines import CoxPHFitter
    from lifelines.statistics import logrank_test
except ImportError:  # pragma: no cover - dependency managed by requirements
    CoxPHFitter = None
    logrank_test = None


def reconstructed_to_frame(arms):
    records = []
    for arm in arms:
        for time_value, event_flag in zip(arm.time, arm.event):
            records.append(
                {
                    "time": float(time_value),
                    "event": int(event_flag),
                    "arm": arm.arm_label,
                }
            )
    return pd.DataFrame(records)


def compute_pairwise_result(arm_a, arm_b, comparison_id):
    frame = reconstructed_to_frame([arm_a, arm_b])
    warnings = list(arm_a.warnings) + list(arm_b.warnings)
    if frame.empty:
        raise ValueError("No reconstructed patient records are available.")

    left = frame[frame["arm"] == arm_a.arm_label]
    right = frame[frame["arm"] == arm_b.arm_label]
    if left["event"].sum() == 0 or right["event"].sum() == 0:
        warnings.append("One arm has no reconstructed events; HR and log-rank may be unstable.")

    if CoxPHFitter is not None and logrank_test is not None:
        lr = logrank_test(
            left["time"],
            right["time"],
            event_observed_A=left["event"],
            event_observed_B=right["event"],
        )

        cox_frame = frame.copy()
        cox_frame["arm_indicator"] = (cox_frame["arm"] == arm_b.arm_label).astype(int)
        cph = CoxPHFitter(penalizer=0.05)
        cph.fit(cox_frame[["time", "event", "arm_indicator"]], duration_col="time", event_col="event")
        summary = cph.summary.loc["arm_indicator"]
        log_rank_p = float(lr.p_value)
        hr = float(math.exp(summary["coef"]))
        hr_ci_low = float(math.exp(summary["coef lower 95%"]))
        hr_ci_high = float(math.exp(summary["coef upper 95%"]))
    else:
        warnings.append("lifelines not installed; using a log-rank and log-HR approximation.")
        log_rank_p, hr, hr_ci_low, hr_ci_high = _manual_pairwise_statistics(frame, arm_a.arm_label, arm_b.arm_label)

    return PairwiseResult(
        comparison_id=comparison_id,
        log_rank_p=float(log_rank_p),
        hr=hr,
        hr_ci_low=hr_ci_low,
        hr_ci_high=hr_ci_high,
        n_reconstructed=int(len(frame)),
        warnings=warnings,
    )


def _manual_pairwise_statistics(frame, arm_a_label, arm_b_label):
    event_times = sorted(frame.loc[frame["event"] == 1, "time"].unique().tolist())
    observed_minus_expected = 0.0
    variance = 0.0
    for time_value in event_times:
        at_risk_a = int(((frame["arm"] == arm_a_label) & (frame["time"] >= time_value)).sum())
        at_risk_b = int(((frame["arm"] == arm_b_label) & (frame["time"] >= time_value)).sum())
        total_at_risk = at_risk_a + at_risk_b
        if total_at_risk <= 1:
            continue

        events_a = int(((frame["arm"] == arm_a_label) & (frame["time"] == time_value) & (frame["event"] == 1)).sum())
        events_b = int(((frame["arm"] == arm_b_label) & (frame["time"] == time_value) & (frame["event"] == 1)).sum())
        total_events = events_a + events_b
        if total_events == 0:
            continue

        expected_b = total_events * (float(at_risk_b) / float(total_at_risk))
        observed_minus_expected += events_b - expected_b

        variance_increment = (
            float(at_risk_a * at_risk_b * total_events * max(total_at_risk - total_events, 0))
            / float((total_at_risk ** 2) * (total_at_risk - 1))
        )
        variance += max(variance_increment, 0.0)

    if variance <= 0:
        return 1.0, 1.0, 1.0, 1.0

    z_score = observed_minus_expected / math.sqrt(variance)
    log_rank_p = math.erfc(abs(z_score) / math.sqrt(2.0))
    log_hr = observed_minus_expected / variance
    se = math.sqrt(1.0 / variance)
    hr = math.exp(log_hr)
    hr_ci_low = math.exp(log_hr - 1.96 * se)
    hr_ci_high = math.exp(log_hr + 1.96 * se)
    return log_rank_p, hr, hr_ci_low, hr_ci_high
