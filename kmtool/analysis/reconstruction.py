import math

import numpy as np

from kmtool.models import ReconstructedArmData


def _interpolate_survival(curve_points, time_point):
    times = np.array([point[0] for point in curve_points], dtype="float64")
    survivals = np.array([point[1] for point in curve_points], dtype="float64")
    if time_point <= times[0]:
        return float(survivals[0])
    if time_point >= times[-1]:
        return float(survivals[-1])
    return float(np.interp(time_point, times, survivals))


def _build_interval_times(curve_points, risk_rows):
    last_time = float(curve_points[-1][0]) if curve_points else 0.0
    if risk_rows:
        times = sorted(set(float(row.time) for row in risk_rows))
        if last_time > times[-1]:
            times.append(last_time)
        return times
    sampled = [float(point[0]) for point in curve_points[:: max(1, len(curve_points) // 40)]]
    if sampled[-1] != last_time:
        sampled.append(last_time)
    return sorted(set(sampled))


def reconstruct_arm_ipd(
    study_id,
    comparison_id,
    curve,
    risk_rows=None,
    fallback_total_n=100,
):
    curve_points = curve.data_points
    if not curve_points:
        raise ValueError("Curve contains no data points.")

    arm_label = curve.arm_label
    warnings = list(curve.warnings)
    risk_rows = risk_rows or []
    arm_specific_rows = [row for row in risk_rows if arm_label in row.arm_counts]
    interval_times = _build_interval_times(curve_points, arm_specific_rows)

    if arm_specific_rows:
        current_at_risk = int(arm_specific_rows[0].arm_counts.get(arm_label, fallback_total_n))
        method = "guyot_interval_approx"
        confidence = min(0.95, curve.confidence + 0.08)
    else:
        current_at_risk = int(fallback_total_n)
        method = "interval_heuristic"
        confidence = max(0.15, curve.confidence - 0.15)
        warnings.append("Risk table unavailable; IPD reconstruction used a heuristic fallback sample size.")

    event_times = []
    event_flags = []
    risk_lookup = {float(row.time): int(row.arm_counts.get(arm_label, current_at_risk)) for row in arm_specific_rows}

    for index in range(len(interval_times) - 1):
        start_time = float(interval_times[index])
        end_time = float(interval_times[index + 1])
        if end_time <= start_time:
            continue
        s0 = max(_interpolate_survival(curve_points, start_time), 1e-6)
        s1 = max(_interpolate_survival(curve_points, end_time), 0.0)
        drop_fraction = max(0.0, min(1.0, (s0 - s1) / s0))
        estimated_events = int(round(current_at_risk * drop_fraction))

        next_risk = risk_lookup.get(end_time)
        if next_risk is not None:
            max_observed_losses = max(0, current_at_risk - next_risk)
            estimated_events = min(max_observed_losses, estimated_events)
            estimated_censored = max(0, current_at_risk - next_risk - estimated_events)
        else:
            estimated_censored = 0
            if index < len(interval_times) - 2 and estimated_events == 0 and s1 < s0 - 0.005:
                estimated_events = 1

        estimated_events = max(0, min(estimated_events, current_at_risk))
        estimated_censored = max(0, min(estimated_censored, current_at_risk - estimated_events))

        event_times.extend([end_time] * estimated_events)
        event_flags.extend([1] * estimated_events)

        if estimated_censored:
            censor_times = np.linspace(start_time, end_time, estimated_censored + 2)[1:-1]
            event_times.extend(float(value) for value in censor_times.tolist())
            event_flags.extend([0] * estimated_censored)

        current_at_risk = current_at_risk - estimated_events - estimated_censored
        if current_at_risk <= 0:
            current_at_risk = 0
            break

    if current_at_risk > 0:
        final_time = float(interval_times[-1])
        event_times.extend([final_time] * current_at_risk)
        event_flags.extend([0] * current_at_risk)

    paired = sorted(zip(event_times, event_flags), key=lambda item: (item[0], -item[1]))
    times = [float(item[0]) for item in paired]
    flags = [int(item[1]) for item in paired]
    return ReconstructedArmData(
        study_id=study_id,
        comparison_id=comparison_id,
        arm_label=arm_label,
        time=times,
        event=flags,
        source_curve_id=curve.curve_id,
        reconstruction_method=method,
        confidence=confidence,
        warnings=warnings,
    )

