import numpy as np
from matplotlib import pyplot as plt

try:
    from lifelines import KaplanMeierFitter
except ImportError:  # pragma: no cover - dependency managed by requirements
    KaplanMeierFitter = None


def plot_curve_series(curves, axis_bounds):
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for curve in curves:
        points = np.array(curve.data_points, dtype="float64")
        if points.size == 0:
            continue
        axis.step(points[:, 0], points[:, 1], where="post", label=curve.arm_label)
    axis.set_xlabel("Time ({0})".format(axis_bounds.time_unit))
    axis.set_ylabel("Survival probability")
    axis.set_ylim(0.0, 1.02)
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    return figure


def plot_overlay(plot_image, curves):
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.imshow(plot_image)
    for curve in curves:
        if not curve.pixel_points:
            continue
        points = np.array(curve.pixel_points, dtype="float64")
        color = np.array(curve.detected_color, dtype="float64") / 255.0
        axis.plot(points[:, 0], points[:, 1], linewidth=2, color=color, label=curve.arm_label)
    axis.set_axis_off()
    axis.legend(loc="upper right")
    figure.tight_layout()
    return figure


def plot_reconstructed_survival(arms):
    figure, axis = plt.subplots(figsize=(7, 4.5))
    if KaplanMeierFitter is not None:
        for arm in arms:
            fitter = KaplanMeierFitter()
            fitter.fit(arm.time, event_observed=arm.event, label=arm.arm_label)
            fitter.plot_survival_function(ax=axis, ci_show=False)
    else:
        for arm in arms:
            times, survivals = _manual_km_curve(arm.time, arm.event)
            axis.step(times, survivals, where="post", label=arm.arm_label)
    axis.set_xlabel("Time")
    axis.set_ylabel("Reconstructed survival probability")
    axis.set_ylim(0.0, 1.02)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def _manual_km_curve(times, events):
    paired = sorted(zip(times, events), key=lambda item: item[0])
    if not paired:
        return [0.0], [1.0]
    at_risk = len(paired)
    survival = 1.0
    x_points = [0.0]
    y_points = [1.0]
    index = 0
    while index < len(paired):
        time_value = float(paired[index][0])
        event_count = 0
        censor_count = 0
        while index < len(paired) and float(paired[index][0]) == time_value:
            if int(paired[index][1]) == 1:
                event_count += 1
            else:
                censor_count += 1
            index += 1
        if event_count:
            x_points.extend([time_value, time_value])
            y_points.extend([survival, survival * (1.0 - float(event_count) / float(at_risk))])
            survival = y_points[-1]
        at_risk -= event_count + censor_count
        if at_risk <= 0:
            break
    if x_points[-1] < float(paired[-1][0]):
        x_points.append(float(paired[-1][0]))
        y_points.append(survival)
    return x_points, y_points
