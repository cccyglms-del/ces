import math

import numpy as np

from kmtool.models import IndirectComparisonResult, StudyEffect


def reported_hr_to_effect(
    study_id,
    comparison_label,
    treatment_left,
    treatment_right,
    hr,
    ci_low,
    ci_high,
    source_method,
    endpoint_text="",
    population_text="",
):
    if hr is None or ci_low is None or ci_high is None:
        raise ValueError("HR and 95% CI are required to compute a log-HR effect.")
    if hr <= 0 or ci_low <= 0 or ci_high <= 0:
        raise ValueError("HR and confidence interval bounds must be positive.")
    log_hr = math.log(float(hr))
    se = (math.log(float(ci_high)) - math.log(float(ci_low))) / (2.0 * 1.96)
    return StudyEffect(
        study_id=study_id,
        comparison_label=comparison_label,
        treatment_left=treatment_left,
        treatment_right=treatment_right,
        log_hr=log_hr,
        se=se,
        source_method=source_method,
        endpoint_text=endpoint_text,
        population_text=population_text,
    )


def orient_effect(effect, numerator, denominator):
    if effect.treatment_left == numerator and effect.treatment_right == denominator:
        return effect
    if effect.treatment_left == denominator and effect.treatment_right == numerator:
        return StudyEffect(
            study_id=effect.study_id,
            comparison_label=effect.comparison_label,
            treatment_left=numerator,
            treatment_right=denominator,
            log_hr=-effect.log_hr,
            se=effect.se,
            source_method=effect.source_method,
            endpoint_text=effect.endpoint_text,
            population_text=effect.population_text,
            warnings=effect.warnings + ["Effect orientation was reversed to match the requested comparison."],
        )
    raise ValueError(
        "Effect {0} does not match requested orientation {1}/{2}.".format(
            effect.study_id,
            numerator,
            denominator,
        )
    )


def pool_fixed_effects(effects):
    if not effects:
        raise ValueError("At least one effect is required.")
    weights = np.array([1.0 / max(effect.se ** 2, 1e-9) for effect in effects], dtype="float64")
    log_hrs = np.array([effect.log_hr for effect in effects], dtype="float64")
    pooled_log_hr = float(np.sum(weights * log_hrs) / np.sum(weights))
    pooled_se = float(math.sqrt(1.0 / np.sum(weights)))
    return pooled_log_hr, pooled_se


def compute_bucher_indirect(request, ab_effects, bc_effects):
    normalized_ab = [orient_effect(effect, request.treatment_a, request.treatment_b) for effect in ab_effects]
    normalized_bc = [orient_effect(effect, request.treatment_b, request.treatment_c) for effect in bc_effects]

    warnings = []
    heterogeneity_notes = []
    if request.endpoint:
        for effect in normalized_ab + normalized_bc:
            if effect.endpoint_text and request.endpoint.lower() not in effect.endpoint_text.lower():
                heterogeneity_notes.append(
                    "Endpoint mismatch for {0}: expected '{1}' but study metadata says '{2}'.".format(
                        effect.study_id,
                        request.endpoint,
                        effect.endpoint_text,
                    )
                )
    if request.population_hint:
        for effect in normalized_ab + normalized_bc:
            if effect.population_text and request.population_hint.lower() not in effect.population_text.lower():
                heterogeneity_notes.append(
                    "Population mismatch for {0}: expected '{1}'.".format(effect.study_id, request.population_hint)
                )

    if heterogeneity_notes and not request.allow_inconsistent:
        raise ValueError("Indirect comparison blocked because selected studies are not clinically consistent.")
    if heterogeneity_notes and request.allow_inconsistent:
        warnings.append("Consistency override enabled; interpret A vs C comparison cautiously.")

    pooled_ab_log_hr, pooled_ab_se = pool_fixed_effects(normalized_ab)
    pooled_bc_log_hr, pooled_bc_se = pool_fixed_effects(normalized_bc)

    ac_log_hr = pooled_ab_log_hr + pooled_bc_log_hr
    ac_se = math.sqrt((pooled_ab_se ** 2) + (pooled_bc_se ** 2))
    ci_low = math.exp(ac_log_hr - 1.96 * ac_se)
    ci_high = math.exp(ac_log_hr + 1.96 * ac_se)

    provenance = [effect.study_id for effect in normalized_ab + normalized_bc]
    return IndirectComparisonResult(
        ac_log_hr=ac_log_hr,
        ac_hr=math.exp(ac_log_hr),
        ci95=(ci_low, ci_high),
        study_provenance=provenance,
        heterogeneity_notes=heterogeneity_notes,
        warnings=warnings,
    )

