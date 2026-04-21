import hashlib
from dataclasses import replace

import pandas as pd

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit(
        "Streamlit is not installed. Run `python -m pip install -r requirements.txt` before launching the app."
    ) from exc

from kmtool.analysis.extraction import extract_km_data, reproject_curves
from kmtool.analysis.indirect import compute_bucher_indirect, reported_hr_to_effect
from kmtool.analysis.ingestion import (
    build_manual_crop,
    load_image_bytes,
    normalize_uploaded_input,
    rank_pdf_pages,
    render_pdf_pages,
)
from kmtool.analysis.literature import search_comparison_candidates
from kmtool.analysis.ocr import parse_manual_risk_table, risk_table_to_csv
from kmtool.analysis.reconstruction import reconstruct_arm_ipd
from kmtool.analysis.statistics import compute_pairwise_result
from kmtool.analysis.visualization import plot_curve_series, plot_overlay, plot_reconstructed_survival
from kmtool.config import AppConfig
from kmtool.models import ArmMapping, AxisBounds, CurveExtractionRequest, IndirectComparisonRequest


@st.cache_data(show_spinner=False, ttl=3600)
def cached_render_pdf_pages(file_bytes, dpi):
    return render_pdf_pages(file_bytes, dpi=dpi)


@st.cache_data(show_spinner=False, ttl=86400)
def cached_literature_search(treatment_left, treatment_right, endpoint, population, retmax=12):
    return search_comparison_candidates(
        treatment_left,
        treatment_right,
        endpoint=endpoint,
        population=population,
        retmax=retmax,
    )


def init_state():
    defaults = {
        "uploaded_digest": "",
        "source_payload": None,
        "pdf_pages": [],
        "pdf_ranked_pages": [],
        "selected_page_index": 0,
        "extraction_result": None,
        "manual_risk_csv": "",
        "reconstructed_arms": [],
        "pairwise_result": None,
        "ab_candidates": [],
        "bc_candidates": [],
        "indirect_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_analysis_state():
    st.session_state["extraction_result"] = None
    st.session_state["manual_risk_csv"] = ""
    st.session_state["reconstructed_arms"] = []
    st.session_state["pairwise_result"] = None
    st.session_state["indirect_result"] = None


def load_source_payload(uploaded_file, config):
    payload = normalize_uploaded_input(uploaded_file)
    digest = hashlib.md5(payload["file_bytes"]).hexdigest()
    if digest == st.session_state["uploaded_digest"]:
        return
    st.session_state["uploaded_digest"] = digest
    st.session_state["source_payload"] = payload
    st.session_state["selected_page_index"] = 0
    reset_analysis_state()

    if payload["source_type"] == "pdf":
        pages, page_texts = cached_render_pdf_pages(payload["file_bytes"], dpi=config.pdf_render_dpi)
        st.session_state["pdf_pages"] = pages
        st.session_state["pdf_ranked_pages"] = rank_pdf_pages(pages, page_texts)
    else:
        st.session_state["pdf_pages"] = []
        st.session_state["pdf_ranked_pages"] = []


def get_active_image():
    payload = st.session_state["source_payload"]
    if not payload:
        return None
    if payload["source_type"] == "image":
        return load_image_bytes(payload["file_bytes"])
    pages = st.session_state.get("pdf_pages") or []
    if not pages:
        return None
    index = st.session_state.get("selected_page_index", 0)
    index = max(0, min(index, len(pages) - 1))
    return pages[index]


def extraction_to_dataframe(candidates):
    rows = []
    for candidate in candidates:
        rows.append(
            {
                "include": candidate.reported_hr is not None,
                "study_id": candidate.study_id,
                "source": candidate.source,
                "year": candidate.year,
                "title": candidate.title,
                "pmid": candidate.pmid,
                "doi": candidate.doi,
                "treatment_left": candidate.treatments[0] if candidate.treatments else "",
                "treatment_right": candidate.treatments[1] if len(candidate.treatments) > 1 else "",
                "hr": candidate.reported_hr,
                "ci_low": candidate.ci_low,
                "ci_high": candidate.ci_high,
                "endpoint_text": candidate.endpoint_text,
                "population_text": candidate.population_text,
                "source_method": "reported_text",
                "warnings": " | ".join(candidate.warnings),
            }
        )
    return pd.DataFrame(rows)


def dataframe_to_effects(frame, comparison_label):
    effects = []
    if frame is None or frame.empty:
        return effects
    included = frame[frame["include"] == True]  # noqa: E712
    for _, row in included.iterrows():
        if pd.isna(row.get("hr")) or pd.isna(row.get("ci_low")) or pd.isna(row.get("ci_high")):
            continue
        effects.append(
            reported_hr_to_effect(
                study_id=str(row["study_id"]),
                comparison_label=comparison_label,
                treatment_left=str(row["treatment_left"]),
                treatment_right=str(row["treatment_right"]),
                hr=float(row["hr"]),
                ci_low=float(row["ci_low"]),
                ci_high=float(row["ci_high"]),
                source_method=str(row.get("source_method", "reported_text")),
                endpoint_text=str(row.get("endpoint_text", "")),
                population_text=str(row.get("population_text", "")),
            )
        )
    return effects


def apply_theme():
    st.set_page_config(page_title="KM Indirect Comparison Lab", page_icon="+/", layout="wide")
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
        html, body, [class*="css"]  { font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif; }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(214, 242, 214, 0.95), transparent 28%),
                radial-gradient(circle at top right, rgba(251, 231, 197, 0.8), transparent 24%),
                linear-gradient(180deg, #f6f2e9 0%, #eef2ea 100%);
        }
        .hero {
            padding: 1.25rem 1.4rem;
            border: 1px solid rgba(38, 64, 43, 0.16);
            border-radius: 18px;
            background: rgba(255, 252, 246, 0.82);
            box-shadow: 0 12px 30px rgba(52, 76, 58, 0.08);
            margin-bottom: 1rem;
        }
        .metric-card {
            padding: 0.9rem 1rem;
            border-radius: 14px;
            background: rgba(255,255,255,0.72);
            border: 1px solid rgba(53, 83, 57, 0.12);
        }
        .small-note {
            color: #355339;
            font-size: 0.93rem;
        }
        h1, h2, h3 {
            letter-spacing: -0.02em;
            color: #1f3422;
        }
        code {
            font-family: 'IBM Plex Mono', monospace;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_upload_tab(config):
    st.subheader("1. Upload / Retrieve")
    uploaded_file = st.file_uploader("Upload a Kaplan-Meier image or article PDF", type=["png", "jpg", "jpeg", "pdf"])
    if uploaded_file is not None:
        try:
            load_source_payload(uploaded_file, config)
        except Exception as exc:
            st.error("Failed to parse the uploaded file: {0}".format(exc))
            return

    payload = st.session_state.get("source_payload")
    if not payload:
        st.info("Upload an image or PDF first. The downstream tabs will unlock after a file is loaded.")
        return

    st.markdown(
        '<div class="metric-card"><strong>Current Input</strong><br><span class="small-note">{0} · {1}</span></div>'.format(
            payload["file_name"],
            payload["source_type"].upper(),
        ),
        unsafe_allow_html=True,
    )
    st.write("")

    if payload["source_type"] == "image":
        image = get_active_image()
        st.image(image, caption="Uploaded Kaplan-Meier image", use_column_width=True)
        return

    ranked_pages = st.session_state.get("pdf_ranked_pages", [])
    if not ranked_pages:
        st.warning("The PDF was uploaded, but no candidate pages were generated.")
        return

    options = {
        "Page {0} | score {1:.1f}".format(item["page_index"] + 1, item["score"]): item["page_index"]
        for item in ranked_pages
    }
    labels = list(options.keys())
    current_index = 0
    for index, label in enumerate(labels):
        if options[label] == st.session_state.get("selected_page_index", 0):
            current_index = index
            break
    selected_label = st.selectbox("Automatically ranked PDF pages", labels, index=current_index)
    st.session_state["selected_page_index"] = options[selected_label]

    top_pages = ranked_pages[:3]
    columns = st.columns(len(top_pages))
    for column, item in zip(columns, top_pages):
        with column:
            st.image(item["image"], caption="Page {0} · score {1:.1f}".format(item["page_index"] + 1, item["score"]))
            st.caption(item["text_excerpt"] or "No extracted text preview")


def render_localization_tab(config):
    st.subheader("2. Figure Localization")
    image = get_active_image()
    if image is None:
        st.info("Upload an input file first. This tab controls localization and extraction settings.")
        return

    st.image(image, caption="Current analysis image", use_column_width=True)
    use_manual_crop = st.checkbox("Enable manual crop", value=False)
    if use_manual_crop:
        columns = st.columns(4)
        left_pct = columns[0].slider("Left boundary %", 0, 95, 5)
        top_pct = columns[1].slider("Top boundary %", 0, 95, 5)
        right_pct = columns[2].slider("Right boundary %", 5, 100, 95)
        bottom_pct = columns[3].slider("Bottom boundary %", 5, 100, 95)
        manual_crop = build_manual_crop(image.width, image.height, (left_pct, top_pct, right_pct, bottom_pct))
        if manual_crop:
            st.image(image.crop(manual_crop), caption="Manual crop preview", use_column_width=True)
    else:
        manual_crop = None

    axis_columns = st.columns(5)
    x_min = axis_columns[0].number_input("x_min", value=0.0, step=1.0)
    x_max = axis_columns[1].number_input("x_max", value=100.0, step=1.0)
    y_min = axis_columns[2].number_input("y_min", value=0.0, step=0.1)
    y_max = axis_columns[3].number_input("y_max", value=1.0, step=0.1)
    time_unit = axis_columns[4].selectbox("Time unit", ["months", "weeks", "days", "years"], index=0)

    study_id = st.text_input("Study ID", value=st.session_state["source_payload"]["file_name"].rsplit(".", 1)[0])
    arm_label_hint = st.text_input("Arm labels (comma-separated)", value="Treatment A,Treatment B")

    if st.button("Run curve extraction", type="primary"):
        request = CurveExtractionRequest(
            source_type=st.session_state["source_payload"]["source_type"],
            file_name=st.session_state["source_payload"]["file_name"],
            study_id=study_id,
            time_unit=time_unit,
            manual_crop=manual_crop,
            manual_axis_bounds=AxisBounds(
                x_min=float(x_min),
                x_max=float(x_max),
                y_min=float(y_min),
                y_max=float(y_max),
                time_unit=time_unit,
            ),
        )
        label_overrides = [value.strip() for value in arm_label_hint.split(",") if value.strip()]
        try:
            result = extract_km_data(request, image, config, arm_label_overrides=label_overrides)
            st.session_state["extraction_result"] = result
            risk_csv = risk_table_to_csv(result["review"].risk_table_rows, [curve.arm_label for curve in result["curves"]])
            st.session_state["manual_risk_csv"] = risk_csv
            st.success("Extraction completed. Review and refine the result in the next tab.")
        except Exception as exc:
            st.error("Curve extraction failed: {0}".format(exc))


def render_review_tab():
    st.subheader("3. Extraction Review")
    extraction_result = st.session_state.get("extraction_result")
    if not extraction_result:
        st.info("Run curve extraction once before opening this tab.")
        return

    review = extraction_result["review"]
    curves = extraction_result["curves"]
    plot_image = extraction_result["plot_image"]

    metric_columns = st.columns(3)
    metric_columns[0].metric("Detected curves", len(curves))
    metric_columns[1].metric("Overall confidence", "{0:.2f}".format(review.confidence_score))
    metric_columns[2].metric("Risk table rows", len(review.risk_table_rows))

    if review.warnings:
        for warning in review.warnings:
            st.warning(warning)

    left, right = st.columns([1, 1])
    with left:
        st.pyplot(plot_overlay(plot_image, curves), use_container_width=True)
    with right:
        st.pyplot(plot_curve_series(curves, review.axis_bounds), use_container_width=True)

    st.caption(
        "You can edit arm labels, axis bounds, and the risk-table CSV below. "
        "These edits directly affect pseudo-IPD reconstruction and downstream statistics."
    )

    label_columns = st.columns(max(1, len(curves)))
    new_labels = []
    for column, curve in zip(label_columns, curves):
        with column:
            new_labels.append(st.text_input(curve.curve_id, value=curve.arm_label))

    axis_columns = st.columns(5)
    axis_bounds = review.axis_bounds
    revised_x_min = axis_columns[0].number_input("Review x_min", value=float(axis_bounds.x_min), step=1.0)
    revised_x_max = axis_columns[1].number_input("Review x_max", value=float(axis_bounds.x_max), step=1.0)
    revised_y_min = axis_columns[2].number_input("Review y_min", value=float(axis_bounds.y_min), step=0.1)
    revised_y_max = axis_columns[3].number_input("Review y_max", value=float(axis_bounds.y_max), step=0.1)
    revised_time_unit = axis_columns[4].selectbox(
        "Review time unit",
        ["months", "weeks", "days", "years"],
        index=["months", "weeks", "days", "years"].index(axis_bounds.time_unit)
        if axis_bounds.time_unit in ["months", "weeks", "days", "years"]
        else 0,
    )

    risk_csv = st.text_area("Risk table CSV", value=st.session_state.get("manual_risk_csv", ""), height=180)
    with st.expander("OCR text"):
        st.code(review.ocr_text or "No OCR output")

    if st.button("Apply reviewed values"):
        updated_axis = AxisBounds(
            x_min=float(revised_x_min),
            x_max=float(revised_x_max),
            y_min=float(revised_y_min),
            y_max=float(revised_y_max),
            time_unit=revised_time_unit,
        )
        updated_curves = reproject_curves(curves, plot_image, updated_axis, arm_label_overrides=new_labels)
        updated_mappings = [
            ArmMapping(
                curve_id=curve.curve_id,
                arm_label=curve.arm_label,
                detected_color=curve.detected_color,
                source="review",
            )
            for curve in updated_curves
        ]
        updated_review = replace(
            review,
            axis_bounds=updated_axis,
            arm_mappings=updated_mappings,
            risk_table_rows=parse_manual_risk_table(risk_csv),
        )
        st.session_state["extraction_result"] = {
            "plot_image": plot_image,
            "plot_bbox": extraction_result["plot_bbox"],
            "risk_bbox": extraction_result.get("risk_bbox"),
            "curves": updated_curves,
            "review": updated_review,
        }
        st.session_state["manual_risk_csv"] = risk_csv
        st.success("Review changes applied.")


def render_pairwise_tab(config):
    st.subheader("4. Pairwise Survival Analysis")
    extraction_result = st.session_state.get("extraction_result")
    if not extraction_result:
        st.info("Complete extraction and review before opening this tab.")
        return

    curves = extraction_result["curves"]
    if len(curves) < 2:
        st.warning("At least two curves are required for pairwise analysis.")
        return

    labels = [curve.arm_label for curve in curves]
    columns = st.columns(4)
    arm_a_label = columns[0].selectbox("Arm A", labels, index=0)
    arm_b_label = columns[1].selectbox("Arm B", labels, index=1 if len(labels) > 1 else 0)
    comparison_id = columns[2].text_input("Comparison ID", value="comparison_1")
    fallback_n = columns[3].number_input("Fallback sample size without risk table", value=int(config.default_sample_size), step=10)

    if st.button("Reconstruct pseudo-IPD and compute log-rank / HR", type="primary"):
        if arm_a_label == arm_b_label:
            st.error("Arm A and Arm B must refer to different curves.")
            return
        curve_a = [curve for curve in curves if curve.arm_label == arm_a_label][0]
        curve_b = [curve for curve in curves if curve.arm_label == arm_b_label][0]
        review = extraction_result["review"]
        study_id = st.session_state["source_payload"]["file_name"]
        try:
            arm_a = reconstruct_arm_ipd(
                study_id=study_id,
                comparison_id=comparison_id,
                curve=curve_a,
                risk_rows=review.risk_table_rows,
                fallback_total_n=int(fallback_n),
            )
            arm_b = reconstruct_arm_ipd(
                study_id=study_id,
                comparison_id=comparison_id,
                curve=curve_b,
                risk_rows=review.risk_table_rows,
                fallback_total_n=int(fallback_n),
            )
            pairwise = compute_pairwise_result(arm_a, arm_b, comparison_id=comparison_id)
            st.session_state["reconstructed_arms"] = [arm_a, arm_b]
            st.session_state["pairwise_result"] = pairwise
            st.success("Pairwise statistics generated.")
        except Exception as exc:
            st.error("Pairwise analysis failed: {0}".format(exc))

    pairwise_result = st.session_state.get("pairwise_result")
    reconstructed_arms = st.session_state.get("reconstructed_arms") or []
    if pairwise_result and reconstructed_arms:
        metrics = st.columns(4)
        metrics[0].metric("log-rank p", "{0:.4g}".format(pairwise_result.log_rank_p))
        metrics[1].metric("HR", "{0:.3f}".format(pairwise_result.hr))
        metrics[2].metric("95% CI", "{0:.3f} - {1:.3f}".format(pairwise_result.hr_ci_low, pairwise_result.hr_ci_high))
        metrics[3].metric("Reconstructed sample size", pairwise_result.n_reconstructed)
        if pairwise_result.warnings:
            for warning in pairwise_result.warnings:
                st.warning(warning)
        st.pyplot(plot_reconstructed_survival(reconstructed_arms), use_container_width=True)
        records = []
        for arm in reconstructed_arms:
            for time_value, event_flag in zip(arm.time[:40], arm.event[:40]):
                records.append({"arm": arm.arm_label, "time": time_value, "event": event_flag})
        st.dataframe(pd.DataFrame(records), use_container_width=True)


def render_literature_tab():
    st.subheader("5. Literature Screening and Indirect A-C Comparison")
    st.caption(
        "This module searches PubMed and Europe PMC for A-B and B-C studies, "
        "then lets you confirm effect sizes and calculate an indirect A-C estimate."
    )

    columns = st.columns(5)
    treatment_a = columns[0].text_input("Treatment A", value="Treatment A")
    treatment_b = columns[1].text_input("Treatment B", value="Treatment B")
    treatment_c = columns[2].text_input("Treatment C", value="Treatment C")
    endpoint = columns[3].text_input("Endpoint", value="overall survival")
    population = columns[4].text_input("Population / disease", value="")

    if st.button("Search A-B and B-C literature"):
        try:
            st.session_state["ab_candidates"] = cached_literature_search(
                treatment_a,
                treatment_b,
                endpoint,
                population,
            )
            st.session_state["bc_candidates"] = cached_literature_search(
                treatment_b,
                treatment_c,
                endpoint,
                population,
            )
            st.success("Candidate studies refreshed.")
        except Exception as exc:
            st.error("Literature search failed: {0}".format(exc))

    ab_candidates = st.session_state.get("ab_candidates") or []
    bc_candidates = st.session_state.get("bc_candidates") or []
    if not ab_candidates and not bc_candidates:
        st.info("After you run the search, editable candidate-study tables will appear here.")
        return

    st.info(
        "If you already reconstructed an HR and 95% CI from a local Kaplan-Meier figure or PDF, "
        "you can overwrite the values directly in the tables below."
    )

    ab_editor = None
    bc_editor = None
    if ab_candidates:
        st.markdown("**A-B candidate studies**")
        ab_editor = st.data_editor(
            extraction_to_dataframe(ab_candidates),
            use_container_width=True,
            num_rows="dynamic",
            key="ab_editor",
        )
    if bc_candidates:
        st.markdown("**B-C candidate studies**")
        bc_editor = st.data_editor(
            extraction_to_dataframe(bc_candidates),
            use_container_width=True,
            num_rows="dynamic",
            key="bc_editor",
        )

    allow_override = st.checkbox("Allow endpoint / population consistency override", value=False)
    if st.button("Compute A vs C using the Bucher method", type="primary"):
        try:
            ab_effects = dataframe_to_effects(ab_editor, "{0} vs {1}".format(treatment_a, treatment_b))
            bc_effects = dataframe_to_effects(bc_editor, "{0} vs {1}".format(treatment_b, treatment_c))
            if not ab_effects or not bc_effects:
                raise ValueError("Both A-B and B-C require at least one selected row containing HR and 95% CI.")
            request = IndirectComparisonRequest(
                treatment_a=treatment_a,
                treatment_b=treatment_b,
                treatment_c=treatment_c,
                endpoint=endpoint,
                selected_ab_studies=[effect.study_id for effect in ab_effects],
                selected_bc_studies=[effect.study_id for effect in bc_effects],
                population_hint=population,
                allow_inconsistent=allow_override,
            )
            st.session_state["indirect_result"] = compute_bucher_indirect(request, ab_effects, bc_effects)
            st.success("Indirect A-C comparison completed.")
        except Exception as exc:
            st.error("Indirect comparison failed: {0}".format(exc))

    indirect_result = st.session_state.get("indirect_result")
    if indirect_result:
        metrics = st.columns(3)
        metrics[0].metric("A vs C HR", "{0:.3f}".format(indirect_result.ac_hr))
        metrics[1].metric("95% CI", "{0:.3f} - {1:.3f}".format(indirect_result.ci95[0], indirect_result.ci95[1]))
        metrics[2].metric("log(HR)", "{0:.3f}".format(indirect_result.ac_log_hr))
        if indirect_result.heterogeneity_notes:
            for note in indirect_result.heterogeneity_notes:
                st.warning(note)
        if indirect_result.warnings:
            for warning in indirect_result.warnings:
                st.warning(warning)
        st.write("Included evidence:", ", ".join(indirect_result.study_provenance))


def main():
    config = AppConfig.from_env()
    apply_theme()
    init_state()

    st.markdown(
        """
        <div class="hero">
            <h1>Kaplan-Meier Indirect Comparison Lab</h1>
            <div class="small-note">
                A research-oriented survival-analysis workspace that extracts event-time signals from Kaplan-Meier
                figures or PDFs, reconstructs pseudo-IPD, computes pairwise survival statistics, and connects
                A-B and B-C studies into a traceable A-C Bucher indirect comparison.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Runtime")
        st.write("Python app: Streamlit single-user MVP")
        st.write("LLM status:", "Configured" if config.llm_enabled else "Not configured")
        st.write("LLM model:", config.llm_model)
        st.write("Default fallback sample size:", config.default_sample_size)
        st.write("PDF DPI:", config.pdf_render_dpi)
        st.caption("If Tesseract is not installed locally, OCR will fall back gracefully.")

    tabs = st.tabs(
        [
            "Upload / Retrieve",
            "Figure Localization",
            "Extraction Review",
            "Pairwise Analysis",
            "Literature and Indirect Comparison",
        ]
    )
    with tabs[0]:
        render_upload_tab(config)
    with tabs[1]:
        render_localization_tab(config)
    with tabs[2]:
        render_review_tab()
    with tabs[3]:
        render_pairwise_tab(config)
    with tabs[4]:
        render_literature_tab()


if __name__ == "__main__":
    main()
