# Kaplan-Meier Research MVP

A Streamlit-based research app for:

- Uploading Kaplan-Meier curve images or article PDFs
- Extracting axes, curve traces, and number-at-risk clues with `CV + OCR + optional multimodal LLM`
- Reconstructing approximate individual-level survival data (`pseudo-IPD`)
- Computing `log-rank p`, `HR`, and `95% CI`
- Searching `PubMed + Europe PMC` and performing Bucher-style indirect `A-B`, `B-C`, then `A-C` comparisons

## Launch

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Environment Variables

- `LLM_API_KEY`
- `LLM_API_BASE`: defaults to `https://api.openai.com/v1`
- `LLM_MODEL`: defaults to `gpt-4.1-mini`
- `LLM_TIMEOUT_SECONDS`: defaults to `45`
- `DEFAULT_SAMPLE_SIZE`: fallback sample size when no risk table is available, default `100`
- `PDF_RENDER_DPI`: PDF rasterization DPI, default `180`

## Notes

- OCR depends on a local `Tesseract OCR` installation. The app still runs without it, but OCR features will degrade gracefully.
- Figure extraction is an approximate research workflow and should not be treated as production-grade evidence synthesis without manual review.
- When number-at-risk tables are missing, image quality is poor, colors are similar, or study populations/endpoints are inconsistent, the app deliberately lowers confidence and surfaces warnings.
