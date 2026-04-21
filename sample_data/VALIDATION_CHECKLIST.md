# Minimal Validation Sample and Screenshot Checkpoints

## Files

- `km_minimal_sample.png`: the simplest two-arm Kaplan-Meier image sample
- `km_minimal_sample.pdf`: a PDF version of the same figure
- `km_minimal_truth.csv`: the reference values for manual review of time points, survival levels, and the risk table

## Ground Truth

- Comparison: `Treatment A` vs `Treatment B`
- Time unit: `months`
- Expected direction: `Treatment A` performs better than `Treatment B`
- Risk-table time points: `0, 6, 12, 18, 24, 30`
- Risk-table endpoints:
  - A: `120 -> 65`
  - B: `120 -> 41`

## Step-by-Step Validation

### 1. Before Launch

Run:

```bash
python -m unittest discover -s tests -v
```

Expected:

- All tests pass

Screenshot checkpoint:

- A terminal screenshot showing `OK` at the bottom

### 2. Upload the Image Sample

Upload `km_minimal_sample.png`

Expected:

- The first tab shows the complete figure
- The chart clearly contains two curves, a legend, and a `Number at risk` section

Screenshot checkpoint:

- A full screenshot of the `Upload / Retrieve` tab

### 3. Figure Localization

In the `Figure Localization` tab, enter:

- `x_min = 0`
- `x_max = 30`
- `y_min = 0`
- `y_max = 1`
- `Time unit = months`
- `Study ID = demo_km_minimal`
- `Arm labels = Treatment A,Treatment B`

Click `Run curve extraction`

Expected:

- Extraction succeeds
- You should not see `No curve candidates were found`

Screenshot checkpoint:

- The success message after clicking the button

### 4. Extraction Review

Expected:

- `Detected curves` is at least `2`
- `Overall confidence` is greater than `0.40`
- The overlay plot shows the blue and red reconstructed traces roughly aligned with the original chart
- The reprojected curves remain monotone decreasing
- The risk-table CSV includes columns for both `Treatment A` and `Treatment B`

Screenshot checkpoints:

- A full screenshot of the `Extraction Review` tab
- Make sure the screenshot includes the confidence score, overlay plot, reprojected plot, and risk-table CSV

### 5. Pairwise Survival Analysis

Select:

- `Arm A = Treatment A`
- `Arm B = Treatment B`
- `Comparison ID = minimal_demo`
- `Fallback sample size without risk table = 100` or leave the default

Click `Reconstruct pseudo-IPD and compute log-rank / HR`

Expected:

- `log-rank p` is produced
- `HR` is produced
- `HR > 1`
- In the reconstructed plot, `Treatment B` declines faster

Explanation:

- In the current app implementation, the HR is interpreted in the `arm_b / arm_a` direction. If `Treatment B` performs worse than `Treatment A`, the displayed HR should be greater than `1`.

Screenshot checkpoint:

- A full screenshot of the `Pairwise Analysis` tab, including `p`, `HR`, `95% CI`, and the reconstructed curves

### 6. PDF Smoke Test

Return to the first tab and upload `km_minimal_sample.pdf`

Expected:

- The ranked PDF candidate pages include the Kaplan-Meier figure
- After selecting the candidate page, the downstream extraction flow still works

Screenshot checkpoint:

- The `Upload / Retrieve` tab with the PDF candidate-page area visible

## Quick Pass / Fail Guide

Pass:

- Both the image and PDF can enter the extraction pipeline
- At least two curves are detected
- The curve direction is correct
- The reconstructed result yields `HR > 1`
- The app does not fail silently

Fail:

- Only one curve is detected
- A curve clearly moves upward
- `Treatment A` and `Treatment B` are swapped
- `HR <= 1` even though the reconstructed plot shows `Treatment B` performing worse
- The PDF candidate-page area is empty or cannot continue into extraction
