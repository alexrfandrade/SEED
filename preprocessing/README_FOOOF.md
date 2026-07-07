# Narrow-band vs Broad-band Power (FOOOF)

## Overview

This script separates the aperiodic (1/f) component of the EEG power spectrum from genuine oscillatory activity, using **FOOOF** (Fitting Oscillations & One Over F, also known as *specparam*). The goal is to avoid misinterpreting a steeper 1/f slope — typically seen in children — as an increase in theta oscillatory power, when it may simply reflect a difference in aperiodic (non-oscillatory) brain activity between age groups.

The pipeline:

1. Loads the continuous (non-epoched), preprocessed EEG dataset.
2. Computes the power spectrum on a frontal theta channel cluster using Welch's method (Hanning windows, 50% overlap).
3. Fits the FOOOF model over 1–40 Hz (both `fixed` and `knee` aperiodic modes are tested; the better-fitting mode is selected automatically based on the R² improvement).
4. Extracts the aperiodic parameters (offset, exponent, and knee if applicable) — the "1/f slope".
5. Extracts oscillatory peaks (center frequency, power, bandwidth) across delta, theta, alpha, beta, and gamma bands.
6. Computes two estimates of theta power: the **raw** total power (biased by the aperiodic component) and the **narrow-band** power (the residual oscillatory power after removing the aperiodic fit).
7. Produces a two-panel figure (raw PSD + aperiodic + FOOOF model; residual oscillatory spectrum) and saves all results to a `.mat` file.

## Input

- `<subject>_preprocessed_continuous.set` — the **continuous** (non-epoched) preprocessed EEG dataset (output of the preprocessing pipeline, saved *before* epoching).

## Output

| File | Description |
|---|---|
| `<subject>_fooof_results.mat` | Contains the aperiodic parameters (offset, exponent, knee), both R² values (fixed and knee mode), detected oscillatory peaks, theta-specific peaks, raw and narrow-band theta power estimates, the fitted spectra (raw, aperiodic fit, residual), and the frontal channel cluster used. |

A figure is also generated on screen (not saved automatically), showing the raw PSD against the FOOOF model and aperiodic fit, and the residual (oscillatory-only) spectrum.

## Dependencies

- **MATLAB**
- **EEGLAB** (tested on version 2025.1.0) — used only to load the `.set` file and access `EEG.data`/`EEG.chanlocs`
- **Python** (tested with `python3.11`), configured via MATLAB's `pyenv`
- **FOOOF** (`fooof` Python package, also known as *specparam*) — install with `pip install fooof`
- **NumPy** (`numpy` Python package), used to pass data to the FOOOF Python object

⚠️ FOOOF is called from MATLAB through Python interoperability (`py.fooof...`). You do not need to interact with Python directly, but a working Python environment with `fooof` and `numpy` installed is required, and MATLAB must be able to locate it via `pyenv`.

## How to use

1. Make sure your continuous, preprocessed `.set` file is available in the target folder.
2. Edit the configuration section at the top of the script:
   - `pyenv('Version', ...)`: path to your Python executable (find yours with `which python3.11` on macOS/Linux, or `where python` on Windows)
   - `eeglab_path`: path to your local EEGLAB installation
   - `dossier`: folder containing the input `.set` file (and where results will be saved)
   - `subject`: subject ID
   - `group`: `'adult'` or `'child'` — used only to annotate console messages (e.g. a note if a child's spectrum was fit with the `fixed` mode), not to change the fitting logic itself
3. Adjust FOOOF fitting parameters if needed (`peak_width_limits`, `max_n_peaks`, `min_peak_height`, `peak_threshold`, `KNEE_THRESHOLD`, `freq_range`).
4. Run the script. No manual/interactive step is required — this script is fully automatic, aside from the R² quality gate described below.

### Quality control built into the script

- The script **stops with an error** if the loaded file is epoched (`EEG.trials > 1`) — FOOOF requires continuous data here.
- The script **stops with an error** if the final model's R² is below `R2_ERROR` (0.90) — the subject should be excluded or the fitting parameters revised.
- The script **prints a warning** if R² is between `R2_ERROR` and `R2_WARN` (0.90–0.95) — a marginal fit that should be visually verified before including the subject in group-level analysis.

## Notes and known limitations

- The aperiodic mode (`fixed` vs `knee`) is chosen automatically based on the R² improvement (`KNEE_THRESHOLD = 0.02`); this threshold is a heuristic and should be reconsidered/documented for your own dataset rather than assumed to generalize.
- Theta power estimates (`theta_total`, `theta_narrowband`) are computed via trapezoidal integration directly on the log-power (dB) spectra. This gives an area-under-the-curve on a logarithmic scale, not a linear-scale power estimate — keep this in mind when comparing these values across studies that may compute theta power differently (e.g. on linear-scale PSD).
- The `ap_fit` variable is extracted via a private FOOOF attribute (`fm._ap_fit`), which is not part of the public, version-guaranteed FOOOF API. If you update your FOOOF/specparam installation and this script breaks, this is the most likely place to check first — record the FOOOF version you used when generating your results.
- A residual amplitude-based artifact rejection step (excluding 2-second windows exceeding a threshold in standard deviations) is present in the script but its output (`clean_data`) is not currently fed into the Welch PSD computation — verify whether this step is intended to be active for your use case before relying on it.
- This script processes one subject at a time; batch/looped processing across subjects is not implemented here.

## Contact / Contribution

Feel free to open an issue or pull request, especially regarding the log-scale vs linear-scale integration choice for theta power, or if you adapt this script to a different channel cluster or frequency range.
