# EEG Preprocessing Pipeline — Per Bloc

## Overview

This script preprocesses raw EEG data (BrainVision format) for a memory experiment composed of several task blocs: **Encoding**, **Gap**, **Old/New**, and **2AFC**. The signal is first sliced into these blocs based on event markers, then each bloc is cleaned, decomposed, and epoched independently using behavioral data from a companion CSV file.

The pipeline follows these steps:

1. Load raw data and initialize EEGLAB
2. Visualize the raw signal and markers
3. Slice the continuous signal into blocs based on marker latencies
4. Manually inspect and remove visually noisy channels (per bloc)
5. Filter and run automatic bad-channel rejection (ASR via `clean_artifacts`)
6. Verify the percentage of data removed by ASR
7. Re-reference to average reference
8. Run ICA (`runica`, extended) followed by ICLabel for automatic artifact-component removal
9. Interpolate previously removed channels (spherical interpolation)
10. Save the continuous (non-epoched) preprocessed signal
11. Enrich event markers with behavioral data (condition, performance, hit/miss, etc.) and epoch each bloc
12. Save the final epoched dataset

## Input

- Raw BrainVision files: `subject.vhdr`, `subject.vmrk`, `subject.eeg`
- Behavioral data file: `subject.csv` (see **Expected CSV format** below)

## Output

| File | Description |
|---|---|
| `subject_enc_after_ICA.set`, `subject_item_after_ICA.set`, `subject_afc_after_ICA.set` | Intermediate save after ICA + ICLabel, before interpolation |
| `subject_[bloc]_preprocessed_continuous.set` | Continuous (non-epoched) preprocessed signal, per bloc |
| `subject_encoding_preprocessed.set` | Final epoched Encoding dataset |
| `subject_oldnew_preprocessed.set` | Final epoched Old/New dataset |
| `subject_afc_preprocessed.set` | Final epoched 2AFC dataset |

## Dependencies

- **MATLAB** (tested on R2023b or later recommended)
- **EEGLAB** (tested on version 2025.1.0) — [https://sccn.ucsd.edu/eeglab](https://sccn.ucsd.edu/eeglab)
- EEGLAB plugins:
  - `bva-io` (BrainVision file import)
  - `clean_rawdata` (ASR / `clean_artifacts`)
  - `ICLabel` (automatic IC classification)
- Signal Processing Toolbox (MATLAB) — not strictly required by this script itself, but recommended for the wider analysis pipeline

## Expected CSV format

The behavioral CSV file must use `,` as delimiter and include (at minimum) the following columns:

- `Object_Encode`, `Condition_Encode`, `Scene`
- `key_AFC.corr`, `Correct_key_AFC`, `key_AFC.keys`
- `key_item.corr`, `OldNew_Condition`, `Correct_Key_Old`, `key_item.keys`

Column names must match exactly (case-sensitive), including the dot notation coming from PsychoPy-style logging (e.g. `key_AFC.corr`).

## How to use

1. Clone or download this repository.
2. Open the script in MATLAB.
3. Edit **only** the configuration section at the top of the script:
   - `eeglab_path`: path to your local EEGLAB installation
   - `input_dir`: folder containing the raw `.vhdr`/`.vmrk`/`.csv` files
   - `output_dir`: folder where preprocessed files will be saved (created automatically if it doesn't exist)
   - `subject`: subject ID (must match the file naming, e.g. `IA3006` for `IA3006.vhdr`)
4. Run the script.

### ⚠️ Manual steps required

This pipeline is **not fully automatic**. At the visual channel-rejection step (section 4), a plot window will open for each bloc (Encoding, Old/New, AFC). You must:
- Visually inspect the signal,
- Select and remove any obviously noisy channel(s),
- Close the plot window to let the script continue.

Do not attempt to run this script unattended (e.g. in batch/background mode) without adapting this step first.

## Notes and known limitations

- The ASR parameters (`ChannelCriterion`, `BurstCriterion`, `WindowCriterion`) and the ICLabel rejection threshold (`thresh = 0.80`) were tuned for this dataset. They may need adjustment for other recordings — check the printed "% data removed" diagnostics after ASR (a warning is printed if more than 20–30% of data is removed).
- Marker labels (`Stim`, `Rating`, `Item`, `Resp_Item`, `AFC`, `Resp_Asso`) are specific to this experiment's marker naming convention. Adapt section 3 if your paradigm uses different marker names.
- Epoch window is fixed at **-0.2 s to +1.0 s** relative to stimulus onset; adjust `epoch_tmin`/`epoch_tmax` if needed.

## Contact / Contribution

Feel free to open an issue or pull request if you adapt this pipeline for a different marker scheme or dataset.
