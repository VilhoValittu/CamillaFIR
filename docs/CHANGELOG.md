# Changelog

All notable changes to **CamillaFIR** are documented in this file.

---

## [4.0.3] - 2026-04-01

### Advanced UI / Workflow
- Added guided preset buttons for Advanced-tab shaping, bass safety, and confidence-pull tuning, with live summaries so the effective settings are visible without opening every fine-tune control.
- Kept the detailed Advanced controls behind collapsible fine-tune sections and blocked guided preset changes while AUTO mode is managing those settings.
- Improved the Files and Target tabs with clearer upload/local-path status cards, preview-source indicators, a target-preview legend hint, and more robust bundled manual/external-link behavior in the NiceGUI app.

### Config / Compatibility
- Introduced shared UI-choice normalization helpers so layout, level mode, level algorithm, and related preset selections accept stable keys, legacy English labels, and translated labels consistently.
- Normalized those stable values through UI collection and pipeline handling while still emitting legacy names where runtime config, summary text, and export diagnostics expect them.
- Kept run/results/status text, manual-target summary output, and export diagnostics aligned with the normalized layout and leveling metadata.

### Preview / Performance
- Added cached TXT/WAV measurement parsing for target-preview uploads and local file paths so repeated preview refreshes do not keep reparsing unchanged inputs.

### Docs & Tests
- Added regression coverage for Advanced guided presets, target-preview caching, file/path status metadata, translation-safe option normalization, and related manual-target/reporting paths.
- Updated README/version references for `4.0.3`.

---

## [4.0.2] - 2026-03-31

### Target preview / Manual mode
- Added direct Manual-mode target dragging in the Target tab preview, so the displayed target curve can be moved vertically with the mouse to update the manual target level.
- Added a dedicated Manual target tilt control using a fixed `1 kHz` pivot, allowing the target curve shape to be tilted intentionally instead of only shifted up/down.
- Added a dedicated right-side tilt handle in the preview so Manual target tilt can also be adjusted with the mouse.
- Added preview-side legend/readout entries for both `Manual level` and `Manual tilt` to make the current manual target settings visible at a glance.

### Config / Runtime
- Kept manual target level and manual target tilt as explicit source-of-truth settings, so preview interaction, saved config, and runtime target loading stay aligned.
- Applied Manual target tilt consistently to preview and runtime house-curve loading, instead of leaving it as a visual-only preview effect.

### Docs & Tests
- Updated README release notes and version references for `4.0.2`.
- Added regression coverage for manual target tilt transforms and preview interaction parsing.

---

## [4.0.1] - 2026-03-31

### Automatic mode
- Continued the AUTO-mode speed-up work by splitting the search flow more clearly into target, refine, finalize, and materialization stages.
- Added lightweight AUTO-mode profiling and separated Optuna telemetry helpers to make performance tuning and troubleshooting easier without changing the normal workflow.

### Architecture
- Introduced an `application/` service layer for run-request assembly, system-health checks, and house-curve loading so UI callbacks stay thinner and module responsibilities remain clearer.
- Removed more NiceGUI migration scaffolding and aligned the runtime entry flow around `src/camillafir/__main__.py`.

### Tests & Docs
- Expanded regression coverage around AUTO-mode selection, stereo leveling, summary/reporting polish, and filter-generation smoke cases.
- Updated README release notes and version references for `4.0.1`.

---

## [4.0.0] - 2026-03-29

### UI
- Replaced the legacy browser UI built on `PyWebIO` with a new `NiceGUI` frontend. The interface is now organized around dedicated NiceGUI tab builders, a shared control registry, NiceGUI-native callbacks, and updated run/results sections.
- Added NiceGUI-specific status/toast handling and refreshed results rendering to match the new frontend flow.

### Architecture
- Split the old monolithic UI and run orchestration into smaller modules such as `ng_app`, `ng_controls`, `ng_mode_controls`, `ng_results_sections`, `engine_build`, `engine_run`, `engine_summary`, and the `workflow/` helpers.
- Continued the auto-mode modularization by exposing the main runtime surface through a dedicated `auto_mode/` package instead of keeping the logic behind legacy import paths.
- Broke export/report generation into focused modules (`export_bundle`, `export_outputs`, `export_scoring`, `export_summary_text`) and added a bundled `User_Manual.md` for the new app layout.

### Tests & Packaging
- Added targeted regression coverage for the NiceGUI application, controls, mode handling, run section, export layout, packaging, and related UI helpers.
- Updated versioned docs/runtime packaging for the `4.0.0` release.

---

## [3.6.6] - 2026-03-29

### DSP / Internal
- Centralized DSP config and telemetry into dedicated adapters (`dsp_config`, `dsp_telemetry`, `mag_telemetry`, `phase_ir_contracts`) so all DSP modules share a single typed configuration surface.
- Added `target_synthesis` and `_pruning` DSP modules to separate target-curve synthesis and Optuna trial pruning from core correction logic.

### Automatic mode
- Updated Optuna legacy compatibility in cache signature handling so older stored study data is recognized correctly across optuna version changes.
- Split monolithic automatic-mode logic into focused orchestrator modules: `orchestrator_target`, `orchestrator_refine`, `orchestrator_finalize`, `optuna_backend`, `scoring_metrics`, `search_entrypoints`, `runtime_context`, `refine_eval`, `winner_polish`, `materialize`, and `protection_seed`.

### Architecture / Refactoring
- Extracted process execution from `camillafir.py` into a dedicated `workflow/` package (`process_run_flow`, `process_support`) to separate run orchestration from entry-point logic.
- Created a `common/` package for shared utilities: `acoustic_stats`, `comparison_stats`, `house_curves`, and `result_postprocess`.
- Split large UI files (`camillafir_ui`, `camillafir_ui_helpers`, `camillafir_plot`, `layout_sections`) into focused modules: `controls_ir_window`, `controls_mode`, `controls_target_preview`, `layout_builders`, `layout_theme`, `plot_generation`, `plot_summary`, `process_run_bridge`, `results_formatters`, `results_sections`, and `ui_state`.
- Added `mode_policy` to the config layer for centralized Basic/Advanced mode policy decisions.

### Tests
- Added tests for DSP config reader, mag post-limits pipeline, phase IR contracts, and filter-pair smoke checks.

### Build & Docs
- Updated release build workflow to Node 24.
- Refreshed `guide.md` with expanded content and updated version references.

---

## [3.6.5] - 2026-03-23

### Automatic mode / Persistence
- Added a dedicated AUTO-mode compatibility version so persistent cache and Optuna study data can be separated from the main program version when persistence compatibility changes.
- Updated automatic-mode cache and Optuna storage paths to use compatibility-version specific filenames when needed, with automatic migration from legacy/unversioned files.
- Hardened AUTO-mode persistence loading so incompatible stored data is ignored safely instead of being mixed into a newer compatibility format.

### UI & Config
- Kept AUTO-mode compatibility metadata transient at runtime so it is not written into normal user config files.
- Locked AUTO-mode TDC and A-FDW controls more safely in the UI to prevent unsupported preset/control combinations during automatic runs.
- Continued the START-button layout cleanup by keeping the action in its own clearer section.

### Docs
- Updated documentation version references for the `3.6.5` release.

---

## [3.6.4] - 2026-03-18

### Automatic mode
- Added a new automatic-mode goal `subwoofers`.
- `subwoofers` behaves like the other AUTO goals, but forces the Smart Scan leveling range to `20-200 Hz`.
- Kept `subwoofers` in the local-refine flow so the new goal still uses the normal automatic-mode search/refinement path.

### UI & Docs
- Added the `subwoofers` goal into the automatic-mode goal selector.
- Updated automatic-mode preview/reporting so the forced `20-200 Hz` Smart Scan window is reflected consistently.
- Refreshed docs version references for the `3.6.4` release and documented the new `subwoofers` goal in the README.

---

## [3.6.3] - 2026-03-16

### Automatic mode
- Added a residual-pass tie-break stage for near-top finalists so AUTO mode can re-check close winners with residual processing before locking the final preset.
- Added winner-polish passes for `phase_limit` and `mag_c_min`, allowing the selected winner to be fine-tuned after the main search when that improves the final ranking.
- Separated cache-ready winner preset data from the fully applied/materialized preset so final UI/export metadata reflects the actual applied winning settings.

### Reporting & UI
- Added phase-limit and `mag_c_min` winner-polish details to the automatic-mode summary export.
- Updated automatic-mode best-preset reporting in UI/export so `mixed_freq` is shown only for mixed filters and `phase_limit` only for linear/asymmetric filters.
- Routed final AUTO results through the materialized applied preset so winner details stay consistent across results, summary export, and saved metadata.

### Plots & Docs
- Improved prediction-plot low-confidence highlighting by merging short gaps and filtering noisy confidence segments before drawing shaded regions.
- Prefer the bundled local Plotly asset when available, falling back to CDN only when needed.
- Cleaned up README structure by removing outdated version-highlight sections and moving the inspiration section to the documentation area.

---

## [3.6.2] - 2026-03-13

### DSP / Stereo link
- Improved stereo-linked leveling with a shared window search and quieter-channel anchoring so linked offset selection stays more stable between channels.
- Exposed stereo leveling window and anchor details in the UI for easier inspection of linked leveling behavior.

### Plots
- Updated delay-compensated filter plots to remove IR peak delay from displayed phase/group-delay views and improved the scaling of those plots for clearer reading.

### Automatic mode / Reporting
- Unified automatic-mode winner score reporting around a single official winner score so summary/export/UI no longer show conflicting `rank_score` values for the same run.
- Added structured winner score component metadata for reporting while keeping the existing optimizer selection behavior unchanged.
- Hardened pre-energy metric handling so unreliable cases remain explicitly unavailable instead of looking like internal calculation failures.
- Polished summary wording for unavailable pre-energy metrics and cleaned up duplicate RT60 band entries in report output.

### Tests
- Extended stereo-linked leveling tests.
- Added coverage for winner score reporting consistency, pre-energy summary formatting, and RT60 band deduplication in reports.

---

## [3.6.1] - 2026-03-12

### Versioning & Export
- Centralized program version resolution in `src/camillafir/version.py` so UI/build version text stays consistent across runtime, packaging, and export paths.
- Added program version and automatic-mode winner rank to exported ZIP/YAML filenames and CamillaDSP YAML titles for easier result identification.
- Included automatic-mode winner rank in the chosen-target/export metadata shown after the run.

### DSP / Leveling
- Reworked leveling window evaluation to use log-balanced medians and more robust window scoring, improving stable offset selection when narrow nulls or drifting sub-window offsets are present.
- Added extra guards so `trans_width=None` no longer breaks the correction fade stage.

### Config
- Normalized persisted `config.json` filter type names back to canonical UI labels such as `Asymmetric`, fixing legacy/internal `asym` naming leakage in saved configs.

## [3.6.0] - 2026-03-12

### Automatic mode
- AUTO mode now keeps the user's HPF selection available instead of forcing HPF off.
- Response-fit HPF is only auto-applied when the user has explicitly enabled HPF; otherwise the estimated HPF is reported as a suggestion without changing the run.

### DSP / Stereo link
- Stabilized stereo-linked target-shift sharing between left and right channels so linked leveling stays more consistent across channels.

### Export
- Added runtime version details (`Python`, `numpy`, `scipy`, `optuna`) to the automatic-mode summary export.

### UI & Docs
- Updated automatic-mode help text to mention the optional HPF selection workflow.

---

## [3.5.9] - 2026-03-12

### Packaging
- Added Optuna submodule collection to the standalone, Linux, and macOS PyInstaller spec files so packaged builds include the automatic-mode Optuna backend more reliably.

### Export
- Added `Optimizer backend` to automatic-mode summary export so saved results now show whether the run used `builtin` or `optuna`.

---

## [3.5.8] - 2026-03-11

### Automatic mode
- Improved phase-1 Optuna search so trial parameters now match the actual evaluated preset values more closely, reducing duplicate/mismatch noise in the sampler model.
- Canonicalized Optuna preset parameter handling for persistent studies, improving duplicate detection and reuse of previously evaluated trials across runs.
- Updated remembered/evaluated trial insertion to prefer Optuna completed-trial style recording before falling back to the older enqueue/ask/tell path.
- Added `max_slope_boost_db_per_oct`, `max_slope_cut_db_per_oct`, and `conf_pull_max_hz` to the main phase-1/global Optuna search space.
- Kept those new secondary parameters restricted to the main/global search after testing showed that letting later refine or micro phases retune them reduced result quality.
- Reduced default phase-3 micro trial count and tightened the micro search shrink range so more of the useful optimization happens earlier in the pipeline.

### Export
- Expanded automatic-mode winner summary export so the additional phase-1/global Optuna parameters are visible in the saved best-preset text.

---

## [3.5.7] - 2026-03-10

- Optuna optimization

---

## [3.5.6] - 2026-03-08

### UI & Results
- Cleared previous results and plots immediately when starting a new run so stale output no longer remains visible during processing.
- Reworked the results view into clearer collapsible sections for acoustic summary, gain/headroom, filter setup, and phase/crossover diagnostics.
- Expanded automatic-mode results with a dedicated winner summary, best-preset table, target-curve top-3 table, and rank-ordered top-5 comparison.
- Moved the completion message into status notices for a cleaner final status presentation.

### Automatic mode
- Fixed exact cache-hit micro-refine finalization so recalculated winner metrics and preset data stay synchronized before saving cache/output metadata.
- Adjusted excursion-penalty waiving so the zero-penalty floor is also honored when automatic excursion protection lands exactly on that floor.

### Multi-rate & Runtime
- Fixed automatic tap scaling for higher sample rates by scaling directly from the selected 44.1 kHz base tap count, preserving FIR time length more accurately in multi-rate mode.
- Updated the automatic multi-rate tap info panel to reflect the current base tap setting instead of fixed default values.
- Reduced default console noise by lowering normal console logging to `WARNING`, while keeping opt-in debug logging available through environment variables.
- Hardened PyWebIO pin access and routed preview/visualization failures through the logger instead of noisy direct console prints.

---

## [3.5.5] - 2026-03-07

### Automatic mode
- Added exact cache-hit reuse for same measurements + settings in both target-curve selection and preset search, so redundant target trials can be skipped when a matching cached winner already exists.
- Expanded Optuna-backed candidate generation across main search, local refine, micro-refine, and target-curve trials while preserving seeded presets inside the study.
- Stored `best_metrics` together with cached presets/targets so cached winners keep their ranking context for later reporting and export summaries.
- Improved target-selection reporting with clearer method labels (`cache hit`, `trial comparison`, `cache wildcard`, etc.) in status text, logs, and export metadata.
- Kept automatic excursion protection seed/final frequency tracking visible in export summaries.

### UI & Defaults
- Simplified automatic-mode result presentation by relying on the winner explanation panel instead of repeating the same winner metrics block.
- Trimmed the automatic-mode top-5 table to emphasize the core comparison columns.
- Updated startup defaults to use `Asymmetric` as the default filter type.
- Added default Optuna sampler settings to config (`multivariate`, `group`, `constant_liar`).

### Docs & Export
- Extended export summary text with selected target-curve method, target-selection grid details, and cached ranking metadata.
- Refreshed documentation version references for the 3.5.5 release.

---

## [3.5.4] - 2026-03-07

### Automatic mode
- Refactored automatic-mode code into smaller modules to improve maintainability and make future changes safer.
- Split automatic-mode logic into clearer parts without changing optimization behavior.

### Notes
- No DSP changes in this release.

---

## [3.5.3] - 2026-03-06

### Automatic mode
- Added adaptive target-curve preselection scoring (fit, boost demand, slope match, asymmetry, mode-band fit).
- Expanded target shortlist logic with spread-based sizing, optional cache wildcard insertion, and milder-step inclusion guards.
- Added optional Optuna pilot sampler for phase-1 candidate generation with automatic fallback to built-in sampler.
- Added `AutoModeConfig` overrides for auto-mode search tuning (trials, refine, hard-gate, micro phase, optuna settings).
- Improved phase-1/phase-2 candidate generation so `mag_c_min` and `low_bass_cut_hz` are actively optimized (also in local refine).
- Added auto-tuned excursion protection finalization from measured zero-penalty floor and propagated seed/final excursion metadata.

### DSP / Scoring
- Added `boost_candidate_min_hz` metric to correction outputs and run stats for auto-mode protection heuristics.
- Updated auto-mode scoring to derive `auto_exc_zero_penalty_hz` and waive excursion-bin penalty component when applicable.

### UI
- Improved automatic-mode status UX with compact phase text and a collapsible `Automatic mode details` history panel.
- Reset automatic-mode status details at run start to avoid stale detail carry-over.
- Added `Target curve top-3` results table (best rank, avg rank, fit RMS, preselect and penalty metrics, trials).

### Export
- Added automatic-mode excursion protection summary line (`seed -> final` frequency) to export summary metadata.

### Dependencies & Docs
- Added `optuna` to `requirements.txt` and `requirements-linux.txt`.
- Updated docs version references to `v3.5.3` in README and official manual.

### Tests
- Expanded automatic-mode tests for config overrides, optimizer backend selection, candidate variability, optional Optuna backend, and excursion-penalty waiving behavior.

---

## [3.5.2] - 2026-03-05

### Automatic mode
- Improved automatic mode performance to complete optimization faster.

---

## [3.5.1] - 2026-03-05

### Automatic mode
- Added target-curve selection mode.
- Fixed AUTO-only control state handling.

### UI
- Fixed status timer flicker by updating status text in place.
- Replaced per-second full scope re-render in `update_status()` with a persistent status DOM node and JS `textContent` updates.
- Kept clear/re-render as fallback if `run_js` is unavailable or fails.

### Paths & Export
- Added shared path helpers in `app_paths.py`.
- Unified default export folder to `Documents/CamillaFIR/filters/<version>` on all platforms.
- Kept writable/safe fallback guards for export directory resolution.
- Included program version in export ZIP filename.
- Passed program version into `save_export_bundle()` for versioned output paths.
- Exposed and showed active automatic-mode cache path in results UI.
- Migrated automatic-mode cache path handling to platform app-data with legacy fallback/migration.
- Localized UI labels (`Paths`, `Item`, `Path`, `Export folder`, `Automatic mode cache`) in EN/FI.

---

## [3.5.0] - 2026-03-04

### Automatic mode (core)
- Added deterministic seeding based on *measurements + key settings* so auto-mode trials are reproducible across runs (same input -> same trial sequence). 
- Upgraded auto-mode cache schema to `v=3` and made it **filter-type aware** (separate buckets for `linear`, `mixed`, `minimum`, `asym`) to prevent cross-filter contamination.
- Added cache **version mismatch** handling: if cached program version differs, cache is ignored and a fresh search is run (with a clear log line).
- Added filter-specific **last-used preset** fallback seeding (per `auto_goal`) when a signature match is not found.

### Automatic mode (search / scoring)
- Added **Phase2 hard-gate** (pre-Pareto) to remove obvious bad candidates by event severity / ripple before building the Pareto front; improves consistency on difficult data without adding trials.
- Added **adaptive search-space shrinking** (derived from phase-1 stability) for phase-2 local refinement, plus an optional **phase-3 micro-refine** pass around the best anchors.
- Added optional **dual-mode detection** (two dominant LF resonances) and mode-ripple-aware scoring/penalties so secondary room modes don’t slip through.

---

## [3.4.2] - 2026-03-03

### Core
- Added persistent automatic-mode cache in `~/.camillafir/camillafir_auto_mode_cache.json` for cross-run reuse of best presets.
- Automatic mode now stores selected target curve in cache (`best_target_curve` / `best_hc_mode`) together with best preset metadata.
- Added measurement-signature based target cache map (`target_by_measurement`) so identical measurements can reuse previous target choice and skip repeated target-curve trial optimization.
- Extended cache entries with `measurement_sig` and bumped cache schema marker to `v=2`.
- Added automatic HPF estimation from measured speaker LF response (AUTO mode), including frequency/slope fit with confidence gating.
- Tuned automatic HPF fitting robustness for real-room responses (trimmed residual fit + conservative slope fallback near LF floor), improving auto-enable and reducing false low-confidence outcomes.

### Target Selection
- Added optional one-step milder target preference for built-in bass ladders (`Harman*`, `BK_*`) to avoid consistently biasing toward the bass-heaviest curve.
- Milder-step switch is guarded by quality thresholds (`rank` drop and `fit_rms` increase limits), so the selected milder curve is used only when quality remains close.
- Added data/config toggle `auto_target_prefer_milder_step` (default enabled via constant) for controlling this behavior.

---

## [3.4.1] - 2026-03-02

### Core
- Upgraded `CamillaFIR automatic mode` to a two-phase search: phase 1 for broad random exploration, phase 2 for local refinement around top candidates with narrower variation.
- Added early-stop logic for non-improving runs to reduce wasted trials and move/finish phases faster.
- Added target preselection (`top-3`, `10 trials/curve`) before main auto optimization.
- If the user selects a custom target curve, automatic built-in target comparison is skipped and the selected target is used directly.
- Added carry-over seeding so the best target-selection preset can initialize main phase-1 search.
- Added automatic LF `-6 dB` estimation from smoothed response data and used it to drive auto protection/correction defaults.
- Added `Harman12` and separate built-in B&K target variants (`BK_Light`, `BK_Medium`, `BK_Strong`) for broader automatic target coverage.

### Scoring
- Improved automatic ranking robustness to avoid `0.000/100` edge cases (notably in Minimum Phase workflows).
- Updated boost penalty behavior so penalty starts above `3 dB` net boost.
- Added/expanded severity-aware DSP event handling in ranking metadata.
- Excursion penalty is now waived when `exc_freq` is auto-derived, so automatic protection frequency selection does not self-penalize.

### UI
- Automatic mode progress text now updates on every trial and shows richer context: selected target, estimated `-6 dB` point, derived `low_bass_cut_hz`, derived `exc_freq`, phase/trial counters, best score, and elapsed time.
- Improved auto-result transparency by exposing winner metrics (rank components and penalties) in top results view.

### Export
- Summary export now includes richer automatic-mode metadata: target selection method and candidates, chosen target and seed preset details, and ranking component breakdown (acoustic average, DSP penalty, excursion penalty, boost, events, severity).

---

## [3.4.0] - 2026-03-02

### Core
- Added `CamillaFIR automatic mode` as a first-class mode (`AUTO`) in the mode system.
- Automatic mode executes 100 trial presets, ranks results, and applies the best preset before final export.
- Trial status text now updates on every run (`1/100`, `2/100`, ...), improving optimization visibility.
- Automatic mode ranking combines acoustic score with DSP quality penalties and stability factors.

### UI
- Added `CamillaFIR automatic mode` into the main mode selector next to `BASIC` and `ADVANCED`.
- Removed separate automatic-mode checkbox from Basic page (mode selector is now the single control).
- Set automatic mode as default mode in UI/config flow.
- Updated mode descriptions and guidance text to a three-mode model: Automatic, BASIC, ADVANCED.

### Config / Compatibility
- Added compatibility mapping between legacy `camillafir_automatic_mode` boolean and new `mode=AUTO` behavior.
- `AUTO` now follows BASIC-style safety policy for guarded controls while running automatic search.

### Export
- Summary export includes automatic-mode metadata (trial counts, best rank score, and selected best preset values).

### Docs
- Updated `docs/README.md` to feature `CamillaFIR automatic mode` as the main v3.4.0 highlight.
- Updated EN/FI translation guide texts for the three-mode workflow.

---

## [3.3.1] - 2026-03-01

### Core
- Add unsafe_raw_dsp checkbox visible only in ADVANCED mode
  - Add bilingual warning text: FOR TEST USE ONLY / VAIN TESTI KÄYTTÖÖN
  - Wire unsafe_raw_dsp through config/model/pipeline persistence
  - In engine, enable true guard bypass in ADVANCED + unsafe_raw_dsp:
  - bypass MAX_SAFE_BOOST cap and disable major correction guard rails
  - Force unsafe_raw_dsp=false in BASIC mode

---

## [3.3.0] - 2026-03-01

### Core
- Bumped app version to `v.3.3.0`.
- Refactored `camillafir.py` into smaller modules to improve maintainability and code manageability.
- Refactored `camillafir_dsp.py` into smaller modules to improve maintainability and code manageability.

### Docs
- Updated documentation version references to `v3.3.0`.
- Updated citation metadata version to `3.3.0`.

---

## [3.2.0] - 2026-02-23

### Core
- Bumped app version to `v.3.2.0`.

### Docs
- Updated documentation version references to `v3.2.0`.
- Updated citation metadata version to `3.2.0`.

---

## [3.1.1.2] - 2026-02-22

### Build / CI
- Added startup smoke tests to release workflow for all build targets:
  - Linux (`dist/CamillaFIR/CamillaFIR` -> `http://127.0.0.1:8080`)
  - Windows (`dist/CamillaFIR/CamillaFIR.exe` -> `http://127.0.0.1:8080`)
  - macOS arm64 (`CamillaFIR.app` binary -> `http://127.0.0.1:8080`)
  - macOS x86_64 (Rosetta run -> `http://127.0.0.1:8080`)
- Release pipeline now fails early if an artifact builds but does not actually start a web server.
- Added failure log tail output in smoke tests to improve diagnostics in GitHub Actions.

### Core
- Bumped app version to `v.3.1.1.2`.

---

## [3.1.1.1] - 2026-02-22

### UI
- Fixed Chromium/Brave chart rendering regression (`Plotly is not defined`) by restoring self-contained embedded Plotly rendering for dashboard and Target Preview
- Fixed Linux chart visibility issue by avoiding fragile Plotly loader mode in embedded UI charts
- Fixed Target Preview measurement smoothing input mapping to use `filter_smooth` (with legacy fallback to `smoothing_level`)
- Added live Target Preview refresh on `filter_smooth` changes

### Build
- Added `pywebio.platform.tornado_websocket` to Linux PyInstaller hidden imports for better websocket backend compatibility

---

## [3.1.1] - 2026-02-22

### DSP
- Added WAV-only ripple attenuation around correction upper-edge transition in correction + phase stages, plus final FIR IR post-polish while preserving phase and peak scale
- Added alignment guard between `delay_samples` and impulse-peak alignment; falls back to peak alignment when estimates disagree significantly
- Aligned WAV parsing closer to TXT baseline with deterministic window/smoothing policy and higher minimum FFT resolution for more consistent results

### IO
- Hardened local path handling by cleaning quoted paths and validating local WAV existence before parsing
- Fixed WAV parser import fallback behavior and added configurable `min_n_fft` support in WAV IR -> FR conversion
- Updated WAV source detection to rely on actual input sources/extensions instead of UI format selection

### UI
- Added elapsed-time status updates and a run timing breakdown table (Read, DSP, ZIP/PNG, Render, Total, per sample rate)
- Improved completion status to show the real output directory path
- Centralized toast behavior through System Health helpers (dedupe + edge-trigger warnings for max boost/taps and preset toasts)

### System Health
- Added explicit L/R measurement source checks (upload/local pair completeness) with dedicated warnings
- Added phase-limit warning when correction limit is set high (above ~800 Hz)
- Improved health gate notifications with centralized, deduplicated toast summaries

### Export & Docs
- Added version stamp to generated summary output (`Version: v.3.1.1`)
- Updated ZIP dashboard PNG path to use combined Matplotlib export when dashboard image export is enabled
- Updated docs/app version references to `v3.1.1` and refreshed EN/FI guide + health translation texts

---

## [3.1.0] - 2026-02-21

### DSP
- Reworked Mixed Phase excess-phase correction with configurable LF-to-HF fade (`low_freq_full_correction_hz` -> `high_freq_no_correction_hz`) and tunable strength (`excess_phase_strength`) via config only (no UI controls)
- Added Mixed-only safety guards for excess correction: absolute excess-delay limiter (`max_excess_delay_ms`) and pre-ringing limiter (`max_pre_ringing_db`) with adaptive down-scaling and diagnostics
- Forced Mixed filters to a fixed startup peak position (90 ms) after window/DC steps for more consistent startup timing
- Changed Manual leveling reference to neutral `0.0 dB` (instead of legacy `75 dB`) and applied manual bias explicitly in correction target shaping

### UI
- Expanded Target Preview to optionally overlay speaker curves (L/R/avg), align them to level window, and show correction band + level-window context
- Improved leveling UX: BASIC now behaves as Smart Scan only, Auto/Manual level ranges are preserved separately, and Manual dB has quick `- / +` step buttons with a neutral-bias hint
- Added dynamic BASIC clamp hints into field help texts so constrained controls clearly show active guard rails
- Refined mode guard rails/defaults (including tighter BASIC boost/phase/FDW limits, forced BASIC stereo-link, and updated default Mixed split/IR-left values)
- Localized System Health issue texts and summary headers through EN/FI translation keys

### Core
- Pipeline now enforces BASIC policy for leveling (`lvl_mode=Auto`) and reapplies mode clamps at run time
- Added config/pipeline model support for Mixed excess-phase tuning fields (`excess_phase_strength`, correction fade range, pre-ringing cap, excess-delay cap`) as config-only parameters (not exposed in UI)
- Added legacy config migration for manual leveling values saved in old absolute-dB style (auto-converted around `~75 dB` baseline)

### Docs
- Bumped docs version references to `v3.1.0`

---

## [3.0.6] - 2026-02-20

### DSP
- Added true fractional-octave gain smoothing (`1/N` on log-frequency axis) and switched non-DF smoothing to use it in correction + phase residual shaping
- Added post-shape filter smoothing and clamp-aware final smoothing so `filter_smooth` / `reg_strength` remain visible even when clamp rails are active
- Added clamp-dominance diagnostics (`NONE`/`LOW`/`MEDIUM`/`HIGH`) with clipped-bin ratio stats for easier tuning and A/B interpretation
- Refined Smart Scan leveling window search to be target-independent and ripple-focused (log-frequency detrended stability scoring)
- Hardened TDC control handling: explicit zero support, safe clamping for `tdc_strength`, and no-op behavior when strength/max reduction are disabled
- Kept theoretical + minimum phase paths unclamped and limited only excess-phase correction with adaptive per-bin clamp (`15..45 deg`) based on confidence and frequency
- Improved Minimum+OFF IR export behavior with zero-padded causal peak shift, optional strict-off forcing, short anti-wrap tail taper, and DC-removal skip in OFF mode
- Reworked output gain handling to auto-level model (`auto_global_gain = -(realized max boost + margin)`) with normalize left as optional extra safety trim
- Added stereo-link shared auto gain override so left/right channels use one common attenuation in linked processing

### UI
- Clarified Filter Type help text and Asymmetric note to separate causal filter-type behavior from IR export windowing behavior
- Improved help readability by separating base filter-type help and Asymmetric note with a clear line break
- Renamed `gain` control to Auto Headroom Margin and surfaced margin/applied auto gain/headroom/final max diagnostics in results + summary output
- Removed manual align toggle from UI (time alignment is now always-on by policy) and kept only stereo-link toggle
- Added plot-only auto-gain compensation so predicted/filter curves remain visually comparable despite output attenuation
- Localized multiple section headers and target preview source labels (EN/FI)

### Core
- Updated pipeline parsing to sanitize and clamp `tdc_strength` before passing configuration forward
- Pipeline now forces `align_opt=True`, clamps gain margin to non-negative values, and maps UI `gain` to `auto_gain_margin_db`
- YAML export now writes `master_gain_db=0.0` and relies on DSP auto-gain path for output level management

### Docs
- Bumped docs version references to `v3.0.6` and moved UI screenshots/TDC image paths under `docs/pics/`

---

## [3.0.5] - 2026-02-18

### DSP
- Removed unused imports
- Changed A-FDW smoothing 2/3 --> 1/3 to make more precise correction on bass
- Fixed summary export fallback error handling (`except Exception as e`) in DSP effective params block

### UI
- Clarified summary.txt content
- Added an Executive Summary section (Score, Match, Confidence, RT60, Worst Event) at the top
- Reduced settings noise by showing only core settings in Summary
- Removed duplicate acoustic events section from Summary
- Standardized Summary output to English-only wording
- Replaced symbol-heavy labels with plain text (`dt`, `Path delta`, `n/a`) for cleaner cross-platform text output

---

## [3.0.4] - 2026-02-18

### DSP
- Refactored DSP pipeline into focused modules for preprocessing, correction, phase/IR, and shared utilities
  - Improves maintainability and separation of concerns
  - Preserves existing processing flow while reducing monolithic engine complexity

- Reworked Mixed Phase blend implementation
  - Replaced crossover-FIR convolution blend with frequency-domain blending
  - Added smooth raised-cosine transition around split frequency
  - Added zero-padded peak alignment (no circular wrap-around)
  - Reduces blend artifacts and improves transition stability

### UI
- Added Mixed Phase blend diagnostics to DSP info
  - Shows per-channel blend split frequency
  - Shows per-channel blend transition width

- Expanded system health checks
  - Adds sanity checks for leveling range, HPF settings, and Mixed Phase parameters
  - Handles latency messaging for Min/Asym low-latency modes
  - Improves correction-range and safety messaging clarity

### Core
- Updated default IR window fallbacks when UI values are missing
  - `ir_window`: 500 ms
  - `ir_window_left`: 120 ms

---

## [3.0.3] - 2026-02-16

### UI
- Added warnings if correction range is too wide

### macOS
- Switched to universal2 build (Intel + Apple Silicon support)
- Fixes "Bad CPU type in executable" on older Intel Macs

---

## [3.0.2] - 2026-02-15

### UI
- Simplified and reorganized UI structure for clearer workflow
- Added real-time house curve preview
- Added project logo to the interface

---

## [3.0.1] - 2026-02-14

### DSP
- GD-gradient limiter redesigned
  - Bass-focused (20–250 Hz)
  - Soft limiting (tanh) replaces hard clipping
  - Relaxed slope limit: 8 → 30 ms/oct
  - Conditionally enabled (bypassed when A-FDW + Bass-first stabilization are active, except in high-risk windowing modes)

### UI
- Fixed A-FDW plot rendering issue with measurement sample rates >48 kHz

---

## [3.0.0] - 2026-02-11

Major DSP engine update and smoothing redesign.

### UI
- Renamed *Psychoacoustic* smoothing to **CamillaFIR Reference**
  - Clarifies that this smoothing is a CamillaFIR-specific reference/safety view
  - Not equivalent to REW-style psychoacoustic smoothing
  - View-only (does not affect DSP calculation)

- Added user-configurable parameters for **Confidence Pull**
  - Available in Advanced mode only
  - Floor threshold
  - Max frequency limit
  - Cut aggressiveness
  - Boost conservativeness
  - Recommended defaults included in help text

### DSP
- Replaced legacy fixed 1/24 octave smoothing in filter calculation
  - New adaptive smoothing:
    - 0–230 Hz: 1/48 octave
    - 230–500 Hz: gradual transition
    - >500 Hz: 1/3 octave
  - Improves LF precision while stabilizing mid/high frequency correction

- Improved Bass-first behavior
  - Better alignment with confidence weighting
  - More predictable low-frequency correction shaping

- Refined confidence pull handling
  - More stable behavior in low-confidence regions
  - Reduced over-aggressive boosts in uncertain bands

### Build
- Linux build switched to `onedir` distribution
  - Improves portability and runtime reliability
  - Avoids common single-file extraction issues

### Behavior changes
- Adjusted default Confidence Floor (0.15 → 0.07)
  - Correction operates more freely before safety pull engages

---

## [2.9.5] - 2026-02-08

### UI
- **Low-bass cut is now toggleable (ON/OFF)**
  - Added an enable checkbox for `low_bass_cut_hz`.
  - When disabled, the Hz input remains visible but is **greyed out (locked)** and cannot be edited.
  - Disabled state uses an **empty value** for the cutoff field to represent “off” (instead of `None`),
    improving UI → pipeline compatibility.
  - UI logic lives in a dedicated helper (`update_low_bass_cut_ui`) and renders inside a scope for clean updates.

---

## [2.9.4] - 2026-02-08

### CFG
-  Fixed low_bass_cut_hz value not saving correctly in config.

---

## [2.9.3] - 2026-02-08

### UI
-  Fixed typo at psychoacoustic plot smoothing code (1/48 / 1/3 ---> 1/6 / 1/3)

---

## [2.9.2] - 2026-02-07

### UI
- **IR windowing hard restrictions clarified and enforced**
  - In **Basic mode**, IR export windowing is now **always forced to Auto**.
    - The windowing mode selector is locked and cannot be changed.
    - A persistent warning message is shown to explain the restriction.
  - When **Filter type = Asymmetric**, IR export windowing is also **locked to Auto**
    in both Basic and Advanced modes.
  - Prevents confusing transient UI states where windowing controls appeared
    editable even though the value was internally forced back to Auto.
  - Ensures UI behavior always matches DSP policy and exported configuration.

---

## [2.9.1] - 2026-02-06

### UI
- **Mixed Phase crossover frequency is now state-dependent**
  - The “Mixed Phase crossover frequency (Hz)” field is active **only** when **Filter type = Mixed Phase**.
  - For other filter types, the field remains visible but is **greyed out (locked)** to prevent invalid configurations.
- **REW Asymmetric IR windowing restricted to Linear Phase filters**
  - The “Asymmetric” IR windowing option (REW Asymmetric export) is selectable **only** when **Filter type = Linear Phase**.
  - For other filter types, the option is shown but **greyed out**, and the UI displays a warning message:
    - `WORKS ONLY WITH LINEAR PHASE FILTERS`

---

## [2.9.0] - 2026-02-06

### UI
- **Windowing mode simplification**
  - Removed **“Symmetric”** and **“Off”** windowing modes.
  - Windowing now offers only:
    - **Auto** – REW-based, adaptive windowing selected automatically from the impulse response.
    - **Asymmetric** – REW-based asymmetric windowing with optional latency reduction.
  - Simplifies the UI and focuses on the most effective and reliable windowing strategies.

### DSP
- **High-pass filter (HPF) magnitude fix**
  - Fixed HPF handling so it is applied as a **true magnitude filter** in the FIR path.
  - HPF is now applied directly to the correction curve (**`gain_db += hpf_db`**),
    instead of being baked into the target response.
  - Ensures **magnitude and phase consistency**.
  - Prevents double-HPF behavior, incorrect low-frequency response,
    and artificial group delay artifacts when HPF is enabled.


---

## [2.8.9] - 2026-02-05

### DSP
- **REW Asymmetric low-latency bass safety**
  - Added automatic safety guards for ultra–low-latency REW Asymmetric mode.
  - When latency target (Left window) is **below 15 ms**:
    - Bass-first (A-FDW confidence shaping) is automatically limited to low frequencies.
  - When latency target is **below 10 ms**:
    - Low-frequency **boosts are disabled** (cuts remain allowed).
  - Prevents unstable bass behavior, excessive ripple, and aggressive FIR boosts
    when time-domain constraints become too tight.
  - Safeguards are automatic, non-configurable, and only active in REW Asymmetric mode.


---

## [2.8.8] - 2026-02-04

### DSP
- **Phase correction safety clamp (±45°)**
  - Room / excess-phase correction is now internally limited to ±45 degrees.
  - The clamp is applied **only to the correction component** (measured − target phase),
    never to loudspeaker minimum-phase or theoretical crossover phase.
  - Prevents excessive phase rotations, pre-ringing, and unstable group delay behavior,
    especially in low-confidence or sparsely measured regions.
  - Improves robustness, repeatability, and subjective transient clarity.
  - No user-facing control; this is a fixed safety default.

### Analysis & Reporting
- Phase correction clamp status is now always reported:
  - Logged during processing (e.g. `max=54.5° -> 45.0°`).
  - Included in `summary.txt` per channel.
  - Shown in **DSP info** section in the UI.

### UI
- Removed slope-limit envelope visualization from magnitude plots.
  - Eliminates confusing shaded artifacts without affecting DSP behavior.

### Notes
- No changes to magnitude targets, A-FDW, TDC, leveling, or IR export behavior.
- Existing presets and workflows remain fully compatible.

---

## [2.8.7] - 2026-02-04

### Fixed
- Psychoacoustic smoothing corrected.
- IR windowing no longer affects filter level.

### UI
- Updated UI text strings and help descriptions.
- Little update to interface look
---

## [2.8.6] - 2026-02-01
 
 ### Changed
- Refactored DSP-related code into a clearer, more modular file structure.
- Separated DSP logic from UI and orchestration layers to improve maintainability.
- Clarified responsibility boundaries between filter generation, windowing, and export logic.
- No functional changes to DSP algorithms or generated filters.

### Internal
- DSP modules are now organized to allow easier future extensions.
- Reduced implicit cross-dependencies between DSP and UI code.
- Improved long-term stability by making DSP behavior less sensitive to UI-side changes.

---

## [2.8.5] - 2026-02-01

### DSP / IR export
- **IR export window edge shape selection (Hann / Tukey)**
  - Added support for selecting IR window edge shape during FIR export.
  - Tukey window includes adjustable alpha parameter (0–1, default 0.25).
  - Window shape is applied only at IR export stage (WAV generation).

- **REW-style asymmetric export placement fix**
  - When `rew_asym` is selected, FIR impulse peak is placed causally
    (early in the impulse) instead of remaining centered.
  - Reduces effective playback latency compared to symmetric placement.

### UI / Pipeline
- **IR window shape & alpha preserved through UI → pipeline → DSP**
  - Prevents silent fallback to legacy Hann window.
  - Selected window shape and alpha are logged immediately after UI collection
    for traceability.

### Notes
- No changes to FIR magnitude targets, phase correction algorithms,
  A-FDW, TDC, or auto-leveling behavior.
- Differences between Hann and Tukey exports are sample-accurate
  and produce non-identical FIR WAV files.

---

## [2.8.4] - 2026-02-01

### Misc
- Github actions now makes running files.

---

## [2.8.3] - 2026-01-31

### DSP update
- **REW-style IR windowing enabled in DSP export path**
  - Adds support for REW-compatible symmetric and asymmetric IR windowing during FIR export.
  - Windowing is applied **only at IR export stage** (WAV generation), not during correction, target fitting, leveling, or scoring.
  - Supported modes:
    - `auto` – automatic window selection (default)
    - `off` – no IR windowing
    - `rew_sym` – REW-style symmetric window
    - `rew_asym` – REW-style asymmetric (causal) window

### IO update
- **IR windowing type is now included in exported filenames**
  - ZIP and FIR WAV filenames include a short windowing tag for traceability and A/B comparisons.
  - Tags: `auto`, `off`, `sym`, `asym`
  - Example: `CamillaFIR_<type>_sym_<timestamp>.zip`
  - Example: `L_<type>_<fs>Hz_<timestamp>_sym.wav`

### Notes
- No change to correction targets, phase modes, FDW, TDC, or leveling behavior.
- This update improves reproducibility and comparability between different IR export strategies.

---

## [2.8.2.3] - 2026-01-30

### Io-update
- Fixed ZIP output when multi-rate is enabled. Generate a single CamillaDSP .yml using $samplerate$

### DSP-update
- More precise leveling tilt used in magnitude calculation

---

## [2.8.2.2] - 2026-01-29

### Ui-update
- updated translations and phase plot to be more clear

---

## [2.8.2.1] - 2026-01-28

## Nothing changed DSP or UI
- changed file structure to more debug-friendly format
- filters go now to ./filters directory

---

## [2.8.2] - 2026-01-27

### Ui-update
- improved robustness of file upload parsing from browser & added xo_help translation

---

## [v2.8.1.2] - 2026-01-27   

### Fixed
- Bug fix for modes selection, that was not saving ui state correctly

---

## [2.8.1.1] - 2026-01-27

### Ui-update
- Added modes selection (Basic & Advanced)

---

## [2.8.1] – 2026-01-25

### Fixed
- **A-FDW bandwidth limits**
  - Corrected incorrect or overly permissive A-FDW bandwidth constraints.
  - Prevents misleading smoothing widths and improves consistency between analysis and visualization.

### Improved
- **A-FDW & TDC guides**
  - Plot guides and annotations now reflect the *effective* (clamped) A-FDW bandwidth.
  - Improves interpretability of confidence masking and decay-based correction limits.

### Notes
- No changes to FIR magnitude targets, phase correction modes, or leveling behavior.
- Maintenance update focused on analysis clarity and safety transparency.

---

## [2.8.0] - 2026-01-24

- **Plot export robustness (ZIP outputs)**
  - Fixed a broken Plotly PNG export path caused by an invalid `try/except` structure.
  - ZIP exports now store dashboard plots as static PNG images generated via Plotly’s native Kaleido backend.
  - Ensures exported plots exactly match the HTML dashboard “Download plot as PNG” output.
  - Eliminates dependency on `.html` files and local `plotly.min.js` for offline ZIP viewing.

---

## [2.7.9] – 2026-01-24

### Fixed
- **Custom house curve upload**
  - Fixed an issue where user-uploaded house curves could fail to load or apply correctly.
  - Improves validation and consistency between UI preview and DSP processing.

### Notes
- No changes to FIR magnitude, phase, or leveling behavior.
- Safe update focused on UI → DSP data integrity.

---


## [2.7.8] – 2026-01-23

### Added
- **Stereo-linked auto-leveling (TXT-compatible default)**
  - SmartScan level window and gain are computed from a shared L/R reference and applied identically to both channels.
  - Eliminates channel-dependent gain drift while preserving automatic delay alignment.

- **Correction-band visualization**
  - Active magnitude correction range (`mag_c_min … mag_c_max`) is now explicitly carried through DSP stats and visualized in plots.
  - Makes it immediately clear where correction is applied and where it is intentionally inactive.

- **Reliability / confidence visualization**
  - Low-confidence frequency regions are visually shaded in plots.
  - Helps explain why certain bands are protected or only lightly corrected (measurement reliability, A-FDW behavior).

### Changed
- **Auto-leveling behavior (default)**
  - Stereo leveling now uses a single shared window and offset instead of independent per-channel SmartScan decisions.
  - Results are deterministic and TXT-compatible by default.

- **Summary.txt clarity**
  - Level window and offset method explicitly indicate stereo-linked operation
    (e.g. `ForcedOffset (StereoLink)`).

### Fixed
- **Auto-align gain drift**
  - Fixed cases where left/right channels could diverge by several dB due to independent leveling window selection.

### Notes
- Auto-align delay estimation is unchanged and remains fully automatic.
- FIR magnitude and phase are unaffected by alignment-only time shifts.

---

## [2.7.7] – 2026-01-20

### Added
- **2058-safe phase mode**
  - Disables room phase correction (confidence/FDW/excess-phase) and uses only theoretical crossover phase and minimum-phase where applicable.

- **Independent slope limits for boost vs cut**
  - Separate dB/oct limits prevent gentle boosts from being flattened while still constraining aggressive cuts.

- **TDC safety brakes**
  - Hard cap on total Temporal Decay Control reduction.
  - Optional slope limit for predictable, stable decay shaping.

- **DF smoothing (experimental)**
  - Gaussian smoothing with approximately constant Hz width across different sample rates and tap counts.

- **Comparison mode**
  - Locks scoring and plots to a fixed analysis grid (fs/taps) for meaningful A/B comparisons.

- **Multi-rate auto-taps mapping**
  - Maintains constant FIR time length across sample rates (44.1 kHz reference).

### Changed
- Refactored leveling logic into a dedicated module for robustness and testability.
- Improved guard logic against unstable phase and excessive corrections.

---

## [2.7.6] and earlier

- Initial public releases and iterative improvements to TDC, confidence masking,
  and the FIR generation pipeline.
- See commit history for detailed technical changes.
