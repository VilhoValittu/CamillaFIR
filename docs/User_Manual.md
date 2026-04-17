# CamillaFIR User Manual

This manual is the practical end-user guide for CamillaFIR. It explains what the program does, which files it expects, how to choose the main settings, what gets exported, and how to deploy the result to a convolution-capable playback system.

CamillaFIR generates room-correction FIR filters from REW exports, built-in sweep measurements, and WAV/IR captures. The typical workflow is:

1. Measure the left and right channels in REW, or use the built-in measurement workflow.
2. Load the measurement files into CamillaFIR.
3. Choose a target curve and operating mode.
4. Generate filters.
5. Export and deploy the output to CamillaDSP, Roon, Equalizer APO, or another FIR-capable DSP.

## 1. What is CamillaFIR?

CamillaFIR is a room-correction FIR filter generator for loudspeaker and listening-room optimization.

It is designed to:

- read REW exports, built-in measurement results, and WAV/IR sources
- compare the measured response against a target curve
- create left and right FIR filters
- export ready-to-use files for convolution playback systems
- keep correction behavior safer and more repeatable through built-in guard rails

CamillaFIR can work in several styles:

- fully automatic with preset search
- guided manual operation with safety clamps
- advanced manual tuning with more direct control

## 2. Requirements

Before you start, you need:

- REW (Room EQ Wizard) or the built-in CamillaFIR measurement tool
- a calibrated measurement microphone
- separate left and right channel measurements or equivalent generated IR sources
- a playback system that supports FIR convolution

Common deployment targets:

- CamillaDSP
- Roon convolution
- Equalizer APO
- other DSP or player software that accepts FIR WAV impulse files

Recommended preparation:

- use the same microphone position and measurement method for both channels
- keep the sample rate consistent through your measurement and playback chain
- verify that REW exports include phase when using text files

## 3. Workflow Overview

The normal workflow inside CamillaFIR is:

1. **Files tab**
   Load left and right measurement files.
2. **Target tab**
   Select a built-in target curve or load a custom target.
3. **Basic tab**
   Choose the operating mode, filter type, sample rate, taps, and the main safety settings.
4. **Advanced / Window / XO tabs**
   Refine advanced correction, timing, windowing, and crossover behavior when needed.
5. **Run tab**
   Start the generation process and review the produced output.

For most systems, start simple:

- load measurements
- choose `CamillaFIR automatic mode (recommended)` or `Basic`
- choose `Asymmetric` or `Mixed Phase`
- generate filters
- listen and re-measure before making more aggressive changes

## 4. Input Files

CamillaFIR supports the two common REW export workflows.

### 4.1 REW text export (`.txt`)

This is the most direct input format for frequency-response processing.

Expected content:

- frequency in Hz
- magnitude in dB
- phase in degrees

Important notes:

- the left and right channels must be loaded separately
- phase data is required if you want full phase-aware correction behavior
- headers are acceptable as long as the export is a normal REW text export

### 4.2 REW impulse export (`.wav`)

CamillaFIR can also import impulse-response WAV files and convert them internally for processing.

Use this when:

- you prefer an IR-based workflow from REW
- you want CamillaFIR to derive the response using its own windowing path

Practical guidance:

- export mono impulse files for each channel
- keep the export settings consistent between left and right
- match the sample rate to the rate you expect to use in playback when practical

### 4.3 AUTO Bass Integration WAV workflow

Use this workflow when subwoofer integration should be part of an automatic run instead of a separate manual crossover step.

Measurement requirements:

- Bass Integration is available only in `CamillaFIR automatic mode`
- use coherent REW IR WAV exports with the same measurement gain and the same acoustic timing reference for every main/sub capture
- do not normalize each file separately
- do not move each file to its own `t=0`

Topology-specific inputs:

- `AVR / Receiver (LFE+Main)` requires four WAV measurements: `L main`, `R main`, `L sub`, `R sub`
- `Direct DAC / CamillaDSP sub output` requires `L main`, `R main`, and `Sub 1`
- `Sub 2` is optional in `Direct DAC` mode and is summed automatically with `Sub 1`

Playback must match the same topology and crossover arrangement that was used during measurement.

## 5. Operating Modes

CamillaFIR has three main operating modes. The DSP engine is the same, but the workflow policy changes.

### 5.1 CamillaFIR automatic mode (recommended)

Use this when you want the program to search for a technically strong preset automatically.

What it does:

- evaluates multiple preset combinations
- scores candidates
- picks the best candidate before final export
- keeps the workflow guarded compared with unrestricted expert tuning
- can use harmonic curves and IACC-aware ranking to avoid overly aggressive or overly symmetric winners

Best for:

- first-pass results
- users who want a strong starting point quickly
- rooms where you do not want to hand-tune every parameter

#### AUTO Bass Integration

Automatic mode also includes an optional Bass Integration workflow for subwoofer systems.

Use it when:

- the subwoofer crossover should be optimized as part of the AUTO run
- you have separate coherent main/sub WAV impulse measurements
- you want crossover guidance and bass-overlap diagnostics in the final summary

Topology options:

- `AVR / Receiver (LFE+Main)` uses separate `L main`, `R main`, `L sub`, and `R sub` measurements, predicts the combined L/R totals, recommends an AVR crossover, and exports the normal L/R FIR filters
- `Direct DAC / CamillaDSP sub output` uses `L main`, `R main`, `Sub 1`, and optional `Sub 2`, can auto-select the `Main/Sub XO`, and can export an aligned mono `Sub_...wav` FIR in addition to the main filters

### 5.2 Basic 

Use this when you want manual control but still want guard rails.

What it does:

- applies conservative defaults
- keeps important safety clamps active
- reduces the chance of extreme or unstable correction

Best for:

- most home systems
- users learning the software
- repeatable day-to-day filter generation

### 5.3 Advanced (expert)

Use this when you already understand the tradeoffs and want broader control.

What it does:

- exposes more direct tuning freedom
- relaxes mode-based safety constraints
- makes you responsible for the result

Best for:

- experienced users
- deliberate A/B testing
- expert tuning of difficult systems

## 6. Target Curves

CamillaFIR includes a large set of built-in target curves, including:

- Harman variants
- Dr. Toole
- B&K variants
- Flat
- cinema and speech-oriented targets
- nearfield and studio-style tilts

You can also load a custom target file.

General guidance:

- a mild house curve is often more natural than perfectly flat in-room playback
- bass-heavy targets can sound impressive, but they increase headroom demand
- if you are unsure, start with a moderate built-in target and listen before pushing bass harder

In automatic mode, CamillaFIR can also select a target automatically when that workflow is enabled.

## 7. Filter Types

CamillaFIR supports four main filter types.

### 7.1 Linear Phase

- strongest timing accuracy
- can produce the highest latency
- can introduce audible pre-ringing in some situations

Latency estimate:

- latency is approximately `taps / (2 * sample_rate)`
- at 44.1 kHz and 65,536 taps, linear-phase latency is about 743 ms

Best for:

- offline listening
- systems where latency is not important
- users prioritizing timing accuracy

### 7.2 Minimum Phase

- very low latency behavior
- magnitude-focused correction
- avoids linear-phase pre-ringing behavior

Best for:

- low-latency systems
- conservative playback chains
- cases where phase correction is not the main goal

### 7.3 Mixed Phase

- linear-style behavior in bass
- minimum-phase style behavior at higher frequencies
- good balance between correction quality and practical use

Best for:

- general home listening
- users who want strong bass correction without full linear-phase behavior everywhere

### 7.4 Asymmetric

- near-linear low-frequency control with earlier impulse placement
- lower perceived delay than full linear phase
- often the most practical all-round choice

Best for:

- mixed music, movie, and desktop use
- users who want strong correction with more practical timing behavior

For many systems, `Asymmetric` is the most sensible first choice.

## 8. Key Settings

These settings have the biggest effect on the result.

### 8.1 Taps

Taps define FIR length.

- more taps improve low-frequency resolution
- more taps also increase latency and CPU load
- short filters are faster, but they can lose low-frequency precision

### 8.2 Sample Rate

Choose a sample rate that fits your playback chain.

- match the playback system when possible
- avoid unnecessary sample-rate conversion if you do not need it
- use `Generate All Rates` when you want one export package for multiple playback rates

### 8.3 Correction Range

The correction band defines the range where CamillaFIR is allowed to work.

Typical use:

- lower limit protects against unnecessary infra-bass correction
- upper limit helps avoid over-correcting poorly measured high frequencies

If the result sounds unnatural, narrowing the correction range is often safer than increasing algorithmic aggressiveness.

### 8.4 Maximum Boost and Cut

These settings limit how strongly the filter can move the response.

- high boost increases headroom risk
- strong cuts can sound overdamped if overused
- safer values are usually better for first runs

### 8.5 Level Matching

Level matching aligns the measured response to the target before correction is generated.

CamillaFIR supports:

- smart automatic scan
- manual window selection

If the target sounds consistently too lean or too heavy, check the leveling window before changing multiple other settings.

### 8.6 HPF and Low-Bass Protection

Low-frequency protection matters for real loudspeakers.

Relevant controls include:

- HPF
- `Low-bass boost lock`
- `Excursion Protection`

These are especially important for:

- small woofers
- bass-reflex systems
- ambitious house curves

## 9. Advanced Features

### 9.1 Temporal Decay Control (TDC)

TDC is designed to reduce excessive ringing and slow modal decay, especially in bass.

In practice it can:

- tighten bass decay
- reduce boom
- improve bass articulation

Use it carefully:

- stronger settings can sound drier
- if bass becomes too lean or over-damped, reduce TDC strength first

### 9.2 Adaptive Frequency-Domain Windowing (A-FDW)

A-FDW changes how finely different frequencies are corrected based on measurement confidence.

In practice:

- high-confidence regions can follow the target more closely
- low-confidence regions are smoothed more conservatively

This helps avoid over-correction in uncertain data.

### 9.3 Bass-first AI

Bass-first AI tells the system to treat low-frequency room modes as meaningful correction targets instead of smoothing them away too aggressively.

Use it when:

- room modes dominate the bass
- you want stronger intentional low-frequency cleanup

Be more conservative if:

- measurements are noisy
- the speakers have limited bass capability

### 9.4 Excursion Protection

Excursion Protection is a dynamic safety feature.

It helps prevent dangerous low-frequency boost by reacting to the measurement and target relationship.

Use it when:

- you are using bass-heavy targets
- your speakers have limited low-frequency headroom
- you want safer automatic bass behavior

### 9.5 Stereo Link

Stereo Link helps preserve left/right balance and phantom center stability.

Keep it enabled unless you have a specific reason to let each channel behave more independently.

### 9.6 IR Windowing

Additional controls affect timing and final impulse behavior:

- IR window style and shape
- asymmetric window placement controls

These matter most when you are refining latency behavior, channel alignment, or export impulse shape.

## 10. Running Filter Generation

When the main settings look correct:

1. verify that both channels are loaded
2. confirm target curve and mode
3. review sample rate, taps, correction range, and safety limits
4. start the run from the `Start` tab

During generation, CamillaFIR evaluates the measurements, builds the target relationship, applies safety logic, and exports the final result.

If you are using automatic mode, the program may take longer because it is comparing multiple candidates instead of generating only one direct configuration.

## 11. Output Files

After a successful run, CamillaFIR produces a result bundle. Common output files include:

- `L_...wav` and `R_...wav`
  Left and right FIR impulse-response files
- `Sub_...wav`
  Optional mono subwoofer FIR file for `Direct DAC / CamillaDSP sub output` Bass Integration runs
- `Summary_...txt`
  Human-readable report with scoring, settings, and diagnostics
- `camilladsp_...yml`
  CamillaDSP-ready configuration file
- ZIP package
  Collected export bundle for convenient deployment

Depending on the workflow, the export package can also contain additional configuration files alongside the WAV filters.

## 12. Reading the Summary

`Summary_...txt` is the main report for understanding what CamillaFIR produced.

Useful sections include:

- acoustic score and target match
- confidence metrics
- core settings used for the run
- Bass Integration mode, recommended crossover, and bass-overlap diagnostics when Bass Integration is enabled
- gain, headroom, and clamp behavior
- TDC, A-FDW, bass-first, and protection status
- alignment and stereo-related diagnostics

Use the summary to compare runs, especially when you change:

- target curve
- filter type
- correction range
- taps
- protection settings

When comparing runs, change one or two variables at a time. This makes the summary much more useful.

## 13. CamillaDSP Integration

CamillaDSP deployment is the most direct workflow.

Typical steps:

1. copy the exported FIR WAV files to your CamillaDSP coefficients folder
2. copy or adapt the generated `camilladsp_...yml`
3. point your CamillaDSP setup to the new filters
4. verify gain staging before normal listening levels

Practical notes:

- confirm that the sample rate matches your playback path
- if you hear clipping or overload, reduce master gain or convolution gain
- always test at moderate volume first

## 14. Roon and Equalizer APO Integration

CamillaFIR also works with other convolution-capable systems.

### 14.1 Roon

- load the exported FIR WAV files into Roon's convolution engine
- make sure the sample-rate handling matches your playback chain
- leave enough headroom if the result includes bass lift

### 14.2 Equalizer APO

- use the generated FIR WAV files in the convolution block
- apply enough preamp reduction to avoid clipping
- re-check channel assignment and sample-rate behavior on Windows

## 15. Health Checks

CamillaFIR performs validation before and during a run.

Typical result classes:

- `WARN`
  Non-critical issues worth reviewing
- `CRIT`
  Critical issues that may block the run, especially in guarded workflows such as Basic or automatic mode

Warnings can indicate:

- risky boost/headroom conditions
- incomplete inputs
- setting combinations that are technically possible but not recommended

If a health check fails, fix the root cause instead of trying to bypass the warning blindly.

## 16. Troubleshooting

### 16.1 The run will not start

Check:

- both channels are loaded
- the files are valid REW exports
- the sample-rate and mode settings are sensible
- any `CRIT` health checks have been resolved

If `Bass Integration` is enabled, also check:

- all required main/sub WAV inputs are loaded for the chosen topology
- every main/sub WAV uses the same timing reference
- the AVR crossover or `Main/Sub XO` matches the measurement setup closely enough to be meaningful

### 16.2 The correction sounds too thin

Check:

- target curve choice
- leveling window
- correction upper range
- TDC strength
- low-bass protection settings

### 16.3 The correction sounds too bright or aggressive

Try:

- lowering the correction upper limit
- using a gentler target
- increasing smoothing or using safer defaults
- moving from Advanced to Basic settings for comparison

### 16.4 Bass is still boomy

Check:

- whether TDC is enabled
- whether Bass-first AI is appropriate for the room
- whether taps are long enough for bass resolution
- whether the measurements themselves are reliable

### 16.5 Bass disappears or becomes weak

Check:

- `Low-bass boost lock`
- `Excursion Protection`
- HPF frequency and slope
- overly aggressive TDC or conservative target selection

### 16.6 Stereo image feels unstable

Check:

- left/right measurements were taken consistently
- `Stereo Link` is enabled
- `Auto-Align L/R` behavior is sensible
- one channel is not using a different file or rate accidentally

## 17. Best Practices

For the most reliable results:

- start with clean measurements
- use one target change at a time
- avoid extreme boost on first runs
- keep Basic mode as your reference workflow
- listen, then re-measure
- compare summaries instead of relying only on memory

The best correction is usually not the most aggressive one. A stable, repeatable, and well-protected filter is usually the better long-term result.

## 18. Recommended First Run

If you want a safe starting point, use this approach:

1. export left and right REW measurements
2. load them into CamillaFIR
3. choose `Basic` or `CamillaFIR automatic mode (recommended)`
4. choose a moderate built-in target
5. choose `Asymmetric` or `Mixed Phase`
6. keep boost conservative
7. generate the filters
8. deploy them at reduced volume first
9. re-measure with the filters active

That is the fastest path to a useful and defensible first result.

## 19. Built-in Measurement

CamillaFIR includes an integrated measurement tool that plays a sine sweep, records the response, and produces IR WAV files ready to import directly into the filter-generation workflow. It can replace a separate REW session when the hardware setup allows it.

### 19.1 What it produces

- Per-channel IR WAV files for Left, Right, Sub1, and Sub2
- Spatial averaging across multiple listening positions (magnitudes averaged; primary-position phase preserved)
- Repeat-based outlier rejection per channel

### 19.2 Configuration

Open the measurement dialog to set:

- **Positions** (1–12): number of listening positions in the room
- **Repeats per channel** (1–12): sweeps captured per channel per position; default is 5
- **Primary position**: which position supplies the reference phase and timing for spatial averaging
- **Channel selection**: checkboxes for Left, Right, Sub1, Sub2
- **Outlier rejection**: enable/disable and strictness level (safe / normal / strict)
- Per-channel settings: audio output and input device, channel index, sample rate, sweep frequency range and length, output gain, optional microphone calibration file

### 19.3 Measurement steps

1. Configure the session and start
2. For each position the tool captures all selected channels in sequence
3. After each position it pauses and prompts you to move the microphone to the next position
4. If Sub1 is selected it pauses before the first sub sweep so you can turn the subwoofer on
5. If Sub2 is also selected it pauses again after Sub1 completes so you can switch subwoofers
6. After all positions are captured it aggregates the results and saves the IR files
7. A summary shows how many takes were kept and rejected per channel

### 19.4 Subwoofer measurement — Windows requirement

> **Subwoofer measurement is only guaranteed to work on Windows.**
>
> The sweep is sent to the LFE channel (channel index 3 in a 5.1 or 7.1 layout). On Windows the output device must be configured for **5.1 or 7.1 multichannel playback** in Windows Sound settings. Without multichannel mode active the LFE channel does not exist at the driver level and the subwoofer will not receive the sweep signal.
>
> To enable multichannel output: open Sound settings → select the playback device → Properties → Advanced or Spatial sound → set the format to 5.1 or 7.1.
>
> On other platforms subwoofer routing may work in some configurations but is not tested or supported.

### 19.5 Outlier rejection

Takes are assessed per channel after all repeats for a position are captured.

- **Hard failures** (clipping, excessive noise, severe peak-timing deviation) are always rejected regardless of strictness
- **Strictness levels** set the median-absolute-deviation threshold for softer outliers:
  - Safe: 3.5 σ — keeps most takes unless they are clearly wrong
  - Normal: 2.75 σ — balanced default
  - Strict: 2.0 σ — discards anything that deviates meaningfully from the group median
- The post-session summary reports how many takes were kept per channel

If many takes are rejected, check the signal level, cable connections, and microphone placement before tightening the strictness setting.

### 19.6 Using the results

- Load the saved IR WAV files via the **REW impulse export (.wav)** path (section 4.2)
- For Bass Integration, use the sub IR files as Sub 1 and Sub 2 inputs (section 4.3)
- Do not normalize individual IR files and do not move them to separate `t=0` references — keep all files from the same session at the same gain so that relative timing and level are preserved
