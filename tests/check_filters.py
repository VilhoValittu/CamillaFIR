import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import os
print("CWD:", os.getcwd())


HERE = os.path.dirname(os.path.abspath(__file__))
FILTERS_DIR = HERE

files = {
    os.path.splitext(fn)[0]: os.path.join(FILTERS_DIR, fn)
    for fn in os.listdir(FILTERS_DIR)
    if fn.lower().endswith(".wav")
}


def to_float_mono(x):
    x = np.asarray(x)
    if x.ndim == 2:
        x = x[:, 0]
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64) / np.iinfo(x.dtype).max
    else:
        x = x.astype(np.float64)
    return x

data = {}
stats = {}

for name, path in files.items():
    fs, x = wav.read(path)
    x = to_float_mono(x)

    pk = int(np.argmax(np.abs(x)))
    peak_ms = pk / fs * 1000.0

    # Normalize per-file by peak for fair overlay
    x_n = x / (np.max(np.abs(x)) + 1e-12)

    # Active region (rough)
    thr = max(1e-8, 1e-5*np.max(np.abs(x_n)))
    nz = np.where(np.abs(x_n) > thr)[0]
    first = int(nz[0]) if nz.size else None
    last  = int(nz[-1]) if nz.size else None

    data[name] = (fs, x_n)
    stats[name] = dict(fs=fs, n=len(x_n), peak_idx=pk, peak_ms=peak_ms, first=first, last=last)

print("=== STATS ===")
for k,v in stats.items():
    print(f"{k:8s} fs={v['fs']} n={v['n']} peak_idx={v['peak_idx']} peak_ms={v['peak_ms']:.3f} "
          f"active=[{v['first']},{v['last']}]")
names = sorted(data.keys())

first = next(iter(stats))
fs = stats[first]["fs"]
n  = stats[first]["n"]

# 1) Early-time overlay (0–800 ms): peak placement
end_ms = 800.0
end_idx = int(end_ms/1000.0*fs)
t_ms = np.arange(end_idx)/fs*1000.0

plt.figure()
for name in names:
    x = data[name][1]
    end_i = min(end_idx, len(x))
    plt.plot(t_ms[:end_i], x[:end_i], label=name)
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude (normalized)")
plt.title("IR comparison (0–800 ms): peak placement")
plt.legend()
plt.show()

# 2) Zoom around each peak (±40 ms), aligned to its own peak
plt.figure()
win = int(0.040*fs)
for name in names:
    x = data[name][1]
    pk = stats[name]["peak_idx"]
    a = max(0, pk-win)
    b = min(len(x), pk+win+1)
    tt = (np.arange(a,b)-pk)/fs*1000.0
    plt.plot(tt, x[a:b], label=name)
plt.xlabel("Time relative to peak (ms)")
plt.ylabel("Amplitude (normalized)")
plt.title("Local shape around peak (±40 ms)")
plt.legend()
plt.show()

# 3) Difference curve between first two files (if present), aligned to the first file peak, ±200 ms
if len(names) >= 2:
    a_name, b_name = names[0], names[1]
    pk_a = stats[a_name]["peak_idx"]
    a = max(0, pk_a-int(0.200*fs))
    b = min(n, pk_a+int(0.200*fs)+1)
    tt = (np.arange(a,b)-pk_a)/fs*1000.0
    diff = data[b_name][1][a:b] - data[a_name][1][a:b]

    plt.figure()
    plt.plot(tt, diff, label=f"{b_name} - {a_name}")
    plt.xlabel(f"Time relative to {a_name} peak (ms)")
    plt.ylabel("Amplitude (normalized)")
    plt.title("Difference near peak (±200 ms)")
    plt.legend()
    plt.show()
