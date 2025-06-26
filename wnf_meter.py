# =============================================================================
# Wow and Flutter Meter - AES6 Analyzer
# 
# Performs Wow and Flutter analysis according to the AES6-2008 (rev. 2012) standard,
# using the preferred method: Two Sigma (2σ) applied to both weighted and unweighted
# speed deviation measurements.
#
# =============================================================================
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, hilbert, lfilter
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
import sys

# --- BASE PATH ---
if getattr(sys, 'frozen', False):
    SCRIPT_DIR = sys._MEIPASS
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------- CONFIG --------------------
TRIM_MS = 20            # ms to trim from inst freq signal
BPF_FACTOR_LOW = 0.1    # mean freq factor to set prefilter low pass frequency 
BPF_FACTOR_HIGH = 1.9   # mean freq factor to set prefillter hi cut frequency 
LPF_FACTOR = 0.4        # mean freq factor to set postfilter cuttof frequency
FS_FILTER = 2000        # must match the weighting filter sample rate
PNG_EXPORT = False      # export results as png image

# -------------------- SELECT AUDIO FILE --------------------
root = tk.Tk()
root.withdraw()
AUDIO_PATH = filedialog.askopenfilename(
    title="Select an audio file",
    filetypes=[("WAV files", "*.wav")]
)

if not AUDIO_PATH:
    print("No file has been selected.")
    exit()

# -------------------- FUNCTIONS --------------------
def bandpass_filter(signal, fs, lowcut, highcut, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)

def lowpass_filter(signal, fs, cutoff, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, signal)

def estimate_freq_zero_crossing(signal, fs):
    t = np.arange(len(signal)) / fs
    zero_crossings = []
    for i in range(1, len(signal)):
        if signal[i-1] < 0 and signal[i] >= 0:
            t0, t1 = t[i-1], t[i]
            y0, y1 = signal[i-1], signal[i]
            crossing_time = t0 - y0 * (t1 - t0) / (y1 - y0)
            zero_crossings.append(crossing_time)
    if len(zero_crossings) < 2:
        return 0.0
    periods = np.diff(zero_crossings)
    return 1.0 / np.mean(periods)

# -------------------- PROCESSING --------------------
fs, data = wavfile.read(AUDIO_PATH)
TRIM = int(fs * TRIM_MS / 1000)
audio = data.astype(np.float32)
if audio.ndim > 1:
    audio = audio[:, 0]

# Step 1: get mean freq using zero-crossing frequency counter
mean_freq = estimate_freq_zero_crossing(audio, fs)

# Step 2: prefilter and get instantaneous frequency signal
bpf_lo = mean_freq * BPF_FACTOR_LOW
bpf_hi = mean_freq * BPF_FACTOR_HIGH
prefiltered = bandpass_filter(audio, fs, bpf_lo, bpf_hi)
analytic = hilbert(prefiltered)
phase = np.unwrap(np.angle(analytic))
inst_freq = np.diff(phase) * fs / (2 * np.pi)
inst_freq = inst_freq[TRIM:-TRIM]

# Step 3: normalize and get ride of FFT artifacts
norm = (inst_freq - mean_freq) / mean_freq
lpf_cutoff = mean_freq * LPF_FACTOR
norm_filt = lowpass_filter(norm, fs, lpf_cutoff)

# Step 4: decimate to weighting filter's sample rate
factor = int(fs / FS_FILTER)
b_lpf, a_lpf = butter(N=2, Wn=400, btype='low', fs=FS_FILTER) # anti-aliasing filter
norm_decimated = lfilter(b_lpf, a_lpf, norm_filt)
norm_decimated = norm_decimated[::factor]

# Step 5: get unweighted WnF
deviation_u = np.abs(norm_decimated)
deviation_u = deviation_u[int(0.4 * FS_FILTER):] # cut the first 0.4 seconds
two_sigma_u = np.percentile(deviation_u, 95)

# Step 6: apply weighting filter
b = np.loadtxt(os.path.join(SCRIPT_DIR, "weighting_filter_b.csv"), delimiter=",")
a = np.loadtxt(os.path.join(SCRIPT_DIR, "weighting_filter_a.csv"), delimiter=",")
norm_weighted = lfilter(b, a, norm_decimated)

# Step 7: get weighted WnF
deviation_w = np.abs(norm_weighted)
deviation_w = deviation_w[int(0.4 * FS_FILTER):]
two_sigma_w = np.percentile(deviation_w, 95)

# -------------------- PLOT RESULTS --------------------
fig = plt.figure(figsize=(14, 10))
fig.canvas.manager.set_window_title('AES6-2008 (r2012) Wow and Flutter Analysis')
base_name = os.path.splitext(os.path.basename(AUDIO_PATH))[0]
fig.suptitle(base_name, fontsize=12)

# Layout
ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
ax2 = plt.subplot2grid((3, 2), (1, 0), colspan=2)
ax3 = plt.subplot2grid((3, 2), (2, 0))
ax4 = plt.subplot2grid((3, 2), (2, 1))

# Plot 1: Instantaneous Frequency
tf = np.arange(len(inst_freq)) / fs
ax1.plot(tf, inst_freq, linewidth=0.3, color='gray')
margin = mean_freq * two_sigma_u * 10
ax1.set_ylim(mean_freq - margin, mean_freq + margin)
ax1.axhline(y=mean_freq, color='purple', linestyle='--', label=f"Mean: {mean_freq:.3f} Hz", linewidth=1)
ax1.set_xlabel("Time (s)")
ax1.set_title("Instantaneous Frequency (Hz)")
ax1.legend()
ax1.grid()

# Plot 2: Frequency Deviation
td = np.arange(len(deviation_u)) / FS_FILTER
ax2.plot(td, deviation_u, color='#edb329', linewidth=1, alpha=0.7)
ax2.plot(td, deviation_w, color='#934226', linewidth=1, alpha=0.7)
ax2.axhline(y=two_sigma_u, color='#e76d14', linestyle='--', label=f"Unweighted Peak WnF (2σ): ± {two_sigma_u * 100:.4f} %", linewidth=1)
ax2.axhline(y=two_sigma_w, color='#6c251e', linestyle='--', label=f"Weighted Peak WnF (2σ): ± {two_sigma_w * 100:.4f} %", linewidth=1)
ax2.set_title("Normalized Frequency Deviation")
ax2.set_xlabel("Time (s)")
ax2.set_ylim(0, two_sigma_u * 3)
ax2.legend()
ax2.grid()

# Plot 3: Spectrum
spectrum_u = np.abs(np.fft.rfft(norm_decimated))
freqs_u = np.fft.rfftfreq(len(norm_decimated), d=1/FS_FILTER)
spectrum_w = np.abs(np.fft.rfft(norm_weighted))
freqs_w = np.fft.rfftfreq(len(norm_weighted), d=1/FS_FILTER)

ax3.semilogx(freqs_u, spectrum_u / np.max(spectrum_u), color='#edb329', label="Unweighted", alpha=0.7)
ax3.semilogx(freqs_u, spectrum_w / np.max(spectrum_w), color='#934226', label="Weighted", alpha=0.7)
ax3.set_title("Normalized WnF Spectrum")
ax3.set_xlabel("Frequency (Hz)")
ax3.legend()
ax3.grid()
ax3.grid(which="minor", color="0.9")
ax3.set_xlim([freqs_u[1], 400])

# Plot 4: Histogram
ax4.hist(norm_filt, bins=1024, density=True, color='gray', alpha=0.7)
ax4.set_title("Histogram")
ax4.grid()

plt.tight_layout(rect=[0, 0, 1, 0.96])
if PNG_EXPORT:
    png_path = os.path.join(SCRIPT_DIR, base_name + ".png")
    plt.savefig(png_path, dpi=300)
    print(f"Plot saved as: {png_path}")

plt.show()
