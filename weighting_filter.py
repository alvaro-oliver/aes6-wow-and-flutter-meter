import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import bilinear, freqz
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# === FILTER SAMPLE RATE ===
fs = 2000  # Hz

# === HPF ===
fc_hpf = 0.6  # cuttof at 0.6 Hz
order_hpf = 3 # 3rd order
w0_hpf = 2 * np.pi * fc_hpf
b_analog_hpf = [1, 0]
a_analog_hpf = [1, w0_hpf]
b_hpf, a_hpf = bilinear(b_analog_hpf, a_analog_hpf, fs=fs)
b_section = b_hpf.copy()
a_section = a_hpf.copy()
for _ in range(order_hpf - 1):
    b_hpf = np.convolve(b_hpf, b_section)
    a_hpf = np.convolve(a_hpf, a_section)

# === LPF ===
fc_lpf = 10  # cutoff at 10 Hz
order_lpf = 1 # 1st order
w0_lpf = 2 * np.pi * fc_lpf
b_analog_lpf = [1]
a_analog_lpf = [1, w0_lpf]
b_lpf, a_lpf = bilinear(b_analog_lpf, a_analog_lpf, fs=fs)

# === COMBINE ===
b_combined = np.convolve(b_hpf, b_lpf)
a_combined = np.convolve(a_hpf, a_lpf)

# === 4 Hz NORMALIZATION ===
_, h_ref = freqz(b_combined, a_combined, worN=[4], fs=fs)
gain_at_4hz = np.abs(h_ref[0])
b_combined /= gain_at_4hz

# === EXPORT COEFICIENTS ===
np.savetxt(os.path.join(SCRIPT_DIR, "weighting_filter_b.csv"), b_combined, delimiter=",")
np.savetxt(os.path.join(SCRIPT_DIR, "weighting_filter_a.csv"), a_combined, delimiter=",")

# === VISUAL FILTER RESPONSE ===
w, h = freqz(b_combined, a_combined, worN=8192, fs=fs)
h_db = 20 * np.log10(np.abs(h) + 1e-12)

# === AES6-2008 WEIGHTING CURVE ===
freqs_aes = np.array([
    0.1, 0.2, 0.315, 0.4, 0.63, 0.8, 1, 1.6, 2, 4,
    6.3, 10, 20, 40, 63, 100, 200
])
gains_db = np.array([
    -48.0, -30.6, -19.7, -15.0, -8.4, -6.0, -4.2, -1.8, -0.9, 0.0,
    -0.9, -2.1, -5.9, -10.4, -14.2, -17.3, -23.0
])

# === PLOT RESULTS ===
plt.figure(figsize=(10, 6))
plt.semilogx(w, h_db, label="HPF × LPF combined IIR Filter (normalized)", linewidth=1.5)
plt.semilogx(freqs_aes, gains_db, 'rx', label="AES6-2008 Curve", markersize=8)
plt.title("AES6-2008 Curve (normalized at 4 Hz)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Gain (dB)")
plt.grid(which='both', linestyle=':')
plt.legend()
plt.xlim([0.05, 1000])
plt.ylim([-55, 5])
plt.tight_layout()
plt.show()

# === PRINT FILTER RESPONSE AT CRITIC FREQS. ===
_, h_check = freqz(b_combined, a_combined, worN=2 * np.pi * freqs_aes / fs)
response_db = 20 * np.log10(np.abs(h_check) + 1e-12)
print("\nFilter response at AES6-2008 points:")
print("Frequency         Gain")
for f, g in zip(freqs_aes, response_db):
    print(f"{f:>7.3f} Hz         {g:>7.2f} dB")