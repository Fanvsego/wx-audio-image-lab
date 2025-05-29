# Audio & Image Processing App

A cross-platform desktop application built with **wxPython** for interactive audio and image processing. The app features two tabs:

* **Sound**: load or record audio, visualize waveforms and spectra, apply filters, perform basis transforms, and mix signals.
* **Image**: load images, apply tensor and eigenvalue operations per channel, mix images, and explore multi-faceted transformations.

---

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Audio Tab Details](#audio-tab-details)
5. [Image Tab Details](#image-tab-details)
6. [License](#license)

---

## Features

### 🎧 Sound Tab

1. **Audio I/O**: Load WAV files or record from microphone.
2. **Visualization**:

   * Time-domain waveform (amplitude vs. time).
   * Frequency spectrum via FFT.
   * Spectrogram display.
3. **Signal Processing**:

   * Block averaging (parameter `n`) to reduce noise.
   * Note detection: find dominant frequencies above an $\alpha\%$ threshold, map to closest musical notes, and display in a table.
   * Low-pass Butterworth filter for bass isolation.
4. **Basis Transforms**:

   * Play signal in a custom 2D basis selected from frequency–amplitude or time–amplitude vectors.
   * Play in the dual basis.
   * Apply the first signal’s basis (direct and dual) to a second loaded signal.
5. **Signal Mixing**:

   * Cross-product mixing: compute vector cross-products of FFT components, project onto a plane, reconstruct audio via inverse FFT.

### 🖼 Image Tab

1. **Image I/O**: Load two RGB images.
2. **Preprocessing**: Crop to a common square size.
3. **Channel Operations** (for each R/G/B):

   * Compute eigenvalues and eigenvectors; display images in each channel’s eigenbasis.
   * Tensor folds: four types of tensor contractions (sum over axes, mean, max).
   * Simple average of both images.
   * Mix via outer product of max-eigenvectors.
   * Reconstruct image from eigenbasis.
   * Mix via invariants (matrix traces).
4. **Visualization**: Display 20 generated images in a 5×4 grid.
5. **Audio-Image Fusion**: Placeholder for future mixed audio–visual effects.

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Fanvsego/wx-audio-image-lab.git
   cd wx-audio-image-lab
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate     # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   *Dependencies include: wxPython, numpy, matplotlib, scipy, Pillow, pyaudio, sounddevice.*

---

## Usage

Run the application:

```bash
python main.py
```

* Switch between **Sound** and **Image** tabs.
* Use the toolbar buttons to load data, visualize transforms, and play or display results.

## Audio Tab Details

* **Recording**: prompt for duration and filename; saves WAV, then loads it.
* **FFT & Spectrogram**: uses `scipy.fft` and `scipy.signal.spectrogram`.
* **Averaging**: groups samples in blocks of size `n`.
* **Note Detection**: Fast Fourier Transform → threshold α% of max amplitude → map peaks to standard note frequencies.
* **Filtering**: 4th-order Butterworth low-pass filter via `scipy.signal.butter`.
* **Basis Transforms**: select two FFT amplitude–frequency or time–amplitude vectors (min & max above 5% threshold), build a 2D basis, project, inverse FFT.
* **Dual Basis**: compute and use the dual basis matrix.
* **Cross-Product Mixing**: compute 3D cross-products (zero z-coordinate), project onto plane normal `n`, inverse FFT.

---

## Image Tab Details

* **Eigenbasis**: for each channel matrix, perform `scipy.linalg.eig`, then reconstruct and normalize.
* **Tensor Folds**: 4 contraction types along specified axes.
* **Max-Eigenvector Mix**: find eigenvector with largest eigenvalue, form outer products across images.
* **Invariant Mix**: compute traces and outer product of trace vectors.
* **Display**: normalize to \[0,255], cast to `uint8`, and plot via `matplotlib`.

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
