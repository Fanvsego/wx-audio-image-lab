
import wx
import wx.grid
import wx.lib.scrolledpanel as scrolled
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from PIL import Image
from scipy.linalg import eig
import pyaudio
import wave
import sounddevice as sd
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import spectrogram, butter, sosfiltfilt
from numpy.linalg import inv


class SoundPanel(wx.Panel):
    """
    Panel implementing audio-related functionality per Task:
    - Load or record audio
    - Plot waveform, FFT spectrum, spectrogram
    - Average signal over n-samples
    - Detect notes based on threshold
    - Apply low-pass filter
    - Play audio in custom basis and dual basis
    - Play second audio in basis of the first
    - Mix both signals using cross product and projection to the plane
    """

    def __init__(self, parent):
        super().__init__(parent)
        
        self.audio_data = None #numpy array for first audio signal
        self.sample_rate = 44100 #default sampling rate
        self.second_audio_data = None #numpy array for second audio signal

        #layout of the application
        vbox = wx.BoxSizer(wx.VERTICAL)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        self.record_btn = wx.Button(self, label="🎤 Record")
        self.upload_btn = wx.Button(self, label="📁 Upload WAV")
        self.upload2_btn = wx.Button(self, label="🔁 Upload Second WAV")
        hbox.Add(self.record_btn, flag=wx.ALL, border=5)
        hbox.Add(self.upload_btn, flag=wx.ALL, border=5)
        hbox.Add(self.upload2_btn, flag=wx.ALL, border=5)

        #Button for plotting and signal processing task
        self.plot_btn = wx.Button(self, label="📈 Plot Amplitude vs Time")
        self.fft_btn = wx.Button(self, label="🔊 FFT Spectrum")
        self.spec_btn = wx.Button(self, label="🌈 Spectrogram")
        self.avg_btn = wx.Button(self, label="⚙️ Average Signal")
        self.notes_btn = wx.Button(self, label="🎵 Detect Notes")
        self.lowpass_btn = wx.Button(self, label="🔌 Lowpass Filter")

        #Button for basis related tasks
        self.playinbasis_btn = wx.Button(self, label="Play audio signal in selected basis")
        self.playindualbasis_btn = wx.Button(self, label="Play audio signal in selected dual basis")
        self.play_second_in_first_basis_btn = wx.Button(self, label="Play second audio signal in basis of first")
        self.mix_two_signals_btn = wx.Button(self, label="Mix two audio signals")

        for btn in [self.plot_btn, self.fft_btn, self.spec_btn, self.avg_btn, self.notes_btn, self.lowpass_btn]:
            hbox.Add(btn, flag=wx.ALL, border=5)
        for btn in [self.playinbasis_btn, self.playindualbasis_btn,self.play_second_in_first_basis_btn, self.mix_two_signals_btn]:
            hbox1.Add(btn, flag=wx.ALL, border=5)
        vbox.Add(hbox)
        vbox.Add(hbox1)

        #canvas for visual output
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self, -1, self.figure)
        vbox.Add(self.canvas, 1, flag=wx.EXPAND)

        #grid for note detection
        self.grid = wx.grid.Grid(self)
        self.grid.CreateGrid(0, 3)
        self.grid.SetColLabelValue(0, "Detected frequency(Hz)")
        self.grid.SetColLabelValue(1, "Assigned frequency(Hz)")
        self.grid.SetColLabelValue(2, "Output note")
        vbox.Add(self.grid, wx.ALL, 5)

        self.SetSizer(vbox)

        #bind on button click event
        self.record_btn.Bind(wx.EVT_BUTTON, self.on_record)
        self.upload_btn.Bind(wx.EVT_BUTTON, self.on_upload)
        self.upload2_btn.Bind(wx.EVT_BUTTON, self.on_upload_second)
        self.plot_btn.Bind(wx.EVT_BUTTON, self.plot_waveform)
        self.fft_btn.Bind(wx.EVT_BUTTON, self.plot_fft)
        self.spec_btn.Bind(wx.EVT_BUTTON, self.plot_spectrogram)
        self.avg_btn.Bind(wx.EVT_BUTTON, self.play_average)
        self.notes_btn.Bind(wx.EVT_BUTTON, self.detect_notes)
        self.lowpass_btn.Bind(wx.EVT_BUTTON, self.apply_lowpass)
        self.playinbasis_btn.Bind(wx.EVT_BUTTON, self.process_in_frequency_basis)
        self.playindualbasis_btn.Bind(wx.EVT_BUTTON, self.process_in_frequency_dual_basis)
        self.play_second_in_first_basis_btn.Bind(wx.EVT_BUTTON, self.play_second_in_first_basis)
        self.mix_two_signals_btn.Bind(wx.EVT_BUTTON, self.mix_two_signals)

    def on_record(self, event):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = self.sample_rate
        RECORD_SECONDS = wx.GetNumberFromUser("Enter duration of an audio-signal.", "", "", value=1)
        OUTPUT_FILENAME = wx.GetTextFromUser("Enter recorded file name", default_value='recorded') + '.wav'

        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

        wx.MessageBox(f"Recording for {RECORD_SECONDS} seconds...", "Info", wx.OK | wx.ICON_INFORMATION)
        frames = [stream.read(CHUNK) for _ in range(int(RATE / CHUNK * RECORD_SECONDS))]

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        self.load_audio(OUTPUT_FILENAME)
        wx.MessageBox("Audio recorded and loaded", "Done", wx.OK | wx.ICON_INFORMATION)

    def on_upload(self, event):
        with wx.FileDialog(self, "Open WAV file", wildcard="WAV files (*.wav)|*.wav",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return
            self.load_audio(fileDialog.GetPath())
            wx.MessageBox("Audio loaded successfully", "Done", wx.OK | wx.ICON_INFORMATION)

    def on_upload_second(self, event):
        with wx.FileDialog(self, "Open Second WAV file", wildcard="WAV files (*.wav)|*.wav",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return
            wf = wave.open(fileDialog.GetPath(), 'rb')
            n_samples = wf.getnframes()
            audio = wf.readframes(n_samples)
            self.second_audio_data = np.frombuffer(audio, dtype=np.int16)
            wf.close()
            wx.MessageBox("Second audio loaded", "Done", wx.OK | wx.ICON_INFORMATION)

    def load_audio(self, filepath):
        wf = wave.open(filepath, 'rb')
        self.sample_rate = wf.getframerate()
        n_samples = wf.getnframes()
        audio = wf.readframes(n_samples)
        self.audio_data = np.frombuffer(audio, dtype=np.int16)
        wf.close()

    def plot_waveform(self, event):
        if self.audio_data is None: return
        self.ax.clear()
        time = np.linspace(0, len(self.audio_data) / self.sample_rate, num=len(self.audio_data))
        self.ax.plot(time, self.audio_data)
        self.ax.set_title("Amplitude vs Time")
        self.ax.set_xlabel("Time [s]")
        self.ax.set_ylabel("Amplitude")
        self.canvas.draw()

    def plot_fft(self, event):
        if self.audio_data is None: return
        self.ax.clear()
        N = len(self.audio_data)
        yf = fft(self.audio_data)
        xf = fftfreq(N, 1 / self.sample_rate)[:N//2]
        self.ax.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
        self.ax.set_title("Amplitude Spectrum (FFT)")
        self.ax.set_xlabel("Frequency [Hz]")
        self.ax.set_ylabel("Amplitude")
        self.canvas.draw()

    def plot_spectrogram(self, event):
        if self.audio_data is None: return
        self.ax.clear()
        f, t, Sxx = spectrogram(self.audio_data, self.sample_rate)
        self.ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
        self.ax.set_title("Spectrogram")
        self.ax.set_xlabel("Time [s]")
        self.ax.set_ylabel("Frequency [Hz]")
        self.canvas.draw()

    def play_average(self, event):
        if self.audio_data is None:
            return
        self.ax.clear()
        n = wx.GetNumberFromUser("Enter n for averaging:", "n =", "Averaging", 1, 1, 10000)
        if n <= 0:
            return
        audio = self.audio_data.astype(np.float32)
        trimmed_len = len(audio) // n * n
        audio_trimmed = audio[:trimmed_len]
        averaged = audio_trimmed.reshape(-1, n).mean(axis=1)
        time = np.linspace(0, len(audio_trimmed) / self.sample_rate, num=len(averaged))
        self.ax.plot(time, averaged)
        self.canvas.draw()
        play_data = np.repeat(averaged, n)
        play_data = np.clip(play_data, -32768, 32767).astype(np.int16)
        sd.play(play_data, self.sample_rate)
        sd.wait()

    def detect_notes(self, event):
        if self.audio_data is None: return
        alpha = wx.GetNumberFromUser("Enter alpha (percentage threshold):", "%", "Note Detection", 10, 1, 100)
        if alpha <= 0: return

        N = len(self.audio_data)
        yf = np.abs(fft(self.audio_data))
        xf = fftfreq(N, 1 / self.sample_rate)

        threshold = (alpha / 100) * np.max(yf)
        detected_freqs = xf[(yf > threshold) & (xf > 0)]

        notes = {
            'C0': 16.35,'D0': 18.35,'E0': 20.6,'F0': 21.8,'G0': 24.5,'A0':27.5,'B0':30.86,
            'C1': 32.70,'D1': 36.7,'E1': 41.2,'F1': 43.65,'G1': 48.99,'A1':55,'B1':61.73,
            'C2': 65.40,'D2': 73.41,'E2': 82.40,'F2': 87.307,'G2': 97.99,'A2':110,'B2':123.47,
            'C3': 130.81,'D3': 146.83,'E3': 164.81,'F3': 174.61,'G3': 196,'A3':220,'B3': 246.94,
            'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23, 'G4': 391.99, 'A4': 440.00, 'B4': 493.88,
            'C5': 523.25, 'D5': 587.33, 'E5': 659.26, 'F5': 698.46, 'G5': 783.99, 'A5': 880, 'B5': 987.77,
            'C6': 261.63*4, 'D6': 293.66*4, 'E6': 329.63*4, 'F6': 349.23*4, 'G6': 391.99*4, 'A6': 440.00*4, 'B6': 493.88*4,
            'C7': 261.63*8, 'D7': 293.66*8, 'E7': 329.63*8, 'F7': 349.23*8, 'G7': 391.99*8, 'A7': 440.00*8, 'B7': 493.88*8,
            'C8': 261.63*16, 'D8': 293.66*16, 'E8': 329.63*16, 'F8': 349.23*16, 'G8': 391.99*16, 'A8': 440.00*16, 'B8': 493.88*16
        }

        for f in detected_freqs:
            closest_note = min(notes.items(), key=lambda x: abs(x[1] - f))
            row = self.grid.GetNumberRows()
            self.grid.AppendRows(1)
            self.grid.SetCellValue(row, 0, str(round(f, 1)))
            self.grid.SetCellValue(row, 1, str(closest_note[1]))
            self.grid.SetCellValue(row, 2, closest_note[0])

    def apply_lowpass(self, event):
        if self.audio_data is None: return
        cutoff = wx.GetNumberFromUser("Enter cutoff frequency for lowpass filter:", "Hz", "Lowpass", 150, 10, 1000)
        sos = butter(1, cutoff / self.sample_rate, output='sos')
        filtered = sosfiltfilt(sos, self.audio_data)
        
        self.ax.clear()
        sd.play(filtered, self.sample_rate)
        sd.wait()
        time = np.linspace(0, len(filtered) / self.sample_rate, num=len(filtered))
        self.ax.plot(time, filtered)
        self.ax.set_title(f"Low-pass Filtered Signal (cutoff = {cutoff} Hz)")
        self.ax.set_xlabel("Time [s]")
        self.ax.set_ylabel("Amplitude")
        self.canvas.draw()

    def process_in_frequency_basis(self, event):
        if self.audio_data is None: return
        choice = wx.GetNumberFromUser(
            "Enter in which basis you want to play an audio-signal.\n"
            "1: Amp-Freq\n2: Amp-Time", "", "", min=1, value=1, max=2)
        
        N = len(self.audio_data)
        if choice == 1:
            Y = fft(self.audio_data)
            freqs = fftfreq(N, 1 / self.sample_rate)
            amps = np.abs(Y)
            phases = np.angle(Y)

            A_max = amps.max()
            threshold = 0.05 * A_max

            idx_max = np.argmax(amps)
            sig_idxs = np.where((amps >= threshold) & (np.arange(N) != idx_max))[0]
            idx_min = sig_idxs[np.argmin(amps[sig_idxs])]

            v_max = np.array([freqs[idx_max], amps[idx_max]], dtype=float)
            v_min = np.array([freqs[idx_min], amps[idx_min]], dtype=float)

            u1 = v_min / np.linalg.norm(v_min)
            u2 = v_max / np.linalg.norm(v_max)
            B = np.column_stack((u1, u2))

            V = np.vstack((freqs, amps))
            C = np.linalg.inv(B) @ V

            amps_rec = C[1, :]


            Y_rec = amps_rec * np.exp(1j * phases)
            data_rec = np.real(ifft(Y_rec)).astype(self.audio_data.dtype)

        else:

            t = np.linspace(0, N / self.sample_rate, N)
            sig = self.audio_data.astype(float)

            A_max = np.max(np.abs(sig))
            threshold = 0.05 * A_max

            idx_max = np.argmax(np.abs(sig))
            sig_idxs = np.where((np.abs(sig) >= threshold) & (np.arange(N) != idx_max))[0]
            idx_min = sig_idxs[np.argmin(np.abs(sig[sig_idxs]))]

            v_max = np.array([t[idx_max], sig[idx_max]], dtype=float)
            v_min = np.array([t[idx_min], sig[idx_min]], dtype=float)

            u1 = v_min / np.linalg.norm(v_min)
            u2 = v_max / np.linalg.norm(v_max)
            B = np.column_stack((u1, u2))

            V = np.vstack((t, sig))
            C = np.linalg.inv(B) @ V

            sig_rec = C[1, :]
            data_rec = sig_rec.astype(self.audio_data.dtype)

        sd.play(data_rec, self.sample_rate)
        sd.wait()

    def process_in_frequency_dual_basis(self, event):
        if self.audio_data is None: return

        choice = wx.GetNumberFromUser(
            "Enter in which basis you want to play an audio-signal.\n"
            "1: Amp-Freq\n2: Amp-Time", "", "", min=1, value=1, max=2)
        
        N = len(self.audio_data)
        
        if choice == 1:
            Y = fft(self.audio_data)
            freqs = fftfreq(N, 1 / self.sample_rate)
            amps = np.abs(Y)
            phases = np.angle(Y)

            A_max = amps.max()
            threshold = 0.05 * A_max

            idx_max = np.argmax(amps)
            sig_idxs = np.where((amps >= threshold) & (np.arange(N) != idx_max))[0]
            idx_min = sig_idxs[np.argmin(amps[sig_idxs])]

            v_max = np.array([freqs[idx_max], amps[idx_max]], dtype=float)
            v_min = np.array([freqs[idx_min], amps[idx_min]], dtype=float)

            u1 = v_min / np.linalg.norm(v_min)
            u2 = v_max / np.linalg.norm(v_max)
            B = np.linalg.matrix_transpose(np.linalg.inv(np.column_stack([u1, u2])))

            V = np.vstack((freqs, amps))
            C = np.linalg.inv(B) @ V

            amps_rec = C[1, :]


            Y_rec = amps_rec * np.exp(1j * phases)
            data_rec = np.real(ifft(Y_rec)).astype(self.audio_data.dtype)

        else:

            t = np.linspace(0, N / self.sample_rate, N)
            sig = self.audio_data.astype(float)

            A_max = np.max(np.abs(sig))
            threshold = 0.05 * A_max

            idx_max = np.argmax(np.abs(sig))
            sig_idxs = np.where((np.abs(sig) >= threshold) & (np.arange(N) != idx_max))[0]
            idx_min = sig_idxs[np.argmin(np.abs(sig[sig_idxs]))]

            v_max = np.array([t[idx_max], sig[idx_max]], dtype=float)
            v_min = np.array([t[idx_min], sig[idx_min]], dtype=float)

            u1 = v_min / np.linalg.norm(v_min)
            u2 = v_max / np.linalg.norm(v_max)
            B = np.linalg.matrix_transpose(np.linalg.inv(np.column_stack([u1, u2])))

            V = np.vstack((t, sig))
            C = np.linalg.inv(B) @ V

            sig_rec = C[1, :]
            data_rec = sig_rec.astype(self.audio_data.dtype)

        sd.play(data_rec, self.sample_rate)
        sd.wait()
        
    def play_second_in_first_basis(self, event):
        if self.audio_data is None: return
        if self.second_audio_data is None:
            self.on_upload_second()
        N = len(self.audio_data)
        Y = fft(self.audio_data)
        freqs = fftfreq(N, 1 / self.sample_rate)
        amps = np.abs(Y)
        phases = np.angle(Y)

        N_sec = len(self.second_audio_data)
        Y_sec = fft(self.second_audio_data)
        freqs_sec = fftfreq(N_sec, 1 / self.sample_rate)
        amps_sec = np.abs(Y_sec)

        A_max = amps.max()
        threshold = 0.05 * A_max

        idx_max = np.argmax(amps)
        sig_idxs = np.where((amps >= threshold) & (np.arange(N) != idx_max))[0]
        idx_min = sig_idxs[np.argmin(amps[sig_idxs])]

        v_max = np.array([freqs[idx_max], amps[idx_max]], dtype=float)
        v_min = np.array([freqs[idx_min], amps[idx_min]], dtype=float)

        u1 = v_min / np.linalg.norm(v_min)
        u2 = v_max / np.linalg.norm(v_max)
        B = np.column_stack((u1, u2))

        V = np.vstack((freqs_sec, amps_sec))
        C = np.linalg.inv(B) @ V


        amps_rec = C[1, :]

        Y_rec = amps_rec * np.exp(1j * phases)
        data_rec = np.real(ifft(Y_rec)).astype(self.audio_data.dtype)
        sd.play(data_rec, self.sample_rate)
        sd.wait()

        B = np.linalg.matrix_transpose(np.linalg.inv(np.column_stack([u1, u2])))

        V = np.vstack((freqs_sec, amps_sec))
        C = np.linalg.inv(B) @ V

        amps_rec = C[1, :]
        wx.MessageBox("playing in dual")
        Y_rec = amps_rec * np.exp(1j * phases)
        data_rec = np.real(ifft(Y_rec)).astype(self.audio_data.dtype)
        sd.play(data_rec, self.sample_rate)
        sd.wait()

    def mix_two_signals(self, event):
        if self.audio_data is None:
            return
        if self.second_audio_data is None:
            self.on_upload_second()
        N      = len(self.audio_data)
        Y      = fft(self.audio_data)
        freqs  = fftfreq(N, 1/self.sample_rate)
        amps   = np.abs(Y)
        phases = np.angle(Y)

        N2     = len(self.second_audio_data)
        Y2     = fft(self.second_audio_data)
        freqs2 = fftfreq(N2, 1/self.sample_rate)
        amps2  = np.abs(Y2)

        M = min(N, N2)
        freqs, amps, phases = freqs[:M], amps[:M], phases[:M]
        freqs2, amps2       = freqs2[:M], amps2[:M]

        V1 = np.column_stack((freqs,  amps,  np.zeros(M)))
        V2 = np.column_stack((freqs2, amps2, np.zeros(M)))

        cross_vecs = np.cross(V1, V2)   # (M,3)

        n = np.array([0,0,1], dtype=float)
        n /= np.linalg.norm(n)

        dots = cross_vecs.dot(n)
        proj = cross_vecs - np.outer(dots, n)

        Y_proj = proj[:,0] + 1j * phases
        y_time = ifft(Y_proj, n=M)

        sd.play(np.real(y_time), samplerate=self.sample_rate)
        sd.wait()

class ImagePanel(wx.Panel):
    """
    Panel implementing image-related functionality per Task:
    - Load two images
    - Prepare, normalize, and mix images via tensor operations
    - Display a suite of generated images and transformations
    """
    def __init__(self, parent):
        super().__init__(parent)

        #placeholders for images and they numpy arrays
        self.img1 = None
        self.img2 = None
        self.img1_array = None
        self.img2_array = None

        #main layout
        vbox = wx.BoxSizer(wx.VERTICAL)
        hbox = wx.BoxSizer(wx.HORIZONTAL)

        #buttons to load images and trigger mixing pipeline
        self.load1_btn = wx.Button(self, label="\U0001F4C2 Load Image 1")
        self.load2_btn = wx.Button(self, label="\U0001F4C2 Load Image 2")
        self.mix_btn = wx.Button(self, label="\U0001F4CA Mix Images")
        hbox.Add(self.load1_btn, flag=wx.ALL, border=5)
        hbox.Add(self.load2_btn, flag=wx.ALL, border=5)
        hbox.Add(self.mix_btn, flag=wx.ALL, border=5)
        vbox.Add(hbox)

        #create a grid of subplots for image output
        self.figure, self.axs = plt.subplots(5, 4, figsize=(16, 12))
        self.canvas = FigureCanvas(self, -1, self.figure)
        vbox.Add(self.canvas, 1, flag=wx.EXPAND)

        self.SetSizer(vbox)
        
        #bind image-loading and mixing
        self.load1_btn.Bind(wx.EVT_BUTTON, self.load_image1)
        self.load2_btn.Bind(wx.EVT_BUTTON, self.load_image2)
        self.mix_btn.Bind(wx.EVT_BUTTON, self.mix_images)

    def load_image1(self, event):
        with wx.FileDialog(self, "Open first image", wildcard="Image files|*.png;*.jpg;*.bmp", 
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.img1 = Image.open(dlg.GetPath()).convert('RGB')
                self.img1_array = np.array(self.img1)

    def load_image2(self, event):
        with wx.FileDialog(self, "Open second image", wildcard="Image files|*.png;*.jpg;*.bmp", 
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.img2 = Image.open(dlg.GetPath()).convert('RGB')
                self.img2_array = np.array(self.img2)

    def mix_images(self, event):
        if self.img1_array is None or self.img2_array is None:
            wx.MessageBox("Load both images first", "Error", wx.OK | wx.ICON_ERROR)
            return

        arr1, arr2 = self.prepare_images(self.img1_array, self.img2_array, max_size=128)
        results, titles = self.tensor_mix_pipeline(arr1, arr2)

        self.figure.clf()
        axs = self.figure.subplots(5, 4)
        self.figure.subplots_adjust(hspace=0.5)

        for ax, img, title in zip(axs.flat, results, titles):
            ax.clear()
            ax.imshow(img)
            ax.set_title(title, fontsize=8)
            ax.axis('off')

        self.canvas.draw()

    def prepare_images(self, arr1, arr2, max_size=128):
        size = min(arr1.shape[0], arr1.shape[1], arr2.shape[0], arr2.shape[1], max_size)
        return arr1[:size, :size], arr2[:size, :size]

    def normalize(self, mat):
        mat = np.real(mat)
        mat = mat - np.min(mat)
        max_val = np.max(mat)
        if max_val == 0:
            return np.zeros_like(mat, dtype=np.uint8)
        mat = mat / max_val * 255
        return np.round(mat).astype(np.uint8)

    def tensor_mix_pipeline(self, img1, img2):
        """
        Core mixing pipeline across R/G/B channels:
        1) Eigen-basis images for each channel of both inputs
        2) Convolutions/folds of tensor products per channel
        3) Simple sum image
        4) Max-eigenvector tensor mix
        5) Reconstruction via original eigenbasis
        6) Mixing via invariants (traces)
        7) Placeholder for audio-image mix demonstration
        Returns list of images and their titles.
        """

        results = []
        titles = []
        ch1 = [img1[:, :, i] for i in range(3)]
        ch2 = [img2[:, :, i] for i in range(3)]

        for idx, ch in enumerate(ch1 + ch2):

            vals, vecs = eig(ch)
            mat_recon = vecs @ np.diag(vals) @ inv(vecs)
            img = self.normalize(mat_recon)
            results.append(img)
            label = f"Img{1 if idx < 3 else 2} {'RGB'[idx % 3]} in EigBasis"
            titles.append(label)


        for i in range(3):

            tensor = np.tensordot(ch1[i], ch2[i], axes=0)
            sum1 = np.sum(tensor, axis=(0, 2))
            sum2 = np.sum(tensor, axis=(1, 3))
            mean = np.mean(tensor, axis=(2, 3))
            maxv = np.max(tensor, axis=(2, 3))

            for arr, name in zip([sum1, sum2, mean, maxv], ["Fold1", "Fold2", "Mean", "Max"]):
                
                img = self.normalize(arr)
                if len(img.shape) == 2:
                    results.append(img)
                    titles.append(f"Tensor {name} {['R','G','B'][i]}")

        summed = np.clip((img1.astype(int) + img2.astype(int)) // 2, 0, 255).astype(np.uint8)
        results.append(summed)
        titles.append("Sum Image")

        def max_eig_vector(mat):
            vals, vecs = eig(mat)
            idx = np.argmax(np.real(vals))
            return np.real(vecs[:, idx])

        mix_tensor = [np.outer(max_eig_vector(ch1[i]), max_eig_vector(ch2[i])) for i in range(3)]
        mix_image = np.stack([self.normalize(m) for m in mix_tensor], axis=2)
        results.append(mix_image)
        titles.append("Max EigVec Mix")

        recon = []
        for i in range(3):
            vals, vecs = eig(ch1[i])
            mat = vecs @ np.diag(vals) @ inv(vecs)
            recon.append(self.normalize(mat))
        recon_image = np.stack(recon, axis=2)
        results.append(recon_image)
        titles.append("Reconstructed (Img1)")

        inv1 = [np.trace(c) for c in ch1]
        inv2 = [np.trace(c) for c in ch2]
        mix = [np.outer([inv1[i]], [inv2[i]]) for i in range(3)]
        inv_image = np.stack([self.normalize(m) for m in mix], axis=2)
        results.append(inv_image)
        titles.append("Invariant Mix")

        audio_img = np.zeros_like(img1)
        results.append(audio_img)
        titles.append("Audio Mix (placeholder)")

        return results, titles

class MyFrame(wx.Frame):
    
    def __init__(self,parent):
        super().__init__(parent, title="Audio & Image Processing App", size=(1200, 800))
        notebook = wx.Notebook(self)
        sound_panel = SoundPanel(notebook)
        image_panel = ImagePanel(notebook)
        notebook.AddPage(sound_panel, "\U0001F3A7 Sound")
        notebook.AddPage(image_panel, "\U0001F5BC Image")

if __name__ == "__main__":
    app = wx.App()
    frame = MyFrame(None)
    frame.Show()
    app.MainLoop()
    del app
