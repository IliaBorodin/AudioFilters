import numpy as np
import tkinter as tk
import scipy.io.wavfile as wavfile
import wave
import os
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import firwin, lfilter
from matplotlib.figure import Figure
from scipy.fftpack import fft
from tkinter import filedialog
from scipy.fft import rfft, irfft


class Window:
    def __init__(self):
        self.__window = tk.Tk()
        self.__load_btn = ttk.Button()
        self.__save_btn = ttk.Button()
        self.__canvas = tk.Canvas()
        self.__canvas1 = tk.Canvas()
        self.__figure = Figure()
        self.__harmonica_btn = tk.Button()
        self.__BPF_filter_btn = tk.Button()
        self.__LPF_filter_btn = tk.Button()
        self.__HPF_filter_btn = tk.Button()
        self.__BSF_filter_btn = tk.Button()
        self.__signal = None

    @property
    def window(self):
        return self.__window

    @property
    def canvas(self):
        return self.__canvas

    @property
    def signal(self):
        return self.__signal

    def init_window(self):
        self.__window = config_window(self.window)  # Задаем конфигурацию окну
        self.__window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.__load_btn = ttk.Button(self.window, text="Загрузить", command=self.load_file)
        self.__load_btn.place(relwidth=0.5)
        self.__save_btn = ttk.Button(self.window, text="Сохранить", command=self.save_file)
        self.__save_btn.place(relx=0.5, relwidth=0.5)
        self.__harmonica_btn = ttk.Button(self.window, text="Создать гармонику", command=self.load_harmonica)
        self.__harmonica_btn.place(relwidth=0.5, y=self.__load_btn.winfo_reqheight())
        self.__BPF_filter_btn = ttk.Button(self.window, text="Пропускающий фильтр", command=self.bpf_get)
        self.__LPF_filter_btn = ttk.Button(self.window, text="Фильтр нижних частот", command=self.lpf_get)
        self.__HPF_filter_btn = ttk.Button(self.window, text="Фильтр высоких частот", command=self.hpf_get)
        self.__BSF_filter_btn = ttk.Button(self.window, text="Подавляющий фильтр", command=self.bsf_get)
        self.__BPF_filter_btn.place(relx=0.5, relwidth=0.125, y=self.__load_btn.winfo_reqheight())
        self.__LPF_filter_btn.place(relx=0.625, relwidth=0.125, y=self.__load_btn.winfo_reqheight())
        self.__HPF_filter_btn.place(relx=0.75, relwidth=0.125, y=self.__load_btn.winfo_reqheight())
        self.__BSF_filter_btn.place(relx=0.875, relwidth=0.125, y=self.__load_btn.winfo_reqheight())

    def on_closing(self):
        self.__window.quit()
        self.__window.destroy()

    def load_harmonica(self):
        self.__signal = Signal()
        figure = Figure(figsize=(10, 10), dpi=100)
        plot_one = figure.add_subplot(2, 1, 1)
        plot_one.plot(self.signal.signal)
        plot_one.set_title("Входной сигнал")
        plot_one.set_xlabel("Время(мс)")
        plot_one.set_ylabel("Амплитуда(Па)")
        plot_one.set_xlim([0, self.signal.signal.size - 1])
        plot = figure.add_subplot(2, 1, 2)
        clc_fft = np.abs(fft(self.signal.signal, self.signal.ts))
        clc_fft = 20 * np.log10(10e-11 + clc_fft / np.max(clc_fft))
        plot.plot(clc_fft, color="green")
        plot.set_title("Входной спектр")
        plot.set_xlabel("Частота(Гц)")
        plot.set_ylabel("Сила звука(дБ)")
        plot.set_xlim([0, self.signal.ts // 2 - 1])
        self.__canvas = FigureCanvasTkAgg(figure, self.window)
        self.__canvas.get_tk_widget().place(relwidth=0.5, y=self.__load_btn.winfo_reqheight() * 2)

    def load_file(self):
        frame_rate, audio = open_file()
        audio_array = np.frombuffer(audio, dtype=np.int16)
        self.__signal = Signal(audio_array, frame_rate)
        figure = Figure(figsize=(10, 10), dpi=100)
        plot_one = figure.add_subplot(2, 1, 1)
        plot_one.plot(self.signal.signal)
        plot_one.set_title("Входной сигнал")
        plot_one.set_title("Входной сигнал")
        plot_one.set_xlabel("Время(мс)")
        plot_one.set_ylabel("Амплитуда(Па)")
        plot_one.set_xlim([0, self.signal.signal.size - 1])
        plot = figure.add_subplot(2, 1, 2)
        clc_fft = np.abs(fft(self.signal.signal, frame_rate))
        clc_fft = 20 * np.log10(10e-11 + clc_fft / np.max(clc_fft))
        plot.plot(clc_fft, color="green")
        plot.set_title("Входной спектр")
        plot.set_title("Входной спектр")
        plot.set_xlabel("Частота(Гц)")
        plot.set_ylabel("Сила звука(дБ)")
        plot.set_xlim([0, frame_rate // 2 - 1])
        self.__canvas = FigureCanvasTkAgg(figure, self.window)
        self.__canvas.get_tk_widget().place(relwidth=0.5, y=self.__load_btn.winfo_reqheight() * 2)

    def save_file(self):
        save_file(self.signal.lpf.filtered, "LPF", self.signal.ts)
        save_file(self.signal.hpf.filtered, "HPF", self.signal.ts)
        save_file(self.signal.bpf.filtered, "BPF", self.signal.ts)
        save_file(self.signal.bsf.filtered, "BSF", self.signal.ts)

    def lpf_get(self):
        figure = Figure(figsize=(10, 10), dpi=100)
        plot = figure.add_subplot(2, 1, 1)
        clc_fft = np.abs(fft(self.signal.lpf.lp, self.signal.ts))
        clc_fft = 20 * np.log10(10e-11 + clc_fft / np.max(clc_fft))
        plot.plot(clc_fft, color="orange")
        plot.set_title("Фильтр")
        plot.set_xlabel("Частота(Гц)")
        plot.set_ylabel("Сила звука(дБ)")
        plot.set_xlim([0, self.signal.ts // 2 - 1])
        plot1 = figure.add_subplot(2, 1, 2)
        clc_fft1 = np.abs(fft(self.signal.lpf.filtered, self.signal.ts))
        clc_fft1 = 20 * np.log10(10e-11 + clc_fft1 / np.max(clc_fft1))
        plot1.plot(clc_fft1, color="purple")
        plot1.set_title("Результат")
        plot1.set_xlabel("Частота(Гц)")
        plot1.set_ylabel("Сила звука(дБ)")
        plot1.set_xlim([0, self.signal.ts // 2 - 1])
        self.__canvas1 = FigureCanvasTkAgg(figure, self.window)
        self.__canvas1.get_tk_widget().place(relwidth=0.5, y=self.__load_btn.winfo_reqheight() * 2, relx=0.5)

    def hpf_get(self):
        figure = Figure(figsize=(10, 10), dpi=100)
        plot = figure.add_subplot(2, 1, 1)
        clc_fft = np.abs(fft(self.signal.hpf.hp, self.signal.ts))
        clc_fft = 20 * np.log10(10e-11 + clc_fft / np.max(clc_fft))
        plot.plot(clc_fft, color="orange")
        plot.set_title("Фильтр")
        plot.set_xlabel("Частота(Гц)")
        plot.set_ylabel("Сила звука(дБ)")
        plot.set_xlim([0, self.signal.ts // 2 - 1])
        plot1 = figure.add_subplot(2, 1, 2)
        clc_fft1 = np.abs(fft(self.signal.hpf.filtered, self.signal.ts))
        clc_fft1 = 20 * np.log10(10e-11 + clc_fft1 / np.max(clc_fft1))
        plot1.plot(clc_fft1, color="purple")
        plot1.set_title("Результат")
        plot1.set_xlabel("Частота(Гц)")
        plot1.set_ylabel("Сила звука(дБ)")
        plot1.set_xlim([0, self.signal.ts // 2 - 1])
        self.__canvas1 = FigureCanvasTkAgg(figure, self.window)
        self.__canvas1.get_tk_widget().place(relwidth=0.5, y=self.__load_btn.winfo_reqheight() * 2, relx=0.5)

    def bpf_get(self):
        figure = Figure(figsize=(10, 10), dpi=100)
        plot = figure.add_subplot(2, 1, 1)
        clc_fft = np.abs(fft(self.signal.bpf.bp, self.signal.ts))
        clc_fft = 20 * np.log10(10e-11 + clc_fft / np.max(clc_fft))
        plot.plot(clc_fft, color="orange")
        plot.set_title("Фильтр")
        plot.set_xlabel("Частота(Гц)")
        plot.set_ylabel("Сила звука(дБ)")
        plot.set_xlim([0, self.signal.ts // 2 - 1])
        plot1 = figure.add_subplot(2, 1, 2)
        clc_fft1 = np.abs(fft(self.signal.bpf.filtered, self.signal.ts))
        clc_fft1 = 20 * np.log10(10e-11 + clc_fft1 / np.max(clc_fft1))
        plot1.plot(clc_fft1, color="purple")
        plot1.set_title("Результат")
        plot1.set_xlabel("Частота(Гц)")
        plot1.set_ylabel("Сила звука(дБ)")
        plot1.set_xlim([0, self.signal.ts // 2 - 1])
        self.__canvas1 = FigureCanvasTkAgg(figure, self.window)
        self.__canvas1.get_tk_widget().place(relwidth=0.5, y=self.__load_btn.winfo_reqheight() * 2, relx=0.5)

    def bsf_get(self):
        figure = Figure(figsize=(10, 10), dpi=100)
        plot = figure.add_subplot(2, 1, 1)
        clc_fft = np.abs(fft(self.signal.bsf.bs, self.signal.ts))
        clc_fft = 20 * np.log10(10e-11 + clc_fft / np.max(clc_fft))
        plot.plot(clc_fft, color="orange")
        plot.set_title("Фильтр")
        plot.set_xlabel("Частота(Гц)")
        plot.set_ylabel("Сила звука(дБ)")
        plot.set_xlim([0, self.signal.ts // 2 - 1])
        plot1 = figure.add_subplot(2, 1, 2)
        clc_fft1 = np.abs(fft(self.signal.bsf.filtered, self.signal.ts))
        clc_fft1 = 20 * np.log10(10e-11 + clc_fft1 / np.max(clc_fft1))
        plot1.plot(clc_fft1, color="purple")
        plot1.set_title("Результат")
        plot1.set_xlabel("Частота(Гц)")
        plot1.set_ylabel("Сила звука(дБ)")
        plot1.set_xlim([0, self.signal.ts // 2 - 1])
        self.__canvas1 = FigureCanvasTkAgg(figure, self.window)
        self.__canvas1.get_tk_widget().place(relwidth=0.5, y=self.__load_btn.winfo_reqheight() * 2, relx=0.5)


class Signal:
    def __init__(self, signal=None, frame_rate=None):
        if signal is None:
            np.random.seed(1)
            self.__f0 = 20  # частота
            self.__ts = 1024  # частота дискретизации
            self.__x = np.linspace(0, 1, self.__ts, endpoint=True)
            self.__signal = ((3 * np.cos(2 * np.pi * 20 * self.__x) + 210 * np.sin(
                2 * np.pi * 117 * self.__x) + 100 * np.sin(2 * np.pi * 184 * self.__x) +
                              380 * np.sin(
                        2 * np.pi * 281 * self.__x))
                             + 5 * np.random.randn(self.__ts))
            self.__lpf_filter = LPF(self.__signal)
            self.__hpf_filter = HPF(self.__signal)
            self.__bpf_filter = BPF(self.__signal)
            self.__bsf_filter = BSF(self.__signal)
        else:
            self.__f0 = 20
            self.__ts = frame_rate
            self.__x = np.arange(0, 1, self.__ts)
            self.__signal = signal
            self.__lpf_filter = LPF(signal)
            self.__hpf_filter = HPF(signal)
            self.__bpf_filter = BPF(signal)
            self.__bsf_filter = BSF(signal)

    @property
    def signal(self):
        return self.__signal

    @property
    def lpf(self):
        return self.__lpf_filter

    @property
    def hpf(self):
        return self.__hpf_filter

    @property
    def bpf(self):
        return self.__bpf_filter

    @property
    def bsf(self):
        return self.__bsf_filter

    @property
    def ts(self):
        return self.__ts


class LPF:
    def __init__(self, signal):
        self.__signal = signal
        self.__M = 161
        self.__fc = 0.1
        self.__lp = firwin(self.__M, self.__fc, window=('kaiser', 9))
        self.__filtered = lfilter(self.__lp, 1, self.__signal)

    @property
    def lp(self):
        return self.__lp

    @property
    def filtered(self):
        return self.__filtered

    @property
    def signal(self):
        return self.__signal

    @signal.setter
    def signal(self, new_signal):
        self.__signal = new_signal

    @property
    def taps(self):
        return self.__M


class HPF:
    def __init__(self, signal):
        self.__signal = signal
        self.__M = 161
        self.__fc = 0.1
        self.__hp = firwin(self.__M, self.__fc, pass_zero=False, window=('kaiser', 9))
        self.__filtered = lfilter(self.__hp, 1, self.__signal)

    @property
    def hp(self):
        return self.__hp

    @property
    def filtered(self):
        return self.__filtered

    @property
    def signal(self):
        return self.__signal

    @signal.setter
    def signal(self, new_signal):
        self.__signal = new_signal

    @property
    def taps(self):
        return self.__M


class BPF:
    def __init__(self, signal):
        self.__signal = signal
        self.__M = 161
        self.__fc_low = 0.05
        self.__fc_high = 0.15
        self.__bp = firwin(self.__M, [self.__fc_low, self.__fc_high], pass_zero=False, window=('kaiser', 9))
        self.__filtered = lfilter(self.__bp, 1, self.__signal)

    @property
    def bp(self):
        return self.__bp

    @property
    def filtered(self):
        return self.__filtered

    @property
    def signal(self):
        return self.__signal

    @signal.setter
    def signal(self, new_signal):
        self.__signal = new_signal

    @property
    def taps(self):
        return self.__M


class BSF:
    def __init__(self, signal):
        self.__signal = signal
        self.__M = 161
        self.__fc_low = 0.05
        self.__fc_high = 0.15
        self.__bs = firwin(self.__M, [self.__fc_low, self.__fc_high], pass_zero=True, window=('kaiser', 9))
        self.__filtered = lfilter(self.__bs, 1, self.__signal)

    @property
    def bs(self):
        return self.__bs

    @property
    def filtered(self):
        return self.__filtered

    @property
    def signal(self):
        return self.__signal

    @signal.setter
    def signal(self, new_signal):
        self.__signal = new_signal

    @property
    def taps(self):
        return self.__M


def config_window(window):
    window.state('zoomed')
    window.resizable(True, True)
    window.title('Фильтрация')
    window.config(bg='#6D7BAB')
    return window


def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Wave files", "*.wav")])
    if file_path:
        # Если файл выбран
        print("Выбранный файл:", file_path)
        try:
            # Открываем файл без контекстного менеджера
            wav_file = wave.open(file_path, 'rb')
            # Чтение параметров аудиофайла
            frame_rate = wav_file.getframerate()
            frames = wav_file.getnframes()
            # Чтение аудио данных
            audio_data = wav_file.readframes(frames)
            # Закрываем файл
            wav_file.close()
            # Возвращаем параметры и аудио данные
            return frame_rate, audio_data
        except wave.Error as e:
            print("Ошибка при открытии файла:", e)
        return None  # Если возникла ошибка, возвращаем None


def save_file(audio, name, frame_rate):
    current_dir = os.path.dirname(__file__)
    result_folder = os.path.join(current_dir, "result")
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    file_name = f"{name}.wav"
    file_path = os.path.join(result_folder, file_name)
    rift_result = rfft(audio)
    rift_result = np.array(rift_result)
    new_audio = irfft(rift_result)
    normalize_audio = np.int16(audio * (32767 / new_audio.max()))
    wavfile.write(file_path, frame_rate, normalize_audio)


def main():
    window = Window()
    window.init_window()
    window.window.mainloop()


if __name__ == "__main__":
    main()
