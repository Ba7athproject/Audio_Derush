import os
import sys
import threading
import platform
import tkinter as tk
from tkinter import filedialog, messagebox, Menu, Toplevel, Text
from ttkbootstrap import Style
from ttkbootstrap.constants import *
from ttkbootstrap import Button, Label, Combobox, Progressbar, Checkbutton
from PIL import Image, ImageTk
import whisperx
import torch
import psutil
from audio_utils import preprocess_audio
import tempfile
import soundfile as sf

AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".ogg", ".flac")

# Gestion optionnelle TinyLlama/llm_corrector
try:
    from llm_corrector import LLMCorr
    HAS_LLAMA = True
except ImportError:
    HAS_LLAMA = False

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def set_favicon(window):
    favicon_path = resource_path("favicon.ico")
    if os.path.exists(favicon_path):
        try:
            ico_img = ImageTk.PhotoImage(Image.open(favicon_path))
            window.iconphoto(False, ico_img)
            window._ico_img = ico_img
        except Exception as e:
            print(f"Erreur chargement favicon : {e}")

class DerushIAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Derush IA - Transcription et Correction")
        self.root.geometry("900x700")
        self.style = Style(theme="darkly")
        set_favicon(self.root)

        self.model_names = [
            "tiny", "tiny.en", "base", "base.en", "small", "small.en",
            "medium", "medium.en", "large-v1", "large-v2", "large-v3"
        ]

        self.selected_file = ""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transcription_result = None
        self.corrected_segments = None
        self.is_rtl = False
        self.stop_flag = threading.Event()
        self.model_cache_dir = os.path.expanduser("~/.cache/whisperx_models")
        if HAS_LLAMA:
            self.llm_corrector = LLMCorr(model_path="models/tinyllama")
        else:
            self.llm_corrector = None
        self.export_srt = tk.BooleanVar(value=True)
        self.use_preprocessing = tk.BooleanVar(value=False)

        # Menu
        self.menu = Menu(self.root)
        self.root.config(menu=self.menu)
        file_menu = Menu(self.menu, tearoff=0)
        file_menu.add_command(label="Choisir un fichier audio", command=self.choisir_fichier_audio)
        file_menu.add_command(label="Choisir dossier d'export", command=self.choisir_dossier_export)
        file_menu.add_separator()
        file_menu.add_command(label="Quitter", command=self.root.quit)
        self.menu.add_cascade(label="Fichier", menu=file_menu)
        help_menu = Menu(self.menu, tearoff=0)
        help_menu.add_command(label="À propos", command=self.afficher_apropos)
        self.menu.add_cascade(label="Aide", menu=help_menu)

        logo_path = resource_path("logo.png")
        try:
            img = Image.open(logo_path).resize((100, 100), Image.LANCZOS)
            self.logo = ImageTk.PhotoImage(img)
            Label(self.root, image=self.logo).place(x=10, y=10)
        except Exception as e:
            print("Erreur chargement logo :", e)

        Label(self.root, text="Modèle WhisperX :").place(x=130, y=30)
        self.model_choice = Combobox(self.root, values=self.model_names, width=20, state="readonly")
        self.model_choice.set("medium")
        self.model_choice.place(x=250, y=30)

        Label(self.root, text="Langue :").place(x=130, y=70)
        self.lang_options = ["Détection automatique", "ar", "fr", "en", "es", "de", "it", "pt"]
        self.lang_choice = Combobox(self.root, values=self.lang_options, width=20, state="readonly")
        self.lang_choice.set("Détection automatique")
        self.lang_choice.place(x=250, y=70)

        self.llm_correction_var = tk.BooleanVar(value=False)
        if HAS_LLAMA:
            Checkbutton(self.root, text="Corriger par IA (TinyLlama)", variable=self.llm_correction_var, bootstyle="info").place(x=130, y=110)
        Checkbutton(self.root, text="Exporter en SRT/VTT", variable=self.export_srt, bootstyle="secondary").place(x=400, y=110)
        Checkbutton(self.root, text="Prétraitement audio", variable=self.use_preprocessing, bootstyle="secondary").place(x=600, y=110)

        Button(self.root, text="Transcrire", command=self.demarrer, bootstyle="success outline").place(x=130, y=150)
        Button(self.root, text="Arrêter", command=self.arreter, bootstyle="danger outline").place(x=250, y=150)

        self.result_box = Text(self.root, wrap="word", height=30, width=110)
        self.result_box.place(x=25, y=200)
        self.result_box.tag_configure('left', justify='left')
        self.result_box.tag_configure('right', justify='right')

        self.progress = Progressbar(self.root, mode='determinate')
        self.progress.place(x=25, y=660, width=850)

        self.export_dir = os.path.abspath(".")
        self.thread = None
        self.stop_flag.clear()
        self.detect_device_and_suggest_model()

    def log(self, msg):
        tag = 'right' if self.is_rtl else 'left'
        self.result_box.insert(tk.END, msg + "\n", tag)
        self.result_box.see(tk.END)

    def choisir_fichier_audio(self):
        filepath = filedialog.askopenfilename(filetypes=[("Fichiers audio", AUDIO_EXTENSIONS)])
        if filepath:
            self.selected_file = filepath
            self.log(f"Fichier choisi : {os.path.basename(filepath)}")
            self.result_box.delete(1.0, tk.END)

    def choisir_dossier_export(self):
        folder = filedialog.askdirectory(initialdir=self.export_dir)
        if folder:
            self.export_dir = folder
            self.log(f"Dossier d'export choisi : {self.export_dir}")

    def detect_device_and_suggest_model(self):
        try:
            if platform.system() != "Windows":
                ram_gb = round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024.**3), 1)
            else:
                ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Aucun GPU"
            self.log(f"Détection matériel : RAM = {ram_gb} Go, GPU = {gpu_name}")
            sugg = "tiny" if ram_gb < 4 else "small" if ram_gb < 8 else "medium" if ram_gb < 12 else "large-v2"
            self.log(f"Modèle suggéré selon matériel : {sugg}")
            self.model_choice.set(sugg)
        except Exception as e:
            self.log(f"Erreur détection matériel : {e}")

    def afficher_apropos(self):
        message = (
            "🎙️ Derush IA - Transcription et Correction\n"
            "Version 1.0 – Août 2025\n\n"
            "Ce programme permet de transcrire des fichiers audio localement à l’aide du modèle WhisperX.\n"
            "Correction facultative avec TinyLlama localement.\n"
            "Développé par معز الباي (Moez Elbey) – Ba7ath-Dev."
        )
        popup = Toplevel(self.root)
        popup.title("À propos")
        popup.geometry("400x220")
        popup.resizable(False, False)
        set_favicon(popup)
        text_widget = Text(popup, wrap="word")
        text_widget.pack(expand=True, fill="both", padx=10, pady=10)
        text_widget.insert(tk.END, message)
        tag = 'right' if self.is_rtl else 'left'
        text_widget.tag_configure(tag, justify=tag)
        text_widget.tag_add(tag, '1.0', 'end')
        text_widget.config(state=tk.DISABLED)

    def arreter(self):
        if self.thread and self.thread.is_alive():
            self.stop_flag.set()
            self.log("Arrêt demandé... Veuillez patienter.")

    def demarrer(self):
        if not self.selected_file:
            messagebox.showwarning("Aucun fichier", "Veuillez choisir un fichier audio d'abord.")
            return
        model_name = self.model_choice.get()
        lang = self.lang_choice.get()
        if lang == "Détection automatique":
            lang = None
        if model_name.startswith("large"):
            if not messagebox.askyesno("Confirmation", f"Le modèle '{model_name}' est volumineux. Voulez-vous continuer ?"):
                return
        self.stop_flag.clear()
        self.progress['value'] = 0
        self.result_box.delete(1.0, tk.END)
        self.log("Début de la transcription...")
        self.thread = threading.Thread(target=self.processus_complet, args=(model_name, lang), daemon=True)
        self.thread.start()

    def estimate_transcription_time(self, duration_sec, model_name):
        speed_factor = {"tiny": 60, "base": 40, "small": 25, "medium": 15, "large": 10}
        for key in speed_factor:
            if key in model_name:
                return round(duration_sec / speed_factor[key], 1)
        return round(duration_sec / 15, 1)

    def processus_complet(self, model_name, lang):
        import pathlib

        temp_path = None

        try:
            self.log(f"Chemin audio soumis au module : {self.selected_file}")
            abs_path = os.path.abspath(self.selected_file)
            self.log(f"Chemin absolu utilisé : {abs_path}")
            self.log(f"os.path.isfile: {os.path.isfile(abs_path)}")
            self.log(f"os.path.exists: {os.path.exists(abs_path)}")
            self.log(f"pathlib.Path.exists: {pathlib.Path(abs_path).exists()}")

            try:
                with open(abs_path, "rb") as f:
                    self.log(f"Ouverture brute Python OK : taille = {len(f.read())} bytes")
            except Exception as e:
                self.log(f"Ouverture brute Python échouée : {e}")

            try:
                d, s = sf.read(abs_path)
                self.log(f"Ouverture soundfile OK, shape={d.shape}")
            except Exception as e:
                self.log(f"Ouverture soundfile échouée : {e}")

            if not os.path.isfile(abs_path):
                self.log(f"Erreur : le fichier n'existe pas à ce chemin : {abs_path}")
                return

            model = self.load_model_with_progress(model_name, self.device)
            self.log("Chargement de l'audio...")

            # Correction : gestion du prétraitement
            if self.use_preprocessing.get():
                audio_array = preprocess_audio(abs_path)
                fd, temp_path = tempfile.mkstemp(suffix=".wav")
                sf.write(temp_path, audio_array, 16000)  # adaptez 16000 au sampling_rate voulu
                os.close(fd)
                audio_path = temp_path
                self.log(f"Prétraitement: audio sauvegardé temporairement dans {audio_path}")
            else:
                audio_path = abs_path

            audio = whisperx.load_audio(audio_path)
            self.log("Transcription en cours...")
            result = model.transcribe(audio, language=lang) if lang else model.transcribe(audio)
            self.transcription_result = result
            detected_lang = result.get("language", "fr")
            self.is_rtl = detected_lang.startswith("ar")
            self.log(f"Langue détectée : {detected_lang.upper()}")
            self.corrected_segments = result["segments"]
            if self.llm_correction_var.get() and self.llm_corrector:
                self.log("Correction par IA en cours (TinyLlama)...")
                texts = [seg["text"] for seg in result["segments"]]
                corrected = self.llm_corrector.correct_segments(texts)
                self.corrected_segments = [
                    {**seg, "text": corrected[i]} for i, seg in enumerate(result["segments"])
                ]
                self.log("Correction IA terminée.")
            base = os.path.splitext(os.path.basename(self.selected_file))[0]
            if self.export_srt.get():
                self.export_to_srt(self.corrected_segments, base + ".srt")
                self.export_to_vtt(self.corrected_segments, base + ".vtt")
            self.export_to_txt(self.corrected_segments, base + ".txt", rtl=self.is_rtl)
            self.log("Export terminé.")
            est = self.estimate_transcription_time(result["segments"][-1]["end"], model_name)
            self.log(f"Temps estimé : {est} minutes")
        except Exception as e:
            import traceback
            self.log(f"Erreur : {e}\n{traceback.format_exc()}")
        finally:
            # Supprime le fichier temporaire créé si besoin
            if temp_path is not None:
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            self.progress.stop()

    def load_model_with_progress(self, model_name, device):
        self.progress['value'] = 0
        for i in range(0, 101, 10):
            if self.stop_flag.is_set():
                raise Exception("Arrêté par l'utilisateur")
            self.progress['value'] = i
            self.root.update_idletasks()
            threading.Event().wait(0.1)
        return whisperx.load_model(model_name, device, compute_type=self.get_compute_type())

    def get_compute_type(self):
        if self.device == "cuda":
            try:
                torch.randn(1, device="cuda", dtype=torch.float16)
                return "float16"
            except Exception:
                return "float32"
        return "float32"

    def export_to_txt(self, segments, filename, rtl=False):
        filepath = os.path.join(self.export_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            if rtl:
                f.write("‏")
            for seg in segments:
                start = self.format_time(seg["start"])
                end = self.format_time(seg["end"])
                f.write(f"[{start} --> {end}] {seg['text'].strip()}\n")
        self.log(f"Exporté en TXT : {filepath}")

    def export_to_srt(self, segments, filename):
        filepath = os.path.join(self.export_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, 1):
                f.write(f"{i}\n{self.srt_time(seg['start'])} --> {self.srt_time(seg['end'])}\n{seg['text'].strip()}\n\n")
        self.log(f"Exporté en SRT : {filepath}")

    def export_to_vtt(self, segments, filename):
        filepath = os.path.join(self.export_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for seg in segments:
                f.write(f"{self.srt_time(seg['start'])} --> {self.srt_time(seg['end'])}\n{seg['text'].strip()}\n\n")
        self.log(f"Exporté en VTT : {filepath}")

    def format_time(self, seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}" if h else f"{m:02d}:{s:06.3f}"

    def srt_time(self, seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) % 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

if __name__ == '__main__':
    root = tk.Tk()
    app = DerushIAApp(root)
    root.mainloop()
