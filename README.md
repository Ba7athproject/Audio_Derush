# Derush IA - Transcription et Correction d'Audio

**Derush IA** est une application locale dotée d'une interface graphique permettant de transcrire des fichiers audio avec une grande précision, en utilisant les modèles d'intelligence artificielle **WhisperX**. Elle intègre également des fonctionnalités de prétraitement audio et offre la possibilité de corriger les transcriptions via un modèle LLM local (TinyLlama).

## 🚀 Fonctionnalités Principales

*   **Interface Graphique Intuitive :** Développée avec `tkinter` et `ttkbootstrap` (thème sombre), offrant une expérience utilisateur fluide et moderne.
*   **Transcription Avancée avec WhisperX :**
    *   Prise en charge de tous les modèles Whisper (de `tiny` à `large-v3`).
    *   Détection automatique du matériel (CPU, GPU, quantité de RAM) pour recommander le modèle le plus adapté à votre système.
    *   Détection automatique de la langue ou sélection manuelle (Français, Anglais, Arabe, etc.).
*   **Prétraitement Audio :** Fonctionnalité optionnelle pour nettoyer et normaliser l'audio avant transcription (conversion mono, normalisation du volume) via un module de traitement audio dédié.
*   **Correction par IA (Optionnel) :** Possibilité d'utiliser un modèle LLM local (`TinyLlama`) pour corriger automatiquement les segments transcrits.
*   **Export Multiple :** Génération des sous-titres et transcriptions sous différents formats :
    *   Texte brut (`.txt`) avec gestion du formatage RTL (Right-to-Left) pour l'arabe.
    *   Sous-titres SubRip (`.srt`).
    *   Sous-titres WebVTT (`.vtt`).
*   **Traitement Asynchrone :** La transcription s'effectue en arrière-plan avec une barre de progression, sans bloquer l'interface.

## 🛠️ Prérequis

*   **Python 3.10 ou supérieur** (L'environnement recommandé est Python 3.13)
*   [FFmpeg](https://ffmpeg.org/download.html) installé et ajouté au PATH de votre système.
*   Environnement matériel avec GPU (NVIDIA avec CUDA) recommandé pour des performances optimales, bien que l'application puisse fonctionner sur CPU.

## 📦 Installation

1.  **Cloner le dépôt** (ou télécharger les fichiers) :
    ```bash
    git clone https://github.com/votre-utilisateur/audio-derush-ia.git
    cd audio-derush-ia
    ```

2.  **Exécuter le script d'initialisation (Windows) :**
    Un script `.bat` est fourni pour configurer et lancer automatiquement l'application.
    ```bash
    Audio_Derush_IA.bat
    ```
    
    *Alternative manuelle :*
    ```bash
    python -m venv venv_ia
    venv_ia\Scripts\activate
    pip install ttkbootstrap Pillow whisperx torch psutil soundfile
    python Derush_IA.py
    ```

## 🎮 Utilisation

1.  Lancez l'application via `Audio_Derush_IA.bat` ou en exécutant `python Derush_IA.py`.
2.  Cliquez sur **Fichier > Choisir un fichier audio** pour sélectionner votre fichier (`.mp3`, `.wav`, `.m4a`, etc.).
3.  Sélectionnez le modèle Whisper (le programme suggérera le meilleur selon votre RAM/GPU).
4.  Cochez les options désirées (Export SRT/VTT, Prétraitement audio, Correction IA).
5.  Cliquez sur **Transcrire**. L'avancement s'affiche dans la fenêtre principale.

## 📂 Structure du projet

*   `Derush_IA.py` : Script principal contenant l'interface graphique et la logique de transcription.
*   `audio_utils.py` : Module gérant le chargement, le prétraitement et la manipulation bas niveau des fichiers audio (librosa, soundfile).
*   `utils.py` : Fonctions utilitaires de vérification des dépendances.
*   `llm_corrector.py` *(optionnel)* : Module permettant l'intégration de TinyLlama.
*   `Audio_Derush_IA.bat` : Script de lancement rapide pour Windows.

## 📝 Licence

Ce projet est développé par **معز الباي (Moez Elbey) – Ba7ath-Dev**.
