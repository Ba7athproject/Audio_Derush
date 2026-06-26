@echo off
REM Aller dans le dossier du projet
cd /d C:\Ba7ath_scripts\Audio_Derush

REM Activer l'environnement virtuel
call venv_ia\Scripts\activate

REM Lancer le script principal
python Derush_IA.py

REM Désactiver le venv (optionnel)
call deactivate

REM Pause pour garder la fenêtre ouverte après exécution
pause
