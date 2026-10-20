@echo off
title Prediksi Harga RAM (Quick Sample)
color 0A
cd /d "%~dp0"
echo ========================================================
echo        PREDIKSI HARGA RAM - CONTOH SAMPLE
echo ========================================================
echo.
python src\predict.py --file sample_input.json
echo.
echo ========================================================
pause
