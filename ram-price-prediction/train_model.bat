@echo off
title Pelatihan Ulang Model RAM Price Prediction
color 0E
cd /d "%~dp0"
echo ========================================================
echo     PELATIHAN ULANG MODEL REGRESI HARGA RAM
echo ========================================================
echo.
python src\train.py
echo.
echo ========================================================
echo Pelatihan selesai! Hasil metrik dan grafik telah diperbarui.
pause
