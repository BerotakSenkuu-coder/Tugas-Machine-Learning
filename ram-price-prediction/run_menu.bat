@echo off
title RAM Price Prediction - Menu Utama
color 0B
cd /d "%~dp0"

:menu
cls
echo ========================================================
echo         RAM PRICE PREDICTION - WINDOWS LAUNCHER
echo ========================================================
echo.
echo  Pilih tindakan yang ingin dijalankan:
echo.
echo  [1] Prediksi Cepat (Gunakan sample_input.json)
echo  [2] Prediksi Interaktif (Ketik Spesifikasi RAM Sendiri)
echo  [3] Latih Ulang Model (Re-Train train.py)
echo  [4] Keluar
echo.
echo ========================================================
set /p opt="Masukkan pilihan (1-4): "

if "%opt%"=="1" goto opt1
if "%opt%"=="2" goto opt2
if "%opt%"=="3" goto opt3
if "%opt%"=="4" goto opt4

echo Pilihan tidak valid. Silakan coba lagi.
timeout /t 2 >nul
goto menu

:opt1
cls
echo [INFO] Menjalankan prediksi dari sample_input.json...
echo.
python src\predict.py --file sample_input.json
echo.
pause
goto menu

:opt2
cls
echo [INFO] Membuka formulir prediksi interaktif...
echo.
python src\predict.py
echo.
pause
goto menu

:opt3
cls
echo [INFO] Melatih ulang 4 model regresi (Linear, Ridge, RandomForest)...
echo Mohon tunggu sebentar...
echo.
python src\train.py
echo.
pause
goto menu

:opt4
exit
