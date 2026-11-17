@echo off
setlocal
cd /d "%~dp0"
title EBI Classifier

set "UV_NO_CONFIG=1"
set "UV_CACHE_DIR=%CD%\.uv-cache"
set "GRADIO_ANALYTICS_ENABLED=False"
set "HF_HUB_OFFLINE=1"
set "PYTHONUTF8=1"
set "PATH=%CD%\runtime\python;%PATH%"

echo.
echo ===============================================
echo             EBI Classifier
echo ===============================================
echo.

if not exist "runtime\uv.exe" goto missing_files
if not exist "runtime\python\python.exe" goto missing_files
if not exist "runtime\wheels" goto missing_files
if not exist "runs\release\final.safetensors" goto missing_files
if not exist "runs\release\final.json" goto missing_files
if not exist "windows\requirements.txt" goto missing_files

if not exist ".venv\Scripts\python.exe" goto create_environment
".venv\Scripts\python.exe" -c "pass" >nul 2>&1
if errorlevel 1 goto create_environment
goto install_packages

:create_environment
echo Preparing the application for this computer...
echo This can take a few minutes the first time.
echo.
"runtime\uv.exe" venv --offline --python "%CD%\runtime\python\python.exe" --clear "%CD%\.venv"
if errorlevel 1 goto setup_failed

:install_packages
echo Checking the application files. Please keep this window open...
"runtime\uv.exe" pip sync --offline --no-index --find-links "%CD%\runtime\wheels" --python "%CD%\.venv\Scripts\python.exe" "%CD%\windows\requirements.txt"
if errorlevel 1 goto setup_failed

echo.
echo Starting the classifier. Your browser should open automatically.
echo If it does not, open the local address shown below in your browser.
echo Keep this window open while using the classifier.
echo To stop, close this window.
echo.
".venv\Scripts\python.exe" app.py
if errorlevel 1 goto app_failed
exit /b 0

:missing_files
echo Some application files are missing.
echo Right-click the downloaded ZIP, choose Extract All, and open the extracted folder.
echo Do not move START_EBI.bat out of that folder.
goto failed

:setup_failed
echo.
echo The application could not be prepared. Check that this folder is writable
echo and that there is enough free disk space. See START_HERE.txt for help.
goto failed

:app_failed
echo.
echo The classifier stopped with an error. See START_HERE.txt for help.

:failed
echo.
echo Please send a photo of this entire window to the person who sent you the app.
pause
exit /b 1
