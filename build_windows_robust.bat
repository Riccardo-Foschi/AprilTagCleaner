@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "PYI_ROOT=%cd%\_pyinstaller_tmp"
set "PYI_DIST=%PYI_ROOT%\dist"
set "PYI_WORK=%PYI_ROOT%\work"
set "PYI_SPEC=%PYI_ROOT%\spec"
set "BUILT_EXE=%PYI_DIST%\AprilTagCleaner.exe"

echo ==========================================
echo   AprilTagCleaner - robust Windows build
echo ==========================================

echo [1/6] Checking Python...
python --version >nul 2>nul
if errorlevel 1 goto :no_python

echo [2/6] Upgrading pip and installing dependencies...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :error

python -m pip install -r requirements.txt
if errorlevel 1 goto :error

if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist "%PYI_ROOT%" rmdir /s /q "%PYI_ROOT%"
for %%F in (*.spec) do del /q "%%F"
if exist release rmdir /s /q release
mkdir release
mkdir "%PYI_DIST%"
mkdir "%PYI_WORK%"
mkdir "%PYI_SPEC%"

echo [3/6] Preparing optional icon...
set "ICON_FILE="
if exist "%cd%\app.ico" set "ICON_FILE=%cd%\app.ico"
if exist "%cd%\assets\app.ico" set "ICON_FILE=%cd%\assets\app.ico"

echo [4/6] Running PyInstaller...
if defined ICON_FILE (
  python -m PyInstaller ^
    --noconfirm ^
    --clean ^
    --onefile ^
    --windowed ^
    --name AprilTagCleaner ^
    --distpath "%PYI_DIST%" ^
    --workpath "%PYI_WORK%" ^
    --specpath "%PYI_SPEC%" ^
    --collect-all pupil_apriltags ^
    --collect-all PIL ^
    --hidden-import cv2 ^
    --hidden-import PIL ^
    --hidden-import numpy ^
    --icon "%ICON_FILE%" ^
    app.py
) else (
  python -m PyInstaller ^
    --noconfirm ^
    --clean ^
    --onefile ^
    --windowed ^
    --name AprilTagCleaner ^
    --distpath "%PYI_DIST%" ^
    --workpath "%PYI_WORK%" ^
    --specpath "%PYI_SPEC%" ^
    --collect-all pupil_apriltags ^
    --collect-all PIL ^
    --hidden-import cv2 ^
    --hidden-import PIL ^
    --hidden-import numpy ^
    app.py
)
if errorlevel 1 goto :error

if not exist "%BUILT_EXE%" (
  echo.
  echo Build failed: PyInstaller did not produce "%BUILT_EXE%".
  goto :error
)

echo [5/6] Assembling release folder...
copy /y "%BUILT_EXE%" "release\AprilTagCleaner.exe" >nul
if errorlevel 1 goto :error

if exist dist rmdir /s /q dist
mkdir dist
copy /y "%BUILT_EXE%" "dist\AprilTagCleaner.exe" >nul
if errorlevel 1 (
  echo.
  echo Warning: could not refresh dist\AprilTagCleaner.exe. Another process may still be locking that file.
  echo The new executable is available in release\AprilTagCleaner.exe.
)

if exist README.md copy /y README.md release\README.txt >nul
if exist requirements.txt copy /y requirements.txt release\requirements.txt >nul

(
  echo AprilTagCleaner build completed.
  echo.
  echo Files in this folder:
  echo - AprilTagCleaner.exe
  echo - README.txt
  echo - requirements.txt
  echo.
  echo Notes:
  echo - This is a onefile executable built with PyInstaller.
  echo - First launch can be slower because PyInstaller extracts bundled files.
  echo - Cleaned images are written next to the source images in the folder apriltag_cleaned.
) > release\HOW_TO_RUN.txt

echo [6/6] Done.
echo.
echo Executable created at:
echo   %cd%\release\AprilTagCleaner.exe
echo.
echo You can distribute the whole release folder or just the EXE.
pause
exit /b 0

:no_python
echo Python was not found in PATH.
echo Install Python 3.10 or 3.11 and make sure "Add Python to PATH" is enabled.
pause
exit /b 1

:error
echo.
echo Build failed. Review the messages above.
if exist "%BUILT_EXE%" echo Partial output available at: "%BUILT_EXE%"
pause
exit /b 1