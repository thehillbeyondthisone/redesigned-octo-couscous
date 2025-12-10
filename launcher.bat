@echo off
setlocal enabledelayedexpansion

:: ============================================
:: Real-Time Spanish Autocomplete Ghost Text Aid
:: Windows Launcher Script
:: ============================================

title Spanish Ghost Text Aid - Launcher

:: Configuration
set VENV_DIR=venv
set PYTHON_CMD=python
set MODEL_PATH=
set EXTRA_ARGS=

:: Colors (Windows 10+)
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "CYAN=[96m"
set "RESET=[0m"

:: Header
echo %CYAN%============================================%RESET%
echo %CYAN% Real-Time Spanish Autocomplete Ghost Text %RESET%
echo %CYAN%============================================%RESET%
echo.

:: Parse command line arguments
:parse_args
if "%~1"=="" goto :done_args
if /i "%~1"=="--model" (
    set MODEL_PATH=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--terminal" (
    set EXTRA_ARGS=%EXTRA_ARGS% --terminal
    shift
    goto :parse_args
)
if /i "%~1"=="--low-vram" (
    set EXTRA_ARGS=%EXTRA_ARGS% --low-vram
    shift
    goto :parse_args
)
if /i "%~1"=="--debug" (
    set EXTRA_ARGS=%EXTRA_ARGS% --debug
    shift
    goto :parse_args
)
if /i "%~1"=="--setup" (
    goto :setup_only
)
if /i "%~1"=="--clean" (
    goto :clean_venv
)
if /i "%~1"=="--help" (
    goto :show_help
)
shift
goto :parse_args
:done_args

:: Check Python installation
echo %YELLOW%[1/4] Checking Python installation...%RESET%
where %PYTHON_CMD% >nul 2>&1
if errorlevel 1 (
    echo %RED%Error: Python not found in PATH%RESET%
    echo Please install Python 3.10+ and add it to PATH
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Display Python version
for /f "tokens=*" %%i in ('%PYTHON_CMD% --version 2^>^&1') do set PYTHON_VERSION=%%i
echo %GREEN%Found: %PYTHON_VERSION%%RESET%

:: Check/Create virtual environment
echo.
echo %YELLOW%[2/4] Setting up virtual environment...%RESET%

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Creating new virtual environment...
    %PYTHON_CMD% -m venv %VENV_DIR%
    if errorlevel 1 (
        echo %RED%Error: Failed to create virtual environment%RESET%
        pause
        exit /b 1
    )
    echo %GREEN%Virtual environment created successfully%RESET%
    set NEED_INSTALL=1
) else (
    echo %GREEN%Virtual environment already exists%RESET%
    set NEED_INSTALL=0
)

:: Activate virtual environment
echo Activating virtual environment...
call %VENV_DIR%\Scripts\activate.bat
if errorlevel 1 (
    echo %RED%Error: Failed to activate virtual environment%RESET%
    pause
    exit /b 1
)

:: Install/Update dependencies
echo.
echo %YELLOW%[3/4] Checking dependencies...%RESET%

:: Check if requirements need to be installed
pip show faster-whisper >nul 2>&1
if errorlevel 1 set NEED_INSTALL=1

if "%NEED_INSTALL%"=="1" (
    echo Installing dependencies... This may take a few minutes.
    echo.

    :: Upgrade pip first
    python -m pip install --upgrade pip

    :: Install PyTorch with CUDA support
    echo Installing PyTorch with CUDA 12.1 support...
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    if errorlevel 1 (
        echo %YELLOW%Warning: CUDA PyTorch install failed, trying CPU version...%RESET%
        pip install torch
    )

    :: Install llama-cpp-python with CUDA support
    echo.
    echo Installing llama-cpp-python with CUDA support...
    echo This requires Visual Studio Build Tools and CUDA Toolkit
    set CMAKE_ARGS=-DGGML_CUDA=on
    pip install llama-cpp-python
    if errorlevel 1 (
        echo %YELLOW%Warning: CUDA llama-cpp-python failed, installing CPU version...%RESET%
        set CMAKE_ARGS=
        pip install llama-cpp-python
    )

    :: Install remaining dependencies
    echo.
    echo Installing remaining dependencies...
    pip install faster-whisper==1.0.3 pyaudio numpy PyQt6 sounddevice silero-vad
    if errorlevel 1 (
        echo %RED%Error: Failed to install dependencies%RESET%
        echo.
        echo If pyaudio fails, you may need to install it manually:
        echo   pip install pipwin
        echo   pipwin install pyaudio
        pause
        exit /b 1
    )

    echo.
    echo %GREEN%All dependencies installed successfully%RESET%
) else (
    echo %GREEN%Dependencies already installed%RESET%
)

:: Launch application
echo.
echo %YELLOW%[4/4] Launching application...%RESET%
echo.

:: Build command
set CMD=python main.py

if not "%MODEL_PATH%"=="" (
    set CMD=%CMD% --model "%MODEL_PATH%"
) else (
    :: Check for model in common locations
    if exist "models\*.gguf" (
        for %%f in (models\*.gguf) do (
            echo %CYAN%Found model: %%f%RESET%
            set CMD=%CMD% --model "%%f"
            goto :found_model
        )
    )
    if exist "*.gguf" (
        for %%f in (*.gguf) do (
            echo %CYAN%Found model: %%f%RESET%
            set CMD=%CMD% --model "%%f"
            goto :found_model
        )
    )
    echo %YELLOW%No LLM model specified. Spanish completion will be disabled.%RESET%
    echo Use --model path\to\model.gguf to enable completions.
    echo.
)
:found_model

set CMD=%CMD%%EXTRA_ARGS%

echo Running: %CMD%
echo.
echo %GREEN%============================================%RESET%
echo %GREEN% Press Ctrl+C to stop, ESC to close overlay %RESET%
echo %GREEN%============================================%RESET%
echo.

%CMD%

:: Deactivate on exit
call %VENV_DIR%\Scripts\deactivate.bat 2>nul
echo.
echo Application closed.
pause
exit /b 0

:: ============================================
:: Helper functions
:: ============================================

:setup_only
echo %CYAN%Running setup only (no launch)...%RESET%
echo.

:: Check Python
where %PYTHON_CMD% >nul 2>&1
if errorlevel 1 (
    echo %RED%Error: Python not found%RESET%
    pause
    exit /b 1
)

:: Create venv
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Creating virtual environment...
    %PYTHON_CMD% -m venv %VENV_DIR%
)

:: Activate and install
call %VENV_DIR%\Scripts\activate.bat
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
set CMAKE_ARGS=-DGGML_CUDA=on
pip install llama-cpp-python
pip install faster-whisper==1.0.3 pyaudio numpy PyQt6 sounddevice silero-vad

echo.
echo %GREEN%Setup complete! Run launcher.bat to start the application.%RESET%
pause
exit /b 0

:clean_venv
echo %YELLOW%Removing virtual environment...%RESET%
if exist "%VENV_DIR%" (
    rmdir /s /q %VENV_DIR%
    echo %GREEN%Virtual environment removed.%RESET%
) else (
    echo No virtual environment found.
)
pause
exit /b 0

:show_help
echo.
echo %CYAN%Usage: launcher.bat [OPTIONS]%RESET%
echo.
echo Options:
echo   --model PATH    Path to GGUF LLM model file
echo   --terminal      Run in terminal mode (no GUI)
echo   --low-vram      Use low VRAM configuration
echo   --debug         Enable debug logging
echo   --setup         Setup only (install dependencies, don't run)
echo   --clean         Remove virtual environment
echo   --help          Show this help message
echo.
echo Examples:
echo   launcher.bat
echo   launcher.bat --model models\llama-3-8b.Q4_K_M.gguf
echo   launcher.bat --terminal --debug
echo   launcher.bat --setup
echo.
echo Model Download:
echo   Download Llama-3-8B-Instruct GGUF from:
echo   https://huggingface.co/TheBloke/Meta-Llama-3-8B-Instruct-GGUF
echo   Recommended: Q4_K_M quantization (~4.9GB)
echo.
pause
exit /b 0
