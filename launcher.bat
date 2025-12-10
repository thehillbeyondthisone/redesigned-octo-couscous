@echo off
setlocal enabledelayedexpansion

:: ============================================
:: Real-Time Spanish Autocomplete Ghost Text Aid
:: Windows Launcher Script with Interactive Menu
:: ============================================

title Spanish Ghost Text Aid - Launcher

:: Configuration
set VENV_DIR=venv
set PYTHON_CMD=python
set MODELS_DIR=models
set CONFIG_FILE=settings.json
set MODEL_PATH=
set EXTRA_ARGS=

:: Colors (Windows 10+)
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "CYAN=[96m"
set "MAGENTA=[95m"
set "WHITE=[97m"
set "RESET=[0m"

:: Check for command line args first
if not "%~1"=="" goto :parse_args

:: No args - show interactive menu
goto :main_menu

:: ============================================
:: Interactive Main Menu
:: ============================================
:main_menu
cls
echo %CYAN%============================================%RESET%
echo %CYAN%  Spanish Ghost Text Aid - Setup Menu      %RESET%
echo %CYAN%============================================%RESET%
echo.
echo  %WHITE%1.%RESET% %GREEN%Quick Start%RESET% (Setup + Run)
echo  %WHITE%2.%RESET% Install/Update Dependencies
echo  %WHITE%3.%RESET% Download Models
echo  %WHITE%4.%RESET% Configure Settings
echo  %WHITE%5.%RESET% Run Application
echo  %WHITE%6.%RESET% Advanced Options
echo  %WHITE%7.%RESET% Help
echo  %WHITE%0.%RESET% Exit
echo.
set /p choice="Select option [1-7, 0]: "

if "%choice%"=="1" goto :quick_start
if "%choice%"=="2" goto :install_deps_menu
if "%choice%"=="3" goto :download_models_menu
if "%choice%"=="4" goto :configure_menu
if "%choice%"=="5" goto :run_app
if "%choice%"=="6" goto :advanced_menu
if "%choice%"=="7" goto :show_help_menu
if "%choice%"=="0" exit /b 0
goto :main_menu

:: ============================================
:: Quick Start - Full Setup + Run
:: ============================================
:quick_start
cls
echo %CYAN%============================================%RESET%
echo %CYAN%  Quick Start - Full Setup                 %RESET%
echo %CYAN%============================================%RESET%
echo.

call :check_python
if errorlevel 1 goto :main_menu

call :setup_venv
call :install_all_deps
call :check_models

echo.
echo %GREEN%Setup complete! Starting application...%RESET%
timeout /t 2 >nul
goto :run_app

:: ============================================
:: Install Dependencies Menu
:: ============================================
:install_deps_menu
cls
echo %CYAN%============================================%RESET%
echo %CYAN%  Install Dependencies                     %RESET%
echo %CYAN%============================================%RESET%
echo.
echo  %WHITE%1.%RESET% Install ALL dependencies (recommended)
echo  %WHITE%2.%RESET% Install Core (PyQt6, sounddevice, numpy)
echo  %WHITE%3.%RESET% Install PyTorch (CUDA 12.1)
echo  %WHITE%4.%RESET% Install PyTorch (CPU only)
echo  %WHITE%5.%RESET% Install faster-whisper (speech recognition)
echo  %WHITE%6.%RESET% Install llama-cpp-python (CUDA)
echo  %WHITE%7.%RESET% Install llama-cpp-python (CPU)
echo  %WHITE%8.%RESET% Check installed packages
echo  %WHITE%0.%RESET% Back to main menu
echo.
set /p dep_choice="Select option: "

if "%dep_choice%"=="1" (
    call :check_python
    call :setup_venv
    call :install_all_deps
    pause
    goto :install_deps_menu
)
if "%dep_choice%"=="2" (
    call :setup_venv
    call :install_core
    pause
    goto :install_deps_menu
)
if "%dep_choice%"=="3" (
    call :setup_venv
    call :install_torch_cuda
    pause
    goto :install_deps_menu
)
if "%dep_choice%"=="4" (
    call :setup_venv
    call :install_torch_cpu
    pause
    goto :install_deps_menu
)
if "%dep_choice%"=="5" (
    call :setup_venv
    call :install_whisper
    pause
    goto :install_deps_menu
)
if "%dep_choice%"=="6" (
    call :setup_venv
    call :install_llama_cuda
    pause
    goto :install_deps_menu
)
if "%dep_choice%"=="7" (
    call :setup_venv
    call :install_llama_cpu
    pause
    goto :install_deps_menu
)
if "%dep_choice%"=="8" (
    call :setup_venv
    call :check_packages
    pause
    goto :install_deps_menu
)
if "%dep_choice%"=="0" goto :main_menu
goto :install_deps_menu

:: ============================================
:: Download Models Menu
:: ============================================
:download_models_menu
cls
echo %CYAN%============================================%RESET%
echo %CYAN%  Download Models                          %RESET%
echo %CYAN%============================================%RESET%
echo.
echo  %MAGENTA%LLM Models (for Spanish completion):%RESET%
echo  %WHITE%1.%RESET% Llama-3-8B-Instruct Q4_K_M (~4.9GB) - Recommended
echo  %WHITE%2.%RESET% Llama-3-8B-Instruct Q5_K_M (~5.5GB) - Better quality
echo  %WHITE%3.%RESET% Llama-3-8B-Instruct Q8_0  (~8.5GB) - Best quality
echo.
echo  %MAGENTA%Whisper Models (auto-downloaded on first run):%RESET%
echo  %WHITE%4.%RESET% Pre-download Whisper large-v3-turbo
echo  %WHITE%5.%RESET% Pre-download Whisper medium
echo  %WHITE%6.%RESET% Pre-download Whisper small
echo.
echo  %WHITE%7.%RESET% Open models folder
echo  %WHITE%8.%RESET% List downloaded models
echo  %WHITE%0.%RESET% Back to main menu
echo.
set /p model_choice="Select option: "

if "%model_choice%"=="1" call :download_llama_q4
if "%model_choice%"=="2" call :download_llama_q5
if "%model_choice%"=="3" call :download_llama_q8
if "%model_choice%"=="4" call :download_whisper large-v3-turbo
if "%model_choice%"=="5" call :download_whisper medium
if "%model_choice%"=="6" call :download_whisper small
if "%model_choice%"=="7" (
    if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"
    explorer "%MODELS_DIR%"
)
if "%model_choice%"=="8" call :list_models
if "%model_choice%"=="0" goto :main_menu
if not "%model_choice%"=="0" pause
goto :download_models_menu

:: ============================================
:: Configure Settings Menu
:: ============================================
:configure_menu
cls
echo %CYAN%============================================%RESET%
echo %CYAN%  Configure Settings                       %RESET%
echo %CYAN%============================================%RESET%
echo.
echo  %WHITE%1.%RESET% Set LLM model path
echo  %WHITE%2.%RESET% Set Whisper model size
echo  %WHITE%3.%RESET% Toggle low VRAM mode
echo  %WHITE%4.%RESET% Toggle debug mode
echo  %WHITE%5.%RESET% View current settings
echo  %WHITE%0.%RESET% Back to main menu
echo.

:: Show current settings
echo  %YELLOW%Current settings:%RESET%
if defined SAVED_MODEL echo    Model: %SAVED_MODEL%
if defined SAVED_WHISPER echo    Whisper: %SAVED_WHISPER%
if defined LOW_VRAM_MODE echo    Low VRAM: %LOW_VRAM_MODE%
if defined DEBUG_MODE echo    Debug: %DEBUG_MODE%
echo.

set /p config_choice="Select option: "

if "%config_choice%"=="1" (
    echo.
    echo Enter path to GGUF model file:
    set /p SAVED_MODEL="Model path: "
    echo %GREEN%Model path saved.%RESET%
    pause
)
if "%config_choice%"=="2" (
    echo.
    echo Whisper model sizes: tiny, base, small, medium, large-v2, large-v3, large-v3-turbo
    set /p SAVED_WHISPER="Whisper model: "
    echo %GREEN%Whisper model saved.%RESET%
    pause
)
if "%config_choice%"=="3" (
    if defined LOW_VRAM_MODE (
        set LOW_VRAM_MODE=
        echo %GREEN%Low VRAM mode disabled.%RESET%
    ) else (
        set LOW_VRAM_MODE=1
        echo %GREEN%Low VRAM mode enabled.%RESET%
    )
    pause
)
if "%config_choice%"=="4" (
    if defined DEBUG_MODE (
        set DEBUG_MODE=
        echo %GREEN%Debug mode disabled.%RESET%
    ) else (
        set DEBUG_MODE=1
        echo %GREEN%Debug mode enabled.%RESET%
    )
    pause
)
if "%config_choice%"=="5" (
    call :show_settings
    pause
)
if "%config_choice%"=="0" goto :main_menu
goto :configure_menu

:: ============================================
:: Advanced Options Menu
:: ============================================
:advanced_menu
cls
echo %CYAN%============================================%RESET%
echo %CYAN%  Advanced Options                         %RESET%
echo %CYAN%============================================%RESET%
echo.
echo  %WHITE%1.%RESET% Clean virtual environment
echo  %WHITE%2.%RESET% Reinstall all dependencies
echo  %WHITE%3.%RESET% Run in terminal mode
echo  %WHITE%4.%RESET% Open command prompt in venv
echo  %WHITE%5.%RESET% Check CUDA availability
echo  %WHITE%6.%RESET% View system info
echo  %WHITE%0.%RESET% Back to main menu
echo.
set /p adv_choice="Select option: "

if "%adv_choice%"=="1" (
    call :clean_venv
    pause
)
if "%adv_choice%"=="2" (
    call :clean_venv
    call :setup_venv
    call :install_all_deps
    pause
)
if "%adv_choice%"=="3" (
    set EXTRA_ARGS=%EXTRA_ARGS% --terminal
    goto :run_app
)
if "%adv_choice%"=="4" (
    call :setup_venv
    echo.
    echo %GREEN%Virtual environment activated. Type 'exit' to return.%RESET%
    cmd /k
)
if "%adv_choice%"=="5" (
    call :setup_venv
    call :check_cuda
    pause
)
if "%adv_choice%"=="6" (
    call :show_system_info
    pause
)
if "%adv_choice%"=="0" goto :main_menu
goto :advanced_menu

:: ============================================
:: Run Application
:: ============================================
:run_app
call :check_python
if errorlevel 1 goto :main_menu

call :setup_venv

:: Build command
set CMD=python main.py

:: Add model path
if defined SAVED_MODEL (
    set CMD=%CMD% --model "%SAVED_MODEL%"
) else (
    :: Auto-detect model
    if exist "%MODELS_DIR%\*.gguf" (
        for %%f in (%MODELS_DIR%\*.gguf) do (
            echo %CYAN%Found model: %%f%RESET%
            set CMD=%CMD% --model "%%f"
            goto :model_found
        )
    )
    if exist "*.gguf" (
        for %%f in (*.gguf) do (
            echo %CYAN%Found model: %%f%RESET%
            set CMD=%CMD% --model "%%f"
            goto :model_found
        )
    )
    echo %YELLOW%No LLM model found. Spanish completion disabled.%RESET%
)
:model_found

:: Add whisper model
if defined SAVED_WHISPER set CMD=%CMD% --whisper-model %SAVED_WHISPER%

:: Add flags
if defined LOW_VRAM_MODE set CMD=%CMD% --low-vram
if defined DEBUG_MODE set CMD=%CMD% --debug
set CMD=%CMD%%EXTRA_ARGS%

cls
echo %GREEN%============================================%RESET%
echo %GREEN%  Starting Spanish Ghost Text Aid          %RESET%
echo %GREEN%============================================%RESET%
echo.
echo Running: %CMD%
echo.
echo Press Ctrl+C to stop
echo.

%CMD%

echo.
echo Application closed.
pause
goto :main_menu

:: ============================================
:: Helper Functions
:: ============================================

:check_python
echo %YELLOW%Checking Python...%RESET%
where %PYTHON_CMD% >nul 2>&1
if errorlevel 1 (
    echo %RED%Error: Python not found in PATH%RESET%
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    exit /b 1
)
for /f "tokens=*" %%i in ('%PYTHON_CMD% --version 2^>^&1') do echo %GREEN%Found: %%i%RESET%
exit /b 0

:setup_venv
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo %YELLOW%Creating virtual environment...%RESET%
    %PYTHON_CMD% -m venv %VENV_DIR%
)
call %VENV_DIR%\Scripts\activate.bat
exit /b 0

:install_all_deps
echo.
echo %CYAN%Installing all dependencies...%RESET%
echo.
python -m pip install --upgrade pip

echo %YELLOW%[1/5] Installing PyTorch (CUDA 12.1)...%RESET%
pip install torch --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo %YELLOW%CUDA install failed, trying CPU...%RESET%
    pip install torch
)

echo %YELLOW%[2/5] Installing faster-whisper...%RESET%
pip install faster-whisper==1.0.3

echo %YELLOW%[3/5] Installing llama-cpp-python (CUDA)...%RESET%
set CMAKE_ARGS=-DGGML_CUDA=on
pip install llama-cpp-python
if errorlevel 1 (
    echo %YELLOW%CUDA build failed, trying CPU...%RESET%
    set CMAKE_ARGS=
    pip install llama-cpp-python
)

echo %YELLOW%[4/5] Installing audio dependencies...%RESET%
pip install pyaudio sounddevice
if errorlevel 1 (
    echo %YELLOW%Trying alternative pyaudio install...%RESET%
    pip install pipwin
    pipwin install pyaudio
)

echo %YELLOW%[5/5] Installing UI and utilities...%RESET%
pip install numpy PyQt6 silero-vad

echo.
echo %GREEN%All dependencies installed!%RESET%
exit /b 0

:install_core
echo %YELLOW%Installing core dependencies...%RESET%
pip install numpy PyQt6 sounddevice pyaudio silero-vad
exit /b 0

:install_torch_cuda
echo %YELLOW%Installing PyTorch with CUDA 12.1...%RESET%
pip install torch --index-url https://download.pytorch.org/whl/cu121
exit /b 0

:install_torch_cpu
echo %YELLOW%Installing PyTorch (CPU)...%RESET%
pip install torch
exit /b 0

:install_whisper
echo %YELLOW%Installing faster-whisper...%RESET%
pip install faster-whisper==1.0.3
exit /b 0

:install_llama_cuda
echo %YELLOW%Installing llama-cpp-python with CUDA...%RESET%
echo This requires Visual Studio Build Tools and CUDA Toolkit
set CMAKE_ARGS=-DGGML_CUDA=on
pip install llama-cpp-python --force-reinstall --no-cache-dir
exit /b 0

:install_llama_cpu
echo %YELLOW%Installing llama-cpp-python (CPU)...%RESET%
set CMAKE_ARGS=
pip install llama-cpp-python --force-reinstall --no-cache-dir
exit /b 0

:check_packages
echo.
echo %CYAN%Installed packages:%RESET%
pip list | findstr /i "torch faster-whisper llama-cpp-python PyQt6 sounddevice pyaudio numpy silero"
exit /b 0

:download_llama_q4
if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"
echo.
echo %CYAN%Downloading Llama-3-8B-Instruct Q4_K_M (~4.9GB)...%RESET%
echo This may take a while depending on your connection.
echo.
curl -L -o "%MODELS_DIR%\Meta-Llama-3-8B-Instruct-Q4_K_M.gguf" ^
    "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
if errorlevel 1 (
    echo %RED%Download failed. Please download manually from:%RESET%
    echo https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF
) else (
    echo %GREEN%Download complete!%RESET%
    set SAVED_MODEL=%MODELS_DIR%\Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
)
exit /b 0

:download_llama_q5
if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"
echo.
echo %CYAN%Downloading Llama-3-8B-Instruct Q5_K_M (~5.5GB)...%RESET%
curl -L -o "%MODELS_DIR%\Meta-Llama-3-8B-Instruct-Q5_K_M.gguf" ^
    "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf"
if errorlevel 1 (
    echo %RED%Download failed.%RESET%
) else (
    echo %GREEN%Download complete!%RESET%
    set SAVED_MODEL=%MODELS_DIR%\Meta-Llama-3-8B-Instruct-Q5_K_M.gguf
)
exit /b 0

:download_llama_q8
if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"
echo.
echo %CYAN%Downloading Llama-3-8B-Instruct Q8_0 (~8.5GB)...%RESET%
curl -L -o "%MODELS_DIR%\Meta-Llama-3-8B-Instruct-Q8_0.gguf" ^
    "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q8_0.gguf"
if errorlevel 1 (
    echo %RED%Download failed.%RESET%
) else (
    echo %GREEN%Download complete!%RESET%
    set SAVED_MODEL=%MODELS_DIR%\Meta-Llama-3-8B-Instruct-Q8_0.gguf
)
exit /b 0

:download_whisper
call :setup_venv
echo.
echo %CYAN%Pre-downloading Whisper %~1 model...%RESET%
python -c "from faster_whisper import WhisperModel; WhisperModel('%~1', device='cpu', compute_type='int8')"
if errorlevel 1 (
    echo %RED%Failed to download. Make sure faster-whisper is installed.%RESET%
) else (
    echo %GREEN%Whisper %~1 model downloaded!%RESET%
)
exit /b 0

:list_models
echo.
echo %CYAN%Downloaded models:%RESET%
echo.
if exist "%MODELS_DIR%\*.gguf" (
    echo %YELLOW%LLM Models (GGUF):%RESET%
    for %%f in (%MODELS_DIR%\*.gguf) do echo   %%~nxf (%%~zf bytes)
) else (
    echo   No GGUF models found in %MODELS_DIR%
)
echo.
echo %YELLOW%Whisper models are cached in:%RESET%
echo   %USERPROFILE%\.cache\huggingface\hub
exit /b 0

:clean_venv
echo %YELLOW%Removing virtual environment...%RESET%
if exist "%VENV_DIR%" (
    rmdir /s /q %VENV_DIR%
    echo %GREEN%Virtual environment removed.%RESET%
) else (
    echo No virtual environment found.
)
exit /b 0

:check_cuda
echo.
echo %CYAN%Checking CUDA availability...%RESET%
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}' if torch.cuda.is_available() else 'N/A'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'N/A')"
exit /b 0

:show_settings
echo.
echo %CYAN%Current Settings:%RESET%
echo   Model path: %SAVED_MODEL%
echo   Whisper model: %SAVED_WHISPER%
echo   Low VRAM mode: %LOW_VRAM_MODE%
echo   Debug mode: %DEBUG_MODE%
echo   Models directory: %MODELS_DIR%
echo   Virtual environment: %VENV_DIR%
exit /b 0

:show_system_info
echo.
echo %CYAN%System Information:%RESET%
echo.
systeminfo | findstr /C:"OS Name" /C:"Total Physical Memory"
echo.
where nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo %YELLOW%GPU Information:%RESET%
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
)
exit /b 0

:show_help_menu
cls
echo %CYAN%============================================%RESET%
echo %CYAN%  Help - Spanish Ghost Text Aid            %RESET%
echo %CYAN%============================================%RESET%
echo.
echo %WHITE%QUICK START:%RESET%
echo   1. Run launcher.bat
echo   2. Select "1. Quick Start" to install everything
echo   3. Download an LLM model (option 3)
echo   4. Run the application (option 5)
echo.
echo %WHITE%REQUIREMENTS:%RESET%
echo   - Python 3.10+
echo   - NVIDIA GPU with CUDA (recommended)
echo   - ~12GB VRAM for full models
echo   - ~8GB disk space for models
echo.
echo %WHITE%MODELS:%RESET%
echo   LLM: Llama-3-8B-Instruct GGUF (Q4_K_M recommended)
echo   Speech: Whisper large-v3-turbo (auto-downloads)
echo.
echo %WHITE%COMMAND LINE:%RESET%
echo   launcher.bat --model path\to\model.gguf
echo   launcher.bat --setup
echo   launcher.bat --terminal --debug
echo   launcher.bat --help
echo.
pause
goto :main_menu

:: ============================================
:: Command Line Argument Parsing
:: ============================================
:parse_args
if "%~1"=="" goto :run_with_args
if /i "%~1"=="--model" (
    set SAVED_MODEL=%~2
    shift & shift
    goto :parse_args
)
if /i "%~1"=="--whisper-model" (
    set SAVED_WHISPER=%~2
    shift & shift
    goto :parse_args
)
if /i "%~1"=="--terminal" (
    set EXTRA_ARGS=%EXTRA_ARGS% --terminal
    shift
    goto :parse_args
)
if /i "%~1"=="--low-vram" (
    set LOW_VRAM_MODE=1
    shift
    goto :parse_args
)
if /i "%~1"=="--debug" (
    set DEBUG_MODE=1
    shift
    goto :parse_args
)
if /i "%~1"=="--setup" goto :quick_start
if /i "%~1"=="--clean" (
    call :clean_venv
    exit /b 0
)
if /i "%~1"=="--help" (
    call :show_help_menu
    exit /b 0
)
shift
goto :parse_args

:run_with_args
goto :run_app
