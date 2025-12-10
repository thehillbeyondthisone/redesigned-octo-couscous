"""
Overlay Module - Component D: The UI (Main Thread)

Enhanced UI with:
- Audio device selection dropdown
- Start/Stop controls
- Debug output panel
- Ghost text display
- Settings dialog
"""

import sys
import json
import logging
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any
from datetime import datetime

from config import UIConfig

logger = logging.getLogger(__name__)

# Settings file path
SETTINGS_FILE = Path("settings.json")

# Try to import PyQt6
try:
    from PyQt6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
        QFrame, QPushButton, QComboBox, QTextEdit, QSplitter,
        QGroupBox, QSizePolicy, QDialog, QLineEdit, QSlider,
        QSpinBox, QDoubleSpinBox, QFileDialog, QTabWidget,
        QFormLayout, QCheckBox, QMessageBox
    )
    from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer
    from PyQt6.QtGui import QFont, QColor, QTextCursor
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    logger.warning("PyQt6 not available, overlay disabled")


def load_settings() -> Dict[str, Any]:
    """Load settings from JSON file."""
    defaults = {
        "llm_model_path": "",
        "whisper_model": "large-v3-turbo",
        "max_tokens": 50,
        "temperature": 0.7,
        "vad_threshold": 0.5,
        "silence_duration_ms": 300,
        "low_vram_mode": False,
        "force_cpu": False,
        "always_on_top": True,
        "auto_start": False,
    }
    try:
        if SETTINGS_FILE.exists():
            with open(SETTINGS_FILE, "r") as f:
                saved = json.load(f)
                defaults.update(saved)
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
    return defaults


def save_settings(settings: Dict[str, Any]):
    """Save settings to JSON file."""
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
        logger.info("Settings saved")
    except Exception as e:
        logger.error(f"Failed to save settings: {e}")


class SettingsDialog(QDialog):
    """Settings dialog for configuring the application."""

    def __init__(self, parent=None, current_settings: Dict[str, Any] = None):
        super().__init__(parent)
        self.settings = current_settings or load_settings()
        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle("Settings")
        self.setMinimumSize(500, 400)
        self.setStyleSheet("""
            QDialog {
                background-color: #1a1a2e;
                color: #ffffff;
            }
            QTabWidget::pane {
                border: 1px solid #3a3a5e;
                border-radius: 6px;
            }
            QTabBar::tab {
                background-color: #2a2a4e;
                color: #ffffff;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background-color: #3a3a5e;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #2a2a4e;
                border: 1px solid #3a3a5e;
                border-radius: 4px;
                padding: 6px;
                color: #ffffff;
            }
            QPushButton {
                background-color: #3a3a5e;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                color: #ffffff;
            }
            QPushButton:hover {
                background-color: #4a4a7e;
            }
            QLabel {
                color: #cccccc;
            }
            QCheckBox {
                color: #ffffff;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #2a2a4e;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #4a9eff;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QGroupBox {
                border: 1px solid #3a3a5e;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
                color: #888888;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)

        layout = QVBoxLayout(self)

        # Tab widget
        tabs = QTabWidget()
        tabs.addTab(self._create_models_tab(), "Models")
        tabs.addTab(self._create_audio_tab(), "Audio & VAD")
        tabs.addTab(self._create_ui_tab(), "Interface")
        layout.addWidget(tabs)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save_and_close)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(save_btn)
        layout.addLayout(btn_layout)

    def _create_models_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # LLM Model
        llm_group = QGroupBox("LLM Model (Spanish Completion)")
        llm_layout = QFormLayout(llm_group)

        self.llm_path_edit = QLineEdit(self.settings.get("llm_model_path", ""))
        self.llm_path_edit.setPlaceholderText("Path to .gguf model file")
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_llm_model)

        path_layout = QHBoxLayout()
        path_layout.addWidget(self.llm_path_edit)
        path_layout.addWidget(browse_btn)
        llm_layout.addRow("Model Path:", path_layout)

        # Max tokens setting
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(10, 200)
        self.max_tokens_spin.setValue(self.settings.get("max_tokens", 50))
        self.max_tokens_spin.setToolTip("Maximum tokens to generate (higher = longer responses)")
        llm_layout.addRow("Max Tokens:", self.max_tokens_spin)

        # Temperature setting
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.1, 2.0)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setValue(self.settings.get("temperature", 0.7))
        self.temperature_spin.setToolTip("Creativity (lower = more focused, higher = more creative)")
        llm_layout.addRow("Temperature:", self.temperature_spin)

        layout.addWidget(llm_group)

        # Whisper Model
        whisper_group = QGroupBox("Whisper Model (Speech Recognition)")
        whisper_layout = QFormLayout(whisper_group)

        self.whisper_combo = QComboBox()
        self.whisper_combo.addItems([
            "tiny", "base", "small", "medium",
            "large-v2", "large-v3", "large-v3-turbo"
        ])
        current_whisper = self.settings.get("whisper_model", "large-v3-turbo")
        idx = self.whisper_combo.findText(current_whisper)
        if idx >= 0:
            self.whisper_combo.setCurrentIndex(idx)
        whisper_layout.addRow("Model Size:", self.whisper_combo)

        self.low_vram_check = QCheckBox("Low VRAM Mode (uses smaller models)")
        self.low_vram_check.setChecked(self.settings.get("low_vram_mode", False))
        whisper_layout.addRow("", self.low_vram_check)

        self.force_cpu_check = QCheckBox("Force CPU Mode (disable CUDA for Whisper)")
        self.force_cpu_check.setChecked(self.settings.get("force_cpu", False))
        whisper_layout.addRow("", self.force_cpu_check)

        layout.addWidget(whisper_group)
        layout.addStretch()
        return widget

    def _create_audio_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # VAD Settings
        vad_group = QGroupBox("Voice Activity Detection")
        vad_layout = QFormLayout(vad_group)

        # VAD Threshold
        self.vad_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.vad_threshold_slider.setRange(10, 90)
        self.vad_threshold_slider.setValue(int(self.settings.get("vad_threshold", 0.5) * 100))
        self.vad_threshold_label = QLabel(f"{self.vad_threshold_slider.value()}%")
        self.vad_threshold_slider.valueChanged.connect(
            lambda v: self.vad_threshold_label.setText(f"{v}%")
        )

        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(self.vad_threshold_slider)
        threshold_layout.addWidget(self.vad_threshold_label)
        vad_layout.addRow("Speech Threshold:", threshold_layout)

        # Silence Duration
        self.silence_spin = QSpinBox()
        self.silence_spin.setRange(100, 2000)
        self.silence_spin.setSuffix(" ms")
        self.silence_spin.setValue(self.settings.get("silence_duration_ms", 300))
        vad_layout.addRow("Silence Duration:", self.silence_spin)

        layout.addWidget(vad_group)
        layout.addStretch()
        return widget

    def _create_ui_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # UI Settings
        ui_group = QGroupBox("Window Settings")
        ui_layout = QFormLayout(ui_group)

        self.always_on_top_check = QCheckBox("Always on top")
        self.always_on_top_check.setChecked(self.settings.get("always_on_top", True))
        ui_layout.addRow("", self.always_on_top_check)

        self.auto_start_check = QCheckBox("Auto-start on launch")
        self.auto_start_check.setChecked(self.settings.get("auto_start", False))
        ui_layout.addRow("", self.auto_start_check)

        layout.addWidget(ui_group)
        layout.addStretch()
        return widget

    def _browse_llm_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select LLM Model",
            str(Path.home()),
            "GGUF Models (*.gguf);;All Files (*)"
        )
        if path:
            self.llm_path_edit.setText(path)

    def _save_and_close(self):
        self.settings["llm_model_path"] = self.llm_path_edit.text()
        self.settings["max_tokens"] = self.max_tokens_spin.value()
        self.settings["temperature"] = self.temperature_spin.value()
        self.settings["whisper_model"] = self.whisper_combo.currentText()
        self.settings["vad_threshold"] = self.vad_threshold_slider.value() / 100.0
        self.settings["silence_duration_ms"] = self.silence_spin.value()
        self.settings["low_vram_mode"] = self.low_vram_check.isChecked()
        self.settings["force_cpu"] = self.force_cpu_check.isChecked()
        self.settings["always_on_top"] = self.always_on_top_check.isChecked()
        self.settings["auto_start"] = self.auto_start_check.isChecked()
        save_settings(self.settings)
        self.accept()

    def get_settings(self) -> Dict[str, Any]:
        return self.settings


class SignalBridge(QObject):
    """
    Signal bridge for thread-safe UI updates.
    Qt signals are thread-safe and can be emitted from any thread.
    """
    update_user_text = pyqtSignal(str)
    update_ghost_text = pyqtSignal(str)
    update_status = pyqtSignal(str)
    append_debug = pyqtSignal(str, str)  # (level, message)
    update_devices = pyqtSignal(list)


class Overlay(QWidget):
    """
    Enhanced overlay with device selection, controls, and debug output.
    """

    def __init__(self, config: UIConfig):
        if not PYQT_AVAILABLE:
            raise RuntimeError("PyQt6 not available")

        super().__init__()
        self.config = config

        # Callbacks
        self._on_start: Optional[Callable] = None
        self._on_stop: Optional[Callable] = None
        self._on_device_change: Optional[Callable[[int], None]] = None
        self._on_settings_change: Optional[Callable[[Dict[str, Any]], None]] = None

        # State
        self._is_running = False
        self._devices: List[dict] = []

        # Signal bridge for thread-safe updates
        self.signals = SignalBridge()
        self.signals.update_user_text.connect(self._set_user_text)
        self.signals.update_ghost_text.connect(self._set_ghost_text)
        self.signals.update_status.connect(self._set_status)
        self.signals.append_debug.connect(self._append_debug)
        self.signals.update_devices.connect(self._update_device_list)

        self._init_ui()

    def _init_ui(self):
        """Initialize the UI components."""
        # Window settings
        self.setWindowTitle("Spanish Ghost Text Aid")
        self.setMinimumSize(600, 450)
        self.resize(700, 500)

        # Window flags - normal window with controls
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Window
        )

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Apply dark theme
        self.setStyleSheet("""
            QWidget {
                background-color: #1a1a2e;
                color: #ffffff;
                font-family: Arial, sans-serif;
            }
            QGroupBox {
                border: 1px solid #3a3a5e;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #888888;
            }
            QPushButton {
                background-color: #3a3a5e;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #4a4a7e;
            }
            QPushButton:pressed {
                background-color: #2a2a4e;
            }
            QPushButton#startBtn {
                background-color: #2d5a3d;
            }
            QPushButton#startBtn:hover {
                background-color: #3d7a4d;
            }
            QPushButton#stopBtn {
                background-color: #5a2d2d;
            }
            QPushButton#stopBtn:hover {
                background-color: #7a3d3d;
            }
            QComboBox {
                background-color: #2a2a4e;
                border: 1px solid #3a3a5e;
                border-radius: 6px;
                padding: 6px 10px;
                min-width: 200px;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #888888;
                margin-right: 10px;
            }
            QComboBox QAbstractItemView {
                background-color: #2a2a4e;
                border: 1px solid #3a3a5e;
                selection-background-color: #4a4a7e;
            }
            QTextEdit {
                background-color: #0d0d1a;
                border: 1px solid #3a3a5e;
                border-radius: 6px;
                padding: 8px;
                font-family: Consolas, monospace;
                font-size: 11px;
            }
            QLabel#statusLabel {
                color: #888888;
                font-size: 12px;
            }
            QLabel#userLabel {
                color: #ffffff;
                font-size: 16px;
                font-weight: bold;
            }
            QLabel#ghostLabel {
                color: #888888;
                font-size: 14px;
                font-style: italic;
            }
        """)

        # === Control Panel ===
        control_group = QGroupBox("Controls")
        control_layout = QHBoxLayout(control_group)
        control_layout.setSpacing(15)

        # Audio device selection
        device_layout = QVBoxLayout()
        device_label = QLabel("Audio Input Device:")
        device_label.setStyleSheet("font-size: 11px; color: #888888;")
        self.device_combo = QComboBox()
        self.device_combo.addItem("Loading devices...")
        self.device_combo.currentIndexChanged.connect(self._on_device_selected)
        device_layout.addWidget(device_label)
        device_layout.addWidget(self.device_combo)
        control_layout.addLayout(device_layout)

        control_layout.addStretch()

        # Start/Stop buttons
        self.start_btn = QPushButton("▶ Start")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.clicked.connect(self._on_start_clicked)

        self.stop_btn = QPushButton("■ Stop")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        self.stop_btn.setEnabled(False)

        # Settings button
        self.settings_btn = QPushButton("⚙ Settings")
        self.settings_btn.clicked.connect(self._on_settings_clicked)

        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.settings_btn)

        main_layout.addWidget(control_group)

        # === Ghost Text Display ===
        ghost_group = QGroupBox("Ghost Text")
        ghost_layout = QVBoxLayout(ghost_group)
        ghost_layout.setSpacing(8)

        # Status indicator
        self.status_label = QLabel("⏸ Stopped")
        self.status_label.setObjectName("statusLabel")

        # User text (English)
        self.user_label = QLabel("Say something in English...")
        self.user_label.setObjectName("userLabel")
        self.user_label.setWordWrap(True)
        self.user_label.setMinimumHeight(30)

        # Ghost text (Spanish)
        self.ghost_label = QLabel("")
        self.ghost_label.setObjectName("ghostLabel")
        self.ghost_label.setWordWrap(True)
        self.ghost_label.setMinimumHeight(25)

        ghost_layout.addWidget(self.status_label)
        ghost_layout.addWidget(self.user_label)
        ghost_layout.addWidget(self.ghost_label)

        main_layout.addWidget(ghost_group)

        # === Debug Output ===
        debug_group = QGroupBox("Debug Output")
        debug_layout = QVBoxLayout(debug_group)

        self.debug_output = QTextEdit()
        self.debug_output.setReadOnly(True)
        self.debug_output.setMinimumHeight(120)
        self.debug_output.setPlaceholderText("Debug messages will appear here...")

        # Clear button
        debug_btn_layout = QHBoxLayout()
        debug_btn_layout.addStretch()
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.debug_output.clear)
        debug_btn_layout.addWidget(clear_btn)

        debug_layout.addWidget(self.debug_output)
        debug_layout.addLayout(debug_btn_layout)

        main_layout.addWidget(debug_group, stretch=1)

        # Enable dragging from title area
        self._drag_position = None

    def _on_device_selected(self, index: int):
        """Handle device selection change."""
        if index < 0 or index >= len(self._devices):
            return

        device = self._devices[index]
        self.append_debug("INFO", f"Selected device: {device['name']}")

        if self._on_device_change:
            self._on_device_change(device['index'])

    def _on_start_clicked(self):
        """Handle start button click."""
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._is_running = True
        self._set_status("▶ Starting...")

        if self._on_start:
            self._on_start()

    def _on_stop_clicked(self):
        """Handle stop button click."""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._is_running = False
        self._set_status("⏸ Stopped")

        if self._on_stop:
            self._on_stop()

    def _on_settings_clicked(self):
        """Handle settings button click."""
        dialog = SettingsDialog(self, load_settings())
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_settings = dialog.get_settings()
            self.append_debug("INFO", "Settings saved. Restart may be required for some changes.")
            if self._on_settings_change:
                self._on_settings_change(new_settings)

    def _set_user_text(self, text: str):
        """Set user text (thread-safe via signal)."""
        self.user_label.setText(text if text else "Say something in English...")

    def _set_ghost_text(self, text: str):
        """Set ghost text (thread-safe via signal)."""
        self.ghost_label.setText(f"→ {text}" if text else "")

    def _set_status(self, status: str):
        """Set status text (thread-safe via signal)."""
        self.status_label.setText(status)

    def _append_debug(self, level: str, message: str):
        """Append debug message (thread-safe via signal)."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # Color code by level
        colors = {
            "DEBUG": "#666666",
            "INFO": "#4a9eff",
            "WARNING": "#ffa500",
            "ERROR": "#ff4444",
        }
        color = colors.get(level.upper(), "#ffffff")

        html = f'<span style="color: #666666;">[{timestamp}]</span> '
        html += f'<span style="color: {color};">[{level.upper()}]</span> '
        html += f'<span style="color: #ffffff;">{message}</span>'

        self.debug_output.append(html)

        # Auto-scroll to bottom
        cursor = self.debug_output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.debug_output.setTextCursor(cursor)

    def _update_device_list(self, devices: list):
        """Update device dropdown (thread-safe via signal)."""
        self._devices = devices
        self.device_combo.clear()

        if not devices:
            self.device_combo.addItem("No audio devices found")
            return

        default_index = 0
        for i, dev in enumerate(devices):
            prefix = "★ " if dev.get('is_default') else ""
            self.device_combo.addItem(f"{prefix}{dev['name']}")
            if dev.get('is_default'):
                default_index = i

        self.device_combo.setCurrentIndex(default_index)

    # === Public API ===

    def set_user_text(self, text: str):
        """Update user text from any thread."""
        self.signals.update_user_text.emit(text)

    def set_ghost_text(self, text: str):
        """Update ghost text from any thread."""
        self.signals.update_ghost_text.emit(text)

    def set_status(self, status: str):
        """Update status from any thread."""
        self.signals.update_status.emit(status)

    def append_debug(self, level: str, message: str):
        """Append debug message from any thread."""
        self.signals.append_debug.emit(level, message)

    def update_devices(self, devices: list):
        """Update device list from any thread."""
        self.signals.update_devices.emit(devices)

    def set_callbacks(
        self,
        on_start: Optional[Callable] = None,
        on_stop: Optional[Callable] = None,
        on_device_change: Optional[Callable[[int], None]] = None,
        on_settings_change: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """Set callback functions for UI events."""
        self._on_start = on_start
        self._on_stop = on_stop
        self._on_device_change = on_device_change
        self._on_settings_change = on_settings_change

    def get_selected_device_index(self) -> Optional[int]:
        """Get currently selected device index."""
        idx = self.device_combo.currentIndex()
        if 0 <= idx < len(self._devices):
            return self._devices[idx]['index']
        return None

    def set_running_state(self, is_running: bool):
        """Update UI to reflect running state."""
        self._is_running = is_running
        self.start_btn.setEnabled(not is_running)
        self.stop_btn.setEnabled(is_running)

    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        super().keyPressEvent(event)


class TerminalOverlay:
    """
    Fallback terminal-based overlay using rich library.
    """

    def __init__(self, config: UIConfig):
        self.config = config
        self._user_text = ""
        self._ghost_text = ""
        self._status = "Stopped"
        self._debug_lines = []

        try:
            from rich.console import Console
            self.console = Console()
            self.rich_available = True
        except ImportError:
            self.rich_available = False

    def set_user_text(self, text: str):
        self._user_text = text
        self._render()

    def set_ghost_text(self, text: str):
        self._ghost_text = text
        self._render()

    def set_status(self, status: str):
        self._status = status
        self._render()

    def append_debug(self, level: str, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._debug_lines.append(f"[{timestamp}] [{level}] {message}")
        if len(self._debug_lines) > 10:
            self._debug_lines = self._debug_lines[-10:]
        self._render()

    def update_devices(self, devices: list):
        pass

    def set_callbacks(self, **kwargs):
        pass

    def get_selected_device_index(self):
        return None

    def set_running_state(self, is_running: bool):
        pass

    def _render(self):
        if self.rich_available:
            from rich.panel import Panel
            from rich.text import Text

            content = Text()
            content.append(f"[{self._status}]\n", style="dim")
            content.append(f"{self._user_text}\n", style="bold white")
            content.append(f"→ {self._ghost_text}\n\n", style="italic dim")

            for line in self._debug_lines[-5:]:
                content.append(f"{line}\n", style="dim")

            self.console.clear()
            self.console.print(Panel(content, title="Spanish Ghost Text Aid"))
        else:
            print("\033[2J\033[H")
            print(f"=== Spanish Ghost Text Aid ===")
            print(f"Status: {self._status}")
            print(f"You: {self._user_text}")
            print(f"Ghost: → {self._ghost_text}")
            print("-" * 30)
            for line in self._debug_lines[-5:]:
                print(line)

    def show(self):
        self._render()

    def close(self):
        pass


class DebugLogHandler(logging.Handler):
    """Custom logging handler that sends logs to the overlay."""

    def __init__(self, overlay):
        super().__init__()
        self.overlay = overlay

    def emit(self, record):
        try:
            msg = self.format(record)
            self.overlay.append_debug(record.levelname, msg)
        except Exception:
            pass


def create_overlay(config: UIConfig, use_terminal: bool = False):
    """Create an overlay instance."""
    if use_terminal or not PYQT_AVAILABLE:
        return TerminalOverlay(config)
    return Overlay(config)


def setup_debug_logging(overlay) -> logging.Handler:
    """Set up logging to output to the overlay debug panel."""
    handler = DebugLogHandler(overlay)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger().addHandler(handler)
    return handler
