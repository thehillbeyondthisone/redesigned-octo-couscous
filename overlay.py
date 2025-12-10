"""
Overlay Module - Component D: The UI (Main Thread)

Non-blocking transparent overlay using PyQt6.
Displays:
- Line 1 (User): English transcription (White)
- Line 2 (Ghost): Spanish completion (Grey/Faded)
"""

import sys
import logging
from typing import Optional

from config import UIConfig

logger = logging.getLogger(__name__)

# Try to import PyQt6
try:
    from PyQt6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QLabel,
        QHBoxLayout, QFrame
    )
    from PyQt6.QtCore import Qt, pyqtSignal, QObject
    from PyQt6.QtGui import QFont, QColor, QPalette
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    logger.warning("PyQt6 not available, overlay disabled")


class SignalBridge(QObject):
    """
    Signal bridge for thread-safe UI updates.

    Qt signals are thread-safe and can be emitted from any thread.
    """
    update_user_text = pyqtSignal(str)
    update_ghost_text = pyqtSignal(str)
    update_status = pyqtSignal(str)


class Overlay(QWidget):
    """
    Transparent always-on-top overlay for displaying ghost text.

    Features:
    - Always on top
    - Semi-transparent background
    - Two-line display (user text + ghost text)
    - Status indicator for speech detection
    """

    def __init__(self, config: UIConfig):
        if not PYQT_AVAILABLE:
            raise RuntimeError("PyQt6 not available")

        super().__init__()
        self.config = config

        # Signal bridge for thread-safe updates
        self.signals = SignalBridge()
        self.signals.update_user_text.connect(self._set_user_text)
        self.signals.update_ghost_text.connect(self._set_ghost_text)
        self.signals.update_status.connect(self._set_status)

        self._init_ui()

    def _init_ui(self):
        """Initialize the UI components."""
        # Window settings
        self.setWindowTitle("Spanish Ghost Text Aid")
        self.setFixedSize(self.config.window_width, self.config.window_height)

        # Always on top, frameless, translucent background
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Main container with rounded corners
        self.container = QFrame(self)
        self.container.setFixedSize(
            self.config.window_width,
            self.config.window_height
        )
        self.container.setStyleSheet(f"""
            QFrame {{
                background-color: rgba(26, 26, 46, {int(self.config.background_opacity * 255)});
                border-radius: 15px;
                border: 2px solid rgba(255, 255, 255, 0.1);
            }}
        """)

        # Layout
        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(8)

        # Status indicator
        self.status_label = QLabel("Listening...")
        self.status_label.setFont(QFont(self.config.font_family, 10))
        self.status_label.setStyleSheet(f"color: #666666;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # User text (English - White)
        self.user_label = QLabel("Say something in English...")
        self.user_label.setFont(QFont(self.config.font_family, self.config.font_size_user))
        self.user_label.setStyleSheet(f"color: {self.config.color_user};")
        self.user_label.setWordWrap(True)
        self.user_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # Ghost text (Spanish - Grey)
        self.ghost_label = QLabel("")
        self.ghost_label.setFont(QFont(self.config.font_family, self.config.font_size_ghost, italic=True))
        self.ghost_label.setStyleSheet(f"color: {self.config.color_ghost};")
        self.ghost_label.setWordWrap(True)
        self.ghost_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # Add to layout
        layout.addWidget(self.status_label)
        layout.addWidget(self.user_label)
        layout.addWidget(self.ghost_label)
        layout.addStretch()

        # Enable dragging
        self._drag_position = None

    def mousePressEvent(self, event):
        """Handle mouse press for dragging."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        """Handle mouse move for dragging."""
        if event.buttons() == Qt.MouseButton.LeftButton and self._drag_position:
            self.move(event.globalPosition().toPoint() - self._drag_position)
            event.accept()

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        self._drag_position = None

    def _set_user_text(self, text: str):
        """Set user text (thread-safe via signal)."""
        self.user_label.setText(text)

    def _set_ghost_text(self, text: str):
        """Set ghost text (thread-safe via signal)."""
        self.ghost_label.setText(f"→ {text}" if text else "")

    def _set_status(self, status: str):
        """Set status text (thread-safe via signal)."""
        self.status_label.setText(status)

    # Public API for thread-safe updates
    def set_user_text(self, text: str):
        """Update user text from any thread."""
        self.signals.update_user_text.emit(text)

    def set_ghost_text(self, text: str):
        """Update ghost text from any thread."""
        self.signals.update_ghost_text.emit(text)

    def set_status(self, status: str):
        """Update status from any thread."""
        self.signals.update_status.emit(status)

    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        super().keyPressEvent(event)


class TerminalOverlay:
    """
    Fallback terminal-based overlay using rich library.

    Used when PyQt6 is not available or in terminal-only mode.
    """

    def __init__(self, config: UIConfig):
        self.config = config
        self._user_text = ""
        self._ghost_text = ""
        self._status = "Listening..."

        # Try to import rich
        try:
            from rich.console import Console
            from rich.live import Live
            from rich.panel import Panel
            from rich.text import Text
            self.console = Console()
            self.rich_available = True
        except ImportError:
            self.rich_available = False
            logger.warning("rich not available, using plain text output")

    def set_user_text(self, text: str):
        """Update user text."""
        self._user_text = text
        self._render()

    def set_ghost_text(self, text: str):
        """Update ghost text."""
        self._ghost_text = text
        self._render()

    def set_status(self, status: str):
        """Update status."""
        self._status = status
        self._render()

    def _render(self):
        """Render the overlay to terminal."""
        if self.rich_available:
            from rich.panel import Panel
            from rich.text import Text

            content = Text()
            content.append(f"[{self._status}]\n", style="dim")
            content.append(f"{self._user_text}\n", style="bold white")
            content.append(f"→ {self._ghost_text}", style="italic dim")

            # Clear and print
            self.console.clear()
            self.console.print(Panel(content, title="Spanish Ghost Text Aid", border_style="blue"))
        else:
            # Plain text fallback
            print("\033[2J\033[H")  # Clear screen
            print(f"=== Spanish Ghost Text Aid ===")
            print(f"Status: {self._status}")
            print(f"You: {self._user_text}")
            print(f"Ghost: → {self._ghost_text}")
            print("=" * 30)

    def show(self):
        """Show the overlay."""
        self._render()

    def close(self):
        """Close the overlay."""
        pass


def create_overlay(config: UIConfig, use_terminal: bool = False) -> Optional[object]:
    """
    Create an overlay instance.

    Args:
        config: UI configuration
        use_terminal: Force terminal mode

    Returns:
        Overlay instance or None if not available
    """
    if use_terminal or not PYQT_AVAILABLE:
        return TerminalOverlay(config)

    return Overlay(config)
