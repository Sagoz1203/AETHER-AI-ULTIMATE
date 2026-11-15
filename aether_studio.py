
from __future__ import annotations

import os
import sys
import re
import json
import traceback
import subprocess
import urllib.request
import urllib.parse
from pathlib import Path
from datetime import datetime

from PySide6.QtCore import Qt, QObject, QThread, Signal, QPointF
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor, QIcon, QFont
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QGridLayout, QStackedWidget, QListWidget,
    QListWidgetItem, QLabel, QPushButton, QLineEdit, QTextEdit,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QFileDialog,
    QMessageBox, QGroupBox, QScrollArea, QProgressBar, QSplitter,
    QFrame, QSlider
)

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:64,expandable_segments:True"
)

try:
    import app as engine
except Exception as exc:
    engine = None
    ENGINE_ERROR = exc
else:
    ENGINE_ERROR = None


APP_NAME = "AETHER AI ULTIMATE STUDIO"
VERSION = "ULTIMATE 2026"


def smooth_video_to_60fps(video_path, callback):
    """Create a motion-interpolated 60 FPS copy using FFmpeg minterpolate."""
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Видео для интерполяции не найдено: {video_path}")
    ffmpeg = engine.get_ffmpeg()
    output_path = video_path.with_name(video_path.stem + "_60fps.mp4")
    if output_path.exists():
        output_path = video_path.with_name(
            f"{video_path.stem}_60fps_{datetime.now():%H%M%S}.mp4"
        )
    command = [
        ffmpeg, "-y", "-i", str(video_path),
        "-map", "0:v:0", "-map", "0:a?",
        "-vf",
        "minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", str(output_path)
    ]
    callback(None, "Интерполяция движения до 60 FPS: FFmpeg minterpolate. Это может занять время.")
    process = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=7200
    )
    if process.returncode != 0 or not output_path.is_file():
        raise RuntimeError(
            "Не удалось создать 60 FPS версию.\n"
            + (process.stderr or process.stdout or "FFmpeg error")[-3500:]
        )
    callback(None, f"Плавная версия 60 FPS сохранена: {output_path}")


def _video_path_from_message(message):
    if not message:
        return None
    match = re.search(r'([A-Za-z]:[\\/][^\r\n]*?\.mp4|/[^\r\n]*?\.mp4)', str(message), re.I)
    if not match:
        return None
    candidate = Path(match.group(1).strip().strip('"'))
    return candidate if candidate.is_file() else None


class Worker(QObject):
    message = Signal(object, str)
    finished = Signal()
    failed = Signal(str)

    def __init__(self, function):
        super().__init__()
        self.function = function

    def run(self):
        try:
            self.function(self.message.emit)
            self.finished.emit()
        except Exception:
            self.failed.emit(traceback.format_exc())


class MaskCanvas(QWidget):
    """Image preview with a paintable inpainting mask."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(420, 360)
        self.setMouseTracking(True)
        self.image = None
        self.image_path = None
        self.strokes = []
        self.current_stroke = None
        self.brush_size = 35

    def load_image(self, path):
        from PIL import Image
        self.image_path = str(path)
        self.image = Image.open(path).convert("RGB")
        self.strokes.clear()
        self.current_stroke = None
        self.update()

    def clear_mask(self):
        self.strokes.clear()
        self.current_stroke = None
        self.update()

    def _display_rect(self):
        if self.image is None:
            return None

        iw, ih = self.image.size
        scale = min(
            self.width() / iw,
            self.height() / ih
        )
        w = iw * scale
        h = ih * scale

        x = (self.width() - w) / 2
        y = (self.height() - h) / 2

        return x, y, w, h

    def _map_position(self, pos):
        rect = self._display_rect()
        if rect is None:
            return None

        x, y, w, h = rect

        px = (pos.x() - x) / w
        py = (pos.y() - y) / h

        if not (0 <= px <= 1 and 0 <= py <= 1):
            return None

        iw, ih = self.image.size

        return (
            int(px * iw),
            int(py * ih)
        )

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton or self.image is None:
            return

        point = self._map_position(event.position())

        if point:
            self.current_stroke = [point]
            self.strokes.append(self.current_stroke)
            self.update()

    def mouseMoveEvent(self, event):
        if self.current_stroke is None:
            return

        point = self._map_position(event.position())

        if point:
            self.current_stroke.append(point)
            self.update()

    def mouseReleaseEvent(self, event):
        self.current_stroke = None

    def get_mask(self):
        from PIL import Image, ImageDraw

        if self.image is None:
            raise RuntimeError("Сначала загрузи исходное изображение.")

        mask = Image.new("L", self.image.size, 0)
        draw = ImageDraw.Draw(mask)

        radius = max(1, int(self.brush_size / 2))

        for stroke in self.strokes:
            if not stroke:
                continue

            if len(stroke) == 1:
                x, y = stroke[0]
                draw.ellipse(
                    (x-radius, y-radius, x+radius, y+radius),
                    fill=255
                )
            else:
                draw.line(
                    stroke,
                    fill=255,
                    width=radius * 2
                )

                for x, y in stroke:
                    draw.ellipse(
                        (x-radius, y-radius, x+radius, y+radius),
                        fill=255
                    )

        return mask

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#101522"))

        if self.image is None:
            painter.setPen(QColor("#8995ad"))
            painter.drawText(
                self.rect(),
                Qt.AlignCenter,
                "Открой изображение для редактирования"
            )
            return

        data = self.image.tobytes("raw", "RGB")
        iw, ih = self.image.size

        qimage = QImage(
            data,
            iw,
            ih,
            iw * 3,
            QImage.Format_RGB888
        ).copy()

        pixmap = QPixmap.fromImage(qimage)
        x, y, w, h = self._display_rect()

        target = self.rect()
        painter.drawPixmap(
            int(x), int(y),
            int(w), int(h),
            pixmap
        )

        painter.setPen(
            QPen(QColor(255, 40, 100, 150),
                 max(1, int(self.brush_size * w / iw)),
                 Qt.SolidLine,
                 Qt.RoundCap,
                 Qt.RoundJoin)
        )

        for stroke in self.strokes:
            if not stroke:
                continue

            points = [
                QPointF(
                    x + px * w / iw,
                    y + py * h / ih
                )
                for px, py in stroke
            ]

            if len(points) == 1:
                painter.drawPoint(points[0])
            else:
                for index in range(len(points)-1):
                    painter.drawLine(
                        points[index],
                        points[index+1]
                    )


class AetherStudio(QMainWindow):

    def __init__(self):
        super().__init__()

        if engine is None:
            raise RuntimeError(
                "Не удалось импортировать app.py:\n"
                + str(ENGINE_ERROR)
            )

        self.setWindowTitle(
            f"{APP_NAME} — {VERSION}"
        )
        self.resize(1540, 960)
        self.setMinimumSize(1120, 740)

        self.fields = {}
        self.pages = {}
        self.timeline_items = []
        self.scene_data = []

        self.worker = None
        self.thread = None
        self.last_image = None
        self._close_when_done = False

        self._build_interface()
        self.refresh_models()
        self.refresh_system()

    # ---------------------------------------------------------
    # GENERAL UI
    # ---------------------------------------------------------

    def _build_interface(self):
        root = QWidget()
        self.setCentralWidget(root)

        main = QVBoxLayout(root)
        main.setContentsMargins(12, 12, 12, 8)
        main.setSpacing(8)

        header = QHBoxLayout()

        title = QLabel("AETHER")
        title.setObjectName("MainTitle")

        subtitle = QLabel(
            "AI ULTIMATE STUDIO  /  LOCAL GENERATION ENVIRONMENT"
        )
        subtitle.setObjectName("Subtitle")

        header.addWidget(title)
        header.addWidget(subtitle)
        header.addStretch()

        self.gpu_label = QLabel("ENGINE: INITIALIZING")
        self.gpu_label.setObjectName("StatusLabel")
        header.addWidget(self.gpu_label)

        main.addLayout(header)

        splitter = QSplitter(Qt.Horizontal)

        self.navigation = QListWidget()
        self.navigation.setObjectName("Navigation")
        self.navigation.setFixedWidth(230)

        self.stack = QStackedWidget()

        splitter.addWidget(self.navigation)
        splitter.addWidget(self.stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main.addWidget(splitter, 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)

        main.addWidget(self.progress)

        bottom = QHBoxLayout()

        self.status = QLabel("Ready")
        self.status.setObjectName("StatusLabel")

        self.busy_label = QLabel("IDLE")
        self.busy_label.setObjectName("StatusLabel")

        bottom.addWidget(self.status, 1)
        bottom.addWidget(self.busy_label)

        main.addLayout(bottom)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(105)
        self.log_box.setPlaceholderText("Activity log")

        main.addWidget(self.log_box)

        self._create_pages()

        self.navigation.currentRowChanged.connect(
            self.stack.setCurrentIndex
        )

        self.navigation.setCurrentRow(0)

        self.setStyleSheet(self._stylesheet())

    def _stylesheet(self):
        return """
        QWidget {
            background: #101522;
            color: #e5eaf5;
            font-family: Segoe UI;
            font-size: 10pt;
        }

        QLabel#MainTitle {
            font-size: 24pt;
            font-weight: 800;
            color: #ffffff;
        }

        QLabel#Subtitle {
            color: #8190ac;
            font-size: 9pt;
        }

        QLabel#StatusLabel {
            color: #8da2c5;
            padding: 5px;
        }

        QListWidget#Navigation {
            background: #151d2d;
            border: 1px solid #27344b;
            border-radius: 12px;
            padding: 8px;
            outline: none;
        }

        QListWidget#Navigation::item {
            min-height: 42px;
            padding: 8px;
            border-radius: 7px;
            margin: 2px;
        }

        QListWidget#Navigation::item:selected {
            background: #294b79;
            color: white;
        }

        QGroupBox {
            background: #151d2d;
            border: 1px solid #293750;
            border-radius: 10px;
            margin-top: 14px;
            padding: 14px;
            font-weight: 700;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 6px;
            color: #9dc7ff;
        }

        QLineEdit, QTextEdit, QComboBox,
        QSpinBox, QDoubleSpinBox {
            background: #0e1625;
            border: 1px solid #33425c;
            border-radius: 6px;
            padding: 8px;
            min-height: 22px;
            selection-background-color: #3266a5;
        }

        QTextEdit {
            min-height: 75px;
        }

        QPushButton {
            background: #254d7d;
            border: 1px solid #3c6da6;
            border-radius: 7px;
            padding: 10px 12px;
            min-height: 24px;
            font-weight: 600;
        }

        QPushButton:hover {
            background: #32649c;
        }

        QPushButton:pressed {
            background: #1b385c;
        }

        QPushButton:disabled {
            background: #303846;
            color: #788398;
        }

        QProgressBar {
            background: #182235;
            border: 1px solid #34445e;
            border-radius: 6px;
            height: 14px;
            text-align: center;
        }

        QProgressBar::chunk {
            background: #3984d6;
            border-radius: 5px;
        }

        QScrollArea {
            border: none;
        }

        QCheckBox {
            spacing: 8px;
            padding: 5px;
        }

        QSlider::groove:horizontal {
            height: 6px;
            background: #293750;
        }

        QSlider::handle:horizontal {
            background: #5ba4f5;
            width: 15px;
            margin: -5px 0;
            border-radius: 7px;
        }
        """

    def _avatar_icon(self):
        pixmap = QPixmap(40, 40)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        painter.setBrush(QColor("#28588a"))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(2, 2, 36, 36)

        painter.setBrush(QColor("#f0c5a0"))
        painter.drawEllipse(13, 8, 14, 14)

        painter.setBrush(QColor("#d9e9ff"))
        painter.drawEllipse(8, 23, 24, 17)

        painter.end()

        return QIcon(pixmap)

    def _create_pages(self):
        definitions = [
            ("🎨  Image Studio", "image"),
            ("🖌  AI Canvas / Edit", "canvas"),
            ("Character Studio", "character"),
            ("🎬  Scene Studio", "scene"),
            ("🎞  Motion / Video", "video"),
            ("✂  Timeline / Export", "timeline"),
            ("⚙  ComfyUI Bridge", "comfy"),
            ("👤  Face Studio", "face"),
            ("✨  Enhancement", "enhance"),
            ("◈  ControlNet", "control"),
            ("🎙  Voice Studio", "voice"),
            ("🖥  Model / System", "system"),
        ]

        for label, key in definitions:
            page = self._new_page(key)
            item = QListWidgetItem(label)

            if key == "character":
                item.setIcon(self._avatar_icon())
                item.setText("  Character Studio")

            self.navigation.addItem(item)
            self.stack.addWidget(page)
            self.pages[key] = page

        self._build_image_page()
        self._build_canvas_page()
        self._build_character_page()
        self._build_scene_page()
        self._build_video_page()
        self._build_timeline_page()
        self._build_comfy_page()
        self._build_face_page()
        self._build_enhance_page()
        self._build_control_page()
        self._build_voice_page()
        self._build_system_page()

    def _new_page(self, key):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(22, 18, 22, 24)
        layout.setSpacing(14)

        scroll.setWidget(content)
        scroll.body = layout

        return scroll

    def _heading(self, key, title, description):
        body = self.pages[key].body

        heading = QLabel(title)
        heading.setStyleSheet(
            "font-size:19pt;font-weight:800;color:#ffffff;"
        )

        desc = QLabel(description)
        desc.setWordWrap(True)
        desc.setStyleSheet("color:#91a1bd;")

        body.addWidget(heading)
        body.addWidget(desc)

    def _group(self, key, title):
        group = QGroupBox(title)
        layout = QFormLayout(group)
        layout.setLabelAlignment(Qt.AlignLeft)
        layout.setHorizontalSpacing(18)
        layout.setVerticalSpacing(12)
        self.pages[key].body.addWidget(group)
        return layout

    def _line(self, form, key, label, default=""):
        widget = QLineEdit(str(default))
        widget.setClearButtonEnabled(True)
        form.addRow(label, widget)
        self.fields[key] = widget
        return widget

    def _text(self, form, key, label, default="", height=100):
        widget = QTextEdit()
        widget.setPlainText(default)
        widget.setMinimumHeight(height)
        form.addRow(label, widget)
        self.fields[key] = widget
        return widget

    def _combo(self, form, key, label, values, default=None):
        widget = QComboBox()
        widget.addItems([str(x) for x in values])

        if default is not None:
            index = widget.findText(str(default))
            if index >= 0:
                widget.setCurrentIndex(index)

        form.addRow(label, widget)
        self.fields[key] = widget
        return widget

    def _file(self, form, key, label, filters="All files (*)"):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)

        edit = QLineEdit()
        edit.setPlaceholderText("Выбери файл...")
        browse = QPushButton("Browse")

        browse.setMaximumWidth(105)
        browse.clicked.connect(
            lambda: self._browse(edit, filters)
        )

        layout.addWidget(edit, 1)
        layout.addWidget(browse)

        form.addRow(label, row)
        self.fields[key] = edit

        return edit

    def _browse(self, edit, filters):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select file",
            str(engine.ROOT),
            filters
        )

        if path:
            edit.setText(path)

    def _button(self, key, label, callback, parent=None):
        button = QPushButton(label)
        button.setMinimumHeight(44)
        button.setSizePolicy(
            button.sizePolicy().horizontalPolicy(),
            button.sizePolicy().verticalPolicy()
        )
        button.clicked.connect(callback)

        self.pages[key].body.addWidget(button)
        return button

    def _checkbox(self, form, key, label, default=False):
        widget = QCheckBox(label)
        widget.setChecked(default)
        form.addRow(widget)
        self.fields[key] = widget
        return widget

    def value(self, key, default=""):
        widget = self.fields.get(key)

        if widget is None:
            return default

        if isinstance(widget, QTextEdit):
            return widget.toPlainText().strip()

        if isinstance(widget, QCheckBox):
            return widget.isChecked()

        return widget.text().strip() if isinstance(widget, QLineEdit) else widget.currentText()

    def number(self, key, default, cast=int):
        try:
            return cast(self.value(key, str(default)))
        except Exception:
            raise ValueError(
                f"Неверное числовое значение: {key}"
            )

    def relative(self, path):
        if not path:
            return None

        p = Path(path)

        try:
            return str(p.resolve().relative_to(engine.ROOT.resolve()))
        except Exception:
            return str(p)

    def _model_values(self):
        return engine.list_checkpoints()

    def _add_model_combo(self, form, key="model"):
        values = self._model_values()

        combo = self._combo(
            form,
            key,
            "Checkpoint",
            values
        )

        if values:
            preferred = getattr(
                engine,
                "PREFERRED_CHECKPOINT",
                ""
            )

            index = combo.findText(preferred)

            if index >= 0:
                combo.setCurrentIndex(index)

        return combo

    # ---------------------------------------------------------
    # IMAGE STUDIO
    # ---------------------------------------------------------

    def _build_image_page(self):
        self._heading(
            "image",
            "IMAGE STUDIO",
            "Локальная генерация изображений, дополнительные компоненты и управление seed."
        )

        form = self._group("image", "Generation")

        self._add_model_combo(form, "image_model")

        self._text(
            form,
            "image_prompt",
            "Prompt",
            "cinematic portrait, detailed realistic lighting, high quality",
            120
        )

        self._text(
            form,
            "image_negative",
            "Negative prompt",
            "blurry, low quality, artifacts, deformed anatomy, extra fingers",
            75
        )

        self._line(form, "image_steps", "Steps", "25")
        self._line(form, "image_cfg", "CFG", "6.5")
        self._line(form, "image_width", "Width", "768")
        self._line(form, "image_height", "Height", "768")
        self._line(form, "image_seed", "Seed (-1 = random)", "-1")

        optional = self._group("image", "Optional components")

        self._combo(
            optional,
            "image_lora",
            "LoRA",
            [""] + [
                str(p.relative_to(engine.ROOT))
                for p in engine.component_files("lora")
            ]
        )

        self._line(optional, "image_lora_scale", "LoRA scale", "0.8")

        self._combo(
            optional,
            "image_vae",
            "VAE",
            [""] + [
                str(p.relative_to(engine.ROOT))
                for p in engine.component_files("vae")
            ]
        )

        self._combo(
            optional,
            "image_embedding",
            "Embedding",
            [""] + [
                str(p.relative_to(engine.ROOT))
                for p in engine.component_files("embedding")
            ]
        )

        self._combo(
            optional,
            "image_ip",
            "IP-Adapter",
            [""] + [
                str(p.relative_to(engine.ROOT))
                for p in engine.component_files("ip_adapter")
            ]
        )

        self._file(
            optional,
            "image_ip_reference",
            "IP-Adapter reference",
            "Images (*.png *.jpg *.jpeg *.webp)"
        )

        self._button(
            "image",
            "GENERATE IMAGE",
            self.generate_image
        )

        self._button(
            "image",
            "Refresh checkpoint list",
            self.refresh_models
        )

    def generate_image(self):
        model = self.value("image_model")
        if not model:
            return self.warning("Image Studio", "Нет доступных checkpoints.")

        prompt = self.value("image_prompt")
        if not prompt:
            return self.warning("Image Studio", "Введи prompt.")

        try:
            steps = self.number("image_steps", 25)
            cfg = self.number("image_cfg", 6.5, float)
            width = self.number("image_width", 768)
            height = self.number("image_height", 768)
            seed = self.number("image_seed", -1)

            if not 1 <= steps <= 100:
                raise ValueError("Steps: 1–100.")

            if not 1 <= cfg <= 30:
                raise ValueError("CFG: 1–30.")

            if not 256 <= width <= 2048 or not 256 <= height <= 2048:
                raise ValueError("Размер: 256–2048.")

        except Exception as exc:
            return self.warning("Parameters", str(exc))

        model = self.relative(model)

        lora = self.value("image_lora") or None
        vae = self.value("image_vae") or None
        embedding = self.value("image_embedding") or None
        ip = self.value("image_ip") or None

        reference = self.value("image_ip_reference") or None
        negative = self.value("image_negative")
        lora_scale = self.number("image_lora_scale", 0.8, float)

        def job(callback):
            engine.run_generation(
                "txt2img",
                model,
                prompt,
                negative,
                None,
                None,
                steps,
                cfg,
                0.0,
                width,
                height,
                seed,
                callback,
                lora_rel=lora,
                lora_scale=lora_scale,
                vae_rel=vae,
                embedding_rel=embedding,
                ip_adapter_rel=ip,
                ip_image_path=reference
            )

        self.start_task("Image generation", job)

    # ---------------------------------------------------------
    # AI CANVAS
    # ---------------------------------------------------------

    def _build_canvas_page(self):
        self._heading(
            "canvas",
            "AI CANVAS / EDIT",
            "Рисуй маску поверх изображения и используй Inpaint, Outpaint или Image-to-Image."
        )

        form = self._group("canvas", "Source and generation")

        self._add_model_combo(form, "canvas_model")

        self._text(
            form,
            "canvas_prompt",
            "Edit prompt",
            "replace only the selected area, preserve the rest of the image",
            100
        )

        self._text(
            form,
            "canvas_negative",
            "Negative prompt",
            "blurry, artifacts, deformed anatomy",
            65
        )

        self._line(form, "canvas_steps", "Steps", "25")
        self._line(form, "canvas_cfg", "CFG", "6.5")
        self._line(form, "canvas_strength", "Strength", "0.35")

        row = QHBoxLayout()

        self.canvas_path = QLineEdit()
        self.canvas_path.setPlaceholderText("Source image")

        browse = QPushButton("Open image")
        browse.clicked.connect(self.open_canvas_image)

        row.addWidget(self.canvas_path, 1)
        row.addWidget(browse)

        self.pages["canvas"].body.addLayout(row)

        self.canvas_widget = MaskCanvas()
        self.pages["canvas"].body.addWidget(self.canvas_widget)

        brush_row = QHBoxLayout()

        brush_row.addWidget(QLabel("Brush size"))

        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(5, 160)
        self.brush_slider.setValue(35)
        self.brush_slider.valueChanged.connect(
            self.set_brush
        )

        brush_row.addWidget(self.brush_slider)

        clear = QPushButton("Clear mask")
        clear.clicked.connect(
            self.canvas_widget.clear_mask
        )

        brush_row.addWidget(clear)
        self.pages["canvas"].body.addLayout(brush_row)

        self._button(
            "canvas",
            "INPAINT — edit masked area",
            self.run_inpaint
        )

        self._button(
            "canvas",
            "OUTPAINT — expand image",
            self.run_outpaint
        )

        self._button(
            "canvas",
            "IMAGE-TO-IMAGE",
            self.run_img2img
        )

    def set_brush(self, value):
        self.canvas_widget.brush_size = value
        self.canvas_widget.update()

    def open_canvas_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open image",
            str(engine.OUTPUT),
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
        )

        if not path:
            return

        try:
            self.canvas_widget.load_image(path)
            self.canvas_path.setText(path)
            self.log(f"Canvas source: {path}")
        except Exception as exc:
            self.warning("Canvas", str(exc))

    def _canvas_args(self):
        if self.canvas_widget.image is None:
            raise RuntimeError("Открой исходное изображение.")

        model = self.value("canvas_model")
        if not model:
            raise RuntimeError("Выбери checkpoint.")

        prompt = self.value("canvas_prompt")
        if not prompt:
            raise RuntimeError("Введи prompt.")

        return (
            self.relative(model),
            prompt,
            self.value("canvas_negative"),
            self.number("canvas_steps", 25),
            self.number("canvas_cfg", 6.5, float),
            self.number("canvas_strength", 0.35, float)
        )

    def run_inpaint(self):
        try:
            model, prompt, negative, steps, cfg, strength = self._canvas_args()

            source = self.canvas_widget.image.copy()
            mask = self.canvas_widget.get_mask()

            if mask.getbbox() is None:
                raise RuntimeError("Нарисуй маску.")

        except Exception as exc:
            return self.warning("Inpaint", str(exc))

        self.start_task(
            "Inpaint",
            lambda cb: engine.run_generation(
                "inpaint", model, prompt, negative,
                source, mask, steps, cfg, strength,
                0, 0, -1, cb
            )
        )

    def run_img2img(self):
        try:
            model, prompt, negative, steps, cfg, strength = self._canvas_args()
            source = self.canvas_widget.image.copy()
        except Exception as exc:
            return self.warning("Img2Img", str(exc))

        self.start_task(
            "Image to Image",
            lambda cb: engine.run_generation(
                "img2img", model, prompt, negative,
                source, None, steps, cfg, strength,
                0, 0, -1, cb
            )
        )

    def run_outpaint(self):
        try:
            from PIL import Image, ImageDraw, ImageOps

            model, prompt, negative, steps, cfg, strength = self._canvas_args()

            source = self.canvas_widget.image.copy()
            width, height = source.size

            border = max(64, int(max(width, height) * 0.20))

            expanded = ImageOps.expand(
                source,
                border=border,
                fill=(0, 0, 0)
            )

            mask = Image.new("L", expanded.size, 255)

            draw = ImageDraw.Draw(mask)
            draw.rectangle(
                (
                    border,
                    border,
                    border + width - 1,
                    border + height - 1
                ),
                fill=0
            )

        except Exception as exc:
            return self.warning("Outpaint", str(exc))

        self.start_task(
            "Outpaint",
            lambda cb: engine.run_generation(
                "inpaint", model, prompt, negative,
                expanded, mask, steps, cfg, strength,
                0, 0, -1, cb
            )
        )

    # ---------------------------------------------------------
    # CHARACTER STUDIO
    # ---------------------------------------------------------

    def _build_character_page(self):
        self._heading(
            "character",
            "CHARACTER STUDIO",
            "Локальная библиотека персонажей. Сохраняй постоянные описания и reference images."
        )

        form = self._group("character", "Character profile")

        self._line(form, "char_name", "Character name", "Character_01")

        self._text(
            form,
            "char_prompt",
            "Appearance / permanent prompt",
            "fictional adult character, consistent facial features, hairstyle, eye color, body proportions, clothing style",
            120
        )

        self._text(
            form,
            "char_negative",
            "Permanent negative prompt",
            "different face, inconsistent identity, distorted anatomy, extra fingers",
            75
        )

        self._file(
            form,
            "char_reference",
            "Reference image",
            "Images (*.png *.jpg *.jpeg *.webp)"
        )

        self._button("character", "SAVE CHARACTER", self.save_character)
        self._button("character", "LOAD CHARACTER", self.load_character)
        self._button("character", "USE PROMPT IN IMAGE STUDIO", self.use_character)

        self.character_status = QLabel(
            f"Profiles folder: {engine.CHARACTERS_DIR}"
        )
        self.character_status.setWordWrap(True)
        self.pages["character"].body.addWidget(self.character_status)

    def save_character(self):
        name = re.sub(
            r"[^A-Za-z0-9_-]+",
            "_",
            self.value("char_name")
        )

        if not name:
            return self.warning("Character", "Укажи имя.")

        data = {
            "name": name,
            "prompt": self.value("char_prompt"),
            "negative_prompt": self.value("char_negative"),
            "reference_image": self.value("char_reference"),
            "created_at": datetime.now().isoformat(timespec="seconds")
        }

        path = engine.CHARACTERS_DIR / f"{name}.json"

        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        self.character_status.setText(f"Saved: {path}")
        self.log(f"Character saved: {path}")

    def load_character(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load character",
            str(engine.CHARACTERS_DIR),
            "Character profiles (*.json)"
        )

        if not path:
            return

        try:
            data = json.loads(
                Path(path).read_text(encoding="utf-8")
            )

            self.fields["char_name"].setText(data.get("name", ""))
            self.fields["char_prompt"].setPlainText(data.get("prompt", ""))
            self.fields["char_negative"].setPlainText(
                data.get("negative_prompt", "")
            )
            self.fields["char_reference"].setText(
                data.get("reference_image", "")
            )

            self.character_status.setText(f"Loaded: {path}")

        except Exception as exc:
            self.warning("Character", str(exc))

    def use_character(self):
        prompt = self.value("char_prompt")
        negative = self.value("char_negative")

        current = self.value("image_prompt")

        self.fields["image_prompt"].setPlainText(
            (prompt + ", " + current).strip(", ")
        )

        if negative:
            old = self.value("image_negative")
            self.fields["image_negative"].setPlainText(
                (old + ", " + negative).strip(", ")
            )

        self.navigation.setCurrentRow(0)

    # ---------------------------------------------------------
    # SCENE STUDIO
    # ---------------------------------------------------------

    def _build_scene_page(self):
        self._heading(
            "scene",
            "SCENE STUDIO",
            "Создание сцен, storyboard и последовательных кадров."
        )

        form = self._group("scene", "Scene setup")

        self._line(form, "scene_project", "Project name", "My_Project")
        self._line(form, "scene_title", "Scene title", "Scene 01")

        self._text(
            form,
            "scene_prompt",
            "Scene prompt",
            "wide cinematic establishing shot, coherent environment, consistent character design",
            100
        )

        self._line(form, "scene_duration", "Duration", "4")
        self._add_model_combo(form, "scene_model")
        self._line(form, "scene_steps", "Generation steps", "20")

        self.scene_list = QListWidget()
        self.scene_list.setMinimumHeight(180)
        self.pages["scene"].body.addWidget(self.scene_list)

        self._button("scene", "ADD SCENE", self.add_scene)
        self._button("scene", "REMOVE SELECTED", self.remove_scene)
        self._button("scene", "SAVE STORYBOARD", self.save_storyboard)
        self._button("scene", "LOAD STORYBOARD", self.load_storyboard)
        self._button(
            "scene",
            "GENERATE STORYBOARD FRAMES",
            self.generate_scene_frames
        )

    def add_scene(self):
        try:
            title = self.value("scene_title") or (
                f"Scene {len(self.scene_data)+1:02d}"
            )

            prompt = self.value("scene_prompt")
            duration = self.number("scene_duration", 4, float)

            if duration <= 0 or duration > 600:
                raise ValueError("Duration: 0–600 seconds.")

            self.scene_data.append({
                "title": title,
                "prompt": prompt,
                "duration": duration
            })

            self.refresh_scene_list()

        except Exception as exc:
            self.warning("Scene", str(exc))

    def refresh_scene_list(self):
        self.scene_list.clear()

        for index, scene in enumerate(self.scene_data):
            self.scene_list.addItem(
                f"{index+1:02d}. {scene['title']} "
                f"— {scene['duration']:g}s | {scene['prompt'][:100]}"
            )

    def remove_scene(self):
        selected = sorted(
            [self.scene_list.row(x) for x in self.scene_list.selectedItems()],
            reverse=True
        )

        for index in selected:
            self.scene_data.pop(index)

        self.refresh_scene_list()

    def save_storyboard(self):
        name = re.sub(
            r"[^A-Za-z0-9_-]+",
            "_",
            self.value("scene_project")
        ) or "storyboard"

        path = engine.SCENES_DIR / f"{name}.json"

        path.write_text(
            json.dumps({
                "project": name,
                "scenes": self.scene_data
            }, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        self.log(f"Storyboard saved: {path}")

    def load_storyboard(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load storyboard",
            str(engine.SCENES_DIR),
            "JSON (*.json)"
        )

        if not path:
            return

        try:
            data = json.loads(
                Path(path).read_text(encoding="utf-8")
            )

            self.scene_data = data.get("scenes", [])

            self.fields["scene_project"].setText(
                data.get("project", "")
            )

            self.refresh_scene_list()

        except Exception as exc:
            self.warning("Storyboard", str(exc))

    def generate_scene_frames(self):
        if not self.scene_data:
            return self.warning("Scene", "Добавь сцены.")

        model = self.value("scene_model")
        if not model:
            return self.warning("Scene", "Выбери checkpoint.")

        model = self.relative(model)
        steps = self.number("scene_steps", 20)
        scenes = [dict(x) for x in self.scene_data]

        def job(callback):
            for index, scene in enumerate(scenes):
                prompt = scene.get("prompt", "").strip()

                if not prompt:
                    continue

                engine.run_generation(
                    "txt2img",
                    model,
                    prompt,
                    "blurry, low quality, inconsistent scene",
                    None,
                    None,
                    steps,
                    6.5,
                    0.0,
                    768,
                    512,
                    -1,
                    callback
                )

                callback(
                    None,
                    f"Storyboard frame {index+1}/{len(scenes)}"
                )

            engine.cleanup()

        self.start_task("Storyboard generation", job)

    # ---------------------------------------------------------
    # MOTION / VIDEO
    # ---------------------------------------------------------

    def _build_video_page(self):
        self._heading(
            "video",
            "MOTION / VIDEO STUDIO",
            "AnimateDiff Text-to-Video and Stable Video Diffusion XT Image-to-Video."
        )

        form = self._group("video", "AnimateDiff")

        self._add_model_combo(form, "video_model")

        motion_dirs = sorted({
            str(p.parent.relative_to(engine.ROOT))
            for p in engine.MOTION.rglob("config.json")
        })

        self._combo(
            form,
            "video_motion",
            "Motion adapter",
            motion_dirs
        )

        motion_loras = [
            str(p.relative_to(engine.ROOT))
            for p in engine.scan_files(
                engine.MOTION / "motion-lora",
                (".safetensors",)
            )
        ]

        self._combo(
            form,
            "video_motion_lora",
            "Motion LoRA",
            [""] + motion_loras
        )

        self._text(
            form,
            "video_prompt",
            "Prompt",
            "cinematic camera movement, natural motion, detailed scene",
            100
        )

        self._text(
            form,
            "video_negative",
            "Negative prompt",
            "flicker, jitter, distorted anatomy, blurry",
            65
        )

        self._line(form, "video_steps", "Steps", "20")
        self._line(form, "video_cfg", "CFG", "7")
        self._line(form, "video_frames", "Frames", "12")
        self._line(form, "video_fps", "Native generation FPS", "8")
        self._checkbox(
            form, "video_smooth60",
            "После генерации создать плавную версию 60 FPS (интерполяция)",
            False
        )

        self._button(
            "video",
            "GENERATE ANIMATEDIFF VIDEO",
            self.generate_animatediff
        )

        svd = self._group("video", "SVD-XT — Image to Video")

        self._file(
            svd,
            "svd_image",
            "Source image",
            "Images (*.png *.jpg *.jpeg *.webp)"
        )

        self._text(
            svd,
            "svd_motion_notes",
            "Motion notes",
            "subtle camera movement, gentle natural motion",
            75
        )

        self._text(
            svd,
            "svd_negative",
            "Negative prompt",
            "flicker, jitter, sudden cuts, face distortion, blurry",
            65
        )

        self._line(svd, "svd_duration", "Duration seconds", "4")
        self._line(svd, "svd_fps", "FPS", "7")
        self._line(svd, "svd_motion", "Motion bucket", "100")
        self._line(svd, "svd_noise", "Noise strength", "0.03")
        self._line(svd, "svd_steps", "Steps", "25")
        self._line(svd, "svd_decode", "Decode chunk", "2")
        self._line(svd, "svd_seed", "Seed", "-1")
        self._checkbox(
            svd, "svd_smooth60",
            "После генерации создать плавную версию 60 FPS (интерполяция)",
            False
        )

        self._button(
            "video",
            "GENERATE SVD-XT VIDEO",
            self.generate_svd
        )

        note = QLabel(
            "RTX 3060 Laptop 6 GB: начинай с 8–12 кадров AnimateDiff "
            "и decode chunk 1–2 для SVD. SVD-XT не является text-to-video моделью."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#e8bd6b;")
        self.pages["video"].body.addWidget(note)

    def generate_animatediff(self):
        try:
            model = self.value("video_model")
            motion = self.value("video_motion")

            if not model or not motion:
                raise RuntimeError("Нужны checkpoint и motion adapter.")

            model = self.relative(model)
            motion = self.relative(motion)

            prompt = self.value("video_prompt")
            negative = self.value("video_negative")

            steps = self.number("video_steps", 20)
            cfg = self.number("video_cfg", 7, float)
            frames = self.number("video_frames", 12)
            fps = self.number("video_fps", 8)

            if not 4 <= frames <= 24:
                raise ValueError("Frames: 4–24.")

            if not 1 <= fps <= 30:
                raise ValueError("FPS: 1–30.")

            motion_lora = self.value("video_motion_lora") or None

        except Exception as exc:
            return self.warning("AnimateDiff", str(exc))

        smooth60 = self.value("video_smooth60")

        def job(callback):
            def receive(image, message):
                callback(image, message)
                if smooth60 and message:
                    output = _video_path_from_message(message)
                    if output:
                        smooth_video_to_60fps(output, callback)

            engine.make_animatediff(
                model, motion, prompt, negative, steps, cfg,
                frames, fps, None, receive, motion_lora
            )

        self.start_task("AnimateDiff", job)

    def generate_svd(self):
        try:
            image = self.value("svd_image")

            if not image or not Path(image).is_file():
                raise RuntimeError("Выбери исходное изображение.")

            duration = self.number("svd_duration", 4, float)
            fps = self.number("svd_fps", 7)
            motion = self.number("svd_motion", 100)
            noise = self.number("svd_noise", 0.03, float)
            steps = self.number("svd_steps", 25)
            decode = self.number("svd_decode", 2)
            seed = self.number("svd_seed", -1)

            if not 0.5 <= duration <= 180:
                raise ValueError("Duration: 0.5–180.")

            if not 3 <= fps <= 15:
                raise ValueError("FPS: 3–15.")

            if not 0 <= motion <= 255:
                raise ValueError("Motion bucket: 0–255.")

            if not 0 <= noise <= 0.5:
                raise ValueError("Noise: 0–0.5.")

            if not 10 <= steps <= 60:
                raise ValueError("Steps: 10–60.")

            if not 1 <= decode <= 8:
                raise ValueError("Decode chunk: 1–8.")

        except Exception as exc:
            return self.warning("SVD-XT", str(exc))

        motion_notes = self.value("svd_motion_notes")
        negative = self.value("svd_negative")
        smooth60 = self.value("svd_smooth60")

        def job(callback):
            def receive(result_image, message):
                callback(result_image, message)
                if smooth60 and message:
                    output = _video_path_from_message(message)
                    if output:
                        smooth_video_to_60fps(output, callback)

            engine.make_svd_video(
                image, motion_notes, negative, duration, fps,
                motion, noise, steps, decode, receive, seed
            )

        self.start_task("SVD-XT", job)

    # ---------------------------------------------------------
    # TIMELINE
    # ---------------------------------------------------------

    def _build_timeline_page(self):
        self._heading(
            "timeline",
            "TIMELINE / VIDEO EXPORT",
            "Добавляй фотографии и видео, меняй порядок, задавай длительность фотографий и экспортируй MP4."
        )

        form = self._group("timeline", "Export settings")

        self._line(form, "timeline_duration", "Default photo duration", "4")
        self._line(form, "timeline_fps", "Output FPS (30 or 60)", "30")
        self._checkbox(
            form, "timeline_smooth60",
            "Экспортировать и дополнительно интерполировать до плавных 60 FPS",
            False
        )

        self._file(
            form,
            "timeline_audio",
            "Audio track",
            "Audio (*.wav *.mp3 *.m4a *.aac)"
        )

        self.timeline_list = QListWidget()
        self.timeline_list.setMinimumHeight(220)
        self.pages["timeline"].body.addWidget(self.timeline_list)

        self._button("timeline", "ADD MEDIA", self.timeline_add)
        self._button("timeline", "REMOVE SELECTED", self.timeline_remove)
        self._button("timeline", "MOVE SELECTED UP", lambda: self.timeline_move(-1))
        self._button("timeline", "MOVE SELECTED DOWN", lambda: self.timeline_move(1))
        self._button("timeline", "SET SELECTED PHOTO DURATION", self.timeline_set_duration)
        self._button("timeline", "EXPORT MP4", self.timeline_export)

    def timeline_add(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add media",
            str(engine.OUTPUT),
            "Media (*.png *.jpg *.jpeg *.webp *.bmp *.mp4 *.mov *.mkv *.avi *.webm)"
        )

        if not paths:
            return

        duration = self.number("timeline_duration", 4, float)

        if duration <= 0:
            return self.warning("Timeline", "Duration должна быть больше нуля.")

        image_ext = {
            ".png", ".jpg", ".jpeg", ".webp", ".bmp"
        }

        for path in paths:
            is_image = Path(path).suffix.lower() in image_ext

            self.timeline_items.append({
                "path": path,
                "duration": duration if is_image else None
            })

        self.refresh_timeline()

    def refresh_timeline(self):
        self.timeline_list.clear()

        for item in self.timeline_items:
            name = Path(item["path"]).name
            duration = item.get("duration")

            label = (
                f"{name} — {duration:g}s"
                if duration is not None
                else f"{name} — full video"
            )

            self.timeline_list.addItem(label)

    def timeline_remove(self):
        selected = sorted(
            [self.timeline_list.row(x) for x in self.timeline_list.selectedItems()],
            reverse=True
        )

        for index in selected:
            self.timeline_items.pop(index)

        self.refresh_timeline()

    def timeline_move(self, delta):
        selected = self.timeline_list.currentRow()

        if selected < 0:
            return

        target = selected + delta

        if not 0 <= target < len(self.timeline_items):
            return

        self.timeline_items[selected], self.timeline_items[target] = (
            self.timeline_items[target],
            self.timeline_items[selected]
        )

        self.refresh_timeline()
        self.timeline_list.setCurrentRow(target)

    def timeline_set_duration(self):
        index = self.timeline_list.currentRow()

        if index < 0:
            return self.warning("Timeline", "Выбери клип.")

        item = self.timeline_items[index]

        if item.get("duration") is None:
            return self.warning(
                "Timeline",
                "Длительность видео пока берётся из исходного файла."
            )

        duration = self.number("timeline_duration", 4, float)

        if duration <= 0:
            return self.warning("Timeline", "Duration должна быть больше нуля.")

        item["duration"] = duration
        self.refresh_timeline()
        self.timeline_list.setCurrentRow(index)

    def timeline_export(self):
        if not self.timeline_items:
            return self.warning("Timeline", "Добавь медиафайлы.")

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export video",
            str(engine.OUTPUT / "aether_export.mp4"),
            "MP4 (*.mp4)"
        )

        if not path:
            return

        items = [dict(x) for x in self.timeline_items]

        for item in items:
            if not Path(item["path"]).is_file():
                return self.warning(
                    "Timeline",
                    f"Файл не найден: {item['path']}"
                )

        audio = self.value("timeline_audio") or None
        fps = self.number("timeline_fps", 30)

        smooth60 = self.value("timeline_smooth60")

        def job(callback):
            exported = engine.export_timeline(items, path, audio, fps)
            callback(None, f"Export finished: {exported}")
            if smooth60 and fps != 60:
                smooth_video_to_60fps(exported, callback)
            elif smooth60 and fps == 60:
                # The export already has 60 frames per second; no second interpolation pass.
                callback(None, "Timeline уже экспортирован в 60 FPS. Дополнительная интерполяция не требуется.")

        self.start_task("Timeline export", job)

    # ---------------------------------------------------------
    # COMFYUI
    # ---------------------------------------------------------

    def _build_comfy_page(self):
        self._heading(
            "comfy",
            "COMFYUI LOCAL BRIDGE",
            "Используй ComfyUI как дополнительный графический движок для сложных workflows."
        )

        form = self._group("comfy", "Local server")

        self._line(
            form,
            "comfy_url",
            "Server URL",
            "http://127.0.0.1:8188"
        )

        self._file(
            form,
            "comfy_workflow",
            "API workflow JSON",
            "JSON (*.json)"
        )

        self._button(
            "comfy",
            "CHECK COMFYUI CONNECTION",
            self.comfy_check
        )

        self._button(
            "comfy",
            "RUN API WORKFLOW",
            self.comfy_run
        )

        self.comfy_status = QLabel(
            "Ожидание локального ComfyUI..."
        )
        self.comfy_status.setWordWrap(True)
        self.pages["comfy"].body.addWidget(self.comfy_status)

        note = QLabel(
            "Требуется API-format JSON, экспортированный из ComfyUI. "
            "Обычный workflow JSON может не подойти."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#e8bd6b;")
        self.pages["comfy"].body.addWidget(note)

    def _comfy_base(self):
        base = self.value("comfy_url").rstrip("/")
        parsed = urllib.parse.urlparse(base)

        if parsed.scheme not in ("http", "https"):
            raise ValueError("Неверная схема URL.")

        if parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("Разрешён только локальный ComfyUI.")

        return base

    def comfy_check(self):
        try:
            base = self._comfy_base()
        except Exception as exc:
            return self.warning("ComfyUI", str(exc))

        def job(callback):
            with urllib.request.urlopen(
                base + "/system_stats",
                timeout=8
            ) as response:
                data = json.loads(response.read().decode("utf-8"))

            callback(
                None,
                "ComfyUI connected:\n" +
                json.dumps(data, ensure_ascii=False)[:2500]
            )

        self.start_task("ComfyUI connection", job)

    def comfy_run(self):
        workflow = self.value("comfy_workflow")

        if not workflow or not Path(workflow).is_file():
            return self.warning("ComfyUI", "Выбери API workflow JSON.")

        try:
            base = self._comfy_base()
        except Exception as exc:
            return self.warning("ComfyUI", str(exc))

        self.start_task(
            "ComfyUI workflow",
            lambda cb: engine.submit_comfyui_workflow(
                base,
                workflow,
                cb
            )
        )

    # ---------------------------------------------------------
    # FACE STUDIO
    # ---------------------------------------------------------

    def _build_face_page(self):
        self._heading(
            "face",
            "FACE STUDIO",
            "Локальные операции с фотографиями и видео. Используй материалы, на которые у тебя есть разрешение."
        )

        form = self._group("face", "Face Swap")

        self._file(
            form,
            "face_source",
            "Source face",
            "Images (*.png *.jpg *.jpeg *.webp)"
        )

        self._file(
            form,
            "face_target",
            "Target image",
            "Images (*.png *.jpg *.jpeg *.webp)"
        )

        self._checkbox(
            form,
            "face_consent",
            "У меня есть разрешение использовать исходное и целевое изображение",
            False
        )

        self._button(
            "face",
            "RUN PHOTO FACE SWAP",
            self.face_swap
        )

        video = self._group("face", "Video Face Swap")

        self._file(
            video,
            "face_video",
            "Target video",
            "Video (*.mp4 *.mov *.mkv *.avi)"
        )

        self._line(video, "face_source_index", "Source face index", "1")
        self._line(video, "face_target_index", "Target face index", "1")

        self._checkbox(
            video,
            "face_video_consent",
            "У меня есть разрешение использовать исходное лицо и целевое видео",
            False
        )

        self._button(
            "face",
            "RUN VIDEO FACE SWAP",
            self.face_swap_video
        )

        fusion = self._group("face", "Optional FaceFusion")

        self._file(
            fusion,
            "fusion_source",
            "Source image",
            "Images (*.png *.jpg *.jpeg *.webp)"
        )

        self._file(
            fusion,
            "fusion_target",
            "Target image / video",
            "Media (*.png *.jpg *.jpeg *.webp *.mp4 *.mov *.mkv)"
        )

        self._button(
            "face",
            "RUN LOCAL FACEFUSION",
            self.facefusion
        )

        portrait = self._group("face", "Portrait restoration")

        self._file(
            portrait,
            "portrait_image",
            "Source portrait",
            "Images (*.png *.jpg *.jpeg *.webp)"
        )

        self._button(
            "face",
            "RESTORE PORTRAIT — GFPGAN",
            self.restore_portrait
        )

    def face_swap(self):
        if not self.value("face_consent"):
            return self.warning("Face Swap", "Подтверди разрешение.")

        source = self.value("face_source")
        target = self.value("face_target")

        if not source or not target:
            return self.warning("Face Swap", "Выбери оба изображения.")

        self.start_task(
            "Photo Face Swap",
            lambda cb: engine.run_face_swap(source, target, cb)
        )

    def face_swap_video(self):
        if not self.value("face_video_consent"):
            return self.warning("Video Face Swap", "Подтверди разрешение.")

        source = self.value("face_source")
        video = self.value("face_video")

        if not source or not video:
            return self.warning("Video Face Swap", "Выбери фото и видео.")

        source_index = self.number("face_source_index", 1)
        target_index = self.number("face_target_index", 1)

        self.start_task(
            "Video Face Swap",
            lambda cb: engine.run_video_face_swap(
                source,
                video,
                cb,
                source_index,
                target_index
            )
        )

    def facefusion(self):
        source = self.value("fusion_source")
        target = self.value("fusion_target")

        if not source or not target:
            return self.warning("FaceFusion", "Выбери source и target.")

        self.start_task(
            "FaceFusion",
            lambda cb: engine.run_facefusion(source, target, cb)
        )

    def restore_portrait(self):
        path = self.value("portrait_image")

        if not path:
            return self.warning("GFPGAN", "Выбери изображение.")

        self.start_task(
            "Portrait restoration",
            lambda cb: engine.restore_portrait(path, cb)
        )

    # ---------------------------------------------------------
    # ENHANCEMENT
    # ---------------------------------------------------------

    def _build_enhance_page(self):
        self._heading(
            "enhance",
            "IMAGE ENHANCEMENT",
            "Увеличение разрешения через локальный Real-ESRGAN NCNN Vulkan."
        )

        form = self._group("enhance", "Real-ESRGAN")

        self._file(
            form,
            "upscale_image",
            "Input image",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
        )

        self._combo(
            form,
            "upscale_model",
            "Model",
            [
                "realesrgan-x4plus",
                "realesrgan-x4plus-anime",
                "realesr-animevideov3"
            ]
        )

        self._combo(
            form,
            "upscale_scale",
            "Scale",
            ["2", "3", "4"],
            "4"
        )

        self._button(
            "enhance",
            "UPSCALE IMAGE",
            self.run_upscale
        )

    def run_upscale(self):
        path = self.value("upscale_image")

        if not path or not Path(path).is_file():
            return self.warning("Upscale", "Выбери изображение.")

        try:
            from PIL import Image
            image = Image.open(path).convert("RGB")

            model = self.value("upscale_model")
            scale = self.number("upscale_scale", 4)

        except Exception as exc:
            return self.warning("Upscale", str(exc))

        self.start_task(
            "Real-ESRGAN",
            lambda cb: engine.upscale_image(
                image,
                cb,
                scale,
                model
            )
        )

    # ---------------------------------------------------------
    # CONTROLNET
    # ---------------------------------------------------------

    def _build_control_page(self):
        self._heading(
            "control",
            "CONTROLNET STUDIO",
            "Контроль структуры и композиции через локальные ControlNet модели."
        )

        form = self._group("control", "ControlNet")

        self._add_model_combo(form, "control_model")

        self._combo(
            form,
            "control_mode",
            "Mode",
            ["Canny", "OpenPose", "Depth", "Tile"]
        )

        self._combo(
            form,
            "control_weights",
            "ControlNet model",
            engine.list_controlnet_models()
        )

        self._file(
            form,
            "control_image",
            "Input image",
            "Images (*.png *.jpg *.jpeg *.webp)"
        )

        self._text(
            form,
            "control_prompt",
            "Prompt",
            "high quality, detailed image",
            90
        )

        self._text(
            form,
            "control_negative",
            "Negative prompt",
            "blurry, artifacts, distorted anatomy",
            65
        )

        self._line(form, "control_steps", "Steps", "20")
        self._line(form, "control_cfg", "CFG", "6.5")
        self._line(form, "control_strength", "Strength", "0.8")

        self._button(
            "control",
            "GENERATE WITH CONTROLNET",
            self.run_controlnet
        )

    def run_controlnet(self):
        model = self.value("control_model")
        control = self.value("control_weights")
        image = self.value("control_image")

        if not model or not control or not image:
            return self.warning(
                "ControlNet",
                "Выбери checkpoint, ControlNet и изображение."
            )

        mode = self.value("control_mode")
        model_rel = self.relative(model)
        control_rel = self.relative(control)
        prompt = self.value("control_prompt")
        negative = self.value("control_negative")
        steps = self.number("control_steps", 20)
        cfg = self.number("control_cfg", 6.5, float)
        strength = self.number("control_strength", 0.8, float)

        self.start_task(
            "ControlNet",
            lambda cb: engine.run_controlnet_mode(
                mode, model_rel, control_rel, image,
                prompt, negative, steps, cfg, strength, cb
            )
        )

    # ---------------------------------------------------------
    # VOICE STUDIO
    # ---------------------------------------------------------

    def _build_voice_page(self):
        self._heading(
            "voice",
            "VOICE STUDIO",
            "Локальный синтез речи, lip-sync и конвертация голоса."
        )

        piper = self._group("voice", "Piper TTS")

        self._text(
            piper,
            "voice_text",
            "Text",
            "Hello! This is a fictional AI character voice.",
            100
        )

        voices = [
            str(p)
            for p in engine.VOICE_MODELS.glob("*.onnx")
        ]

        self._combo(
            piper,
            "voice_model",
            "Voice model",
            voices
        )

        self._button(
            "voice",
            "GENERATE WAV",
            self.generate_voice
        )

        lip = self._group("voice", "Lip Sync")

        self._file(
            lip,
            "lipsync_video",
            "Face video",
            "Video (*.mp4 *.mov *.mkv)"
        )

        self._file(
            lip,
            "lipsync_audio",
            "Audio",
            "Audio (*.wav *.mp3 *.m4a)"
        )

        self._button(
            "voice",
            "CREATE LIP-SYNC VIDEO",
            self.run_lipsync
        )

        rvc = self._group("voice", "RVC")

        self._file(
            rvc,
            "rvc_audio",
            "Input audio",
            "Audio (*.wav *.mp3 *.flac)"
        )

        self._file(
            rvc,
            "rvc_model",
            "RVC model",
            "RVC model (*.pth)"
        )

        self._file(
            rvc,
            "rvc_index",
            "Index (optional)",
            "Index (*.index)"
        )

        self._button(
            "voice",
            "RUN RVC CONVERSION",
            self.run_rvc
        )

    def generate_voice(self):
        text_value = self.value("voice_text")
        model = self.value("voice_model")

        if not text_value or not model:
            return self.warning("Piper", "Нужны текст и voice model.")

        self.start_task(
            "Piper TTS",
            lambda cb: engine.synthesize_voice_piper(
                text_value,
                model,
                cb
            )
        )

    def run_lipsync(self):
        video = self.value("lipsync_video")
        audio = self.value("lipsync_audio")

        if not video or not audio:
            return self.warning("Lip Sync", "Выбери видео и аудио.")

        self.start_task(
            "Lip Sync",
            lambda cb: engine.create_lipsync_video(
                video,
                audio,
                cb
            )
        )

    def run_rvc(self):
        audio = self.value("rvc_audio")
        model = self.value("rvc_model")
        index = self.value("rvc_index")

        if not audio or not model:
            return self.warning("RVC", "Выбери аудио и модель.")

        self.start_task(
            "RVC",
            lambda cb: engine.run_rvc_conversion(
                audio,
                model,
                index,
                cb
            )
        )

    # ---------------------------------------------------------
    # SYSTEM / MODEL MANAGER
    # ---------------------------------------------------------

    def _build_system_page(self):
        self._heading(
            "system",
            "SYSTEM / MODEL MANAGER",
            "Проверка установленных моделей, путей, GPU и внешних инструментов."
        )

        self.system_info = QTextEdit()
        self.system_info.setReadOnly(True)
        self.system_info.setMinimumHeight(260)

        self.pages["system"].body.addWidget(self.system_info)

        self._button(
            "system",
            "REFRESH DIAGNOSTICS",
            self.refresh_system
        )

        self._button(
            "system",
            "RESCAN MODELS",
            self.refresh_models
        )

        self._button(
            "system",
            "CLEAR GPU CACHE",
            self.clear_cache
        )

        self._button(
            "system",
            "OPEN OUTPUT FOLDER",
            lambda: self.open_folder(engine.OUTPUT)
        )

        self._button(
            "system",
            "OPEN MODELS FOLDER",
            lambda: self.open_folder(engine.MODELS)
        )

        self._button(
            "system",
            "OPEN PROJECT FOLDER",
            lambda: self.open_folder(engine.ROOT)
        )

    def refresh_models(self):
        try:
            models = engine.list_checkpoints()

            for key in (
                "image_model",
                "canvas_model",
                "scene_model",
                "video_model",
                "control_model"
            ):
                widget = self.fields.get(key)

                if not isinstance(widget, QComboBox):
                    continue

                current = widget.currentText()

                widget.clear()
                widget.addItems(models)

                index = widget.findText(current)

                if index >= 0:
                    widget.setCurrentIndex(index)

            optional_map = {
                "image_lora": "lora",
                "image_vae": "vae",
                "image_embedding": "embedding",
                "image_ip": "ip_adapter",
            }
            for key, kind in optional_map.items():
                widget = self.fields.get(key)
                if isinstance(widget, QComboBox):
                    current = widget.currentText()
                    widget.clear()
                    widget.addItems([""] + engine.component_relpaths(kind))
                    index = widget.findText(current)
                    if index >= 0:
                        widget.setCurrentIndex(index)

            self.log(
                f"Checkpoints found: {len(models)}; optional components refreshed."
            )

        except Exception as exc:
            self.log(f"Model scan error: {exc}")

        self.refresh_system()

    def refresh_system(self):
        if engine is None:
            return

        try:
            import torch

            cuda = torch.cuda.is_available()

            if cuda:
                gpu = torch.cuda.get_device_name(0)
                memory = (
                    torch.cuda.get_device_properties(0).total_memory
                    / 1024**3
                )
            else:
                gpu = "CPU"
                memory = 0

            ffmpeg = engine.get_ffmpeg()

            checkpoints = engine.list_checkpoints()

            lines = [
                APP_NAME,
                VERSION,
                "",
                f"Project: {engine.ROOT}",
                f"Python: {sys.version.split()[0]}",
                f"PyTorch: {torch.__version__}",
                f"CUDA available: {cuda}",
                f"GPU: {gpu}",
                f"VRAM total: {memory:.2f} GB",
                f"Checkpoints: {len(checkpoints)}",
                f"LoRA: {len(engine.component_files('lora'))}",
                f"VAE: {len(engine.component_files('vae'))}",
                f"Embeddings: {len(engine.component_files('embedding'))}",
                f"IP-Adapter: {len(engine.component_files('ip_adapter'))}",
                f"ControlNet: {len(engine.list_controlnet_models())}",
                f"Motion: {len(engine.scan_files(engine.MOTION))}",
                f"Voice: {len(engine.scan_files(engine.VOICE_MODELS, ('.onnx',)))}",
                f"FFmpeg: {ffmpeg}",
                "",
                "EXTERNAL COMPONENTS",
                f"GFPGAN weights: {(engine.ROOT / 'GFPGANv1.4.pth').is_file()}",
                f"FaceSwap model: {(engine.INSIGHT / 'inswapper_128.onnx').is_file()}",
                f"FaceFusion: {(engine.TOOLS_DIR / 'facefusion' / 'facefusion.py').is_file()}",
                f"Wav2Lip: {(engine.TOOLS_DIR / 'Wav2Lip' / 'inference.py').is_file()}",
                f"RVC CLI: {(engine.TOOLS_DIR / 'RVC' / 'infer_cli.py').is_file()}",
                "",
                "VIDEO TOOLS",
                f"Real-ESRGAN: {(engine.ROOT / 'external' / 'realesrgan' / 'portable' / 'realesrgan-ncnn-vulkan.exe').is_file()}",
                f"Blender: {(engine.ROOT / 'external' / 'blender' / 'blender.exe').is_file()}",
            ]

            if cuda:
                lines += [
                    f"VRAM allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB",
                    f"VRAM reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB"
                ]

            self.system_info.setPlainText("\n".join(lines))

            self.gpu_label.setText(
                f"{gpu}  |  {memory:.1f} GB VRAM"
            )

        except Exception as exc:
            self.system_info.setPlainText(
                "Diagnostics failed:\n" + str(exc)
            )

    def clear_cache(self):
        self.start_task(
            "GPU cleanup",
            lambda cb: (
                engine.cleanup(drop_pipeline=False),
                cb(None, "GPU cache cleanup finished.")
            )
        )

    def open_folder(self, path):
        try:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)

            if sys.platform == "win32":
                os.startfile(str(path))
            else:
                import subprocess
                subprocess.Popen(["xdg-open", str(path)])

        except Exception as exc:
            self.warning("Open folder", str(exc))

    # ---------------------------------------------------------
    # WORKERS / RESULTS / LOGGING
    # ---------------------------------------------------------

    def start_task(self, title, function):
        if self.thread is not None and self.thread.isRunning():
            return self.warning(
                "Busy",
                "Дождись завершения текущей задачи."
            )

        self.status.setText(title)
        self.busy_label.setText("RUNNING")
        self.progress.setValue(0)

        self.log(f"Started: {title}")

        self.thread = QThread()
        self.worker = Worker(function)

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)

        self.worker.message.connect(self.on_message)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.on_finished)

        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)

        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self._thread_stopped)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def on_message(self, image, message):
        if image is not None:
            try:
                if hasattr(image, "convert"):
                    image = image.convert("RGB")

                    data = image.tobytes("raw", "RGB")

                    qimage = QImage(
                        data,
                        image.width,
                        image.height,
                        image.width * 3,
                        QImage.Format_RGB888
                    ).copy()

                    self.last_image = QPixmap.fromImage(qimage)

                elif isinstance(image, str):
                    self.last_image = QPixmap(image)

            except Exception as exc:
                self.log(f"Preview error: {exc}")

        self.log(message)

        if message:
            match = re.search(
                r"([A-Za-z]:\\[^\\r\\n]+\\.(?:png|jpg|jpeg|webp|mp4|wav))",
                message,
                re.IGNORECASE
            )

            if match:
                path = Path(match.group(1))

                if path.is_file() and path.suffix.lower() in (
                    ".png", ".jpg", ".jpeg", ".webp"
                ):
                    self.last_image = QPixmap(str(path))

        if self.last_image and not self.last_image.isNull():
            self.status.setText(
                f"Result received: {self.last_image.width()} × {self.last_image.height()}"
            )

        self.progress.setValue(
            min(95, self.progress.value() + 10)
        )

    def on_finished(self):
        self.progress.setValue(100)
        self.status.setText("Task completed")
        self.busy_label.setText("IDLE")
        self.log("Task finished.")

    def on_failed(self, error):
        self.busy_label.setText("ERROR")
        self.status.setText("Task failed")
        self.log(error)

        QMessageBox.critical(
            self,
            "AETHER task failed",
            error[-5000:]
        )

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")

        self.log_box.append(
            f"[{timestamp}] {message}"
        )

    def warning(self, title, message):
        QMessageBox.warning(
            self,
            title,
            str(message)
        )

    def _thread_stopped(self):
        self.worker = None
        self.thread = None
        if self._close_when_done:
            self._close_when_done = False
            self.close()

    def closeEvent(self, event):
        if self.thread is not None and self.thread.isRunning():
            answer = QMessageBox.question(
                self,
                "Task running",
                "Задача ещё выполняется. Дождаться завершения и закрыть AETHER?"
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self._close_when_done = True
            self.status.setText("Закрытие после завершения текущей задачи…")
            event.ignore()
            return
        event.accept()


def main():
    application = QApplication(sys.argv)

    application.setApplicationName(APP_NAME)
    application.setFont(QFont("Segoe UI", 10))

    try:
        window = AetherStudio()
        window.show()

    except Exception:
        QMessageBox.critical(
            None,
            "AETHER startup error",
            traceback.format_exc()
        )
        return 1

    return application.exec()


if __name__ == "__main__":
    sys.exit(main())
