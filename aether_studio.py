
# -*- coding: utf-8 -*-
"""
AETHER AI ULTIMATE STUDIO
Unified PySide6 interface for the existing AIExoticLocal engine.

Existing app.py is preserved and imported as the backend.
"""

from __future__ import annotations

import os
import sys
import traceback
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output_media"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# ------------------------------------------------------------
# BACKEND
# ------------------------------------------------------------

try:
    import app as engine
    ENGINE_ERROR = None
except Exception as exc:
    engine = None
    ENGINE_ERROR = f"{type(exc).__name__}: {exc}"

# ------------------------------------------------------------
# QT
# ------------------------------------------------------------

from PySide6.QtCore import (
    Qt, QObject, QThread, Signal, QSize
)

from PySide6.QtGui import (
    QPixmap, QAction, QFont
)

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFrame,
    QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGridLayout, QStackedWidget, QFileDialog,
    QLineEdit, QTextEdit, QComboBox,
    QSpinBox, QDoubleSpinBox, QProgressBar,
    QMessageBox, QGroupBox, QFormLayout,
    QCheckBox
)

# ------------------------------------------------------------
# WORKER
# ------------------------------------------------------------

class Worker(QObject):

    finished = Signal(object)
    failed = Signal(str)
    message = Signal(str)

    def __init__(self, function, *args, **kwargs):
        super().__init__()

        self.function = function
        self.args = args
        self.kwargs = kwargs

    def callback(self, *args):

        message = " ".join(
            str(x) for x in args
            if x is not None
        )

        if message:
            self.message.emit(message)

    def run(self):

        try:
            result = self.function(
                *self.args,
                **self.kwargs
            )

            self.finished.emit(result)

        except Exception:
            self.failed.emit(traceback.format_exc())


# ------------------------------------------------------------
# FILE PICKER
# ------------------------------------------------------------

class FilePicker(QWidget):

    def __init__(
        self,
        file_filter="All files (*.*)",
        save=False
    ):

        super().__init__()

        self.save_mode = save
        self.file_filter = file_filter

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.edit = QLineEdit()
        self.edit.setPlaceholderText("Select file...")

        self.button = QPushButton("Browse")
        self.button.setObjectName("secondary")

        layout.addWidget(self.edit, 1)
        layout.addWidget(self.button)

        self.button.clicked.connect(self.browse)

    def browse(self):

        if self.save_mode:

            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save file",
                str(OUTPUT_DIR),
                self.file_filter
            )

        else:

            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select file",
                str(ROOT),
                self.file_filter
            )

        if path:
            self.edit.setText(path)

    def text(self):
        return self.edit.text().strip()


# ------------------------------------------------------------
# PAGE
# ------------------------------------------------------------

class StudioPage(QWidget):

    def __init__(self, title, description=""):

        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 22)
        layout.setSpacing(15)

        heading = QLabel(title)
        heading.setObjectName("pageTitle")

        layout.addWidget(heading)

        if description:

            subtitle = QLabel(description)
            subtitle.setObjectName("subtitle")
            subtitle.setWordWrap(True)

            layout.addWidget(subtitle)

        self.body = QVBoxLayout()
        self.body.setSpacing(14)

        layout.addLayout(self.body)
        layout.addStretch(1)


# ------------------------------------------------------------
# MAIN WINDOW
# ------------------------------------------------------------

class AetherStudio(QMainWindow):

    worker_log = Signal(str)

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "AETHER AI ULTIMATE STUDIO"
        )

        self.resize(1500, 950)
        self.setMinimumSize(1100, 700)

        self.thread = None
        self.worker = None

        self.pages = {}
        self.navigation = []

        self._build_interface()
        self.worker_log.connect(self.log)
        self.refresh_status()

    # --------------------------------------------------------
    # UI BUILD
    # --------------------------------------------------------

    def _build_interface(self):

        central = QWidget()
        self.setCentralWidget(central)

        main = QHBoxLayout(central)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(235)

        side = QVBoxLayout(sidebar)
        side.setContentsMargins(18, 22, 18, 18)
        side.setSpacing(8)

        brand = QLabel("AETHER")
        brand.setObjectName("brand")

        side.addWidget(brand)

        subtitle = QLabel("AI ULTIMATE STUDIO")
        subtitle.setObjectName("brandSub")

        side.addWidget(subtitle)
        side.addSpacing(22)

        self.stack = QStackedWidget()

        items = [
            ("⌂", "Overview", self.page_overview),
            ("✦", "Image Studio", self.page_image),
            ("◉", "Portrait / Upscale", self.page_restore),
            ("◎", "Face Studio", self.page_face),
            ("▶", "Video Studio", self.page_video),
            ("⌁", "ControlNet", self.page_controlnet),
            ("♫", "Voice Studio", self.page_voice),
            ("◌", "Lip Sync", self.page_lipsync),
            ("▤", "Timeline", self.page_timeline),
        ]

        for icon, title, builder in items:

            button = QPushButton(
                f"{icon}    {title}"
            )

            button.setObjectName("navButton")
            button.setCursor(Qt.PointingHandCursor)

            button.clicked.connect(
                lambda checked=False, b=builder, btn=button:
                self.navigate(b, btn)
            )

            side.addWidget(button)
            self.navigation.append(button)

        side.addStretch(1)

        self.gpu_status = QLabel("GPU: checking")
        self.engine_status = QLabel("ENGINE: checking")

        self.gpu_status.setObjectName("sideStatus")
        self.engine_status.setObjectName("sideStatus")

        side.addWidget(self.gpu_status)
        side.addWidget(self.engine_status)

        main.addWidget(sidebar)

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(0)

        main.addLayout(right, 1)

        topbar = QFrame()
        topbar.setObjectName("topbar")

        top_layout = QHBoxLayout(topbar)
        top_layout.setContentsMargins(24, 12, 24, 12)

        self.section_title = QLabel("Overview")
        self.section_title.setObjectName("topTitle")

        top_layout.addWidget(self.section_title)
        top_layout.addStretch(1)

        self.status_badge = QLabel("READY")
        self.status_badge.setObjectName("readyPill")

        top_layout.addWidget(self.status_badge)

        right.addWidget(topbar)

        right.addWidget(self.stack, 1)

        # ACTIVITY PANEL

        activity = QFrame()
        activity.setObjectName("bottomPanel")

        activity_layout = QVBoxLayout(activity)
        activity_layout.setContentsMargins(20, 10, 20, 12)

        row = QHBoxLayout()

        title = QLabel("ACTIVITY LOG")
        title.setObjectName("smallHeading")

        row.addWidget(title)
        row.addStretch(1)

        self.progress_label = QLabel("Idle")
        self.progress_label.setObjectName("muted")

        row.addWidget(self.progress_label)

        activity_layout.addLayout(row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(5)

        activity_layout.addWidget(self.progress)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(100)

        activity_layout.addWidget(self.log_box)

        right.addWidget(activity)

        # CREATE PAGES

        builders = [
            self.page_overview,
            self.page_image,
            self.page_restore,
            self.page_face,
            self.page_video,
            self.page_controlnet,
            self.page_voice,
            self.page_lipsync,
            self.page_timeline,
        ]

        for builder in builders:

            page = builder()

            self.pages[builder.__name__] = page
            self.stack.addWidget(page)

        self.navigate(
            self.page_overview,
            self.navigation[0]
        )

        # MENU

        file_menu = self.menuBar().addMenu("File")

        open_output = QAction(
            "Open output folder", self
        )

        open_output.triggered.connect(
            self.open_output
        )

        file_menu.addAction(open_output)

        refresh = QAction(
            "Refresh system status", self
        )

        refresh.triggered.connect(
            self.refresh_status
        )

        file_menu.addAction(refresh)

    # --------------------------------------------------------
    # COMMON COMPONENTS
    # --------------------------------------------------------

    def card(self, title, description=""):

        box = QGroupBox(title)
        box.setObjectName("moduleCard")

        layout = QVBoxLayout(box)
        layout.setSpacing(10)

        if description:

            label = QLabel(description)
            label.setObjectName("muted")
            label.setWordWrap(True)

            layout.addWidget(label)

        form = QFormLayout()
        form.setSpacing(10)

        layout.addLayout(form)

        return box, form

    def navigate(self, builder, button=None):

        page = self.pages.get(builder.__name__)

        if page is None:
            return

        self.stack.setCurrentWidget(page)

        names = {
            "page_overview": "Overview",
            "page_image": "Image Studio",
            "page_restore": "Portrait / Upscale",
            "page_face": "Face Studio",
            "page_video": "Video Studio",
            "page_controlnet": "ControlNet",
            "page_voice": "Voice Studio",
            "page_lipsync": "Lip Sync",
            "page_timeline": "Timeline",
        }

        self.section_title.setText(
            names.get(builder.__name__, "AETHER")
        )

        for item in self.navigation:

            item.setProperty(
                "active",
                item is button
            )

            item.style().unpolish(item)
            item.style().polish(item)

    def required(self, *values):

        if any(
            not str(value).strip()
            for value in values
        ):

            raise ValueError(
                "Please complete all required fields."
            )

    def backend(self):

        if engine is None:

            raise RuntimeError(
                "Could not import app.py:\n"
                + str(ENGINE_ERROR)
            )

        return engine

    def model_relative(self, value):

        value = str(value).strip()

        self.required(value)

        path = Path(value)

        if path.is_absolute():

            try:

                return str(
                    path.relative_to(MODELS_DIR)
                )

            except ValueError:

                return value

        return value

    # --------------------------------------------------------
    # MODEL DISCOVERY
    # --------------------------------------------------------

    def populate_models(self, combo):

        combo.clear()

        if engine is None:

            combo.addItem("Engine unavailable")
            return

        found = []

        try:

            result = engine.list_checkpoints()

            if isinstance(result, (list, tuple)):

                for item in result:

                    if isinstance(item, (str, Path)):
                        found.append(str(item))

                    elif isinstance(item, dict):

                        found.append(
                            str(
                                item.get("path")
                                or item.get("name")
                            )
                        )

        except Exception:
            pass

        if not found and MODELS_DIR.exists():

            for path in MODELS_DIR.rglob("*"):

                if path.is_file() and path.suffix.lower() in {
                    ".safetensors",
                    ".ckpt",
                    ".gguf"
                }:

                    try:

                        found.append(
                            str(path.relative_to(MODELS_DIR))
                        )

                    except ValueError:
                        pass

                elif path.is_dir() and (
                    path / "model_index.json"
                ).exists():

                    found.append(
                        str(path.relative_to(MODELS_DIR))
                    )

        for item in sorted(set(found), key=str.lower):
            combo.addItem(item)

        if not found:
            combo.addItem("No models discovered")

    # --------------------------------------------------------
    # OVERVIEW
    # --------------------------------------------------------

    def page_overview(self):

        page = StudioPage(
            "Welcome to AETHER",
            "Your local AI creative workstation."
        )

        grid = QGridLayout()
        grid.setSpacing(12)

        modules = [
            (
                "IMAGE STUDIO",
                "Text-to-image, image-to-image and inpainting."
            ),
            (
                "PORTRAIT STUDIO",
                "GFPGAN restoration and Real-ESRGAN upscaling."
            ),
            (
                "FACE STUDIO",
                "InsightFace image and video face swap."
            ),
            (
                "VIDEO STUDIO",
                "AnimateDiff and Stable Video Diffusion."
            ),
            (
                "CONTROLNET",
                "Canny, Depth, OpenPose and Tile."
            ),
            (
                "VOICE STUDIO",
                "Piper, RVC and audio-driven lip sync."
            ),
        ]

        for i, (title, description) in enumerate(modules):

            box, _ = self.card(title, description)

            grid.addWidget(
                box,
                i // 3,
                i % 3
            )

        page.body.addLayout(grid)

        system, form = self.card(
            "SYSTEM INFORMATION",
            "Local engine connection."
        )

        self.overview_info = QLabel()
        self.overview_info.setWordWrap(True)

        form.addRow(self.overview_info)

        page.body.addWidget(system)

        button = QPushButton("Open output folder")
        button.clicked.connect(self.open_output)

        page.body.addWidget(button)

        return page

    # --------------------------------------------------------
    # IMAGE STUDIO
    # --------------------------------------------------------

    def page_image(self):

        page = StudioPage(
            "Image Studio",
            "Generate images using the local Diffusers engine."
        )

        box, form = self.card(
            "IMAGE GENERATION",
            "Uses run_generation() from the existing app.py."
        )

        self.gen_model = QComboBox()
        self.populate_models(self.gen_model)

        self.gen_task = QComboBox()
        self.gen_task.addItems([
            "txt2img",
            "img2img",
            "inpaint",
            "outpaint"
        ])

        self.gen_prompt = QTextEdit()
        self.gen_prompt.setFixedHeight(90)
        self.gen_prompt.setPlaceholderText(
            "Describe your image..."
        )

        self.gen_negative = QLineEdit()
        self.gen_negative.setPlaceholderText(
            "Negative prompt"
        )

        self.gen_source = FilePicker(
            "Images (*.png *.jpg *.jpeg *.webp)"
        )

        self.gen_mask = FilePicker(
            "Images (*.png *.jpg *.jpeg *.webp)"
        )

        self.gen_steps = QSpinBox()
        self.gen_steps.setRange(1, 100)
        self.gen_steps.setValue(25)

        self.gen_cfg = QDoubleSpinBox()
        self.gen_cfg.setRange(0.1, 30)
        self.gen_cfg.setValue(7)

        self.gen_strength = QDoubleSpinBox()
        self.gen_strength.setRange(0.01, 1)
        self.gen_strength.setValue(0.65)

        self.gen_width = QSpinBox()
        self.gen_width.setRange(256, 1536)
        self.gen_width.setSingleStep(64)
        self.gen_width.setValue(768)

        self.gen_height = QSpinBox()
        self.gen_height.setRange(256, 1536)
        self.gen_height.setSingleStep(64)
        self.gen_height.setValue(768)

        self.gen_seed = QSpinBox()
        self.gen_seed.setRange(-1, 2147483647)
        self.gen_seed.setValue(-1)

        self.gen_lora = QLineEdit()
        self.gen_lora.setPlaceholderText(
            "Optional LoRA path relative to models"
        )

        self.gen_lora_scale = QDoubleSpinBox()
        self.gen_lora_scale.setRange(-2, 2)
        self.gen_lora_scale.setValue(1)

        self.gen_vae = QLineEdit()
        self.gen_vae.setPlaceholderText(
            "Optional VAE path relative to models"
        )

        form.addRow("Model", self.gen_model)
        form.addRow("Task", self.gen_task)
        form.addRow("Prompt", self.gen_prompt)
        form.addRow("Negative prompt", self.gen_negative)
        form.addRow("Source image", self.gen_source)
        form.addRow("Mask image", self.gen_mask)
        form.addRow("Steps", self.gen_steps)
        form.addRow("CFG", self.gen_cfg)
        form.addRow("Strength", self.gen_strength)
        form.addRow("Width", self.gen_width)
        form.addRow("Height", self.gen_height)
        form.addRow("Seed (-1 = random)", self.gen_seed)
        form.addRow("LoRA path", self.gen_lora)
        form.addRow("LoRA scale", self.gen_lora_scale)
        form.addRow("VAE path", self.gen_vae)

        run = QPushButton("Generate image")
        run.clicked.connect(self.run_generation)

        form.addRow(run)

        page.body.addWidget(box)

        self.image_preview = QLabel(
            "Generated image preview"
        )

        self.image_preview.setObjectName("preview")
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setMinimumHeight(230)

        page.body.addWidget(self.image_preview)

        refresh = QPushButton("Refresh preview")
        refresh.setObjectName("secondary")
        refresh.clicked.connect(self.refresh_preview)

        page.body.addWidget(refresh)

        return page

    def run_generation(self):

        try:

            e = self.backend()

            prompt = self.gen_prompt.toPlainText().strip()
            self.required(prompt)

            task = self.gen_task.currentText()

            source = self.gen_source.text() or None
            mask = self.gen_mask.text() or None

            if task != "txt2img" and not source:

                raise ValueError(
                    "This task requires a source image."
                )

            if task == "inpaint" and not mask:

                raise ValueError(
                    "Inpainting requires a mask image."
                )

            self.start_task(
                e.run_generation,
                task,
                self.model_relative(
                    self.gen_model.currentText()
                ),
                prompt,
                self.gen_negative.text(),
                source,
                mask,
                self.gen_steps.value(),
                self.gen_cfg.value(),
                self.gen_strength.value(),
                self.gen_width.value(),
                self.gen_height.value(),
                self.gen_seed.value(),
                self.task_callback,
                lora_rel=self.gen_lora.text().strip() or None,
                lora_scale=self.gen_lora_scale.value(),
                vae_rel=self.gen_vae.text().strip() or None,
                embedding_rel=None,
                ip_adapter_rel=None,
                ip_image_path=None
            )

        except Exception as exc:

            self.warning("Image generation", exc)

    # --------------------------------------------------------
    # PORTRAIT / UPSCALE
    # --------------------------------------------------------

    def page_restore(self):

        page = StudioPage(
            "Portrait & Upscale",
            "Connects to the tested GFPGAN and Real-ESRGAN functions."
        )

        box, form = self.card(
            "GFPGAN RESTORATION",
            "Restore portrait details with GFPGANv1.4."
        )

        self.restore_file = FilePicker(
            "Images (*.png *.jpg *.jpeg *.webp)"
        )

        form.addRow("Input image", self.restore_file)

        button = QPushButton("Restore portrait")
        button.clicked.connect(self.run_restore)

        form.addRow(button)

        page.body.addWidget(box)

        up, uf = self.card(
            "REAL-ESRGAN UPSCALER",
            "Uses the portable Real-ESRGAN executable."
        )

        self.upscale_file = FilePicker(
            "Images (*.png *.jpg *.jpeg *.webp)"
        )

        self.upscale_scale = QComboBox()
        self.upscale_scale.addItems(["2", "3", "4"])

        self.upscale_model = QComboBox()
        self.upscale_model.addItems([
            "realesrgan-x4plus",
            "realesrgan-x4plus-anime",
            "realesr-animevideov3-x2",
            "realesr-animevideov3-x3",
            "realesr-animevideov3-x4"
        ])

        uf.addRow("Input image", self.upscale_file)
        uf.addRow("Scale", self.upscale_scale)
        uf.addRow("Model", self.upscale_model)

        run = QPushButton("Upscale image")
        run.clicked.connect(self.run_upscale)

        uf.addRow(run)

        page.body.addWidget(up)

        self.restore_preview = QLabel("Output preview")
        self.restore_preview.setObjectName("preview")
        self.restore_preview.setAlignment(Qt.AlignCenter)
        self.restore_preview.setMinimumHeight(220)

        page.body.addWidget(self.restore_preview)

        return page

    def run_restore(self):

        try:

            e = self.backend()
            self.required(self.restore_file.text())

            self.start_task(
                e.restore_portrait,
                self.restore_file.text(),
                self.task_callback
            )

        except Exception as exc:

            self.warning("GFPGAN", exc)

    def run_upscale(self):

        try:

            e = self.backend()
            self.required(self.upscale_file.text())

            from PIL import Image

            image = Image.open(
                self.upscale_file.text()
            ).convert("RGB")

            scale = int(
                self.upscale_scale.currentText()
            )

            model = self.upscale_model.currentText()

            self.start_task(
                e.upscale_image,
                image,
                self.task_callback,
                scale,
                model
            )

        except Exception as exc:

            self.warning("Real-ESRGAN", exc)

    # --------------------------------------------------------
    # FACE STUDIO
    # --------------------------------------------------------

    def page_face(self):

        page = StudioPage(
            "Face Studio",
            "Use only your own face or identities you have permission to use."
        )

        box, form = self.card(
            "IMAGE FACE SWAP",
            "InsightFace image processing."
        )

        self.face_source = FilePicker(
            "Images (*.png *.jpg *.jpeg *.webp)"
        )

        self.face_target = FilePicker(
            "Images (*.png *.jpg *.jpeg *.webp)"
        )

        self.face_consent = QCheckBox(
            "I have the necessary rights and consent."
        )

        form.addRow("Source face", self.face_source)
        form.addRow("Target image", self.face_target)
        form.addRow(self.face_consent)

        button = QPushButton("Run image face swap")
        button.clicked.connect(self.run_face_swap)

        form.addRow(button)

        page.body.addWidget(box)

        video, vf = self.card(
            "VIDEO FACE SWAP",
            "InsightFace video processing."
        )

        self.video_source = FilePicker(
            "Images (*.png *.jpg *.jpeg)"
        )

        self.video_target = FilePicker(
            "Videos (*.mp4 *.mov *.mkv *.avi)"
        )

        self.video_consent = QCheckBox(
            "I have the necessary rights and consent."
        )

        vf.addRow("Source face", self.video_source)
        vf.addRow("Target video", self.video_target)
        vf.addRow(self.video_consent)

        run = QPushButton("Run video face swap")
        run.clicked.connect(self.run_video_face_swap)

        vf.addRow(run)

        page.body.addWidget(video)

        unavailable, uf = self.card(
            "FACEFUSION",
            "FaceFusion runtime is not currently connected."
        )

        label = QLabel(
            "NOT INSTALLED / NOT CONNECTED"
        )

        label.setObjectName("warningText")
        uf.addRow(label)

        page.body.addWidget(unavailable)

        return page

    def run_face_swap(self):

        try:

            e = self.backend()

            self.required(
                self.face_source.text(),
                self.face_target.text()
            )

            if not self.face_consent.isChecked():

                raise ValueError(
                    "Confirm that you have the necessary rights and consent."
                )

            self.start_task(
                e.run_face_swap,
                self.face_source.text(),
                self.face_target.text(),
                self.task_callback
            )

        except Exception as exc:

            self.warning("Face swap", exc)

    def run_video_face_swap(self):

        try:

            e = self.backend()

            self.required(
                self.video_source.text(),
                self.video_target.text()
            )

            if not self.video_consent.isChecked():

                raise ValueError(
                    "Confirm that you have the necessary rights and consent."
                )

            self.start_task(
                e.run_video_face_swap,
                self.video_source.text(),
                self.video_target.text(),
                self.task_callback
            )

        except Exception as exc:

            self.warning("Video face swap", exc)

    # --------------------------------------------------------
    # VIDEO STUDIO
    # --------------------------------------------------------

    def page_video(self):

        page = StudioPage(
            "Video Studio",
            "Video generation can be demanding on a 6 GB GPU. Start with modest frame counts."
        )

        box, form = self.card(
            "ANIMATEDIFF",
            "Animate a still image with a motion adapter."
        )

        self.anim_model = FilePicker(
            "Models (*.safetensors *.ckpt)"
        )

        self.anim_motion = FilePicker(
            "Models (*.safetensors *.ckpt)"
        )

        self.anim_image = FilePicker(
            "Images (*.png *.jpg *.jpeg *.webp)"
        )

        self.anim_prompt = QLineEdit()
        self.anim_prompt.setPlaceholderText("Motion prompt")

        self.anim_negative = QLineEdit()
        self.anim_negative.setPlaceholderText("Negative prompt")

        self.anim_steps = QSpinBox()
        self.anim_steps.setRange(1, 50)
        self.anim_steps.setValue(15)

        self.anim_cfg = QDoubleSpinBox()
        self.anim_cfg.setRange(1, 20)
        self.anim_cfg.setValue(7)

        self.anim_frames = QSpinBox()
        self.anim_frames.setRange(4, 32)
        self.anim_frames.setValue(16)

        self.anim_fps = QSpinBox()
        self.anim_fps.setRange(4, 30)
        self.anim_fps.setValue(8)

        form.addRow("Base model", self.anim_model)
        form.addRow("Motion adapter", self.anim_motion)
        form.addRow("Initial image", self.anim_image)
        form.addRow("Prompt", self.anim_prompt)
        form.addRow("Negative prompt", self.anim_negative)
        form.addRow("Steps", self.anim_steps)
        form.addRow("CFG", self.anim_cfg)
        form.addRow("Frames", self.anim_frames)
        form.addRow("FPS", self.anim_fps)

        run = QPushButton("Generate AnimateDiff")
        run.clicked.connect(self.run_animatediff)

        form.addRow(run)

        page.body.addWidget(box)

        svd, sf = self.card(
            "STABLE VIDEO DIFFUSION",
            "Generate a video from an image."
        )

        self.svd_image = FilePicker(
            "Images (*.png *.jpg *.jpeg *.webp)"
        )

        self.svd_notes = QLineEdit()
        self.svd_notes.setPlaceholderText("Motion description")

        self.svd_duration = QDoubleSpinBox()
        self.svd_duration.setRange(1, 10)
        self.svd_duration.setValue(3)

        self.svd_fps = QSpinBox()
        self.svd_fps.setRange(4, 30)
        self.svd_fps.setValue(8)

        self.svd_steps = QSpinBox()
        self.svd_steps.setRange(1, 50)
        self.svd_steps.setValue(20)

        sf.addRow("Input image", self.svd_image)
        sf.addRow("Motion notes", self.svd_notes)
        sf.addRow("Duration (seconds)", self.svd_duration)
        sf.addRow("FPS", self.svd_fps)
        sf.addRow("Steps", self.svd_steps)

        run_svd = QPushButton("Generate SVD video")
        run_svd.clicked.connect(self.run_svd)

        sf.addRow(run_svd)

        page.body.addWidget(svd)

        return page

    def run_animatediff(self):

        try:

            e = self.backend()

            self.required(
                self.anim_model.text(),
                self.anim_motion.text(),
                self.anim_image.text(),
                self.anim_prompt.text()
            )

            self.start_task(
                e.make_animatediff,
                self.model_relative(self.anim_model.text()),
                self.model_relative(self.anim_motion.text()),
                self.anim_prompt.text(),
                self.anim_negative.text(),
                self.anim_steps.value(),
                self.anim_cfg.value(),
                self.anim_frames.value(),
                self.anim_fps.value(),
                self.anim_image.text(),
                self.task_callback
            )

        except Exception as exc:

            self.warning("AnimateDiff", exc)

    def run_svd(self):

        try:

            e = self.backend()
            self.required(self.svd_image.text())

            self.start_task(
                e.make_svd_video,
                self.svd_image.text(),
                self.svd_notes.text(),
                "",
                self.svd_duration.value(),
                self.svd_fps.value(),
                127,
                0.02,
                20,
                4,
                self.task_callback,
                -1
            )

        except Exception as exc:

            self.warning("SVD", exc)

    # --------------------------------------------------------
    # CONTROLNET
    # --------------------------------------------------------

    def page_controlnet(self):

        page = StudioPage(
            "ControlNet",
            "Canny, Depth, OpenPose and Tile."
        )

        box, form = self.card(
            "CONTROLNET GENERATION",
            "Uses run_controlnet_mode() from app.py."
        )

        self.cn_mode = QComboBox()
        self.cn_mode.addItems([
            "canny",
            "depth",
            "openpose",
            "tile"
        ])

        self.cn_base = QLineEdit()
        self.cn_base.setPlaceholderText(
            "Base model path relative to models"
        )

        self.cn_control = QLineEdit()
        self.cn_control.setPlaceholderText(
            "Control model path relative to models"
        )

        self.cn_image = FilePicker(
            "Images (*.png *.jpg *.jpeg *.webp)"
        )

        self.cn_prompt = QTextEdit()
        self.cn_prompt.setFixedHeight(80)

        self.cn_negative = QLineEdit()

        self.cn_steps = QSpinBox()
        self.cn_steps.setRange(1, 80)
        self.cn_steps.setValue(25)

        self.cn_cfg = QDoubleSpinBox()
        self.cn_cfg.setRange(0.1, 30)
        self.cn_cfg.setValue(7)

        self.cn_strength = QDoubleSpinBox()
        self.cn_strength.setRange(0.01, 1)
        self.cn_strength.setValue(0.8)

        form.addRow("Mode", self.cn_mode)
        form.addRow("Base model", self.cn_base)
        form.addRow("Control model", self.cn_control)
        form.addRow("Input image", self.cn_image)
        form.addRow("Prompt", self.cn_prompt)
        form.addRow("Negative prompt", self.cn_negative)
        form.addRow("Steps", self.cn_steps)
        form.addRow("CFG", self.cn_cfg)
        form.addRow("Strength", self.cn_strength)

        run = QPushButton("Run ControlNet")
        run.clicked.connect(self.run_controlnet)

        form.addRow(run)

        page.body.addWidget(box)

        return page

    def run_controlnet(self):

        try:

            e = self.backend()

            self.required(
                self.cn_base.text(),
                self.cn_control.text(),
                self.cn_image.text(),
                self.cn_prompt.toPlainText()
            )

            self.start_task(
                e.run_controlnet_mode,
                self.cn_mode.currentText(),
                self.model_relative(self.cn_base.text()),
                self.model_relative(self.cn_control.text()),
                self.cn_image.text(),
                self.cn_prompt.toPlainText(),
                self.cn_negative.text(),
                self.cn_steps.value(),
                self.cn_cfg.value(),
                self.cn_strength.value(),
                self.task_callback
            )

        except Exception as exc:

            self.warning("ControlNet", exc)

    # --------------------------------------------------------
    # VOICE STUDIO
    # --------------------------------------------------------

    def page_voice(self):

        page = StudioPage(
            "Voice Studio",
            "Local Piper synthesis and RVC conversion."
        )

        piper, pf = self.card(
            "PIPER TEXT TO SPEECH",
            "Synthesize speech using a local Piper model."
        )

        self.piper_text = QTextEdit()
        self.piper_text.setFixedHeight(100)

        self.piper_model = FilePicker(
            "Piper models (*.onnx)"
        )

        pf.addRow("Text", self.piper_text)
        pf.addRow("Voice model", self.piper_model)

        run = QPushButton("Synthesize speech")
        run.clicked.connect(self.run_piper)

        pf.addRow(run)

        page.body.addWidget(piper)

        rvc, rf = self.card(
            "RVC VOICE CONVERSION",
            "Convert an audio file using an RVC model."
        )

        self.rvc_audio = FilePicker(
            "Audio (*.wav *.mp3 *.flac *.ogg *.m4a)"
        )

        self.rvc_model = FilePicker(
            "RVC models (*.pth)"
        )

        self.rvc_index = FilePicker(
            "Index files (*.index)"
        )

        rf.addRow("Audio input", self.rvc_audio)
        rf.addRow("Model", self.rvc_model)
        rf.addRow("Index", self.rvc_index)

        convert = QPushButton("Run RVC conversion")
        convert.clicked.connect(self.run_rvc)

        rf.addRow(convert)

        page.body.addWidget(rvc)

        return page

    def run_piper(self):

        try:

            e = self.backend()

            self.required(
                self.piper_text.toPlainText(),
                self.piper_model.text()
            )

            self.start_task(
                e.synthesize_voice_piper,
                self.piper_text.toPlainText(),
                self.piper_model.text(),
                self.task_callback
            )

        except Exception as exc:

            self.warning("Piper", exc)

    def run_rvc(self):

        try:

            e = self.backend()

            self.required(
                self.rvc_audio.text(),
                self.rvc_model.text(),
                self.rvc_index.text()
            )

            self.start_task(
                e.run_rvc_conversion,
                self.rvc_audio.text(),
                self.rvc_model.text(),
                self.rvc_index.text(),
                self.task_callback
            )

        except Exception as exc:

            self.warning("RVC", exc)

    # --------------------------------------------------------
    # LIP SYNC
    # --------------------------------------------------------

    def page_lipsync(self):

        page = StudioPage(
            "Lip Sync",
            "Synchronize a face video with an audio track."
        )

        box, form = self.card(
            "AUDIO DRIVEN LIP SYNC",
            "Uses create_lipsync_video() from app.py."
        )

        self.lip_video = FilePicker(
            "Videos (*.mp4 *.mov *.mkv *.avi)"
        )

        self.lip_audio = FilePicker(
            "Audio (*.wav *.mp3 *.flac *.ogg *.m4a)"
        )

        form.addRow("Face video", self.lip_video)
        form.addRow("Audio", self.lip_audio)

        run = QPushButton("Create lip-sync video")
        run.clicked.connect(self.run_lipsync)

        form.addRow(run)

        page.body.addWidget(box)

        return page

    def run_lipsync(self):

        try:

            e = self.backend()

            self.required(
                self.lip_video.text(),
                self.lip_audio.text()
            )

            self.start_task(
                e.create_lipsync_video,
                self.lip_video.text(),
                self.lip_audio.text(),
                self.task_callback
            )

        except Exception as exc:

            self.warning("Lip Sync", exc)

    # --------------------------------------------------------
    # TIMELINE
    # --------------------------------------------------------

    def page_timeline(self):

        page = StudioPage(
            "Timeline",
            "Combine media files using the existing FFmpeg exporter."
        )

        box, form = self.card(
            "TIMELINE EXPORT",
            "Enter one media file path per line in sequence order."
        )

        self.timeline_items = QTextEdit()
        self.timeline_items.setFixedHeight(120)

        self.timeline_items.setPlaceholderText(
            "E:\\AIExoticLocal\\output_media\\clip1.mp4\n"
            "E:\\AIExoticLocal\\output_media\\clip2.mp4"
        )

        self.timeline_output = FilePicker(
            "MP4 video (*.mp4)",
            save=True
        )

        self.timeline_audio = FilePicker(
            "Audio (*.wav *.mp3 *.flac *.m4a)"
        )

        self.timeline_fps = QSpinBox()
        self.timeline_fps.setRange(1, 60)
        self.timeline_fps.setValue(30)

        form.addRow("Media files", self.timeline_items)
        form.addRow("Output file", self.timeline_output)
        form.addRow("Optional audio", self.timeline_audio)
        form.addRow("FPS", self.timeline_fps)

        run = QPushButton("Export timeline")
        run.clicked.connect(self.run_timeline)

        form.addRow(run)

        page.body.addWidget(box)

        return page

    def run_timeline(self):

        try:

            e = self.backend()

            items = [
                line.strip().strip('"')
                for line in self.timeline_items.toPlainText().splitlines()
                if line.strip()
            ]

            self.required(
                *items,
                self.timeline_output.text()
            )

            self.start_task(
                e.export_timeline,
                items,
                self.timeline_output.text(),
                self.timeline_audio.text() or None,
                self.timeline_fps.value()
            )

        except Exception as exc:

            self.warning("Timeline", exc)

    # --------------------------------------------------------
    # TASK EXECUTION
    # --------------------------------------------------------

    def task_callback(self, *args):

        message = " ".join(
            str(x) for x in args
            if x is not None
        )

        if message:
            self.worker_log.emit(message)

    def start_task(self, function, *args, **kwargs):

        if self.thread is not None and self.thread.isRunning():

            QMessageBox.information(
                self,
                "AETHER is busy",
                "Wait until the current task finishes."
            )

            return

        self.progress.setValue(0)
        self.progress_label.setText("Working...")
        self.status_badge.setText("RUNNING")

        self.log(
            "Starting: " + function.__name__
        )

        self.thread = QThread(self)

        self.worker = Worker(
            function,
            *args,
            **kwargs
        )

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(
            self.worker.run
        )

        self.worker.message.connect(
            self.log
        )

        self.worker.finished.connect(
            self.task_finished
        )

        self.worker.failed.connect(
            self.task_failed
        )

        self.worker.finished.connect(
            self.thread.quit
        )

        self.worker.failed.connect(
            self.thread.quit
        )

        self.thread.finished.connect(
            self.thread_finished
        )

        self.thread.start()

    def thread_finished(self):

        self.thread = None
        self.worker = None

    def task_finished(self, result):

        self.progress.setValue(100)
        self.progress_label.setText("Completed")
        self.status_badge.setText("READY")

        if result is not None:

            self.log(
                "Result: " + str(result)
            )

        self.refresh_preview()

    def task_failed(self, error):

        self.progress_label.setText("Failed")
        self.status_badge.setText("ERROR")

        self.log(error)

        QMessageBox.critical(
            self,
            "Task failed",
            error[-3500:]
        )

    def warning(self, title, error):

        QMessageBox.warning(
            self,
            title,
            str(error)
        )

    def log(self, message):

        self.log_box.append(str(message))

        self.progress_label.setText(
            str(message).splitlines()[-1][:120]
        )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    def refresh_status(self):

        if engine is None:

            self.engine_status.setText(
                "ENGINE: IMPORT ERROR"
            )

            self.gpu_status.setText(
                "GPU: unavailable"
            )

            if hasattr(self, "overview_info"):

                self.overview_info.setText(
                    "Backend import error:\n"
                    + str(ENGINE_ERROR)
                )

            return

        self.engine_status.setText(
            "ENGINE: CONNECTED"
        )

        gpu = "CPU"

        try:

            import torch

            if torch.cuda.is_available():

                gpu = torch.cuda.get_device_name(0)

                memory = (
                    torch.cuda.get_device_properties(0)
                    .total_memory / (1024 ** 3)
                )

                gpu += f" ({memory:.1f} GB)"

        except Exception:
            pass

        self.gpu_status.setText(
            "GPU: " + gpu
        )

        if hasattr(self, "overview_info"):

            self.overview_info.setText(
                "Interface: PySide6\n"
                f"Backend: {Path(engine.__file__).name}\n"
                f"Models: {MODELS_DIR}\n"
                f"Output: {OUTPUT_DIR}\n"
                "GFPGAN: existing backend function\n"
                "Real-ESRGAN: existing backend function\n"
                "FaceFusion: not connected"
            )

    # --------------------------------------------------------
    # PREVIEW
    # --------------------------------------------------------

    def refresh_preview(self):

        if not hasattr(self, "image_preview"):
            return

        images = []

        try:

            images = sorted(
                [
                    p for p in OUTPUT_DIR.iterdir()
                    if p.suffix.lower() in {
                        ".png", ".jpg", ".jpeg", ".webp"
                    }
                ],
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )

        except Exception:
            pass

        if not images:

            self.image_preview.setText(
                "No generated images yet."
            )

            return

        pixmap = QPixmap(str(images[0]))

        if pixmap.isNull():
            return

        scaled = pixmap.scaled(
            QSize(850, 350),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.image_preview.setPixmap(scaled)

        if hasattr(self, "restore_preview"):

            self.restore_preview.setPixmap(
                pixmap.scaled(
                    QSize(750, 260),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    def open_output(self):

        try:

            if sys.platform == "win32":

                os.startfile(str(OUTPUT_DIR))

            else:

                subprocess.Popen([
                    "xdg-open",
                    str(OUTPUT_DIR)
                ])

        except Exception as exc:

            self.warning("Output folder", exc)


# ------------------------------------------------------------
# THEME
# ------------------------------------------------------------

def apply_theme(application):

    application.setStyle("Fusion")

    application.setFont(
        QFont("Segoe UI", 10)
    )

    application.setStyleSheet("""
        QWidget {
            background: #11151d;
            color: #e8edf5;
        }

        QMainWindow {
            background: #0d1118;
        }

        QMenuBar {
            background: #11151d;
            color: #dce5f2;
            padding: 5px;
        }

        QMenu {
            background: #171e29;
            border: 1px solid #2a3545;
        }

        #sidebar {
            background: #0b1017;
            border-right: 1px solid #222c3a;
        }

        #brand {
            color: #f2f6ff;
            font-size: 27px;
            font-weight: 800;
            letter-spacing: 3px;
        }

        #brandSub {
            color: #8194ad;
            font-size: 9px;
            letter-spacing: 2px;
        }

        #navButton {
            text-align: left;
            padding: 12px;
            border: 0;
            border-radius: 8px;
            color: #aab8ca;
            background: transparent;
        }

        #navButton:hover {
            background: #182333;
            color: white;
        }

        #navButton[active="true"] {
            background: #1d3046;
            color: #80c7ff;
            font-weight: 700;
        }

        #sideStatus {
            color: #8194ad;
            font-size: 10px;
        }

        #topbar {
            background: #111720;
            border-bottom: 1px solid #222c3a;
        }

        #topTitle {
            font-size: 13px;
            font-weight: 700;
        }

        #readyPill {
            color: #7ee0b1;
            background: #173328;
            border-radius: 10px;
            padding: 5px 12px;
            font-size: 10px;
            font-weight: 700;
        }

        #pageTitle {
            color: #f4f7fc;
            font-size: 27px;
            font-weight: 750;
        }

        #subtitle {
            color: #8e9db1;
            font-size: 12px;
        }

        QGroupBox {
            border: 1px solid #2a3545;
            border-radius: 11px;
            margin-top: 10px;
            padding: 13px;
            background: #171e29;
            font-weight: 700;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 5px;
            color: #9fcfff;
        }

        QLabel#muted {
            color: #8e9db1;
            font-size: 11px;
            font-weight: 400;
        }

        #smallHeading {
            color: #9eb1c8;
            font-size: 10px;
            font-weight: 800;
        }

        #bottomPanel {
            background: #0d121a;
            border-top: 1px solid #222c3a;
        }

        QTextEdit,
        QLineEdit,
        QComboBox,
        QSpinBox,
        QDoubleSpinBox {
            background: #0e141d;
            border: 1px solid #303d4e;
            border-radius: 6px;
            padding: 8px;
            color: #eaf0f8;
        }

        QTextEdit:focus,
        QLineEdit:focus,
        QComboBox:focus {
            border: 1px solid #4c9bd2;
        }

        QPushButton {
            background: #2779b9;
            border: 0;
            border-radius: 7px;
            color: white;
            padding: 10px 16px;
            font-weight: 700;
        }

        QPushButton:hover {
            background: #348dce;
        }

        QPushButton#secondary {
            background: #263447;
            color: #cbd8e8;
        }

        QPushButton#secondary:hover {
            background: #34485f;
        }

        QCheckBox {
            color: #aebdd0;
            spacing: 8px;
        }

        QProgressBar {
            background: #202a37;
            border: 0;
            border-radius: 2px;
        }

        QProgressBar::chunk {
            background: #42a5e8;
            border-radius: 2px;
        }

        #preview {
            background: #0b1017;
            border: 1px dashed #3a4b60;
            border-radius: 8px;
            color: #71839a;
            padding: 10px;
        }

        #warningText {
            color: #f0c477;
            font-weight: 700;
        }
    """)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    application = QApplication(sys.argv)

    apply_theme(application)

    window = AetherStudio()
    window.show()

    if engine is None:

        QMessageBox.critical(
            window,
            "Backend error",
            "Could not import app.py:\n"
            + str(ENGINE_ERROR)
        )

    sys.exit(application.exec())


if __name__ == "__main__":
    main()
