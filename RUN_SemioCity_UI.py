from __future__ import annotations
import os, sys, time, traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, Dict

import numpy as np
from PIL import Image

# Import generation tracker
try:
    from generation_tracker import tracker, start_generation, log_step1, log_step2, log_step3, complete_generation
except Exception:
    # Soft fallback if tracker is missing
    class _Dummy:
        csv_path = "generation_log.csv"
    def _log(*a, **k): pass
    tracker = _Dummy()
    start_generation = lambda **kw: "session"
    log_step1 = _log; log_step2 = _log; log_step3 = _log; complete_generation = _log

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QPushButton,
    QProgressBar, QFileDialog, QMessageBox, QGroupBox, QCheckBox, QSpinBox,
    QSplitter, QScrollArea, QFrame, QMainWindow, QTabWidget, QSizePolicy
)
from PyQt6.QtGui import QPixmap, QFont, QPalette, QColor
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl

# Import ChatController for AI integration
try:
    import sys
    ui_agent_path = str(Path(__file__).parent / "00_ui_agent")
    if ui_agent_path not in sys.path:
        sys.path.insert(0, ui_agent_path)
    from chat_controller import ChatController
except ImportError:
    ChatController = None

# Try to import WebEngine, fallback to regular widget if not available
try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    WEBENGINE_AVAILABLE = True
except ImportError:
    QWebEngineView = None
    WEBENGINE_AVAILABLE = False

# Optional heavy deps guarded
try:
    import torch
except Exception:
    torch = None

try:
    from transformers import pipeline as hf_pipeline
except Exception:
    hf_pipeline = None

# Hunyuan drop-in if present
try:
    from hunyuan3d21.dropin import image_to_3d  # type: ignore
    HUNYUAN_DROPIN_AVAILABLE = True
except Exception:
    image_to_3d = None
    HUNYUAN_DROPIN_AVAILABLE = False


# ---------- Path discovery & sys.path wiring ----------

@dataclass
class Paths:
    root: Path
    ui_agent: Optional[Path]
    ai_gen: Optional[Path]
    remove_bg: Optional[Path]
    twoD_to_3D: Optional[Path]
    backend_base: Optional[Path]  # 3d_backend
    hy3dshape_inner: Optional[Path]
    hy3dpaint: Optional[Path]
    diff_renderer: Optional[Path]


def find_paths(start: Path) -> Paths:
    root = start.resolve()
    # Candidates
    ui_agent = (root / "00_ui_agent")
    ai_gen = (root / "01_ai_generation")
    remove_bg = (root / "02_remove_background")

    # Prefer the known-good 06 copy if present, then 03 as fallback
    six = None
    for name in ["06_2D_to_3D", "06_2d_to_3d"]:
        p = root / name
        if p.exists():
            six = p
            break

    three = None
    for name in ["03_2D_to_3D", "03_2d_to_3d", "03_to_3d", "03_2D-3D", "03-2D_to_3D"]:
        p = root / name
        if p.exists():
            three = p
            break

    # backend_base under 06, root, or 03 (that order)
    backend_base = None
    for bb in [
        (six / "3d_backend") if six else None,
        root / "3d_backend",
        (three / "3d_backend") if three else None,
    ]:
        if bb and bb.exists():
            backend_base = bb
            break

    hy3dshape_inner = None
    hy3dpaint = None
    diff_renderer = None
    if backend_base:
        b = backend_base
        # Direct path to hy3dshape (single layer)
        hy3dshape_inner = b / "hy3dshape"
        hy3dpaint = b / "hy3dpaint"
        diff_renderer = hy3dpaint / "DifferentiableRenderer"

    # Put viable dirs on sys.path
    for p in [ui_agent, ai_gen, remove_bg, three, backend_base, hy3dshape_inner, hy3dpaint, diff_renderer]:
        if p and p.exists():
            sp = str(p.resolve())
            if sp not in sys.path:
                sys.path.insert(0, sp)

    return Paths(
        root=root,
        ui_agent=ui_agent if ui_agent.exists() else None,
        ai_gen=ai_gen if ai_gen.exists() else None,
        remove_bg=remove_bg if remove_bg.exists() else None,
        twoD_to_3D=three if (three and three.exists()) else None,
        backend_base=backend_base if (backend_base and backend_base.exists()) else None,
        hy3dshape_inner=hy3dshape_inner if (hy3dshape_inner and hy3dshape_inner.exists()) else None,
        hy3dpaint=hy3dpaint if (hy3dpaint and hy3dpaint.exists()) else None,
        diff_renderer=diff_renderer if (diff_renderer and diff_renderer.exists()) else None,
    )


HERE = Path(__file__).resolve().parent
PATHS = find_paths(HERE)


# Compatibility shim for backend name drift
import sys as _sys, importlib
try:
    m = importlib.import_module("hy3dshape.models.denoisers.hunyuan3ddit")
    # Make the config's module name resolve to the one you actually have
    _sys.modules.setdefault("hy3dshape.models.denoisers.hunyuandit", m)

    # Ensure the expected class name exists - direct alias
    if hasattr(m, "Hunyuan3DDiT") and not hasattr(m, "HunYuanDiTPlain"):
        setattr(m, "HunYuanDiTPlain", getattr(m, "Hunyuan3DDiT"))
except Exception:
    # Leave silently; the import might already be correct in some setups
    pass

# Try importing backend after wiring sys.path
try:
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline  # type: ignore
except Exception:
    Hunyuan3DDiTFlowMatchingPipeline = None

try:
    from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline  # type: ignore
except Exception:
    Hunyuan3DPaintPipeline = None


# ---------- Tiny in-process orchestrator with streaming ----------

class InProcessOrchestrator:
    def __init__(self, paths: Paths, out_root: Path):
        self.paths = paths
        self.out_root = out_root

    # 01: image generation
    def generate_image_stream(self, prompt: str, steps: int = 30, seed: int = 123456789,
                              model: str = "black-forest-labs/FLUX.1-dev", style: str = None,
                              use_lora: bool = True, on_log: Callable[[str], None] | None = None) -> Path:
        out = self.out_root / "images"
        out.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_path = out / f"gen_{ts}.png"
        if on_log: on_log(f"Loading FLUX pipeline: {model}")
        # Import on demand to avoid import errors if user doesn't need this step
        from run_flux_workflow import main as flux_main  # type: ignore
        flux_main(prompt, str(out_path), steps, seed, model, use_lora=use_lora)
        if on_log: on_log(f"Generated image -> {out_path}")
        return out_path

    # 02: background/texture removal (monochrome clean)
    def monochrome_clean_stream(self, image_path: Path, prompt: Optional[str] = None,
                                on_log: Callable[[str], None] | None = None) -> Path:
        out = self.out_root / "images"
        out.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_path = out / f"mono_{ts}.png"
        # Validate input image path with clear logging
        try:
            resolved_in = Path(image_path).resolve()
        except Exception:
            resolved_in = Path(str(image_path))
        if on_log: on_log(f"Step 2: input image = {resolved_in}")
        if not resolved_in.exists():
            if on_log: on_log(f"Step 2 ERROR: input image not found: {resolved_in}")
            raise FileNotFoundError(f"Kontext input not found: {resolved_in}")
        if on_log: on_log("Loading FLUX Kontext pipeline")

        try:
            from flux_kontext_simple import run_kontext, DEFAULT_PROMPT  # type: ignore
            if on_log: on_log(f"Running Kontext → {out_path}")
            run_kontext(str(resolved_in), str(out_path), prompt or DEFAULT_PROMPT)
            if on_log: on_log(f"Monochrome image -> {out_path}")
            return out_path
        except Exception as e:
            if on_log: on_log(f"Kontext failed: {e}")
            raise

    # 03: 2D -> 3D (shape + optional texture)
    def generate_3d_stream(self, image_path: Path, enable_texture: bool = True,
                           on_log: Callable[[str], None] | None = None) -> Dict[str, str]:
        out_mesh = self.out_root / "meshes"
        out_tex = self.out_root / "textured"
        for p in [out_mesh, out_tex]:
            p.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")

        # Prefer drop-in
        if HUNYUAN_DROPIN_AVAILABLE:
            if on_log: on_log("Using Hunyuan drop-in")
            res = image_to_3d(
                image_path=str(image_path),
                out_dir=str(out_mesh.parent),
                enable_texture=enable_texture,
                max_num_view=6,
                texture_res=512
            )
            # Normalize keys to our expected names
            result = {
                "out_dir": str(self.out_root),
            }
            if isinstance(res, dict):
                if "mesh_shape_glb" in res:
                    result["mesh_shape_glb"] = res["mesh_shape_glb"]
                if "mesh_textured_glb" in res:
                    result["mesh_textured_glb"] = res["mesh_textured_glb"]
            return result

        # Fallback: direct pipelines
        if Hunyuan3DDiTFlowMatchingPipeline is None:
            raise RuntimeError("hy3dshape not importable; check 3d_backend paths")

        if on_log: on_log("Generating 3D shape (Hunyuan3D-2.1)")
        shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1")
        mesh = shape_pipe(image=str(image_path))[0]
        shape_path = out_mesh / f"mesh_{ts}.glb"
        try:
            mesh.export(str(shape_path))
        except Exception:
            # Try trimesh as fallback
            import trimesh
            if isinstance(mesh, trimesh.Trimesh):
                mesh.export(str(shape_path))
            else:
                raise

        textured_path = None
        if enable_texture:
            if Hunyuan3DPaintPipeline is None:
                if on_log: on_log("Texture pipeline missing; skipping texture")
            else:
                if on_log: on_log("Generating texture (Hunyuan3D-Paint-2.1)")
                try:
                    paint_pipe = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-Paint-2.1")
                    textured = paint_pipe(mesh, image=str(image_path))
                    textured_path = out_tex / f"textured_{ts}.glb"
                    textured.export(str(textured_path))
                except Exception as e:
                    if on_log: on_log(f"Texture generation failed: {e}")

        result = {
            "out_dir": str(self.out_root),
            "mesh_shape_glb": str(shape_path),
        }
        if textured_path:
            result["mesh_textured_glb"] = str(textured_path)
        return result


# ---------- Qt workers ----------

class WorkerStream(QThread):
    log = pyqtSignal(str)
    finished = pyqtSignal(str, object)
    errored = pyqtSignal(str)

    def __init__(self, fn: Callable, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            def _on_log(msg: str):
                self.log.emit(msg)
            res = self.fn(*self.args, on_log=_on_log, **self.kwargs)
            self.finished.emit(self.fn.__name__, res)
        except Exception as e:
            tb = traceback.format_exc()
            self.errored.emit(tb)


class PipelineWorker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    errored = pyqtSignal(str)
    intermediate = pyqtSignal(str, dict)  # NEW: for intermediate step updates

    def __init__(self, orch: InProcessOrchestrator, prompt: Optional[str], image_path: Optional[str],
                 enable_texture: bool, steps: int = 10, use_lora: bool = True):
        super().__init__()
        self.orch = orch
        self.prompt = prompt
        self.image_path = image_path
        self.enable_texture = enable_texture
        self.steps = steps
        self.use_lora = use_lora

    def run(self):
        try:
            self.progress.emit(5); self.log.emit("Pipeline: preparing")
            result = {}

            # Start tracking session
            session_id = start_generation(
                prompt=self.prompt or "No prompt",
                texture_enabled=self.enable_texture,
                depth_threshold=100,  # Default values
                depth_max=200
            )
            self.log.emit(f"Started tracking session: {session_id}")
            
            # Validate we have either a prompt or an image
            if not self.prompt and not self.image_path:
                self.log.emit("Pipeline ERROR: No prompt or image provided")
                raise ValueError("Pipeline requires either a prompt or an image")

            # Step 1
            if self.prompt and not self.image_path:
                self.progress.emit(10); self.log.emit("Step 1: generate image")
                img = self.orch.generate_image_stream(self.prompt, steps=self.steps, use_lora=self.use_lora)  # Use configured steps
                self.image_path = str(img)
                result["image_generated"] = self.image_path
                log_step1(self.image_path)  # Track step 1
                # Emit intermediate signal for step 1
                self.intermediate.emit("step1", {"image_generated": self.image_path})
            else:
                self.log.emit("Step 1: skipped (image provided)")
                result["image_generated"] = self.image_path
                if self.image_path:
                    log_step1(self.image_path)  # Track provided image

            # Step 2
            self.progress.emit(35); self.log.emit("Step 2: background/texture cleanup")
            try:
                # Ensure we have a valid image path for Step 2
                if not self.image_path:
                    self.log.emit("Step 2 ERROR: No image available for background removal")
                    raise ValueError("No image available for Step 2")
                
                self.log.emit(f"Step 2: Processing image: {self.image_path}")
                mono = self.orch.monochrome_clean_stream(Path(self.image_path), on_log=self.log.emit)
                result["image_cleaned"] = str(mono)
                log_step2(str(mono))  # Track step 2
                self.log.emit(f"Step 2 completed: {mono}")
                # Emit intermediate signal for step 2
                self.intermediate.emit("step2", {"image_cleaned": result["image_cleaned"]})
            except Exception as e:
                self.log.emit(f"Step 2 failed: {e}")
                raise

            # Step 3
            self.progress.emit(65); self.log.emit("Step 3: 2D -> 3D")
            res = self.orch.generate_3d_stream(Path(mono), enable_texture=self.enable_texture)

            # Track step 3 results
            mesh_shape = res.get("mesh_shape_glb") if isinstance(res, dict) else None
            mesh_textured = res.get("mesh_textured_glb") if isinstance(res, dict) else None
            mesh_final = mesh_textured or mesh_shape  # Prefer textured if available

            log_step3(
                mesh_shape_path=mesh_shape,
                mesh_textured_path=mesh_textured,
                final_mesh_path=mesh_final
            )

            # Merge 3D results with intermediate results
            if isinstance(res, dict):
                result.update(res)
            else:
                result["mesh_result"] = res

            self.progress.emit(100); self.log.emit("Pipeline finished")

            # Complete tracking session
            complete_generation(status='success', notes='Full pipeline completed')

            self.finished.emit(result)
        except Exception as e:
            # Complete tracking session with error
            complete_generation(status='error', error_message=str(e), notes='Pipeline failed')
            self.errored.emit(traceback.format_exc())


# ---------- UI ----------

@dataclass
class Config:
    output_dir: str = "output"


# New: square preview label to enforce 1:1 containers without changing names
class SquareLabel(QLabel):
    def __init__(self, text: str = "", side: int = 320, parent: QWidget | None = None):
        super().__init__(text, parent)
        self._side = side
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Make it visually square and obvious as a container
        self.setMinimumSize(side, side)
        self.setMaximumHeight(side)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet("border: 2px solid #404040; background-color: #2d2d2d; border-radius: 8px;")

    def resizeEvent(self, ev):  # keep square feel when container grows
        w = self.width()
        if self.maximumHeight() != w:
            # clamp height to width so it remains ~square
            self.setMaximumHeight(w)
        # Don't auto-rescale images here - let the _set_preview method handle it properly
        return super().resizeEvent(ev)


class SemioAgentUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SemioCity — Unified Agent UI (Dark Theme)")
        self.setGeometry(100, 100, 1600, 1000)

        # Configuration
        self.cfg = Config()
        self.out_root = (PATHS.root / self.cfg.output_dir).resolve()
        self.out_root.mkdir(exist_ok=True)
        self.orch = InProcessOrchestrator(PATHS, self.out_root)

        self.busy = False
        self.current_image = None
        self.current_mesh_path = None

        # Initialize ChatController for AI integration
        self.chat = ChatController() if ChatController else None

        # Apply dark theme
        self._apply_dark_theme()
        self._setup_ui()

    def _apply_dark_theme(self):
        """Apply elegant dark theme with improved styling"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
                color: #ffffff;
                font-family: 'Inter', sans-serif;
            }

            /* GroupBoxes — clean, minimal borders */
            QGroupBox {
                font-family: 'Inter', sans-serif;
                font-weight: 600;
                font-size: 16px;
                border: 1px solid #333333;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 8px;
                background-color: #252525;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px 0 8px;
                color: #cccccc;
                background-color: transparent;
                font-family: 'Inter', sans-serif;
                font-weight: 600;
                font-size: 16px;
            }

            /* Buttons — modern, flat design */
            QPushButton {
                background-color: #2d2d2d;
                border: 1px solid #404040;
                border-radius: 6px;
                padding: 10px 16px;
                min-width: 120px;
                color: #ffffff;
                font-family: 'Inter', sans-serif;
                font-weight: 500;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
                border-color: #505050;
            }
            QPushButton:pressed {
                background-color: #1f1f1f;
                border-color: #2a2a2a;
            }
            QPushButton:disabled {
                background-color: #1a1a1a;
                color: #666666;
                border-color: #2a2a2a;
            }

            /* Primary buttons — blue accent */
            QPushButton[primary="true"] {
                background-color: #0078d4;
                border-color: #106ebe;
                color: #ffffff;
                font-weight: 600;
            }
            QPushButton[primary="true"]:hover {
                background-color: #106ebe;
                border-color: #005a9e;
            }
            QPushButton[primary="true"]:pressed {
                background-color: #005a9e;
                border-color: #004578;
            }

            /* Labels */
            QLabel {
                color: #ffffff;
                font-family: 'Inter', sans-serif;
                font-size: 13px;
            }

            /* Text inputs */
            QTextEdit {
                background-color: #2d2d2d;
                border: 1px solid #404040;
                border-radius: 6px;
                color: #ffffff;
                padding: 8px;
                font-family: 'Inter', sans-serif;
                font-size: 13px;
                selection-background-color: #0078d4;
            }
            QTextEdit:focus {
                border-color: #0078d4;
            }

            /* Progress bar */
            QProgressBar {
                border: 1px solid #404040;
                border-radius: 6px;
                text-align: center;
                background-color: #2d2d2d;
                color: #ffffff;
                font-weight: 500;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 5px;
            }

            /* SpinBoxes */
            QSpinBox {
                background-color: #2d2d2d;
                border: 1px solid #404040;
                border-radius: 6px;
                color: #ffffff;
                padding: 6px;
                font-size: 13px;
            }
            QSpinBox:focus {
                border-color: #0078d4;
            }

            /* Checkboxes */
            QCheckBox {
                color: #ffffff;
                font-size: 13px;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
            }
            QCheckBox::indicator:unchecked {
                border: 2px solid #404040;
                background-color: #2d2d2d;
            }
            QCheckBox::indicator:checked {
                border: 2px solid #0078d4;
                background-color: #0078d4;
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iMTIiIHZpZXdCb3g9IjAgMCAxMiAxMiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEwIDNMNC41IDguNUwyIDYiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+Cjwvc3ZnPgo=);
            }

            /* Tabs */
            QTabWidget::pane {
                border: 1px solid #333333;
                background-color: #252525;
                border-radius: 8px;
            }
            QTabBar::tab {
                background-color: #2d2d2d;
                border: 1px solid #404040;
                border-bottom: none;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                color: #cccccc;
                font-family: 'Inter', sans-serif;
                font-weight: 500;
            }
            QTabBar::tab:selected {
                background-color: #0078d4;
                color: #ffffff;
                border-color: #106ebe;
            }
            QTabBar::tab:hover:!selected {
                background-color: #3a3a3a;
                color: #ffffff;
            }

            /* Scrollbars */
            QScrollBar:vertical {
                background-color: #2d2d2d;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #404040;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #505050;
            }
        """)

    def _create_title_header(self):
        """Create the main title header with SEMIOCITY WORLD"""
        header = QWidget()
        header.setFixedHeight(60)
        header.setStyleSheet("""
            QWidget {
                background-color: #1a1a1a;
                border-bottom: 1px solid #333333;
            }
        """)

        layout = QVBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Main title label
        title_label = QLabel("SEMIOCITY WORLD")
        title_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        title_label.setStyleSheet("""
            QLabel {
                font-family: 'Inter', sans-serif;
                font-size: 22px;
                font-weight: bold;
                color: #ffffff;
                background-color: transparent;
                padding: 15px 0px;
                margin: 0px;
            }
        """)

        layout.addWidget(title_label)
        return header

    def _setup_ui(self):
        """Setup the main UI layout with tabs for Pipeline and Chatbot"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout with tabs
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Main title header
        self.title_header = self._create_title_header()
        main_layout.addWidget(self.title_header)

        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Pipeline tab
        self.pipeline_tab = self._create_pipeline_tab()
        self.tab_widget.addTab(self.pipeline_tab, "2D→3D")

        # Chatbot tab
        self.chatbot_tab = self._create_chatbot_tab()
        self.tab_widget.addTab(self.chatbot_tab, "Semiocity Agent")

    def _create_pipeline_tab(self):
        """Create the main pipeline tab with improved layout"""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Left panel for controls
        left_panel = self._create_controls_panel()
        layout.addWidget(left_panel, stretch=1)

        # Right panel for results and 3D viewer
        right_panel = self._create_results_panel()
        layout.addWidget(right_panel, stretch=2)

        return tab

    def _create_controls_panel(self):
        """Create the left controls panel with improved layout"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Step 1: Image Generation
        step1 = QGroupBox("Step 1 — Image Generation")
        s1_layout = QVBoxLayout(step1)
        s1_layout.setSpacing(8)

        # Prompt input
        s1_layout.addWidget(self._muted_label("AI Prompt"))
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Describe the image you want…")
        self.prompt_input.setMaximumHeight(120)
        s1_layout.addWidget(self.prompt_input)

        # Generation steps control
        steps_layout = QHBoxLayout()
        steps_layout.addWidget(self._muted_label("Steps:"))
        self.steps_spinbox = QSpinBox()
        self.steps_spinbox.setRange(1, 50)
        self.steps_spinbox.setValue(10)
        self.steps_spinbox.setToolTip("More steps = better quality but slower generation")
        steps_layout.addWidget(self.steps_spinbox)
        steps_layout.addStretch()
        s1_layout.addLayout(steps_layout)

        # LoRa option
        self.lora_cb = QCheckBox("Use SemioCity LoRa (architectural style)")
        self.lora_cb.setChecked(True)
        self.lora_cb.setToolTip("Enable SemioCity LoRa for enhanced architectural generation")
        s1_layout.addWidget(self.lora_cb)

        # Image loading
        self.load_btn = QPushButton("📁 Load Image")
        self.load_btn.clicked.connect(self.load_image)
        s1_layout.addWidget(self.load_btn)

        self.gen_btn = QPushButton("🎨 Generate Image (FLUX)")
        self.gen_btn.setProperty("primary", True)
        self.gen_btn.clicked.connect(self.generate_image)
        s1_layout.addWidget(self.gen_btn)

        layout.addWidget(step1)

        # Step 2: Background Removal
        step2 = QGroupBox("Step 2 — Background Removal")
        s2_layout = QVBoxLayout(step2)
        s2_layout.setSpacing(8)

        self.clean_btn = QPushButton("🧹 Remove Background (Kontext)")
        self.clean_btn.setProperty("primary", True)
        self.clean_btn.clicked.connect(self.remove_background)
        self.clean_btn.setEnabled(False)
        s2_layout.addWidget(self.clean_btn)

        layout.addWidget(step2)

        # Step 3: 3D Generation
        step3 = QGroupBox("Step 3 — 3D Generation")
        s3_layout = QVBoxLayout(step3)
        s3_layout.setSpacing(8)

        # Texture option
        self.texture_cb = QCheckBox("Generate textured mesh")
        self.texture_cb.setChecked(True)
        s3_layout.addWidget(self.texture_cb)

        # Depth controls (created for compatibility but hidden; defaults applied)
        depth_group = QGroupBox("Depth Settings")
        depth_layout = QVBoxLayout(depth_group)

        depth_thresh_layout = QHBoxLayout()
        depth_thresh_layout.addWidget(QLabel("Depth Threshold:"))
        self.depth_threshold = QSpinBox()
        self.depth_threshold.setRange(50, 500)
        self.depth_threshold.setValue(50)  # default embedded
        depth_thresh_layout.addWidget(self.depth_threshold)
        depth_layout.addLayout(depth_thresh_layout)

        depth_max_layout = QHBoxLayout()
        depth_max_layout.addWidget(QLabel("Depth Max:"))
        self.depth_max = QSpinBox()
        self.depth_max.setRange(100, 500)
        self.depth_max.setValue(500)  # default embedded
        depth_max_layout.addWidget(self.depth_max)
        depth_layout.addLayout(depth_max_layout)

        depth_group.setVisible(False)  # hidden per spec
        s3_layout.addWidget(depth_group)

        # 3D generation button
        self.mesh_btn = QPushButton("🔷 Generate 3D Mesh")
        self.mesh_btn.setProperty("primary", True)
        self.mesh_btn.clicked.connect(self.generate_3d)
        self.mesh_btn.setEnabled(False)
        s3_layout.addWidget(self.mesh_btn)

        layout.addWidget(step3)

        # Full pipeline
        pipe_group = QGroupBox("Full Pipeline")
        pipe_layout = QVBoxLayout(pipe_group)

        self.full_btn = QPushButton("🚀 Run Complete 2D→3D Pipeline")
        self.full_btn.clicked.connect(self.run_full)
        self.full_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                font-weight: bold;
                font-size: 14px;
                padding: 12px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
        """)
        # Reset + Run row
        self.reset_btn = QPushButton("↺ Reset UI")
        self.reset_btn.setToolTip("Reset the UI state (files stay on disk)")
        self.reset_btn.clicked.connect(self.reset_ui)
        buttons_row = QHBoxLayout()
        buttons_row.addWidget(self.reset_btn)
        buttons_row.addStretch(1)
        buttons_row.addWidget(self.full_btn)
        pipe_layout.addLayout(buttons_row)

        layout.addWidget(pipe_group)

        # Progress and status
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status = QTextEdit()
        self.status.setReadOnly(True)
        self.status.setMaximumHeight(200)
        self.status.setPlaceholderText("Status messages will appear here...")
        layout.addWidget(self.status)

        return panel

    def _muted_label(self, text: str) -> QLabel:
        """Create a muted label with consistent styling"""
        label = QLabel(text)
        label.setStyleSheet("color: #cccccc; font-family: 'Inter', sans-serif; font-size: 12px; font-weight: 500;")
        return label

    def _create_results_panel(self):
        """Create the right results panel with Hi‑Fi layout (previews row → viewer → buttons bottom)"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Previews group — two square cards side-by-side
        preview_group = QGroupBox("Image Pipeline")
        preview_group_layout = QHBoxLayout(preview_group)
        preview_group_layout.setContentsMargins(12, 12, 12, 12)
        preview_group_layout.setSpacing(12)
        preview_group_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        # Left card: Step 1
        left_col = QVBoxLayout()
        left_col.setSpacing(6)
        left_col.addWidget(self._muted_label("Step 1 — Generated Image"))
        self.prev1 = SquareLabel(side=320)
        left_col.addWidget(self.prev1, alignment=Qt.AlignmentFlag.AlignTop)

        # Right card: Step 2
        right_col = QVBoxLayout()
        right_col.setSpacing(6)
        right_col.addWidget(self._muted_label("Step 2 — Background Removed"))
        self.prev2 = SquareLabel(side=320)
        right_col.addWidget(self.prev2, alignment=Qt.AlignmentFlag.AlignTop)

        preview_group_layout.addStretch(1)
        preview_group_layout.addLayout(left_col)
        preview_group_layout.addLayout(right_col)
        preview_group_layout.addStretch(1)

        layout.addWidget(preview_group)

        # 3D Viewer section (expands), with buttons pinned to bottom
        viewer_group = QGroupBox("3D Viewer")
        viewer_layout = QVBoxLayout(viewer_group)
        viewer_layout.setSpacing(8)

        # Create 3D viewer (web-based if available, fallback to label)
        if WEBENGINE_AVAILABLE and QWebEngineView:
            self.viewer_web = QWebEngineView()
            self.viewer_web.setMinimumHeight(320)
            self.viewer_web.setStyleSheet("border: 2px solid #404040; border-radius: 8px;")

            # --- NEW: enable local file + remote access + WebGL ---
            try:
                from PyQt6.QtWebEngineCore import QWebEngineSettings
                s = self.viewer_web.settings()
                s.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
                s.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
                s.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
                s.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
                s.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)
            except Exception as e:
                self.log(f"WebEngine settings warning: {e}")

            # Initialize viewer ready state and pending mesh URL
            self.viewer_ready = False
            self._pending_mesh_url = None

            # Connect loadFinished signal for proper timing
            self.viewer_web.loadFinished.connect(self._on_viewer_loaded)

            # Load the 3D viewer HTML (try proper Three.js version first)
            viewer_html_path = os.path.join(os.path.dirname(__file__), "3d_viewer_with_threejs.html")
            if not os.path.exists(viewer_html_path):
                viewer_html_path = os.path.join(os.path.dirname(__file__), "3d_viewer_basic_working.html")
                if not os.path.exists(viewer_html_path):
                    viewer_html_path = os.path.join(os.path.dirname(__file__), "3d_viewer_proper.html")
                    if not os.path.exists(viewer_html_path):
                        viewer_html_path = os.path.join(os.path.dirname(__file__), "3d_viewer_self_contained.html")
                        if not os.path.exists(viewer_html_path):
                            viewer_html_path = os.path.join(os.path.dirname(__file__), "3d_viewer_simple_working.html")
                            if not os.path.exists(viewer_html_path):
                                viewer_html_path = os.path.join(os.path.dirname(__file__), "3d_viewer_local_embedded.html")
                                if not os.path.exists(viewer_html_path):
                                    viewer_html_path = os.path.join(os.path.dirname(__file__), "3d_viewer.html")
                                    if not os.path.exists(viewer_html_path):
                                        viewer_html_path = os.path.join(os.path.dirname(__file__), "3d_viewer_offline.html")

            if os.path.exists(viewer_html_path):
                self.viewer_web.load(QUrl.fromLocalFile(os.path.abspath(viewer_html_path)))
                viewer_layout.addWidget(self.viewer_web)
            else:
                # Fallback to label if HTML not found
                self.viewer_label = QLabel("3D mesh will appear here after generation\n(3D viewer HTML not found)")
                self.viewer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.viewer_label.setMinimumHeight(320)
                self.viewer_label.setStyleSheet("border: 2px solid #404040; background-color: #2d2d2d; border-radius: 8px;")
                viewer_layout.addWidget(self.viewer_label)
                self.viewer_web = None
        else:
            # Fallback to mesh viewer or label
            try:
                from mesh_viewer_fallback import MeshViewerFallback
                self.viewer_fallback = MeshViewerFallback()
                self.viewer_fallback.setMinimumHeight(320)
                viewer_layout.addWidget(self.viewer_fallback)
                self.viewer_web = None
                self.viewer_label = None
            except ImportError:
                # Final fallback to label
                self.viewer_label = QLabel("3D mesh will appear here after generation\n(WebEngine not available - install PyQt6-WebEngine)")
                self.viewer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.viewer_label.setMinimumHeight(320)
                self.viewer_label.setStyleSheet("border: 2px solid #404040; background-color: #2d2d2d; border-radius: 8px;")
                viewer_layout.addWidget(self.viewer_label)
                self.viewer_web = None
                self.viewer_fallback = None

        # Status line
        self.viewer_status = QLabel("Ready • 0/3")
        self.viewer_status.setStyleSheet("color: #999999; font-size: 12px;")
        viewer_layout.addWidget(self.viewer_status)

        # Push action buttons to the bottom edge
        viewer_layout.addStretch(1)

        # 3D action buttons pinned to bottom
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)

        # Left side - Open Mesh
        self.open_mesh_btn = QPushButton("📁 Open Mesh File")
        self.open_mesh_btn.clicked.connect(self.open_mesh)
        self.open_mesh_btn.setEnabled(False)
        self.open_mesh_btn.setFixedSize(140, 40)
        button_layout.addWidget(self.open_mesh_btn)

        # Debug button for testing 3D viewer
        self.debug_3d_btn = QPushButton("🔧 Test 3D")
        self.debug_3d_btn.clicked.connect(self.debug_3d_viewer)
        self.debug_3d_btn.setFixedSize(80, 40)
        self.debug_3d_btn.setToolTip("Test 3D Viewer with latest mesh")
        button_layout.addWidget(self.debug_3d_btn)

        # Test Step 2 button
        self.test_step2_btn = QPushButton("🧪 Test Step 2")
        self.test_step2_btn.clicked.connect(self.test_step2_direct)
        self.test_step2_btn.setFixedSize(100, 40)
        self.test_step2_btn.setToolTip("Test Step 2 directly on current image")
        button_layout.addWidget(self.test_step2_btn)

        # Add stretch to push right buttons to the right
        button_layout.addStretch(1)

        # Right side - smaller square buttons
        self.download_btn = QPushButton("⬇️")
        self.download_btn.clicked.connect(self.download_mesh)
        self.download_btn.setEnabled(False)
        self.download_btn.setFixedSize(40, 40)
        self.download_btn.setToolTip("Download Mesh")
        button_layout.addWidget(self.download_btn)

        self.csv_btn = QPushButton("📊")
        self.csv_btn.clicked.connect(self.view_generation_log)
        self.csv_btn.setFixedSize(40, 40)
        self.csv_btn.setToolTip("Generations")
        button_layout.addWidget(self.csv_btn)

        viewer_layout.addLayout(button_layout)

        # Let viewer group fill remaining space so buttons hug panel's bottom
        layout.addWidget(viewer_group, stretch=1)

        return panel

    def _create_chatbot_tab(self):
        """Create the chatbot tab (placeholder for future chatbot integration)"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Chatbot placeholder
        chatbot_group = QGroupBox("AI Chatbot")
        chatbot_layout = QVBoxLayout(chatbot_group)
        chatbot_layout.setSpacing(8)

        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setPlaceholderText("Chat with the AI assistant will appear here...")
        chatbot_layout.addWidget(self.chat_display)

        # Chat input
        input_layout = QHBoxLayout()
        self.chat_input = QTextEdit()
        self.chat_input.setMaximumHeight(80)
        self.chat_input.setPlaceholderText("Type your message here...")
        input_layout.addWidget(self.chat_input)

        self.send_btn = QPushButton("Send")
        self.send_btn.setProperty("primary", True)
        self.send_btn.clicked.connect(self.send_chat_message)
        input_layout.addWidget(self.send_btn)

        chatbot_layout.addLayout(input_layout)
        layout.addWidget(chatbot_group)

        return tab

    def log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.status.append(f"[{ts}] {msg}")
        self.status.verticalScrollBar().setValue(self.status.verticalScrollBar().maximum())

    def send_chat_message(self):
        """Send a chat message to the AI assistant"""
        message = self.chat_input.toPlainText().strip()
        if not message:
            return
        self.chat_display.append(f"You: {message}")
        self.chat_input.clear()

        # Check if ChatController is available
        if not self.chat:
            self.chat_display.append("AI: ChatController not available. Please check the 00_ui_agent module.")
            return

        # Run the agent in a worker so the UI doesn't freeze
        def _chat_call(user_text, current_img, on_log=None):
            ctrl = getattr(self, "chat", None) or ChatController()
            return ctrl.ask(user_text, image_path=current_img)

        self.worker = WorkerStream(_chat_call, message, self.current_image)
        self.worker.log.connect(self.log)

        def _finish(_, res):
            # Show a compact assistant note
            note = res.get("notes", "")
            self.chat_display.append(f"AI: {note}")

            # If vision analysis arrived, pretty-print summary
            if "analysis" in res:
                a = res["analysis"]
                self.chat_display.append(
                    f"summary: {a.get('summary','')}\n"
                    f"denotation: {a.get('denotation','')}\n"
                    f"connotation: {a.get('connotation','')}\n"
                    f"myth: {a.get('myth','')}"
                )

            # Dispatch actions to the pipeline
            act = res.get("action", "noop")
            args = res.get("args", {})
            if act == "generate_image" and args.get("prompt"):
                self.prompt_input.setPlainText(args["prompt"])
                # Check if user wants full pipeline or just image generation
                user_msg = message.lower()
                if any(word in user_msg for word in ["pipeline", "full", "complete", "all steps", "step 1 to 3", "2d to 3d"]):
                    self.run_full()  # Run complete pipeline
                else:
                    self.generate_image()  # Just generate image
            elif act == "monochrome_clean" and self.current_image:
                self.remove_background()
            elif act == "generate_3d" and self.current_image:
                self.generate_3d()
            elif act == "analyze_image":
                pass  # already handled

        self.worker.finished.connect(_finish)
        self.worker.errored.connect(lambda e: self.chat_display.append(f"AI error: {e}"))
        self.worker.start()

    def open_mesh(self):
        """Open the generated mesh file"""
        if self.current_mesh_path and os.path.exists(self.current_mesh_path):
            os.startfile(self.current_mesh_path)
        else:
            QMessageBox.warning(self, "No Mesh", "No mesh file available to open.")

    def download_mesh(self):
        """Download the generated mesh file"""
        if self.current_mesh_path and os.path.exists(self.current_mesh_path):
            # Open file dialog to save the mesh
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Mesh File", 
                os.path.basename(self.current_mesh_path),
                "3D Files (*.glb *.obj *.ply);;All Files (*)"
            )
            if file_path:
                import shutil
                shutil.copy2(self.current_mesh_path, file_path)
                self.log(f"Mesh saved to: {file_path}")
        else:
            QMessageBox.warning(self, "No Mesh", "No mesh file available to download.")

    def test_load_mesh(self):
        """Test loading an existing mesh file"""
        # Use the most recent mesh file
        mesh_dir = self.out_root / "meshes"
        if mesh_dir.exists():
            mesh_files = list(mesh_dir.glob("*.glb"))
            if mesh_files:
                # Get the most recent file
                latest_mesh = max(mesh_files, key=os.path.getmtime)
                self.current_mesh_path = str(latest_mesh)
                self.log(f"Testing with mesh: {os.path.basename(latest_mesh)}")

                # Update the 3D viewer
                files = [f"Test: {os.path.basename(latest_mesh)}"]
                self._update_3d_viewer(files)
                self.open_mesh_btn.setEnabled(True)
                self.download_btn.setEnabled(True)
            else:
                self.log("No mesh files found in output/meshes/")
        else:
            self.log("No meshes directory found")

    def view_generation_log(self):
        """View the generation log CSV"""
        try:
            from generation_tracker import get_recent_generations
            recent = get_recent_generations(20)  # Get last 20 generations

            if not recent:
                QMessageBox.information(self, "Generation Log", "No generations found in log.")
                return

            # Create a simple text display of recent generations
            log_text = "Recent Generations:\n" + "="*50 + "\n\n"

            for i, gen in enumerate(recent[-10:], 1):  # Show last 10
                log_text += f"{i}. Session: {gen['session_id']}\n"
                log_text += f"   Prompt: {gen['prompt'][:60]}...\n"
                log_text += f"   Status: {gen['status']}\n"
                log_text += f"   Duration: {gen['pipeline_duration']}s\n"
                log_text += f"   Step 1: {os.path.basename(gen['step1_image']) if gen['step1_image'] else 'N/A'}\n"
                log_text += f"   Step 2: {os.path.basename(gen['step2_clean_image']) if gen['step2_clean_image'] else 'N/A'}\n"
                log_text += f"   Step 3: {os.path.basename(gen['step3_mesh_final']) if gen['step3_mesh_final'] else 'N/A'}\n"
                log_text += f"   Time: {gen['timestamp']}\n"
                log_text += "\n"

            # Show in a dialog
            dialog = QMessageBox(self)
            dialog.setWindowTitle("Generation Log")
            dialog.setText(log_text)
            dialog.setDetailedText(f"Full CSV available at: {tracker.csv_path}")
            dialog.exec()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load generation log: {e}")

    def _update_3d_viewer(self, files):
        """Update the 3D viewer with mesh files"""
        self.log(f"Updating 3D viewer with files: {files}")
        self.log(f"Current mesh path: {self.current_mesh_path}")
        self.log(f"Viewer web available: {self.viewer_web is not None}")
        self.log(f"Viewer fallback available: {hasattr(self, 'viewer_fallback') and self.viewer_fallback is not None}")

        if self.viewer_web and files and self.current_mesh_path:
            # Convert Windows path to file URL for web viewer
            mesh_url = QUrl.fromLocalFile(os.path.abspath(self.current_mesh_path)).toString()
            self.log(f"Mesh URL: {mesh_url}")

            if not self.viewer_ready:
                self._pending_mesh_url = mesh_url
                self.log("Viewer not ready yet; will load when ready.")
            else:
                self._inject_load_model(mesh_url)
                self.log(f"Loading 3D model: {os.path.basename(self.current_mesh_path)}")
        elif hasattr(self, 'viewer_fallback') and self.viewer_fallback and files and self.current_mesh_path:
            # Use fallback mesh viewer
            self.viewer_fallback.set_mesh_path(self.current_mesh_path)
            self.log(f"3D model ready in fallback viewer: {os.path.basename(self.current_mesh_path)}")
        elif hasattr(self, 'viewer_label') and self.viewer_label:
            # Fallback to text display
            if files:
                self.viewer_label.setText(f"3D Mesh Generated!\n" + "\n".join(files))
                self.log(f"Updated viewer label with: {files}")
            else:
                self.viewer_label.setText("No 3D files generated")
                self.log("Updated viewer label: No 3D files generated")
        else:
            self.log("No suitable 3D viewer available")

    def set_busy(self, busy: bool):
        self.busy = busy
        for b in [self.gen_btn, self.clean_btn, self.mesh_btn, self.full_btn, self.load_btn]:
            b.setEnabled(not busy)
        self.progress.setVisible(busy)

    def _set_preview(self, label: QLabel, path: str):
        """Set preview image with proper error handling and optimal scaling"""
        if not path or not os.path.exists(path):
            self.log(f"⚠️ Preview image not found: {path}")
            label.setText("Image not found")
            return

        pm = QPixmap(path)
        if pm.isNull():
            self.log(f"⚠️ Failed to load preview image: {path}")
            label.setText("Failed to load image")
            return

        # Get original image dimensions
        orig_width = pm.width()
        orig_height = pm.height()
        self.log(f"📏 Original image: {orig_width}x{orig_height}")

        # Get label dimensions (use the square size from SquareLabel)
        label_width = label.width()
        label_height = label.height()

        # If label dimensions are invalid, use the minimum size
        if label_width <= 0 or label_height <= 0:
            label_width = 320  # Default square size
            label_height = 320

        self.log(f"📏 Label dimensions: {label_width}x{label_height}")

        # For square images (1024x1024) in square containers (320x320), optimize scaling
        if orig_width == orig_height and label_width == label_height:
            # Both are square - perfect match scenario
            scale = label_width / orig_width
            new_size = int(orig_width * scale)

            self.log(f"📏 Square-to-square: {orig_width}x{orig_height} → {new_size}x{new_size} (scale: {scale:.3f})")

            # For 1024x1024 → 320x320, scale = 0.3125 (perfect fit)
            if 0.1 < scale < 10.0:
                scaled_pm = pm.scaled(new_size, new_size,
                                     Qt.AspectRatioMode.IgnoreAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)
                label.setPixmap(scaled_pm)
            else:
                self.log(f"⚠️ Extreme scale factor {scale:.3f}, using original image")
                label.setPixmap(pm)
        else:
            # Non-square images - use aspect ratio preserving scaling
            scale_x = label_width / orig_width
            scale_y = label_height / orig_height
            scale = min(scale_x, scale_y)  # Use the smaller scale to fit completely

            # Calculate new dimensions
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)

            self.log(f"📏 Scaled image: {new_width}x{new_height} (scale: {scale:.3f})")

            # Only scale if the scale factor is reasonable (not too small or too large)
            if 0.1 < scale < 10.0:
                # Use high-quality scaling
                scaled_pm = pm.scaled(new_width, new_height,
                                     Qt.AspectRatioMode.IgnoreAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)
                label.setPixmap(scaled_pm)
            else:
                # If scale is extreme, just use original
                self.log(f"⚠️ Extreme scale factor {scale:.3f}, using original image")
                label.setPixmap(pm)

    # --- Reset UI ---
    def reset_ui(self):
        """Reset UI state: clear previews, viewer, paths, and disable actions.
        Keeps files on disk untouched.
        """
        # Stop any busy state
        self.set_busy(False)

        # Clear in-memory pointers
        self.current_image = None
        self.current_mesh_path = None

        # Reset previews
        try:
            self.prev1.clear()
            self.prev1.setText("No image loaded")
        except Exception:
            pass
        try:
            self.prev2.clear()
            self.prev2.setText("No image processed")
        except Exception:
            pass

        # Reset viewer
        self.viewer_status.setText("Ready • 0/3")
        if hasattr(self, "viewer_web") and self.viewer_web:
            # Reload viewer or blank it
            viewer_html_path = os.path.join(os.path.dirname(__file__), "3d_viewer_offline.html")
            if not os.path.exists(viewer_html_path):
                viewer_html_path = os.path.join(os.path.dirname(__file__), "3d_viewer.html")
            if os.path.exists(viewer_html_path):
                self.viewer_web.load(QUrl.fromLocalFile(os.path.abspath(viewer_html_path)))
            else:
                self.viewer_web.setUrl(QUrl("about:blank"))
            self.viewer_ready = False
            self._pending_mesh_url = None
        elif hasattr(self, "viewer_fallback") and self.viewer_fallback:
            try:
                if hasattr(self.viewer_fallback, "clear"):
                    self.viewer_fallback.clear()
            except Exception:
                pass
        elif hasattr(self, "viewer_label") and self.viewer_label:
            self.viewer_label.setText("3D mesh will appear here after generation")

        # Reset controls
        self.prompt_input.clear()
        self.texture_cb.setChecked(True)

        # Depth defaults (controls hidden but keep values consistent)
        if hasattr(self, "depth_threshold"):
            self.depth_threshold.setValue(50)
        if hasattr(self, "depth_max"):
            self.depth_max.setValue(500)

        # Disable actions back to initial state
        self.clean_btn.setEnabled(False)
        self.mesh_btn.setEnabled(False)
        self.open_mesh_btn.setEnabled(False)
        self.download_btn.setEnabled(False)

        # Progress + status area
        self.progress.reset()
        self.progress.setVisible(False)
        self.status.clear()

        self.log("UI reset: previews cleared, viewer reset, actions disabled (files remain on disk).")

    # --- callbacks ---
    def load_image(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)")
        if not p: return
        self.current_image = p
        self._set_preview(self.prev1, p)
        self.clean_btn.setEnabled(True)
        self.mesh_btn.setEnabled(True)
        self.log(f"Loaded {os.path.basename(p)}")

    def generate_image(self):
        if self.busy: return
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "No prompt", "Enter a prompt or load an image")
            return
        steps = self.steps_spinbox.value()
        self.set_busy(True); self.log(f"Generating image with {steps} steps...")
        # Pass use_lora explicitly; avoid shifting positional args into the seed slot
        self.worker = WorkerStream(self.orch.generate_image_stream, prompt, steps, use_lora=self.lora_cb.isChecked())
        self.worker.log.connect(self.log)
        def _finish(_, path):
            self.set_busy(False)
            self.current_image = str(path)
            self._set_preview(self.prev1, self.current_image)
            self.clean_btn.setEnabled(True); self.mesh_btn.setEnabled(True)
            self.log(f"Image ready: {path}")
        self.worker.finished.connect(_finish)
        self.worker.errored.connect(lambda e: (self.set_busy(False), QMessageBox.critical(self, "Error", e), self.log(e)))
        self.worker.start()

    def remove_background(self):
        if self.busy or not self.current_image: return
        self.set_busy(True); self.log("Cleaning image (Kontext)...")
        self.worker = WorkerStream(self.orch.monochrome_clean_stream, Path(self.current_image))
        self.worker.log.connect(self.log)
        def _finish(_, path):
            self.set_busy(False)
            self.current_image = str(path)
            self._set_preview(self.prev2, self.current_image)
            self.mesh_btn.setEnabled(True)
            self.log(f"Cleaned image: {path}")
        self.worker.finished.connect(_finish)
        self.worker.errored.connect(lambda e: (self.set_busy(False), QMessageBox.critical(self, "Error", e), self.log(e)))
        self.worker.start()

    def generate_3d(self):
        if self.busy or not self.current_image: return
        self.set_busy(True); self.log("Generating 3D...")

        # Pass depth parameters to the orchestrator (values exist, even if hidden)
        depth_params = {
            'depth_threshold': self.depth_threshold.value(),
            'depth_max': self.depth_max.value()
        }
        # (currently not consumed downstream; preserved for compatibility)

        self.worker = WorkerStream(self.orch.generate_3d_stream, Path(self.current_image), self.texture_cb.isChecked())
        self.worker.log.connect(self.log)
        def _finish(_, res):
            self.set_busy(False)
            self.log("3D step complete")
            self.log(f"3D result: {res}")
            files = []
            if isinstance(res, dict):
                if res.get("mesh_shape_glb"): 
                    files.append(f"Shape: {os.path.basename(res['mesh_shape_glb'])}")
                    self.current_mesh_path = res['mesh_shape_glb']
                    self.log(f"Set mesh path to: {self.current_mesh_path}")
                if res.get("mesh_textured_glb"): 
                    files.append(f"Textured: {os.path.basename(res['mesh_textured_glb'])}")
                    self.current_mesh_path = res['mesh_textured_glb']
                    self.log(f"Set mesh path to: {self.current_mesh_path}")

            # Update 3D viewer
            if files:
                self.log(f"Updating 3D viewer with {len(files)} files")
                self._update_3d_viewer(files)
                self.open_mesh_btn.setEnabled(True)
                self.download_btn.setEnabled(True)
            else:
                self.log("No mesh files found in result")
                self._update_3d_viewer([])
        self.worker.finished.connect(_finish)
        self.worker.errored.connect(lambda e: (self.set_busy(False), QMessageBox.critical(self, "Error", e), self.log(e)))
        self.worker.start()

    def run_full(self):
        if self.busy: return
        prompt = self.prompt_input.toPlainText().strip() or None
        img = None if prompt and not self.current_image else self.current_image
        self.set_busy(True); self.log("Running full pipeline...")
        
        # Disable viewer accel before starting; we'll re-enable on finish/error
        self._park_viewer()
        self._set_viewer_accel(False)
        
        steps = self.steps_spinbox.value()
        self.pworker = PipelineWorker(self.orch, prompt, img, self.texture_cb.isChecked(), steps, self.lora_cb.isChecked())
        self.pworker.log.connect(self.log)
        self.pworker.progress.connect(lambda p: self.progress.setValue(p))
        self.pworker.intermediate.connect(self._on_intermediate)
        self.pworker.finished.connect(lambda res: (self._set_viewer_accel(True), self._full_finish(res)))
        self.pworker.errored.connect(lambda e: (self._set_viewer_accel(True), self._full_error(e)))
        self.pworker.start()

    def _full_finish(self, res):
        """Handle full pipeline completion"""
        self.set_busy(False)
        self.log("Full pipeline done")

        # Update image previews if we have the intermediate results
        if res.get("image_generated"):
            self._set_preview(self.prev1, res["image_generated"])
            self.log(f"Updated Step 1 preview: {res['image_generated']}")

        if res.get("image_cleaned"):
            self._set_preview(self.prev2, res["image_cleaned"])
            self.log(f"Updated Step 2 preview: {res['image_cleaned']}")

        # Restore 3D viewer and load mesh
        self._restore_viewer()
        
        # Update 3D viewer with mesh
        files = []
        if res.get("mesh_shape_glb"): 
            files.append(f"Shape: {os.path.basename(res['mesh_shape_glb'])}")
            self.current_mesh_path = res['mesh_shape_glb']
        if res.get("mesh_textured_glb"): 
            files.append(f"Textured: {os.path.basename(res['mesh_textured_glb'])}")
            self.current_mesh_path = res['mesh_textured_glb']

        if files:
            # Wait a moment for viewer to load, then update with mesh
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(2000, lambda: self._update_3d_viewer(files))
            self.open_mesh_btn.setEnabled(True)
            self.download_btn.setEnabled(True)
        else:
            self._update_3d_viewer([])

    def _full_error(self, e):
        """Handle full pipeline error"""
        self.set_busy(False)
        QMessageBox.critical(self, "Error", e)
        self.log(e)

    def _on_intermediate(self, stage, payload):
        """Handle intermediate step updates for immediate UI refresh"""
        if stage == "step1" and payload.get("image_generated"):
            self._set_preview(self.prev1, payload["image_generated"])
            self.log(f"Step 1 preview updated: {payload['image_generated']}")
        elif stage == "step2" and payload.get("image_cleaned"):
            self._set_preview(self.prev2, payload["image_cleaned"])
            self.log(f"Step 2 preview updated: {payload['image_cleaned']}")

    def _on_viewer_loaded(self, ok):
        """Handle viewer page load completion"""
        self.viewer_ready = ok
        if ok and getattr(self, "_pending_mesh_url", None):
            self._inject_load_model(self._pending_mesh_url)
            self._pending_mesh_url = None
            self.log("3D viewer loaded and pending model injected")

    def _inject_load_model(self, mesh_url: str):
        """Inject 3D model loading JavaScript into the viewer"""
        # Convert Windows path to proper file URL
        if mesh_url.startswith('file:///'):
            # Already a file URL
            clean_url = mesh_url
        else:
            # Convert to file URL
            clean_url = mesh_url.replace('\\', '/')
            if not clean_url.startswith('file://'):
                clean_url = f"file:///{clean_url}"

        # Escape quotes in the URL to prevent JavaScript syntax errors
        clean_url = clean_url.replace("'", "\\'").replace('"', '\\"')

        # Convert Python boolean to JavaScript boolean
        viewer_ready_js = "true" if self.viewer_ready else "false"

        js_code = f"""
            console.log('Python attempting to load model: {clean_url}');
            console.log('Viewer ready state:', {viewer_ready_js});
            console.log('loadModel function available:', typeof window.loadModel);
            console.log('Current model path:', window.currentModelPath);

            // Force update status
            if (document.getElementById('status')) {{
                document.getElementById('status').textContent = 'Loading from Python...';
            }}

            if (window.loadModel && typeof window.loadModel === 'function') {{
                try {{
                    console.log('Calling loadModel with:', '{clean_url}');
                    window.loadModel('{clean_url}');
                    console.log('Model load command sent successfully');
                }} catch (error) {{
                    console.error('Error calling loadModel:', error);
                    if (document.getElementById('status')) {{
                        document.getElementById('status').textContent = 'Error: ' + error.message;
                    }}
                }}
            }} else {{
                console.error('loadModel function not available');
                if (document.getElementById('status')) {{
                    document.getElementById('status').textContent = 'loadModel function not available';
                }}
                if (typeof initViewer === 'function') {{
                    console.log('Attempting to initialize viewer...');
                    initViewer();
                    setTimeout(function() {{
                        if (window.loadModel) {{
                            console.log('Retrying model load after init...');
                            window.loadModel('{clean_url}');
                        }}
                    }}, 1000);
                }}
            }}
        """
        try:
            self.viewer_web.page().runJavaScript(js_code)
            self.log(f"3D model injected into viewer: {os.path.basename(self.current_mesh_path) if self.current_mesh_path else 'Unknown'}")
        except Exception as e:
            self.log(f"JavaScript execution error: {e}")

    def debug_3d_viewer(self):
        """Debug function to test 3D viewer with existing mesh files"""
        # Find the most recent mesh file
        mesh_dir = self.out_root / "meshes"
        if mesh_dir.exists():
            mesh_files = list(mesh_dir.glob("*.glb"))
            if mesh_files:
                # Get the most recent file
                latest_mesh = max(mesh_files, key=os.path.getmtime)
                self.current_mesh_path = str(latest_mesh)
                self.log(f"🔧 Testing 3D viewer with: {os.path.basename(latest_mesh)}")

                # Test the viewer
                if self.viewer_web:
                    mesh_url = QUrl.fromLocalFile(os.path.abspath(self.current_mesh_path)).toString()
                    self.log(f"🔧 Mesh URL: {mesh_url}")
                    self.log(f"🔧 Viewer ready: {self.viewer_ready}")

                    # Force load the model
                    self._inject_load_model(mesh_url)

                    # Enable buttons
                    self.open_mesh_btn.setEnabled(True)
                    self.download_btn.setEnabled(True)
                else:
                    self.log("🔧 No 3D viewer available")
            else:
                self.log("🔧 No mesh files found in output/meshes/")
        else:
            self.log("🔧 No meshes directory found")

    def test_step2_direct(self):
        """Test Step 2 directly without full pipeline"""
        if not self.current_image:
            self.log("❌ No current image to test Step 2")
            return
        
        self.log("🧪 Testing Step 2 directly...")
        try:
            # Park viewer and disable GPU acceleration to free VRAM
            self._park_viewer()
            self._set_viewer_accel(False)
            
            mono = self.orch.monochrome_clean_stream(Path(self.current_image), on_log=self.log)
            self.log(f"✅ Step 2 test successful: {mono}")
            self._set_preview(self.prev2, str(mono))
            
            # Re-enable GPU acceleration
            self._set_viewer_accel(True)
        except Exception as e:
            self.log(f"❌ Step 2 test failed: {e}")
            # Re-enable GPU acceleration even on error
            self._set_viewer_accel(True)

    def _set_viewer_accel(self, enable: bool):
        """Toggle GPU acceleration for WebEngine viewer only"""
        if not getattr(self, "viewer_web", None):
            return
        try:
            from PyQt6.QtWebEngineCore import QWebEngineSettings
            s = self.viewer_web.settings()
            s.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, enable)
            s.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, enable)
            self.log(f"🔧 WebEngine GPU acceleration: {'enabled' if enable else 'disabled'}")
        except Exception as e:
            self.log(f"⚠️ Could not toggle WebEngine GPU: {e}")

    def _park_viewer(self):
        """Drop WebGL content so GL textures are released"""
        if getattr(self, "viewer_web", None):
            # Don't reset viewer_ready - just clear the content
            self.viewer_web.setHtml(
                "<html><body style='background:#1a1a1a;color:#aaa;"
                "font-family:sans-serif;padding:1rem'>Viewer paused during processing…</body></html>"
            )
            self.log("🔧 Viewer parked to free VRAM")

    def _restore_viewer(self):
        """Restore the 3D viewer after processing"""
        if getattr(self, "viewer_web", None):
            # Use the improved 3D viewer with better lighting and auto-rotation
            viewer_html_path = os.path.join(os.path.dirname(__file__), "3d_viewer_improved.html")
            if not os.path.exists(viewer_html_path):
                # Fallback to simple working viewer
                viewer_html_path = os.path.join(os.path.dirname(__file__), "3d_viewer_simple_working.html")
                if not os.path.exists(viewer_html_path):
                    viewer_html_path = os.path.join(os.path.dirname(__file__), "3d_viewer_self_contained_fixed.html")
            
            if os.path.exists(viewer_html_path):
                self.viewer_web.load(QUrl.fromLocalFile(os.path.abspath(viewer_html_path)))
                self.log("🔧 3D viewer restored with improved lighting")
            else:
                self.log("⚠️ Could not find 3D viewer HTML to restore")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for better dark theme support
    w = SemioAgentUI()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()