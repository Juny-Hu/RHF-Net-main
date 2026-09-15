"""Desktop GUI for five-fold PRH-Net four-class tea classification.

Inputs are a paired RGB image/array and a 20-channel PCA-HSI NumPy array.
The application averages class-wise softmax probabilities from the five final
fold checkpoints and reports the predicted picking-period class.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QFontDatabase, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


PROJECT_ROOT = Path(__file__).resolve().parent
PRH_ROOT = PROJECT_ROOT
DEFAULT_CHECKPOINT_ROOT = PROJECT_ROOT / "weights" / "fusion"
DEFAULT_RGB_CHECKPOINT_ROOT = PROJECT_ROOT / "weights" / "rgb"
DEFAULT_HSI_CHECKPOINT_ROOT = PROJECT_ROOT / "weights" / "hsi"
DEFAULT_PCA_TRANSFORM = PROJECT_ROOT / "artifacts" / "pca20_training_transform.npz"
PREVIEW_LONG_EDGE = 300
CLASS_INFO = {
    "P1": "Before Qingming",
    "P2": "0-15 days after Qingming",
    "P3": "16-30 days after Qingming",
    "P4": "31-45 days after Qingming",
}
CLASS_NAMES = list(CLASS_INFO)
RGB_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
RGB_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]


class ResNet18FeatureMap(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        model = resnet18(weights=None)
        if channels != 3:
            model.conv1 = nn.Conv2d(
                channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layers = nn.Sequential(model.layer1, model.layer2, model.layer3, model.layer4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(self.stem(x)).mean((2, 3))


class PRHNet(nn.Module):
    """Final ordinary RHF-Net: RGB-only, HSI-only, or feature concatenation."""
    def __init__(self, mode: str, hsi_channels: int = 20, num_classes: int = 4):
        super().__init__()
        if mode not in {"rgb", "hsi", "fusion"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = mode
        if mode in {"rgb", "fusion"}:
            self.rgb_backbone = ResNet18FeatureMap(3)
        if mode in {"hsi", "fusion"}:
            self.hsi_backbone = ResNet18FeatureMap(hsi_channels)
        feature_dim = 1024 if mode == "fusion" else 512
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feature_dim), nn.Dropout(0.3), nn.Linear(feature_dim, num_classes)
        )

    def forward(self, rgb: torch.Tensor, hsi: torch.Tensor) -> torch.Tensor:
        if self.mode == "rgb":
            feature = self.rgb_backbone(rgb)
        elif self.mode == "hsi":
            feature = self.hsi_backbone(hsi)
        else:
            feature = torch.cat((self.rgb_backbone(rgb), self.hsi_backbone(hsi)), dim=1)
        return self.classifier(feature)


def load_array(path: Path, channels: int) -> torch.Tensor:
    raw = np.load(path, allow_pickle=False)
    array = np.nan_to_num(raw.astype(np.float32))
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D array, but received shape {array.shape}")
    if array.shape[0] != channels and array.shape[-1] == channels:
        array = np.moveaxis(array, -1, 0)
    if array.shape[0] != channels:
        raise ValueError(f"Expected {channels} channels, but received shape {array.shape}")
    return torch.from_numpy(np.ascontiguousarray(array))


def load_rgb(path: Path) -> tuple[torch.Tensor, np.ndarray]:
    if path.suffix.lower() == ".npy":
        raw = np.load(path, allow_pickle=False)
        array = np.nan_to_num(raw.astype(np.float32))
        if array.ndim != 3:
            raise ValueError(f"Expected a 3D RGB array, but received shape {array.shape}")
        if array.shape[0] != 3 and array.shape[-1] == 3:
            array = np.moveaxis(array, -1, 0)
        if array.shape[0] != 3:
            raise ValueError(f"Expected 3 RGB channels, but received shape {array.shape}")
        if raw.dtype == np.uint8 or float(array.max()) > 1.5:
            array = array / 255.0
        tensor = torch.from_numpy(np.ascontiguousarray(array)).clamp(0, 1)
    else:
        image = Image.open(path).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    preview = tensor.permute(1, 2, 0).numpy()
    resized = F.interpolate(
        tensor.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False,
        antialias=True,
    )[0]
    normalized = (resized - RGB_MEAN) / RGB_STD
    return normalized, preview


def load_hsi(
    path: Path, transform_path: Path = DEFAULT_PCA_TRANSFORM
) -> tuple[torch.Tensor, np.ndarray]:
    raw = np.load(path, allow_pickle=False)
    array = np.nan_to_num(raw.astype(np.float32))
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D HSI array, but received shape {array.shape}")
    if array.shape[-1] in (20, 75):
        array_hwc = array
    elif array.shape[0] in (20, 75):
        array_hwc = np.moveaxis(array, 0, -1)
    else:
        raise ValueError(f"Expected 20 or 75 HSI channels, but received shape {array.shape}")

    source_channels = array_hwc.shape[-1]
    if source_channels == 75:
        if not transform_path.exists():
            raise FileNotFoundError(f"Missing recovered PCA20 transform: {transform_path}")
        transform = np.load(transform_path, allow_pickle=False)
        matrix = transform["matrix"].astype(np.float32)
        bias = transform["bias"].astype(np.float32)
        if matrix.shape != (75, 20) or bias.shape != (20,):
            raise ValueError(f"Invalid PCA20 transform shapes: {matrix.shape}, {bias.shape}")
        pca_hwc = np.tensordot(array_hwc, matrix, axes=([-1], [0])) + bias
        preview_source = array_hwc[..., [12, 37, 62]]
    else:
        pca_hwc = array_hwc
        preview_source = array_hwc[..., :3]

    tensor = torch.from_numpy(np.ascontiguousarray(np.moveaxis(pca_hwc, -1, 0)))
    preview = preview_source
    display = np.zeros_like(preview, dtype=np.float32)
    for channel in range(3):
        low, high = np.percentile(preview[..., channel], [1, 99])
        display[..., channel] = np.clip(
            (preview[..., channel] - low) / max(float(high - low), 1e-12), 0, 1
        )
    resized = F.interpolate(
        tensor.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False,
        antialias=True,
    )[0]
    return resized, display


def load_state_dict_file(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_fold_checkpoint(model: PRHNet, checkpoint: Path, device: torch.device) -> None:
    state = load_state_dict_file(checkpoint, device)
    model.load_state_dict(state, strict=True)


class FiveFoldEnsemble:
    def __init__(self, checkpoint_root: Path, device_name: str = "auto"):
        self.checkpoint_root = checkpoint_root
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.models: list[PRHNet] = []

    def load(self, progress=None) -> str:
        checkpoints = [self.checkpoint_root / f"fold_{fold}" / "best.pt" for fold in range(1, 6)]
        missing = [str(path) for path in checkpoints if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing model checkpoints:\n" + "\n".join(missing))
        self.models.clear()
        for index, checkpoint in enumerate(checkpoints, start=1):
            model = PRHNet("fusion").to(self.device)
            load_fold_checkpoint(model, checkpoint, self.device)
            model.eval()
            self.models.append(model)
            if progress:
                progress(index, len(checkpoints))
        return f"{self.device.type.upper()}"

    def predict(self, rgb_path: Path, hsi_path: Path) -> dict:
        if len(self.models) != 5:
            raise RuntimeError("The five-fold ensemble has not finished loading")
        start = time.perf_counter()
        rgb, _ = load_rgb(rgb_path)
        hsi, _ = load_hsi(hsi_path)
        rgb = rgb.unsqueeze(0).to(self.device)
        hsi = hsi.unsqueeze(0).to(self.device)
        fold_probabilities = []
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        with torch.inference_mode():
            for model in self.models:
                fold_probabilities.append(torch.softmax(model(rgb, hsi), dim=1)[0].cpu())
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        probabilities = torch.stack(fold_probabilities)
        mean = probabilities.mean(dim=0)
        std = probabilities.std(dim=0, unbiased=True)
        prediction = int(mean.argmax().item())
        fold_rows = []
        for index, values in enumerate(probabilities, start=1):
            fold_prediction = int(values.argmax().item())
            fold_rows.append({
                "fold": index,
                "class": CLASS_NAMES[fold_prediction],
                "confidence": float(values[fold_prediction].item()),
                "probabilities": values.tolist(),
            })
        return {
            "rgb_file": str(rgb_path),
            "hsi_file": str(hsi_path),
            "predicted_class": CLASS_NAMES[prediction],
            "description": CLASS_INFO[CLASS_NAMES[prediction]],
            "confidence": float(mean[prediction].item()),
            "mean_probabilities": mean.tolist(),
            "std_probabilities": std.tolist(),
            "fold_results": fold_rows,
            "elapsed_seconds": time.perf_counter() - start,
            "device": str(self.device),
            "mode": "fusion",
        }


class SingleModeEnsemble:
    def __init__(self, mode: str, checkpoint_root: Path, device_name: str = "auto"):
        self.mode = mode
        self.checkpoint_root = checkpoint_root
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.models: list[PRHNet] = []

    def load(self) -> None:
        checkpoints = [self.checkpoint_root / f"fold_{fold}" / "best.pt" for fold in range(1, 6)]
        missing = [str(path) for path in checkpoints if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing model checkpoints:\n" + "\n".join(missing))
        self.models.clear()
        for checkpoint in checkpoints:
            model = PRHNet(self.mode).to(self.device)
            model.load_state_dict(load_state_dict_file(checkpoint, self.device), strict=True)
            model.eval()
            self.models.append(model)

    def predict(self, path: Path) -> dict:
        if len(self.models) != 5:
            raise RuntimeError(f"The five-fold {self.mode.upper()} ensemble is not loaded")
        start = time.perf_counter()
        inputs = load_rgb(path)[0] if self.mode == "rgb" else load_hsi(path)[0]
        inputs = inputs.unsqueeze(0).to(self.device)
        fold_probabilities = []
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        with torch.inference_mode():
            for model in self.models:
                if self.mode == "rgb":
                    logits = model(inputs, torch.zeros((1, 20, 224, 224), device=self.device))
                else:
                    logits = model(torch.zeros((1, 3, 224, 224), device=self.device), inputs)
                fold_probabilities.append(torch.softmax(logits, dim=1)[0].cpu())
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        probabilities = torch.stack(fold_probabilities)
        mean = probabilities.mean(dim=0)
        std = probabilities.std(dim=0, unbiased=True)
        prediction = int(mean.argmax().item())
        return {
            "rgb_file": str(path) if self.mode == "rgb" else None,
            "hsi_file": str(path) if self.mode == "hsi" else None,
            "predicted_class": CLASS_NAMES[prediction],
            "description": CLASS_INFO[CLASS_NAMES[prediction]],
            "confidence": float(mean[prediction].item()),
            "mean_probabilities": mean.tolist(),
            "std_probabilities": std.tolist(),
            "elapsed_seconds": time.perf_counter() - start,
            "device": str(self.device),
            "mode": self.mode,
        }


class ProbabilityRow(tk.Frame):
    def __init__(self, master, code: str, description: str, color: str, font_family: str):
        super().__init__(master, bg="#FFFFFF")
        self.color = color
        self.code_label = tk.Label(
            self, text=code, bg="#FFFFFF", fg="#1F2933",
            font=(font_family, 14, "bold"), width=3, anchor="w",
        )
        self.code_label.grid(row=0, column=0, rowspan=2, sticky="nw", padx=(0, 8))
        self.description_label = tk.Label(
            self, text=description, bg="#FFFFFF", fg="#5B6573",
            font=(font_family, 12, "bold"), anchor="w",
        )
        self.description_label.grid(row=0, column=1, sticky="ew")
        self.value_label = tk.Label(
            self, text="0.0000", bg="#FFFFFF", fg="#1F2933",
            font=(font_family, 12, "bold"), width=7, anchor="e",
        )
        self.value_label.grid(row=0, column=2, sticky="e")
        self.canvas = tk.Canvas(self, height=8, bg="#E8EDF2", highlightthickness=0)
        self.canvas.grid(row=1, column=1, columnspan=2, sticky="ew", pady=(5, 0))
        self.columnconfigure(1, weight=1)
        self.value = 0.0
        self.canvas.bind("<Configure>", lambda _event: self._draw())

    def set_value(self, value: float, selected: bool = False) -> None:
        self.value = min(max(value, 0.0), 1.0)
        self.value_label.configure(text=f"{self.value:.4f}")
        self.code_label.configure(fg=self.color if selected else "#1F2933")
        self._draw()

    def _draw(self) -> None:
        self.canvas.delete("all")
        width = max(self.canvas.winfo_width(), 1)
        self.canvas.create_rectangle(0, 0, width * self.value, 8, fill=self.color, outline="")


class PRHApplication(tk.Tk):
    COLORS = {
        "background": "#F3F5F7",
        "surface": "#FFFFFF",
        "ink": "#17212B",
        "muted": "#65717E",
        "border": "#D9E0E6",
        "navy": "#243B53",
        "blue": "#3F76A6",
        "green": "#4E8A69",
        "coral": "#C45D4C",
        "amber": "#B78435",
        "danger": "#A6403A",
    }

    def __init__(self, checkpoint_root: Path, device: str = "auto", autoload: bool = True):
        super().__init__()
        self.title("RHF-Net | Lu'an Guapian Picking-Period Classification")
        self.geometry("1080x620")
        self.minsize(960, 590)
        self.configure(bg=self.COLORS["background"])
        self.font_family = self._choose_font()
        self.checkpoint_root = checkpoint_root
        self.ensemble = FiveFoldEnsemble(checkpoint_root, device)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cagf-inference")
        self.models_ready = False
        self.rgb_path: Path | None = None
        self.hsi_path: Path | None = None
        self.rgb_photo = None
        self.hsi_photo = None
        self.last_result = None
        self._configure_styles()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        if autoload:
            self.after(150, self._start_model_loading)

    def _choose_font(self) -> str:
        return "Arial"

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(
            "Primary.TButton", font=(self.font_family, 13, "bold"),
            foreground="#FFFFFF", background=self.COLORS["coral"],
            borderwidth=0, padding=(15, 10),
        )
        style.map("Primary.TButton", background=[("active", "#AC4D40"), ("disabled", "#C7CDD3")])
        style.configure(
            "Secondary.TButton", font=(self.font_family, 13, "bold"),
            foreground=self.COLORS["ink"], background="#E9EEF2",
            borderwidth=0, padding=(11, 8),
        )
        style.map("Secondary.TButton", background=[("active", "#DCE4EA")])
        style.configure(
            "RGB.TButton", font=(self.font_family, 13, "bold"),
            foreground="#FFFFFF", background=self.COLORS["blue"],
            borderwidth=0, padding=(12, 8),
        )
        style.map("RGB.TButton", background=[("active", "#315F87")])
        style.configure(
            "HSI.TButton", font=(self.font_family, 13, "bold"),
            foreground="#FFFFFF", background=self.COLORS["green"],
            borderwidth=0, padding=(12, 8),
        )
        style.map("HSI.TButton", background=[("active", "#3E7054")])

    def _panel(self, parent) -> tk.Frame:
        return tk.Frame(
            parent, bg=self.COLORS["surface"], highlightthickness=1,
            highlightbackground=self.COLORS["border"], bd=0,
        )

    def _build_ui(self) -> None:
        header = tk.Frame(self, bg=self.COLORS["surface"], height=94)
        header.pack(fill="x")
        header.pack_propagate(False)
        title_box = tk.Frame(header, bg=self.COLORS["surface"])
        title_box.pack(side="left", padx=28, pady=12)
        tk.Label(
            title_box, text="RHF-Net", bg=self.COLORS["surface"], fg=self.COLORS["navy"],
            font=(self.font_family, 30, "bold"), anchor="w",
        ).pack(anchor="w")
        tk.Label(
            title_box, text="Lu'an Guapian Picking-Period Classification", bg=self.COLORS["surface"], fg=self.COLORS["muted"],
            font=(self.font_family, 14, "bold"), anchor="w",
        ).pack(anchor="w", pady=(2, 0))
        self.model_status = tk.Label(
            header, text="Preparing models", bg="#EEF2F5", fg=self.COLORS["muted"],
            font=(self.font_family, 13, "bold"), padx=14, pady=8,
        )
        self.model_status.pack(side="right", padx=28)

        separator = tk.Frame(self, bg=self.COLORS["border"], height=1)
        separator.pack(fill="x")

        body = tk.Frame(self, bg=self.COLORS["background"])
        body.pack(fill="both", expand=True, padx=24, pady=14)
        body.columnconfigure(0, weight=2, uniform="body")
        body.columnconfigure(1, weight=1, uniform="body")
        body.rowconfigure(0, weight=1)

        self.input_panel = self._panel(body)
        self.input_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.result_panel = self._panel(body)
        self.result_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        self._build_input_panel()
        self._build_result_panel()

        footer = tk.Frame(self, bg=self.COLORS["navy"], height=32)
        footer.pack(fill="x")
        footer.pack_propagate(False)
        self.footer_status = tk.Label(
            footer, text="Initializing models", bg=self.COLORS["navy"], fg="#E8EDF2",
            font=(self.font_family, 12, "bold"), anchor="w",
        )
        self.footer_status.pack(fill="both", padx=24)

    def _section_title(self, parent, text: str, subtitle: str = "") -> None:
        tk.Label(
            parent, text=text, bg=self.COLORS["surface"], fg=self.COLORS["ink"],
            font=(self.font_family, 18, "bold"), anchor="w",
        ).pack(fill="x")
        if subtitle:
            tk.Label(
                parent, text=subtitle, bg=self.COLORS["surface"], fg=self.COLORS["muted"],
                font=(self.font_family, 12, "bold"), anchor="w",
            ).pack(fill="x", pady=(3, 0))

    def _build_input_panel(self) -> None:
        panel = self.input_panel
        heading = tk.Frame(panel, bg=self.COLORS["surface"])
        heading.pack(fill="x", padx=22, pady=(16, 10))
        self._section_title(heading, "Paired Inputs", "RGB and raw HSI files must have the same filename")

        file_area = tk.Frame(panel, bg=self.COLORS["surface"])
        file_area.pack(fill="x", padx=22)
        file_area.columnconfigure(1, weight=1)
        self.rgb_var = tk.StringVar()
        self.hsi_var = tk.StringVar()
        for row, (label, variable, style_name, command) in enumerate((
            ("RGB", self.rgb_var, "RGB.TButton", self._choose_rgb),
            ("Raw HSI", self.hsi_var, "HSI.TButton", self._choose_hsi),
        )):
            tk.Label(
                file_area, text=label, bg=self.COLORS["surface"], fg=self.COLORS["ink"],
                font=(self.font_family, 13, "bold"), width=8, anchor="w",
            ).grid(row=row, column=0, sticky="w", pady=6)
            entry = tk.Entry(
                file_area, textvariable=variable, state="readonly", readonlybackground="#F7F9FA",
                fg=self.COLORS["muted"], relief="flat", bd=0, font=(self.font_family, 12, "bold"),
            )
            entry.grid(row=row, column=1, sticky="ew", padx=(0, 10), ipady=8, pady=6)
            ttk.Button(file_area, text="Browse", style=style_name, command=command).grid(
                row=row, column=2, pady=6
            )

        self.pair_status = tk.Label(
            panel, text="Load RGB , HSI , or a paired sample", bg=self.COLORS["surface"], fg=self.COLORS["muted"],
            font=(self.font_family, 12, "bold"), anchor="w",
        )
        self.pair_status.pack(fill="x", padx=22, pady=(6, 12))

        previews = tk.Frame(panel, bg=self.COLORS["surface"])
        previews.pack(fill="x", padx=22, pady=(0, 12))
        previews.columnconfigure(0, weight=1, uniform="preview")
        previews.columnconfigure(1, weight=1, uniform="preview")
        tk.Label(
            previews, text="RGB Preview", bg=self.COLORS["surface"], fg=self.COLORS["blue"],
            font=(self.font_family, 13, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 7))
        tk.Label(
            previews, text="Raw HSI Preview", bg=self.COLORS["surface"], fg=self.COLORS["green"],
            font=(self.font_family, 13, "bold"),
        ).grid(row=0, column=1, sticky="w", padx=(12, 0), pady=(0, 7))
        self.rgb_preview = tk.Label(
            previews, text="RGB", bg="#E9EEF2", fg="#8A96A3",
            font=(self.font_family, 16, "bold"), relief="flat", width=38, height=6,
        )
        self.rgb_preview.grid(row=1, column=0, sticky="n", ipadx=0, ipady=0)
        self.hsi_preview = tk.Label(
            previews, text="HSI", bg="#E9EEF2", fg="#8A96A3",
            font=(self.font_family, 16, "bold"), relief="flat", width=38, height=6,
        )
        self.hsi_preview.grid(row=1, column=1, sticky="n", padx=(12, 0))

        actions = tk.Frame(panel, bg=self.COLORS["surface"])
        actions.pack(fill="x", padx=22, pady=(0, 14))
        self.predict_button = ttk.Button(
            actions, text="Predict", style="Primary.TButton", command=self._start_prediction,
            state="disabled",
        )
        self.predict_button.pack(side="left")
        ttk.Button(actions, text="Clear", style="Secondary.TButton", command=self._clear).pack(
            side="left", padx=10
        )
        self.export_button = ttk.Button(
            actions, text="Export", style="Secondary.TButton", command=self._export_result,
            state="disabled",
        )
        self.export_button.pack(side="right")

    def _build_result_panel(self) -> None:
        panel = self.result_panel
        heading = tk.Frame(panel, bg=self.COLORS["surface"])
        heading.pack(fill="x", padx=22, pady=(16, 10))
        self._section_title(heading, "Final Prediction")

        prediction = tk.Frame(panel, bg="#F7F9FA", highlightthickness=1, highlightbackground="#E3E8EC")
        prediction.pack(fill="x", padx=22, pady=(0, 16))
        self.result_code = tk.Label(
            prediction, text="--", bg="#F7F9FA", fg=self.COLORS["coral"],
            font=(self.font_family, 40, "bold"), width=3,
        )
        self.result_code.pack(side="left", padx=(18, 10), pady=15)
        result_text = tk.Frame(prediction, bg="#F7F9FA")
        result_text.pack(side="left", fill="both", expand=True, pady=14)
        self.result_description = tk.Label(
            result_text, text="Awaiting prediction", bg="#F7F9FA", fg=self.COLORS["ink"],
            font=(self.font_family, 16, "bold"), anchor="w",
        )
        self.result_description.pack(fill="x")
        self.result_confidence = tk.Label(
            result_text, text="Mean confidence  --", bg="#F7F9FA", fg=self.COLORS["muted"],
            font=(self.font_family, 13, "bold"), anchor="w",
        )
        self.result_confidence.pack(fill="x", pady=(5, 0))

        probability_box = tk.Frame(panel, bg=self.COLORS["surface"])
        probability_box.pack(fill="x", padx=22)
        tk.Label(
            probability_box, text="Class Probabilities", bg=self.COLORS["surface"], fg=self.COLORS["ink"],
            font=(self.font_family, 14, "bold"), anchor="w",
        ).pack(fill="x", pady=(0, 8))
        colors = [self.COLORS["blue"], self.COLORS["green"], self.COLORS["amber"], self.COLORS["coral"]]
        self.probability_rows = {}
        for code, color in zip(CLASS_NAMES, colors):
            row = ProbabilityRow(probability_box, code, CLASS_INFO[code], color, self.font_family)
            row.pack(fill="x", pady=5)
            self.probability_rows[code] = row

        self.timing_label = tk.Label(
            panel, text="Inference time  --", bg=self.COLORS["surface"], fg=self.COLORS["muted"],
            font=(self.font_family, 12, "bold"), anchor="e",
        )
        self.timing_label.pack(fill="x", padx=22, pady=(14, 12))

    def _set_status(self, text: str, kind: str = "normal") -> None:
        colors = {
            "normal": ("#EEF2F5", self.COLORS["muted"]),
            "ready": ("#E8F2EC", self.COLORS["green"]),
            "busy": ("#FFF3E3", self.COLORS["amber"]),
            "error": ("#F8E9E7", self.COLORS["danger"]),
        }
        background, foreground = colors[kind]
        self.model_status.configure(text=text, bg=background, fg=foreground)
        self.footer_status.configure(text=text)

    def _start_model_loading(self) -> None:
        self._set_status("Loading models 0/5", "busy")

        def progress(index, total):
            self.after(0, lambda: self._set_status(f"Loading models {index}/{total}", "busy"))

        future = self.executor.submit(self.ensemble.load, progress)
        future.add_done_callback(lambda item: self.after(0, self._finish_model_loading, item))

    def _finish_model_loading(self, future: Future) -> None:
        try:
            text = future.result()
        except Exception as error:
            self.models_ready = False
            self._set_status("Model loading failed", "error")
            messagebox.showerror("Model loading failed", str(error), parent=self)
        else:
            self.models_ready = True
            self._set_status(text, "ready")
        self._update_predict_state()

    def _choose_rgb(self) -> None:
        filename = filedialog.askopenfilename(
            parent=self, title="Select RGB File",
            filetypes=[("RGB data", "*.npy *.png *.jpg *.jpeg *.tif *.tiff"), ("All files", "*.*")],
        )
        if not filename:
            return
        try:
            _, preview = load_rgb(Path(filename))
            self._show_preview(self.rgb_preview, preview, "rgb")
        except Exception as error:
            messagebox.showerror("Invalid RGB file", str(error), parent=self)
            return
        self.rgb_path = Path(filename)
        self.rgb_var.set(Path(filename).name)
        self._update_pair_status()

    def _choose_hsi(self) -> None:
        filename = filedialog.askopenfilename(
            parent=self, title="Select HSI File",
            filetypes=[("Raw HSI NumPy data", "*.npy"), ("All files", "*.*")],
        )
        if not filename:
            return
        try:
            _, preview = load_hsi(Path(filename))
            self._show_preview(self.hsi_preview, preview, "hsi")
        except Exception as error:
            messagebox.showerror("Invalid raw HSI file", str(error), parent=self)
            return
        self.hsi_path = Path(filename)
        self.hsi_var.set(Path(filename).name)
        self._update_pair_status()

    def _show_preview(self, label: tk.Label, array: np.ndarray, modality: str) -> None:
        image = Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8))
        photo = ImageTk.PhotoImage(image)
        label.configure(
            image=photo,
            text="",
            width=image.width,
            height=image.height,
        )
        if modality == "rgb":
            self.rgb_photo = photo
        else:
            self.hsi_photo = photo

    def _files_are_paired(self) -> bool:
        return bool(
            self.rgb_path and self.hsi_path and self.rgb_path.stem == self.hsi_path.stem
        )

    def _update_pair_status(self) -> None:
        if not self.rgb_path or not self.hsi_path:
            self.pair_status.configure(text="Load the matching file for the other modality", fg=self.COLORS["muted"])
        elif self._files_are_paired():
            self.pair_status.configure(
                text=f"Pair verified | {self.rgb_path.stem}", fg=self.COLORS["green"]
            )
        else:
            self.pair_status.configure(
                text="Filenames do not match. Select RGB and raw HSI data from the same sample.",
                fg=self.COLORS["danger"],
            )
        self._update_predict_state()

    def _update_predict_state(self) -> None:
        state = "normal" if self.models_ready and self._files_are_paired() else "disabled"
        self.predict_button.configure(state=state)

    def _start_prediction(self) -> None:
        if not self._files_are_paired():
            return
        self.predict_button.configure(state="disabled")
        self.export_button.configure(state="disabled")
        self._set_status("Running five-fold ensemble", "busy")
        future = self.executor.submit(self.ensemble.predict, self.rgb_path, self.hsi_path)
        future.add_done_callback(lambda item: self.after(0, self._finish_prediction, item))

    def _finish_prediction(self, future: Future) -> None:
        try:
            result = future.result()
        except Exception as error:
            self._set_status("Prediction failed", "error")
            messagebox.showerror("Prediction failed", str(error), parent=self)
        else:
            self.last_result = result
            predicted = result["predicted_class"]
            self.result_code.configure(text=predicted)
            self.result_description.configure(text=result["description"])
            self.result_confidence.configure(
                text=f'Mean confidence  {result["confidence"]:.4f}'
            )
            for code, value in zip(CLASS_NAMES, result["mean_probabilities"]):
                self.probability_rows[code].set_value(value, selected=code == predicted)
            self.timing_label.configure(
                text=f'Inference time  {result["elapsed_seconds"]:.3f} s  |  {result["device"].upper()}'
            )
            self._set_status("Prediction complete", "ready")
            self.export_button.configure(state="normal")
        self._update_predict_state()

    def _clear(self) -> None:
        self.rgb_path = self.hsi_path = None
        self.rgb_var.set("")
        self.hsi_var.set("")
        self.rgb_preview.configure(image="", text="RGB", width=38, height=6)
        self.hsi_preview.configure(image="", text="HSI", width=38, height=6)
        self.rgb_photo = self.hsi_photo = None
        self.last_result = None
        self.result_code.configure(text="--")
        self.result_description.configure(text="Awaiting prediction")
        self.result_confidence.configure(text="Mean confidence  --")
        for row in self.probability_rows.values():
            row.set_value(0.0)
        self.timing_label.configure(text="Inference time  --")
        self.export_button.configure(state="disabled")
        self._update_pair_status()

    def _export_result(self) -> None:
        if not self.last_result:
            return
        filename = filedialog.asksaveasfilename(
            parent=self, title="Export Prediction", defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("CSV", "*.csv")],
            initialfile=f'{Path(self.last_result["rgb_file"]).stem}_prediction.json',
        )
        if not filename:
            return
        path = Path(filename)
        if path.suffix.lower() == ".csv":
            with path.open("w", newline="", encoding="utf-8-sig") as file:
                writer = csv.writer(file)
                writer.writerow(["class", "mean_probability", "std_probability"])
                for code, mean, std in zip(
                    CLASS_NAMES,
                    self.last_result["mean_probabilities"],
                    self.last_result["std_probabilities"],
                ):
                    writer.writerow([code, f"{mean:.6f}", f"{std:.6f}"])
        else:
            path.write_text(
                json.dumps(self.last_result, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        self._set_status(f"Result exported | {path.name}", "ready")

    def _on_close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.destroy()


class QtProbabilityRow(QWidget):
    def __init__(self, code: str, description: str, color: str):
        super().__init__()
        self.color = color
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 3, 0, 3)
        layout.setHorizontalSpacing(10)
        self.code_label = QLabel(code)
        self.code_label.setObjectName("probabilityCode")
        self.description_label = QLabel(description)
        self.description_label.setObjectName("probabilityDescription")
        self.value_label = QLabel("0.0000")
        self.value_label.setObjectName("probabilityValue")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.bar = QProgressBar()
        self.bar.setRange(0, 10000)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(10)
        layout.addWidget(self.code_label, 0, 0, 2, 1)
        layout.addWidget(self.description_label, 0, 1)
        layout.addWidget(self.value_label, 0, 2)
        layout.addWidget(self.bar, 1, 1, 1, 2)
        layout.setColumnStretch(1, 1)
        self.set_value(0.0)

    def set_value(self, value: float, selected: bool = False) -> None:
        value = min(max(float(value), 0.0), 1.0)
        self.value_label.setText(f"{value:.4f}")
        self.bar.setValue(round(value * 10000))
        code_color = self.color if selected else "#17212B"
        self.code_label.setStyleSheet(f"color: {code_color};")
        self.bar.setStyleSheet(
            "QProgressBar {background: #E8EDF2; border: none;}"
            f"QProgressBar::chunk {{background: {self.color};}}"
        )


class PRHQtApplication(QMainWindow):
    COLORS = {
        "background": "#F3F5F7",
        "surface": "#FFFFFF",
        "ink": "#17212B",
        "muted": "#65717E",
        "border": "#D9E0E6",
        "navy": "#243B53",
        "blue": "#3F76A6",
        "green": "#4E8A69",
        "coral": "#C45D4C",
        "amber": "#B78435",
        "danger": "#A6403A",
    }

    def __init__(
        self,
        checkpoint_root: Path,
        device: str = "auto",
        autoload: bool = True,
        rgb_checkpoint_root: Path = DEFAULT_RGB_CHECKPOINT_ROOT,
        hsi_checkpoint_root: Path = DEFAULT_HSI_CHECKPOINT_ROOT,
    ):
        super().__init__()
        self.checkpoint_root = checkpoint_root
        self.ensemble = FiveFoldEnsemble(checkpoint_root, device)
        self.rgb_ensemble = SingleModeEnsemble("rgb", rgb_checkpoint_root, device)
        self.hsi_ensemble = SingleModeEnsemble("hsi", hsi_checkpoint_root, device)
        self.models_ready = False
        self.rgb_path: Path | None = None
        self.hsi_path: Path | None = None
        self.last_result = None
        self.setWindowTitle("RHF-Net | Lu'an Guapian Picking-Period Classification")
        self.resize(1100, 640)
        self.setMinimumSize(1020, 610)
        self._build_ui()
        if autoload:
            QTimer.singleShot(100, self._load_models)

    def _panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("panel")
        return panel

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(108)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(30, 15, 30, 15)
        title_box = QVBoxLayout()
        title_box.setSpacing(1)
        title = QLabel("RHF-Net")
        title.setObjectName("mainTitle")
        subtitle = QLabel("Lu'an Guapian Picking-Period Classification")
        subtitle.setObjectName("mainSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header_layout.addLayout(title_box)
        header_layout.addStretch()
        self.model_status = QLabel("Preparing models")
        self.model_status.setObjectName("modelStatus")
        header_layout.addWidget(self.model_status)
        outer.addWidget(header)

        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(24, 16, 24, 16)
        content_layout.setSpacing(18)
        self.input_panel = self._panel()
        self.result_panel = self._panel()
        content_layout.addWidget(self.input_panel, 2)
        content_layout.addWidget(self.result_panel, 1)
        outer.addWidget(content, 1)
        self._build_input_panel()
        self._build_result_panel()

        self.footer_status = QLabel("Initializing models")
        self.footer_status.setObjectName("footer")
        self.footer_status.setFixedHeight(36)
        self.footer_status.setContentsMargins(26, 0, 0, 0)
        outer.addWidget(self.footer_status)

    def _section_heading(self, text: str, subtitle: str = "") -> QVBoxLayout:
        layout = QVBoxLayout()
        layout.setSpacing(3)
        title = QLabel(text)
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        if subtitle:
            detail = QLabel(subtitle)
            detail.setObjectName("sectionSubtitle")
            layout.addWidget(detail)
        return layout

    def _build_input_panel(self) -> None:
        layout = QVBoxLayout(self.input_panel)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(12)
        layout.addLayout(self._section_heading(
            "Model Inputs", "Load RGB, HSI, or a paired RGB-HSI sample"
        ))

        files = QGridLayout()
        files.setHorizontalSpacing(10)
        files.setVerticalSpacing(10)
        self.rgb_entry = QLineEdit()
        self.hsi_entry = QLineEdit()
        for entry in (self.rgb_entry, self.hsi_entry):
            entry.setReadOnly(True)
            entry.setMinimumHeight(40)
        rgb_button = QPushButton("Browse")
        rgb_button.setObjectName("rgbButton")
        rgb_button.clicked.connect(self._choose_rgb)
        hsi_button = QPushButton("Browse")
        hsi_button.setObjectName("hsiButton")
        hsi_button.clicked.connect(self._choose_hsi)
        files.addWidget(QLabel("RGB"), 0, 0)
        files.addWidget(self.rgb_entry, 0, 1)
        files.addWidget(rgb_button, 0, 2)
        files.addWidget(QLabel("Raw HSI"), 1, 0)
        files.addWidget(self.hsi_entry, 1, 1)
        files.addWidget(hsi_button, 1, 2)
        files.setColumnStretch(1, 1)
        layout.addLayout(files)

        self.pair_status = QLabel("Select RGB, HSI, or both")
        self.pair_status.setObjectName("pairStatus")
        layout.addWidget(self.pair_status)

        preview_titles = QHBoxLayout()
        rgb_title = QLabel("RGB Preview")
        rgb_title.setObjectName("rgbTitle")
        hsi_title = QLabel("Raw HSI Preview")
        hsi_title.setObjectName("hsiTitle")
        preview_titles.addWidget(rgb_title, 1)
        preview_titles.addWidget(hsi_title, 1)
        layout.addLayout(preview_titles)

        previews = QHBoxLayout()
        previews.setSpacing(14)
        self.rgb_preview = self._preview_placeholder("RGB")
        self.hsi_preview = self._preview_placeholder("HSI")
        previews.addWidget(self.rgb_preview, 0, Qt.AlignmentFlag.AlignTop)
        previews.addWidget(self.hsi_preview, 0, Qt.AlignmentFlag.AlignTop)
        previews.addStretch()
        layout.addLayout(previews)

        actions = QHBoxLayout()
        self.predict_button = QPushButton("Predict")
        self.predict_button.setObjectName("predictButton")
        self.predict_button.clicked.connect(self._predict)
        self.predict_button.setEnabled(False)
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self._clear)
        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self._export_result)
        self.export_button.setEnabled(False)
        actions.addWidget(self.predict_button)
        actions.addWidget(clear_button)
        actions.addStretch()
        actions.addWidget(self.export_button)
        layout.addLayout(actions)
        layout.addStretch()

    def _preview_placeholder(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("preview")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setFixedSize(PREVIEW_LONG_EDGE, 100)
        return label

    def _build_result_panel(self) -> None:
        layout = QVBoxLayout(self.result_panel)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(14)
        layout.addLayout(self._section_heading("Final Prediction"))

        result_box = QFrame()
        result_box.setObjectName("resultBox")
        result_layout = QHBoxLayout(result_box)
        result_layout.setContentsMargins(16, 14, 16, 14)
        self.result_code = QLabel("--")
        self.result_code.setObjectName("resultCode")
        result_layout.addWidget(self.result_code)
        result_text = QVBoxLayout()
        self.result_description = QLabel("Awaiting prediction")
        self.result_description.setObjectName("resultDescription")
        self.result_confidence = QLabel("Mean confidence  --")
        self.result_confidence.setObjectName("resultConfidence")
        result_text.addWidget(self.result_description)
        result_text.addWidget(self.result_confidence)
        result_layout.addLayout(result_text, 1)
        layout.addWidget(result_box)

        probability_title = QLabel("Class Probabilities")
        probability_title.setObjectName("probabilityTitle")
        layout.addWidget(probability_title)
        colors = [self.COLORS["blue"], self.COLORS["green"], self.COLORS["amber"], self.COLORS["coral"]]
        self.probability_rows = {}
        for code, color in zip(CLASS_NAMES, colors):
            row = QtProbabilityRow(code, CLASS_INFO[code], color)
            layout.addWidget(row)
            self.probability_rows[code] = row
        self.timing_label = QLabel("Inference time  --")
        self.timing_label.setObjectName("timing")
        self.timing_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.timing_label)
        layout.addStretch()

    def _set_status(self, text: str, kind: str = "normal") -> None:
        colors = {
            "normal": ("#EEF2F5", self.COLORS["muted"]),
            "ready": ("#E8F2EC", self.COLORS["green"]),
            "busy": ("#FFF3E3", self.COLORS["amber"]),
            "error": ("#F8E9E7", self.COLORS["danger"]),
        }
        background, foreground = colors[kind]
        self.model_status.setText(text)
        self.model_status.setStyleSheet(
            f"background: {background}; color: {foreground}; padding: 9px 16px;"
        )
        self.footer_status.setText(text)

    def _load_models(self) -> None:
        self._set_status("Loading models", "busy")
        QApplication.processEvents()
        try:
            status = self.ensemble.load()
            self.rgb_ensemble.load()
            self.hsi_ensemble.load()
        except Exception as error:
            self._set_status("Model loading failed", "error")
            QMessageBox.critical(self, "Model loading failed", str(error))
        else:
            self.models_ready = True
            self._set_status(status, "ready")
        self._update_predict_state()

    def _choose_rgb(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select RGB File", "",
            "RGB data (*.npy *.png *.jpg *.jpeg *.tif *.tiff);;All files (*)",
        )
        if not filename:
            return
        try:
            _, preview = load_rgb(Path(filename))
            self._show_preview(self.rgb_preview, preview)
        except Exception as error:
            QMessageBox.critical(self, "Invalid RGB file", str(error))
            return
        self.rgb_path = Path(filename)
        self.rgb_entry.setText(self.rgb_path.name)
        self._update_pair_status()

    def _choose_hsi(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select HSI File", "", "HSI NumPy data (*.npy);;All files (*)"
        )
        if not filename:
            return
        try:
            _, preview = load_hsi(Path(filename))
            self._show_preview(self.hsi_preview, preview)
        except Exception as error:
            QMessageBox.critical(self, "Invalid HSI file", str(error))
            return
        self.hsi_path = Path(filename)
        self.hsi_entry.setText(self.hsi_path.name)
        self._update_pair_status()

    def _show_preview(self, label: QLabel, array: np.ndarray) -> None:
        rgb = np.ascontiguousarray((np.clip(array, 0, 1) * 255).astype(np.uint8))
        height, width, _ = rgb.shape
        image = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            PREVIEW_LONG_EDGE,
            PREVIEW_LONG_EDGE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setFixedSize(pixmap.size())
        label.setPixmap(pixmap)
        label.setText("")

    def _files_are_paired(self) -> bool:
        return bool(self.rgb_path and self.hsi_path and self.rgb_path.stem == self.hsi_path.stem)

    def _prediction_mode(self) -> str | None:
        if self.rgb_path and self.hsi_path:
            return "fusion" if self._files_are_paired() else None
        if self.rgb_path:
            return "rgb"
        if self.hsi_path:
            return "hsi"
        return None

    def _update_pair_status(self) -> None:
        if self.rgb_path and not self.hsi_path:
            self.pair_status.setText("RGB-only prediction mode")
            self.pair_status.setStyleSheet(f"color: {self.COLORS['blue']};")
        elif self.hsi_path and not self.rgb_path:
            self.pair_status.setText("HSI-only prediction mode | HSI to PCA20")
            self.pair_status.setStyleSheet(f"color: {self.COLORS['green']};")
        elif self._files_are_paired():
            self.pair_status.setText(f"Fusion prediction mode | {self.rgb_path.stem}")
            self.pair_status.setStyleSheet(f"color: {self.COLORS['green']};")
        elif not self.rgb_path and not self.hsi_path:
            self.pair_status.setText("Select RGB, HSI, or both")
            self.pair_status.setStyleSheet(f"color: {self.COLORS['muted']};")
        else:
            self.pair_status.setText("Filenames do not match. Select data from the same sample.")
            self.pair_status.setStyleSheet(f"color: {self.COLORS['danger']};")
        self._update_predict_state()

    def _update_predict_state(self) -> None:
        self.predict_button.setEnabled(self.models_ready and self._prediction_mode() is not None)

    def _predict(self) -> None:
        mode = self._prediction_mode()
        if mode is None:
            return
        self.predict_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self._set_status("Running prediction", "busy")
        QApplication.processEvents()
        try:
            if mode == "fusion":
                result = self.ensemble.predict(self.rgb_path, self.hsi_path)
            elif mode == "rgb":
                result = self.rgb_ensemble.predict(self.rgb_path)
            else:
                result = self.hsi_ensemble.predict(self.hsi_path)
        except Exception as error:
            self._set_status("Prediction failed", "error")
            QMessageBox.critical(self, "Prediction failed", str(error))
        else:
            self.last_result = result
            predicted = result["predicted_class"]
            self.result_code.setText(predicted)
            self.result_description.setText(result["description"])
            self.result_confidence.setText(
                f'{result["mode"].upper()} | Mean confidence  {result["confidence"]:.4f}'
            )
            for code, value in zip(CLASS_NAMES, result["mean_probabilities"]):
                self.probability_rows[code].set_value(value, code == predicted)
            self.timing_label.setText(
                f'Inference time  {result["elapsed_seconds"]:.3f} s  |  {result["device"].upper()}'
            )
            self._set_status(f'{result["mode"].upper()} prediction complete', "ready")
            self.export_button.setEnabled(True)
        self._update_predict_state()

    def _clear(self) -> None:
        self.rgb_path = self.hsi_path = None
        self.rgb_entry.clear()
        self.hsi_entry.clear()
        for label, text in ((self.rgb_preview, "RGB"), (self.hsi_preview, "HSI")):
            label.clear()
            label.setText(text)
            label.setFixedSize(PREVIEW_LONG_EDGE, 100)
        self.last_result = None
        self.result_code.setText("--")
        self.result_description.setText("Awaiting prediction")
        self.result_confidence.setText("Mean confidence  --")
        for row in self.probability_rows.values():
            row.set_value(0.0)
        self.timing_label.setText("Inference time  --")
        self.export_button.setEnabled(False)
        self._update_pair_status()

    def _export_result(self) -> None:
        if not self.last_result:
            return
        source_file = self.last_result.get("rgb_file") or self.last_result.get("hsi_file")
        initial = f'{Path(source_file).stem}_prediction.json'
        filename, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Prediction", initial, "JSON (*.json);;CSV (*.csv)"
        )
        if not filename:
            return
        path = Path(filename)
        if "CSV" in selected_filter and path.suffix.lower() != ".csv":
            path = path.with_suffix(".csv")
        elif not path.suffix:
            path = path.with_suffix(".json")
        if path.suffix.lower() == ".csv":
            with path.open("w", newline="", encoding="utf-8-sig") as file:
                writer = csv.writer(file)
                writer.writerow(["class", "mean_probability", "std_probability"])
                for code, mean, std in zip(
                    CLASS_NAMES,
                    self.last_result["mean_probabilities"],
                    self.last_result["std_probabilities"],
                ):
                    writer.writerow([code, f"{mean:.6f}", f"{std:.6f}"])
        else:
            path.write_text(
                json.dumps(self.last_result, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        self._set_status(f"Result exported | {path.name}", "ready")


def configure_qt_application(app: QApplication) -> None:
    for font_path in (
        "/mnt/c/Windows/Fonts/arial.ttf",
        "/mnt/c/Windows/Fonts/arialbd.ttf",
    ):
        if Path(font_path).exists():
            QFontDatabase.addApplicationFont(font_path)
    app.setFont(QFont("Arial", 13, QFont.Weight.Bold))
    app.setStyleSheet("""
        QWidget { font-family: Arial; font-size: 17px; font-weight: 700; color: #17212B; }
        QWidget#root { background: #F3F5F7; }
        QFrame#header, QFrame#panel { background: #FFFFFF; }
        QFrame#panel { border: 1px solid #D9E0E6; }
        QLabel#mainTitle { color: #243B53; font-size: 38px; font-weight: 700; }
        QLabel#mainSubtitle { color: #65717E; font-size: 19px; font-weight: 700; }
        QLabel#modelStatus { font-size: 17px; font-weight: 700; }
        QLabel#sectionTitle { font-size: 24px; font-weight: 700; }
        QLabel#sectionSubtitle { color: #65717E; font-size: 16px; font-weight: 700; }
        QLabel#pairStatus { color: #65717E; font-size: 16px; font-weight: 700; }
        QLabel#rgbTitle { color: #3F76A6; font-size: 17px; font-weight: 700; }
        QLabel#hsiTitle { color: #4E8A69; font-size: 17px; font-weight: 700; }
        QLabel#preview { background: #E9EEF2; color: #65717E; font-size: 18px; font-weight: 700; }
        QLineEdit { background: #F7F9FA; border: 1px solid #D9E0E6; padding: 7px 9px; font-size: 16px; }
        QPushButton { background: #E9EEF2; border: none; padding: 10px 17px; font-size: 17px; font-weight: 700; }
        QPushButton:hover { background: #DCE4EA; }
        QPushButton:disabled { color: #8A96A3; background: #E7EBEE; }
        QPushButton#rgbButton { color: white; background: #3F76A6; }
        QPushButton#hsiButton { color: white; background: #4E8A69; }
        QPushButton#predictButton { color: white; background: #C45D4C; }
        QFrame#resultBox { background: #F7F9FA; border: 1px solid #E3E8EC; }
        QLabel#resultCode { color: #C45D4C; font-size: 46px; font-weight: 700; }
        QLabel#resultDescription { font-size: 19px; font-weight: 700; }
        QLabel#resultConfidence { color: #65717E; font-size: 16px; font-weight: 700; }
        QLabel#probabilityTitle { font-size: 19px; font-weight: 700; }
        QLabel#probabilityCode { font-size: 18px; font-weight: 700; }
        QLabel#probabilityDescription, QLabel#probabilityValue { font-size: 16px; font-weight: 700; }
        QLabel#timing { color: #65717E; font-size: 15px; font-weight: 700; }
        QLabel#footer { color: #E8EDF2; background: #243B53; font-size: 15px; font-weight: 700; }
    """)


def run_smoke_test(checkpoint_root: Path, rgb_path: Path, hsi_path: Path, device: str) -> None:
    if rgb_path.stem != hsi_path.stem:
        raise ValueError("RGB and raw HSI filenames must match")
    ensemble = FiveFoldEnsemble(checkpoint_root, device)
    ensemble.load()
    result = ensemble.predict(rgb_path, hsi_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--rgb-checkpoint-root", type=Path, default=DEFAULT_RGB_CHECKPOINT_ROOT)
    parser.add_argument("--hsi-checkpoint-root", type=Path, default=DEFAULT_HSI_CHECKPOINT_ROOT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--rgb", type=Path)
    parser.add_argument("--hsi", type=Path)
    parser.add_argument("--ui-smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        if not args.rgb or not args.hsi:
            parser.error("--smoke-test requires --rgb and --hsi")
        run_smoke_test(args.checkpoint_root, args.rgb, args.hsi, args.device)
        return
    qt_app = QApplication.instance() or QApplication([])
    configure_qt_application(qt_app)
    window = PRHQtApplication(
        args.checkpoint_root,
        args.device,
        autoload=not args.ui_smoke,
        rgb_checkpoint_root=args.rgb_checkpoint_root,
        hsi_checkpoint_root=args.hsi_checkpoint_root,
    )
    window.show()
    if args.ui_smoke:
        QTimer.singleShot(800, qt_app.quit)
    qt_app.exec()


if __name__ == "__main__":
    main()
