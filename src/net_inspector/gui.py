"""Tkinter GUI focused on segmentation and live camera preview."""

from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

from net_inspector.config import AppConfig
from net_inspector.segmenter import Segmenter, render_overlay
from net_inspector.synth.generate import generate_demo_image
from net_inspector.utils.io import ensure_dir, save_image, timestamp_id


class NetInspectorGUI:
    """Segmentation-first GUI with optional live camera."""

    def __init__(self) -> None:
        self.config = AppConfig()
        self.segmenter = Segmenter()

        self.root = tk.Tk()
        self.root.title("Net Inspector")
        self.root.minsize(1040, 720)

        self.current_image: Optional[np.ndarray] = None
        self.segmented_image: Optional[np.ndarray] = None

        self._camera_thread: Optional[threading.Thread] = None
        self._segment_thread: Optional[threading.Thread] = None
        self._camera_running = False
        self._stop_event = threading.Event()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_seg: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._frame_id = 0
        self._seg_frame_id = -1
        self._seg_fps = 0.0
        self._seg_last_time = time.time()
        self._seg_count = 0
        self._fps = 0.0
        self._fps_last_time = time.time()
        self._fps_count = 0

        self._apply_style()
        self._build_ui()
        self._start_ui_loop()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _apply_style(self) -> None:
        self.root.configure(bg="#f7f8fa")
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure("TFrame", background="#f7f8fa")
        style.configure("Card.TFrame", background="#ffffff")
        style.configure(
            "Header.TLabel",
            background="#f7f8fa",
            foreground="#0f172a",
            font=("Segoe UI", 18, "bold"),
        )
        style.configure(
            "SubHeader.TLabel",
            background="#f7f8fa",
            foreground="#64748b",
            font=("Segoe UI", 10),
        )
        style.configure(
            "Badge.TLabel",
            background="#eef2ff",
            foreground="#3730a3",
            font=("Segoe UI", 9, "bold"),
            padding=(8, 2),
        )
        style.configure(
            "TLabel",
            background="#ffffff",
            foreground="#0f172a",
            font=("Segoe UI", 10),
        )
        style.configure(
            "Status.TLabel",
            background="#ffffff",
            foreground="#475569",
            font=("Segoe UI", 9),
        )
        style.configure(
            "TLabelframe",
            background="#ffffff",
            foreground="#0f172a",
            bordercolor="#e2e8f0",
            relief="solid",
            borderwidth=1,
        )
        style.configure(
            "TLabelframe.Label",
            background="#ffffff",
            foreground="#334155",
            font=("Segoe UI", 9, "bold"),
        )
        style.configure(
            "TButton",
            background="#2563eb",
            foreground="#ffffff",
            borderwidth=0,
            focusthickness=0,
            padding=(10, 6),
        )
        style.map(
            "TButton",
            background=[("active", "#1d4ed8"), ("disabled", "#94a3b8")],
            foreground=[("disabled", "#f8fafc")],
        )
        style.configure(
            "TCheckbutton",
            background="#ffffff",
            foreground="#0f172a",
            font=("Segoe UI", 9),
            padding=(4, 2),
        )
        style.map("TCheckbutton", foreground=[("disabled", "#94a3b8")])

    def _build_ui(self) -> None:
        """Build the Tkinter layout.

        @return None
        """
        main = ttk.Frame(self.root, padding=16)
        main.pack(fill=tk.BOTH, expand=True)

        self.var_debris = tk.BooleanVar(value=True)
        self.var_tear = tk.BooleanVar(value=True)
        self.var_fire = tk.BooleanVar(value=True)
        self.var_sky = tk.BooleanVar(value=False)
        self.var_net = tk.BooleanVar(value=True)
        self.var_debris_detect = tk.BooleanVar(value=True)
        self.var_fire_detect = tk.BooleanVar(value=True)
        self.var_legend = tk.BooleanVar(value=True)
        self.var_camera = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value=self._segment_status_text())
        self._extra_status = ""
        self.report_var = tk.StringVar(value="Debris: 0.0%")

        self.h_min_var = tk.IntVar(value=self.config.heuristic.green_h_min)
        self.h_max_var = tk.IntVar(value=self.config.heuristic.green_h_max)
        self.s_min_var = tk.IntVar(value=self.config.heuristic.green_s_min)
        self.v_min_var = tk.IntVar(value=self.config.heuristic.green_v_min)

        header = ttk.Frame(main)
        header.pack(fill=tk.X, pady=(0, 12))
        header_left = ttk.Frame(header)
        header_left.pack(side=tk.LEFT, anchor=tk.W)
        title_row = ttk.Frame(header_left)
        title_row.pack(side=tk.TOP, anchor=tk.W)
        ttk.Label(title_row, text="Net Inspector", style="Header.TLabel").pack(
            side=tk.LEFT, anchor=tk.W
        )
        ttk.Label(title_row, text="SEGMENTATION", style="Badge.TLabel").pack(
            side=tk.LEFT, padx=10
        )
        ttk.Label(
            header_left,
            text="Live camera preview with net overlay and fast segmentation refresh.",
            style="SubHeader.TLabel",
        ).pack(side=tk.TOP, anchor=tk.W, pady=(2, 0))

        controls = ttk.Frame(main, style="Card.TFrame", padding=12)
        controls.pack(fill=tk.X)

        options = ttk.LabelFrame(controls, text="Synthetic elements", padding=8)
        options.pack(side=tk.LEFT, padx=(0, 12), pady=4)
        ttk.Checkbutton(options, text="Debris", variable=self.var_debris).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Checkbutton(options, text="Tear", variable=self.var_tear).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Checkbutton(options, text="Fire", variable=self.var_fire).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Checkbutton(options, text="Sky", variable=self.var_sky).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Checkbutton(options, text="Net overlay", variable=self.var_net).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Checkbutton(options, text="Legend", variable=self.var_legend).pack(
            side=tk.LEFT, padx=4
        )

        actions = ttk.LabelFrame(controls, text="Actions", padding=8)
        actions.pack(side=tk.LEFT, padx=(0, 12), pady=4)
        ttk.Button(actions, text="Generate demo image", command=self._on_generate).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(actions, text="Load image", command=self._on_load).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(actions, text="Run Segmentation", command=self._on_segment).pack(
            side=tk.LEFT, padx=4
        )

        camera = ttk.LabelFrame(controls, text="Live camera", padding=8)
        camera.pack(side=tk.LEFT, padx=(0, 12), pady=4)
        ttk.Checkbutton(
            camera, text="Enable", variable=self.var_camera, command=self._toggle_camera
        ).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(
            camera, text="Debris", variable=self.var_debris_detect
        ).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(
            camera, text="Fire", variable=self.var_fire_detect
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(camera, text="Stop", command=self._stop_camera).pack(side=tk.LEFT, padx=4)
        ttk.Label(camera, textvariable=self.status_var, style="Status.TLabel").pack(
            side=tk.LEFT, padx=8
        )

        report = ttk.LabelFrame(controls, text="Debris report", padding=8)
        report.pack(side=tk.LEFT, padx=(0, 12), pady=4)
        ttk.Label(report, textvariable=self.report_var, style="Status.TLabel").pack(
            side=tk.LEFT, padx=4
        )

        sliders = ttk.LabelFrame(controls, text="Net HSV", padding=8)
        sliders.pack(side=tk.LEFT, padx=(0, 12), pady=4)
        self._add_slider(sliders, "H min", self.h_min_var, 0, 179)
        self._add_slider(sliders, "H max", self.h_max_var, 0, 179)
        self._add_slider(sliders, "S min", self.s_min_var, 0, 255)
        self._add_slider(sliders, "V min", self.v_min_var, 0, 255)

        images = ttk.Frame(main, padding=0)
        images.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        input_frame = ttk.LabelFrame(images, text="Input", padding=8)
        input_frame.pack(side=tk.LEFT, padx=(0, 12), fill=tk.BOTH, expand=True)
        output_frame = ttk.LabelFrame(images, text="Segmentation output", padding=8)
        output_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.input_label = ttk.Label(
            input_frame, text="No image loaded", anchor=tk.CENTER, justify=tk.CENTER
        )
        self.input_label.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.output_label = ttk.Label(
            output_frame,
            text="Run segmentation to see overlay",
            anchor=tk.CENTER,
            justify=tk.CENTER,
        )
        self.output_label.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

    def _segment_status_text(self) -> str:
        """Return segmentation status string with FPS.

        @return Status string.
        """
        ready = "Ready" if self.segmenter.available() else "Missing weights"
        return f"Segmentation: {ready} | {self._seg_fps:.1f} FPS"

    def _camera_status_text(self) -> str:
        """Return camera status string with FPS.

        @return Status string.
        """
        cam_state = "On" if self._camera_running else "Off"
        fps_part = f" | {self._fps:.1f} FPS" if self._camera_running else ""
        return f"Camera: {cam_state}{fps_part}"

    def _on_generate(self) -> None:
        """Generate a synthetic demo image and show it.

        @return None
        """
        result = generate_demo_image(
            add_debris=self.var_debris.get(),
            add_tear=self.var_tear.get(),
            add_fire=self.var_fire.get(),
            add_sky=self.var_sky.get(),
        )
        self.current_image = result.image
        self.segmented_image = None
        self._set_image(self.input_label, self.current_image)
        self._clear_output()

    def _on_load(self) -> None:
        """Load an image from disk and show it.

        @return None
        """
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All", "*.*")],
        )
        if not path:
            return
        image = cv2.imread(path)
        if image is None:
            messagebox.showerror("Error", "Failed to load image")
            return
        self.current_image = image
        self.segmented_image = None
        self._set_image(self.input_label, self.current_image)
        self._clear_output()

    def _on_segment(self) -> None:
        """Run segmentation on the current image.

        @return None
        """
        if self.current_image is None:
            messagebox.showwarning("No image", "Load or generate an image first.")
            return
        if not self.segmenter.available():
            messagebox.showerror(
                "Segmentation unavailable",
                "PyTorch segmentation weights not found. Place models/deeplabv3_resnet50.pth",
            )
            return
        mask = self.segmenter.segment(self.current_image)
        overlay = render_overlay(
            self.current_image,
            mask,
            self.segmenter.labels,
            show_legend=self.var_legend.get(),
        )
        overlay, stats = self._apply_extra_overlays(overlay, self.current_image)
        self._extra_status = stats
        self.segmented_image = overlay
        self._set_image(self.output_label, self.segmented_image)
        self._save_segmentation(self.segmented_image)

    def _toggle_camera(self) -> None:
        """Toggle live camera on/off.

        @return None
        """
        if self.var_camera.get():
            self._start_camera()
        else:
            self._stop_camera()

    def _start_camera(self) -> None:
        """Start camera and segmentation threads.

        @return None
        """
        if self._camera_running:
            return
        self._camera_running = True
        self._stop_event.clear()
        self._fps = 0.0
        self._fps_last_time = time.time()
        self._fps_count = 0
        self._camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._camera_thread.start()
        self._segment_thread = threading.Thread(target=self._segment_loop, daemon=True)
        self._segment_thread.start()

    def _stop_camera(self) -> None:
        """Stop camera and segmentation threads.

        @return None
        """
        self._camera_running = False
        self._stop_event.set()
        self.var_camera.set(False)
        self._fps = 0.0
        if self._camera_thread and self._camera_thread.is_alive():
            self._camera_thread.join(timeout=1.0)
        if self._segment_thread and self._segment_thread.is_alive():
            self._segment_thread.join(timeout=1.0)
        with self._frame_lock:
            self._latest_frame = None
            self._latest_seg = None

    def _camera_loop(self) -> None:
        """Camera capture loop (runs on its own thread).

        @return None
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self._camera_running = False
            self.root.after(
                0, lambda: messagebox.showerror("Camera error", "Unable to open camera 0")
            )
            return
        while self._camera_running and not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            with self._frame_lock:
                self._latest_frame = frame
                self._frame_id += 1
            self._fps_count += 1
            now = time.time()
            if now - self._fps_last_time >= 1.0:
                self._fps = self._fps_count / max(0.001, now - self._fps_last_time)
                self._fps_last_time = now
                self._fps_count = 0
            time.sleep(0.001)
        cap.release()

    def _segment_loop(self) -> None:
        """Segmentation loop (runs on its own thread).

        @return None
        """
        while self._camera_running and not self._stop_event.is_set():
            if not self.segmenter.available():
                time.sleep(0.1)
                continue
            with self._frame_lock:
                if self._latest_frame is None or self._seg_frame_id == self._frame_id:
                    frame = None
                else:
                    frame = self._latest_frame.copy()
                    self._seg_frame_id = self._frame_id
            if frame is None:
                time.sleep(0.01)
                continue
            mask = self.segmenter.segment(frame)
            overlay = render_overlay(
                frame,
                mask,
                self.segmenter.labels,
                show_legend=self.var_legend.get(),
            )
            overlay, stats = self._apply_extra_overlays(overlay, frame)
            self._extra_status = stats
            with self._frame_lock:
                self._latest_seg = overlay
            self._seg_count += 1
            if self._seg_count % 5 == 0:
                now = time.time()
                elapsed = now - self._seg_last_time
                if elapsed > 0:
                    self._seg_fps = 5.0 / elapsed
                self._seg_last_time = now

    def _start_ui_loop(self) -> None:
        """Periodic UI refresh loop.

        @return None
        """
        base = self._segment_status_text()
        camera = self._camera_status_text()
        extra = f" | {self._extra_status}" if self._extra_status else ""
        self.status_var.set(f"{base} | {camera}{extra}")
        if self._latest_frame is not None:
            self._set_image(self.input_label, self._latest_frame)
        if self._latest_seg is not None:
            self._set_image(self.output_label, self._latest_seg)
        self.root.after(50, self._start_ui_loop)

    def _set_image(self, label: ttk.Label, image_bgr: np.ndarray) -> None:
        """Render an image into a Tk label.

        @param label Tk label to update.
        @param image_bgr Image in BGR.
        @return None
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(image_rgb)
        pil.thumbnail((540, 420))
        tk_image = ImageTk.PhotoImage(pil)
        label.configure(image=tk_image, text="")
        label.image = tk_image

    def _clear_output(self) -> None:
        """Clear the output panel.

        @return None
        """
        self.output_label.configure(image="", text="Run segmentation to see overlay")
        self.output_label.image = None

    def _save_segmentation(self, overlay: np.ndarray) -> None:
        """Save segmentation overlay to outputs/annotated.

        @param overlay Segmentation overlay image.
        @return None
        """
        ensure_dir(self.config.outputs_annotated)
        stamp = timestamp_id()
        out_path = self.config.outputs_annotated / f"seg_{stamp}.jpg"
        save_image(out_path, overlay)

    def _apply_extra_overlays(
        self, overlay: np.ndarray, image_bgr: np.ndarray
    ) -> tuple[np.ndarray, str]:
        """Apply net/debris/fire overlays with conservative confidence.

        @param overlay Current overlay image.
        @param image_bgr Source image (BGR).
        @return (overlay image, status text).
        """
        status_parts = []

        if self.var_net.get():
            net_mask = self._compute_net_mask(image_bgr)
            overlay = self._overlay_mask(overlay, net_mask, (0, 255, 0), 0.25)
            net_ratio = float(np.count_nonzero(net_mask)) / float(net_mask.size)
            status_parts.append(f"Net {net_ratio*100:.1f}%")

        if self.var_debris_detect.get():
            net_mask = self._compute_net_mask(image_bgr)
            debris_mask = self._compute_debris_mask(image_bgr, net_mask)
            overlay = self._overlay_mask(overlay, debris_mask, (0, 140, 255), 0.25)
            debris_ratio = float(np.count_nonzero(debris_mask)) / float(debris_mask.size)
            status_parts.append(f"Debris {debris_ratio*100:.1f}% (low)")
            self.report_var.set(f"Debris: {debris_ratio*100:.1f}% (low)")
        else:
            self.report_var.set("Debris: off")

        if self.var_fire_detect.get():
            fire_mask = self._compute_fire_mask(image_bgr)
            overlay = self._overlay_mask(overlay, fire_mask, (0, 0, 255), 0.25)
            fire_ratio = float(np.count_nonzero(fire_mask)) / float(fire_mask.size)
            status_parts.append(f"Fire {fire_ratio*100:.1f}% (low)")

        return overlay, " | ".join(status_parts)

    def _compute_debris_mask(self, image_bgr: np.ndarray, net_mask: np.ndarray) -> np.ndarray:
        """Compute a conservative debris mask within net regions.

        @param image_bgr Input image (BGR).
        @param net_mask Net mask.
        @return Binary debris mask.
        """
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        cfg = self.config.heuristic
        green = cv2.inRange(
            hsv,
            (cfg.green_h_min, cfg.green_s_min, cfg.green_v_min),
            (cfg.green_h_max, 255, 255),
        )
        non_green = cv2.bitwise_not(green)
        debris = cv2.bitwise_and(non_green, net_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        debris = cv2.morphologyEx(debris, cv2.MORPH_OPEN, kernel, iterations=1)
        return debris

    def _compute_fire_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        """Compute a conservative fire mask (low confidence).

        @param image_bgr Input image (BGR).
        @return Binary fire mask.
        """
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        fire1 = cv2.inRange(hsv, (0, 140, 140), (25, 255, 255))
        fire2 = cv2.inRange(hsv, (160, 140, 140), (179, 255, 255))
        fire = cv2.bitwise_or(fire1, fire2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fire = cv2.morphologyEx(fire, cv2.MORPH_OPEN, kernel, iterations=1)
        return fire

    def _overlay_mask(self, image_bgr: np.ndarray, mask: np.ndarray, color, alpha: float) -> np.ndarray:
        """Overlay a binary mask on an image.

        @param image_bgr Input image (BGR).
        @param mask Binary mask.
        @param color BGR color.
        @param alpha Overlay alpha.
        @return Image with overlay.
        """
        if mask is None or mask.size == 0:
            return image_bgr
        overlay = image_bgr.copy()
        overlay[mask > 0] = color
        return cv2.addWeighted(image_bgr, 1 - alpha, overlay, alpha, 0)

    def _compute_net_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        """Compute a simple net mask using green HSV thresholds.

        @param image_bgr Input image (BGR).
        @return Binary net mask.
        """
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        lower = (self.h_min_var.get(), self.s_min_var.get(), self.v_min_var.get())
        upper = (self.h_max_var.get(), 255, 255)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return mask

    def _add_slider(
        self,
        parent: ttk.LabelFrame,
        label: str,
        var: tk.IntVar,
        minimum: int,
        maximum: int,
    ) -> None:
        """Add a labeled slider row.

        @param parent Parent frame.
        @param label Label text.
        @param var IntVar bound to slider.
        @param minimum Slider min.
        @param maximum Slider max.
        @return None
        """
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, style="Status.TLabel").pack(side=tk.LEFT)
        scale = tk.Scale(
            row,
            from_=minimum,
            to=maximum,
            orient=tk.HORIZONTAL,
            variable=var,
            length=120,
            showvalue=True,
            resolution=1,
        )
        scale.pack(side=tk.LEFT, padx=6)

    def _on_close(self) -> None:
        """Handle window close event.

        @return None
        """
        self._stop_camera()
        self.root.quit()
        self.root.destroy()


def launch_gui() -> None:
    """Launch the Tkinter GUI.

    @return None
    """
    app = NetInspectorGUI()
    app.root.mainloop()


if __name__ == "__main__":
    launch_gui()
