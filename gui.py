"""
GeoGrain Analyzer Professional v9.2
Scientific Dashboard + Modern Hover UI + GIS Layer System
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import threading
import cv2
import os
import numpy as np
import pandas as pd
from tkinter import simpledialog


from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from modules.sam_detector import detect_grains
from modules.report_generator import (
    create_dcurve,
    create_log_curve,
    create_frequency_curve   # ✅ ADDED
)
from modules.uav_scale import get_pixel_to_mm


# =====================================================
# BUTTON
# =====================================================
def create_button(parent, text, command, bg, hover_bg):

    btn = tk.Button(
        parent,
        text=text,
        command=command,
        bg=bg,
        fg="white",
        activebackground=hover_bg,
        relief="flat",
        width=14
    )

    def on_enter(e):
        btn["bg"] = hover_bg

    def on_leave(e):
        btn["bg"] = bg

    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)

    return btn


# =====================================================
# MAIN GUI
# =====================================================
class GeoGrainGUI:

    def __init__(self, root):

        self.root = root
        self.root.title("GeoGrain Analyzer Professional v9.2")
        self.root.geometry("1450x880")
        self.root.configure(bg="#1E1E1E")

        self.image_path = None
        self.pixel_to_mm = 12.0
        self.result = None

        self.layers = []
        self.active_layer_index = None

        # GIS
        self.layer_vars = []
        self.layer_checks = []

        self.build_ui()

    # =====================================================
    # BUILD UI
    # =====================================================
    def build_ui(self):

        # MENU
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Layers", menu=file_menu)
        file_menu.add_command(label="Add Image", command=self.load_image)
        file_menu.add_command(label="Remove Image", command=self.remove_layer)
        file_menu.add_command(label="Clear All", command=self.clear_workspace)

        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)

        analysis_menu.add_command(
            label="Number-Weighted Grain Size Distribution",
            command=self.show_frequency_curve
        )
        analysis_menu.add_command(
            label="Grain Size Percentile Distribution",
            command=self.show_dcurve
        )
        analysis_menu.add_command(
            label="Volume-Weighted Grain Size Distribution",
            command=self.show_log_curve
        )

        results_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Results", menu=results_menu)
        results_menu.add_command(label="Open CSV", command=self.open_csv)
        results_menu.add_command(label="Open PDF", command=self.open_pdf)

        # HEADER
        header = tk.Frame(self.root, bg="#252526")
        header.pack(fill=tk.X)

        create_button(header, "Select Image", self.load_image,
                      "#2979FF", "#5393FF").pack(side=tk.LEFT, padx=5, pady=8)

        create_button(header, "Remove Image", self.remove_layer,
                      "#D32F2F", "#FF6659").pack(side=tk.LEFT, padx=5)

        create_button(header, "Clear All", self.clear_workspace,
                      "#FF9800", "#FFB74D").pack(side=tk.LEFT, padx=5)

        create_button(header, "Run Detection", self.run_detection,
                      "#00C853", "#33D975").pack(side=tk.LEFT, padx=5)
        
        create_button(header, "Calibrate Scale", self.start_calibration,
              "#9C27B0", "#BA68C8").pack(side=tk.LEFT, padx=5)

        # MAIN
        main_frame = tk.Frame(self.root, bg="#1E1E1E")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # LAYERS
        layer_panel = tk.Frame(main_frame, bg="#2A2A2A", width=220)
        layer_panel.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(layer_panel, text="Layers",
                 bg="#2A2A2A", fg="#00C853",
                 font=("Arial", 12, "bold")).pack(pady=8)

        self.layer_frame = tk.Frame(layer_panel, bg="#2A2A2A")
        self.layer_frame.pack(fill=tk.BOTH, expand=True)

        # IMAGE
        self.image_panel = tk.Frame(main_frame, bg="#1E1E1E")
        self.image_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(self.image_panel, bg="#1E1E1E")
        self.image_label.pack(padx=10, pady=10)

        # RIGHT PANEL
        right_panel = tk.Frame(main_frame, bg="#252526", width=420)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(right_panel, text="Morphological Statistics",
                 bg="#252526", fg="#00C853",
                 font=("Arial", 14, "bold")).pack(pady=12)

        self.stats_text = tk.Label(
            right_panel, text="No data",
            bg="#252526", fg="#E0E0E0",
            justify=tk.LEFT, font=("Consolas", 10)
        )
        self.stats_text.pack(pady=5)

        tk.Label(right_panel, text="Graph View",
                 bg="#252526", fg="#00C853",
                 font=("Arial", 13, "bold")).pack(pady=15)

        self.graph_frame = tk.Frame(right_panel, bg="#252526")
        self.graph_frame.pack(fill=tk.BOTH, expand=True)

        # STATUS
        self.status = tk.Label(
            self.root,
            text="Ready",
            anchor="w",
            bg="#111111",
            fg="#E0E0E0"
        )
        self.status.pack(fill=tk.X, side=tk.BOTTOM)

        # PROGRESS
        self.progress = ttk.Progressbar(
            self.root,
            orient="horizontal",
            mode="indeterminate"
        )
        self.progress.pack(fill=tk.X, side=tk.BOTTOM)

    # =====================================================
    # LOAD IMAGE
    # =====================================================
    def load_image(self):

        paths = filedialog.askopenfilenames(
            filetypes=[("Images", "*.jpg *.png *.tif *.jpeg")]
        )

        if not paths:
            return

        for path in paths:
            name = os.path.basename(path)
            self.layers.append((name, path))

            var = tk.BooleanVar(value=True)
            self.layer_vars.append(var)

            chk = tk.Checkbutton(
                self.layer_frame,
                text=name,
                variable=var,
                bg="#2A2A2A",
                fg="white",
                selectcolor="#444444",
                anchor="w",
                command=self.update_layer_visibility
            )
            chk.pack(fill=tk.X)

            self.layer_checks.append(chk)

        self.update_layer_visibility()

    def update_layer_visibility(self):

        visible = [
            path for (name, path), var in zip(self.layers, self.layer_vars)
            if var.get()
        ]

        if not visible:
            self.image_label.config(image="")
            self.image_label.image = None
            return

        top = visible[-1]
        self.image_path = top

        image = cv2.imread(top)
        self.display_image(image)

    def remove_layer(self):

        if not self.layers:
            return

        self.layers.pop(-1)
        self.layer_checks[-1].destroy()
        self.layer_checks.pop()
        self.layer_vars.pop()

        self.update_layer_visibility()

    def clear_workspace(self):

        self.layers.clear()

        for chk in self.layer_checks:
            chk.destroy()

        self.layer_checks.clear()
        self.layer_vars.clear()

        self.image_label.config(image="")
        self.image_label.image = None

        self.clear_graph()
        self.stats_text.config(text="No data")

        self.status.config(text="Workspace cleared")

    # =====================================================
    # DETECTION
    # =====================================================
    def run_detection(self):

        if not self.image_path:
            messagebox.showerror("Error", "Please select image first")
            return

        self.status.config(text="Detecting...")
        self.progress.start(10)

        threading.Thread(target=self.process_detection, daemon=True).start()

    def process_detection(self):

        try:
            result = detect_grains(
                self.image_path,
                self.pixel_to_mm
            )

            self.root.after(0, lambda: self.finish_detection(result))

        except Exception as e:
            self.progress.stop()
            messagebox.showerror("Error", str(e))

    def finish_detection(self, result):

        self.progress.stop()

        self.result = result

        self.grain_sizes = result["grain_sizes"]
        self.grain_volumes = result.get("grain_volumes", None)

        self.display_image(result["overlay"])
        self.update_stats()

        self.status.config(text=f"Done | Rocks: {result['stats']['count']}")

    # =====================================================
    # DISPLAY
    # =====================================================
    def display_image(self, image):

        if image is None:
            return

        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            pass

        image = Image.fromarray(image)
        image = image.resize((920, 620))

        image_tk = ImageTk.PhotoImage(image)
        self.image_label.config(image=image_tk)
        self.image_label.image = image_tk

    # =====================================================
    # STATS
    # =====================================================
    def update_stats(self):

        if not self.result:
            return

        stats = self.result["stats"]

        df = pd.read_csv(self.result["csv_path"])

        mean_aspect = df["Aspect_Ratio_a_by_b"].mean()
        elongated_percent = (df["Aspect_Ratio_a_by_b"] > 1.5).mean() * 100

        mean_a = df["a_axis_mm"].mean()
        mean_b = df["b_axis_mm"].mean()
        mean_orientation = df["Orientation_angle_deg"].mean()

        text = f"""
Total Grains        : {stats['count']}
Mean Size (mm)      : {stats['mean']:.2f}
D50 (mm)            : {stats['d50']:.2f}

Mean a-axis (mm)    : {mean_a:.2f}
Mean b-axis (mm)    : {mean_b:.2f}
Mean Aspect Ratio   : {mean_aspect:.3f}
Elongated (>1.5)    : {elongated_percent:.1f} %

Mean Orientation    : {mean_orientation:.1f}°
"""

        self.stats_text.config(text=text)

    # =====================================================
    # GRAPHS
    # =====================================================
    def clear_graph(self):
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

    def show_frequency_curve(self):
        if not self.result:
            return
        self.clear_graph()
        fig = create_frequency_curve(self.grain_sizes)
        FigureCanvasTkAgg(fig, self.graph_frame).get_tk_widget().pack()

    def show_dcurve(self):
        if not self.result:
            return
        self.clear_graph()
        fig = create_dcurve(self.result["stats"])
        FigureCanvasTkAgg(fig, self.graph_frame).get_tk_widget().pack()

    def show_log_curve(self):
        if not self.result:
            return
        self.clear_graph()

        if self.grain_volumes is not None:
            fig = create_log_curve(self.grain_sizes, self.grain_volumes)
        else:
            fig = create_log_curve(self.grain_sizes)

        FigureCanvasTkAgg(fig, self.graph_frame).get_tk_widget().pack()

    def show_overlay(self):
        if not self.result:
            return
        overlay = cv2.imread(self.result["overlay_path"])
        self.display_image(overlay)

    def open_csv(self):
        if self.result:
            os.startfile(self.result["csv_path"])

    def open_pdf(self):
        if self.result:
            os.startfile(self.result["pdf_path"])

    # =====================================================
    # CALIBRATION (ADDED ONLY)
    # =====================================================

    def start_calibration(self):

        if not self.image_path:
            messagebox.showerror("Error", "Load image first")
            return

        self.calibration_points = []
        self.calibrating = True

        self.image_label.bind("<Button-1>", self.capture_point)

        self.status.config(text="Click 2 points for calibration")

    def capture_point(self, event):

        if not hasattr(self, "calibrating") or not self.calibrating:
            return

        self.calibration_points.append((event.x, event.y))

        if len(self.calibration_points) == 2:

            self.image_label.unbind("<Button-1>")
            self.calibrating = False

            (x1, y1), (x2, y2) = self.calibration_points

            pixel_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            if pixel_distance == 0:
                messagebox.showerror("Error", "Invalid points selected")
                return

            real_distance = simpledialog.askfloat(
                "Input",
                "Enter real-world distance (mm):"
            )

            if not real_distance:
                return

            self.pixel_to_mm = real_distance / pixel_distance

            self.status.config(
                text=f"Scale set: {self.pixel_to_mm:.4f} mm/pixel"
            )