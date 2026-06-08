import json
import os
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageTk
    from pupil_apriltags import Detector
except Exception as exc:
    print("Missing dependency:", exc)
    raise

APP_VERSION = "1.0.0"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
ROBUST_INPAINT_METHOD = "Shift-Map (robust)"
ARUCO_APRILTAG_FAMILIES = {
    "tag16h5": "DICT_APRILTAG_16h5",
    "tag25h9": "DICT_APRILTAG_25h9",
    "tag36h11": "DICT_APRILTAG_36h11",
}
APRILTAG_FAMILY_MAX_ID = {
    "tag16h5": 29,
    "tag25h9": 34,
    "tag36h11": 586,
}
SCAN_INDEX_FILENAME = "scan_index.json"


class Tooltip:
    """Simple tooltip that appears below a widget on hover."""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind("<Enter>", self._on_enter, add="+")
        widget.bind("<Leave>", self._on_leave, add="+")

    def _on_enter(self, _event=None):
        if self.tip_window or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw, text=self.text, justify="left",
            background="#ffffe0", relief="solid", borderwidth=1,
            font=("Segoe UI", 9), wraplength=340,
        )
        label.pack(ipadx=4, ipady=2)

    def _on_leave(self, _event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


class AprilTagCleanerApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"AprilTag Cleaner  v{APP_VERSION}")
        self.root.geometry("1320x860")
        self.root.minsize(1100, 760)

        self.detector = Detector(families="tag36h11 tag25h9 tag16h5 tagStandard41h12 tagStandard52h13", quad_decimate=1.0, nthreads=min(4, os.cpu_count() or 1))
        self.opencv_detectors = {}
        self.input_files = []
        self.results = []
        self.current_index = -1
        self.preview_photo = None
        self.preview_transform = None
        self.preview_cache_index = None
        self.preview_source_image = None
        self.preview_zoom = 1.0
        self.preview_pan_x = 0.0
        self.preview_pan_y = 0.0
        self.preview_drag_start = None
        self.cancel_requested = False
        self.worker_running = False
        self.ignore_file_selection = False
        self.edit_mode = None
        self.edit_points = []
        self.last_edit_mode = None          # last used polygon mode ("add" or "erase")
        self.mask_dirty = False             # True when current mask has unsaved manual edits
        self._right_pan_start = None        # for right-click pan tracking
        self._right_drag_occurred = False   # distinguish right-click vs right-drag
        self._scan_index_lock = threading.Lock()
        self._rect_drag_start = None         # for left-click rectangle drawing
        self._rect_drag_current = None
        self._rect_dragging = False

        self.family_var = tk.StringVar(value="tag36h11")       # default: tag36h11
        self.quad_decimate_var = tk.DoubleVar(value=1.0)
        self.inpaint_method_var = tk.StringVar(value="Navier-Stokes")
        self.radius_var = tk.DoubleVar(value=12.0)             # default: 12
        self.expansion_pct_var = tk.DoubleVar(value=45.0)
        self.save_overlays_var = tk.BooleanVar(value=False)
        self.output_folder_var = tk.StringVar(value="")
        self.rename_sequential_var = tk.BooleanVar(value=True)
        self.reduce_res_var = tk.BooleanVar(value=False)
        self.reduce_res_scale_var = tk.IntVar(value=50)
        self.status_var = tk.StringVar(value="Select images, run a scan to build masks, review them, then apply cleanup.")
        self.preview_hint_var = tk.StringVar(value="Run a scan to review masks. Use Left/Right arrows to move between images.")
        self.preview_index_var = tk.StringVar(value="No scan results yet.")
        self.preview_zoom_var = tk.StringVar(value="1.0x")

        self._build_ui()
        self.root.bind("<Left>", self.on_left_key)
        self.root.bind("<Right>", self.on_right_key)
        self.root.bind("<Escape>", self.cancel_polygon)
        self.root.bind("<m>", self.on_key_add_mask)
        self.root.bind("<M>", self.on_key_add_mask)
        self.root.bind("<e>", self.on_key_erase_mask)
        self.root.bind("<E>", self.on_key_erase_mask)
        self.root.bind("<Return>", self.on_key_enter)
        self.update_controls()

    def _build_ui(self):
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, padding=12)
        left.grid(row=0, column=0, sticky="nsw")
        right = ttk.Frame(self.root, padding=(0, 12, 12, 12))
        right.grid(row=0, column=1, sticky="nsew")

        for i in range(30):
            left.rowconfigure(i, weight=0)
        left.columnconfigure(0, weight=1)

        # ── Print Tags PDF (collapsible) ───────────────────────────────────
        self._pdf_section_open = False
        self._pdf_toggle_btn = ttk.Button(
            left, text="▶  Print Tags PDF",
            command=self._toggle_pdf_section,
        )
        self._pdf_toggle_btn.grid(row=0, column=0, sticky="ew", pady=(0, 4))

        self._pdf_frame = ttk.Frame(left, relief="groove", borderwidth=1, padding=(8, 6))
        # not grid-ed yet – shown/hidden by toggle

        pdf_inner = self._pdf_frame
        pdf_inner.columnconfigure(1, weight=1)

        ttk.Label(pdf_inner, text="Tag family:").grid(row=0, column=0, sticky="w", pady=2)
        self.pdf_family_var = tk.StringVar(value="tag36h11")
        family_cb = ttk.Combobox(pdf_inner, textvariable=self.pdf_family_var,
                                  values=list(ARUCO_APRILTAG_FAMILIES.keys()),
                                  state="readonly", width=11)
        family_cb.grid(row=0, column=1, sticky="w", padx=(6, 0))
        family_cb.bind("<<ComboboxSelected>>", self._on_pdf_family_changed)

        ttk.Label(pdf_inner, text="Tag IDs – from:").grid(row=1, column=0, sticky="w", pady=2)
        id_frame = ttk.Frame(pdf_inner)
        id_frame.grid(row=1, column=1, sticky="ew", padx=(6, 0))
        self.pdf_id_from_var = tk.IntVar(value=0)
        self.pdf_id_to_var = tk.IntVar(value=19)
        ttk.Spinbox(id_frame, from_=0, to=586, increment=1, textvariable=self.pdf_id_from_var, width=6).grid(row=0, column=0)
        ttk.Label(id_frame, text="  to:").grid(row=0, column=1)
        ttk.Spinbox(id_frame, from_=0, to=586, increment=1, textvariable=self.pdf_id_to_var, width=6).grid(row=0, column=2)

        ttk.Label(pdf_inner, text="Tag size (mm):").grid(row=2, column=0, sticky="w", pady=2)
        self.pdf_tag_size_var = tk.IntVar(value=50)
        ttk.Spinbox(pdf_inner, from_=10, to=200, increment=5, textvariable=self.pdf_tag_size_var, width=8).grid(row=2, column=1, sticky="w", padx=(6, 0))

        ttk.Label(pdf_inner, text="Page size:").grid(row=3, column=0, sticky="w", pady=2)
        self.pdf_page_var = tk.StringVar(value="A4")
        ttk.Combobox(pdf_inner, textvariable=self.pdf_page_var, values=["A4", "Letter"], state="readonly", width=9).grid(row=3, column=1, sticky="w", padx=(6, 0))

        ttk.Label(pdf_inner, text="Tag border (mm):").grid(row=4, column=0, sticky="w", pady=2)
        self.pdf_margin_var = tk.IntVar(value=5)
        ttk.Spinbox(pdf_inner, from_=0, to=50, increment=1, textvariable=self.pdf_margin_var, width=8).grid(row=4, column=1, sticky="w", padx=(6, 0))

        self.pdf_label_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(pdf_inner, text="Print tag ID label in border", variable=self.pdf_label_var).grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))

        ttk.Label(pdf_inner, text="Label size (% of border):").grid(row=6, column=0, sticky="w", pady=2)
        self.pdf_label_size_var = tk.IntVar(value=85)
        ttk.Spinbox(pdf_inner, from_=10, to=200, increment=5, textvariable=self.pdf_label_size_var, width=8).grid(row=6, column=1, sticky="w", padx=(6, 0))

        btn_gen_pdf = ttk.Button(pdf_inner, text="Generate PDF…", command=self._generate_apriltag_pdf)
        btn_gen_pdf.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        Tooltip(btn_gen_pdf, "Generate a printable PDF with the selected range of tag36h11 AprilTags at the specified physical size.")

        # ── Inputs ────────────────────────────────────────────────────────
        ttk.Label(left, text="Inputs", font=("Segoe UI", 12, "bold")).grid(row=2, column=0, sticky="w")

        btn_add_images = ttk.Button(left, text="Add images", command=self.select_files)
        btn_add_images.grid(row=3, column=0, sticky="ew", pady=(8, 4))
        Tooltip(btn_add_images, "Add one or more image files to the input list.")

        btn_add_folder = ttk.Button(left, text="Add folder", command=self.select_folder)
        btn_add_folder.grid(row=4, column=0, sticky="ew", pady=4)
        Tooltip(btn_add_folder, "Add all images from a selected folder to the input list.")

        btn_clear = ttk.Button(left, text="Clear list", command=self.clear_files)
        btn_clear.grid(row=5, column=0, sticky="ew", pady=4)
        Tooltip(btn_clear, "Remove all images from the input list and reset the session.")

        self.file_list = tk.Listbox(left, height=14, exportselection=False)
        self.file_list.grid(row=6, column=0, sticky="nsew", pady=(8, 8))
        self.file_list.bind("<<ListboxSelect>>", self.on_file_select)
        left.rowconfigure(6, weight=1)

        # ── Output (above Detection workflow) ─────────────────────────────
        ttk.Label(left, text="Output", font=("Segoe UI", 12, "bold")).grid(row=7, column=0, sticky="w", pady=(4, 0))

        out_folder_frame = ttk.Frame(left)
        out_folder_frame.grid(row=8, column=0, sticky="ew", pady=(4, 0))
        out_folder_frame.columnconfigure(0, weight=1)
        ttk.Label(out_folder_frame, text="Output folder (blank = auto subfolder):").grid(row=0, column=0, columnspan=2, sticky="w")
        self.out_folder_entry = ttk.Entry(out_folder_frame, textvariable=self.output_folder_var)
        self.out_folder_entry.grid(row=1, column=0, sticky="ew")
        btn_browse = ttk.Button(out_folder_frame, text="Browse\u2026", command=self.browse_output_folder)
        btn_browse.grid(row=1, column=1, padx=(4, 0))
        Tooltip(btn_browse, "Select a custom output folder for cleaned images and masks.")

        ttk.Checkbutton(left, text="Sequential renaming (00000, 00001\u2026)", variable=self.rename_sequential_var).grid(row=9, column=0, sticky="w", pady=(6, 0))

        reduce_frame = ttk.Frame(left)
        reduce_frame.grid(row=10, column=0, sticky="ew", pady=(4, 0))
        ttk.Checkbutton(reduce_frame, text="Reduce output resolution to", variable=self.reduce_res_var, command=self.on_reduce_res_change).grid(row=0, column=0, sticky="w")
        self.res_scale_spin = ttk.Spinbox(reduce_frame, from_=10, to=90, increment=5, textvariable=self.reduce_res_scale_var, width=5)
        self.res_scale_spin.grid(row=0, column=1, padx=(6, 2))
        ttk.Label(reduce_frame, text="% of original").grid(row=0, column=2, sticky="w")

        btn_open_out = ttk.Button(left, text="Open output folder", command=self.open_output_folder)
        btn_open_out.grid(row=11, column=0, sticky="ew", pady=(8, 4))
        Tooltip(btn_open_out, "Open the output folder in Windows Explorer.")

        # ── Detection workflow ─────────────────────────────────────────────
        ttk.Label(left, text="Detection workflow", font=("Segoe UI", 12, "bold")).grid(row=12, column=0, sticky="w", pady=(8, 0))

        params = ttk.Frame(left)
        params.grid(row=13, column=0, sticky="ew", pady=(8, 0))
        params.columnconfigure(1, weight=1)

        ttk.Label(params, text="Family preset").grid(row=0, column=0, sticky="w")
        fam = ttk.Combobox(params, textvariable=self.family_var, state="readonly", values=[
            "tag36h11", "tag25h9", "tag16h5", "tagStandard41h12", "tagStandard52h13"
        ])
        fam.grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=2)

        ttk.Label(params, text="Inpaint method").grid(row=1, column=0, sticky="w")
        meth = ttk.Combobox(params, textvariable=self.inpaint_method_var, state="readonly", values=[ROBUST_INPAINT_METHOD, "Telea", "Navier-Stokes"])
        meth.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=2)
        meth.bind("<<ComboboxSelected>>", self.on_inpaint_method_change)

        ttk.Label(params, text="Radius (Telea/NS only)").grid(row=2, column=0, sticky="w")
        self.radius_spin = ttk.Spinbox(params, from_=1, to=50, increment=1, textvariable=self.radius_var)
        self.radius_spin.grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=2)

        ttk.Label(params, text="Mask expansion (%)").grid(row=3, column=0, sticky="w")
        ttk.Spinbox(params, from_=0, to=200, increment=1, textvariable=self.expansion_pct_var).grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=2)

        ttk.Label(params, text="Quad decimate (1=best)").grid(row=4, column=0, sticky="w")
        ttk.Spinbox(params, from_=1.0, to=4.0, increment=0.5, textvariable=self.quad_decimate_var, format="%.1f").grid(row=4, column=1, sticky="ew", padx=(8, 0), pady=2)

        cb_overlays = ttk.Checkbutton(left, text="Export overlays on cleanup", variable=self.save_overlays_var)
        cb_overlays.grid(row=14, column=0, sticky="w", pady=(8, 0))
        Tooltip(cb_overlays, "Save side-by-side overlay images (original with mask highlighted) to an 'overlays' subfolder when running Cleanup tags.")

        self.scan_button = ttk.Button(left, text="Scan listed images", command=self.start_scan)
        self.scan_button.grid(row=16, column=0, sticky="ew", pady=(10, 4))
        Tooltip(self.scan_button, "Detect AprilTags in all listed images and build masks.\nIf cached masks are found in the output folder they will be used automatically.")

        self.apply_cleanup_button = ttk.Button(left, text="Cleanup tags from all images", command=self.start_cleanup)
        self.apply_cleanup_button.grid(row=17, column=0, sticky="ew", pady=4)
        Tooltip(self.apply_cleanup_button, "Apply inpainting cleanup to all images using the current masks and save results to the output folder.")

        self.cancel_button = ttk.Button(left, text="Cancel", command=self.cancel_processing)
        self.cancel_button.grid(row=18, column=0, sticky="ew", pady=4)
        Tooltip(self.cancel_button, "Stop the current processing operation after the current image finishes.")

        self.progress = ttk.Progressbar(left, mode="determinate")
        self.progress.grid(row=19, column=0, sticky="ew", pady=(12, 4))
        ttk.Label(left, textvariable=self.status_var, wraplength=300, foreground="#444").grid(row=20, column=0, sticky="ew", pady=(4, 0))

        # ── Right panel ────────────────────────────────────────────────────
        right.columnconfigure(0, weight=1)
        right.rowconfigure(5, weight=1)
        right.rowconfigure(8, weight=1)

        ttk.Label(right, text="Preview", font=("Segoe UI", 12, "bold")).grid(row=0, column=0, sticky="w")

        nav_frame = ttk.Frame(right)
        nav_frame.grid(row=1, column=0, sticky="ew", pady=(8, 4))
        nav_frame.columnconfigure(1, weight=1)
        self.prev_button = ttk.Button(nav_frame, text="Previous", command=self.show_previous_result)
        self.prev_button.grid(row=0, column=0, sticky="w")
        Tooltip(self.prev_button, "Show previous image.  (← Arrow key)")
        ttk.Label(nav_frame, textvariable=self.preview_index_var, anchor="center").grid(row=0, column=1, sticky="ew", padx=12)
        self.next_button = ttk.Button(nav_frame, text="Next", command=self.show_next_result)
        self.next_button.grid(row=0, column=2, sticky="e")
        Tooltip(self.next_button, "Show next image.  (→ Arrow key)")

        edit_frame = ttk.Frame(right)
        edit_frame.grid(row=2, column=0, sticky="ew", pady=(0, 4))

        self.add_mask_button = ttk.Button(edit_frame, text="Add new mask", command=lambda: self.start_polygon_mode("add"))
        self.add_mask_button.grid(row=0, column=0, sticky="w")
        Tooltip(self.add_mask_button, "Draw a polygon to add a masked region.  (M)\nLeft-click to place points, then right-click or Enter to apply.")

        self.erase_mask_button = ttk.Button(edit_frame, text="Erase existing mask", command=lambda: self.start_polygon_mode("erase"))
        self.erase_mask_button.grid(row=0, column=1, sticky="w", padx=(6, 0))
        Tooltip(self.erase_mask_button, "Draw a polygon to erase a masked region.  (E)\nLeft-click to place points, then right-click or Enter to apply.")

        self.apply_polygon_button = ttk.Button(edit_frame, text="Apply mask", command=self.apply_polygon_edit)
        self.apply_polygon_button.grid(row=0, column=2, sticky="w", padx=(6, 0))
        Tooltip(self.apply_polygon_button, "Apply the drawn polygon to the mask.  (Enter  or  Right-click on canvas)")

        ttk.Label(edit_frame, text="Zoom").grid(row=0, column=4, sticky="w", padx=(18, 6))
        self.zoom_out_button = ttk.Button(edit_frame, text="-", width=3, command=self.zoom_out)
        self.zoom_out_button.grid(row=0, column=5, sticky="w")
        Tooltip(self.zoom_out_button, "Zoom out.  (Mouse wheel ↓)")

        ttk.Label(edit_frame, textvariable=self.preview_zoom_var, width=6, anchor="center").grid(row=0, column=6, sticky="w", padx=(6, 6))

        self.zoom_in_button = ttk.Button(edit_frame, text="+", width=3, command=self.zoom_in)
        self.zoom_in_button.grid(row=0, column=7, sticky="w")
        Tooltip(self.zoom_in_button, "Zoom in.  (Mouse wheel ↑)")

        self.zoom_fit_button = ttk.Button(edit_frame, text="Fit", command=self.reset_preview_view)
        self.zoom_fit_button.grid(row=0, column=8, sticky="w", padx=(6, 0))
        Tooltip(self.zoom_fit_button, "Fit image to view.  (Double-click middle mouse button)")

        ttk.Label(right, text="\u2713 Masks are auto-saved in the background whenever you switch image.", foreground="#666").grid(row=3, column=0, sticky="w", pady=(0, 2))

        ttk.Label(right, textvariable=self.preview_hint_var, wraplength=880, foreground="#444").grid(row=4, column=0, sticky="ew", pady=(0, 4))

        self.preview_canvas = tk.Canvas(right, background="#111", highlightthickness=1, highlightbackground="#666", width=900, height=520)
        self.preview_canvas.grid(row=5, column=0, sticky="nsew", pady=(0, 4))
        self.preview_canvas.bind("<ButtonPress-1>", self.on_preview_press)
        self.preview_canvas.bind("<B1-Motion>", self.on_preview_b1_motion)
        self.preview_canvas.bind("<ButtonRelease-1>", self.on_preview_b1_release)
        self.preview_canvas.bind("<Double-Button-1>", self.on_preview_complete_polygon)
        self.preview_canvas.bind("<Configure>", self.on_preview_resize)
        self.preview_canvas.bind("<MouseWheel>", self.on_preview_mousewheel)
        self.preview_canvas.bind("<Button-4>", self.on_preview_mousewheel)
        self.preview_canvas.bind("<Button-5>", self.on_preview_mousewheel)
        # Shift+left drag = pan (legacy)
        self.preview_canvas.bind("<Shift-ButtonPress-1>", self.start_preview_pan)
        self.preview_canvas.bind("<Shift-B1-Motion>", self.on_preview_pan)
        self.preview_canvas.bind("<Shift-ButtonRelease-1>", self.end_preview_pan)
        # Right-click drag = pan; right-click (no drag) = apply polygon or recall last mode
        self.preview_canvas.bind("<ButtonPress-3>", self.on_right_button_press)
        self.preview_canvas.bind("<B3-Motion>", self.on_right_drag)
        self.preview_canvas.bind("<ButtonRelease-3>", self.on_right_button_release)
        # Double-click middle mouse button = zoom fit
        self.preview_canvas.bind("<Double-Button-2>", self.on_middle_double_click)

        self.cleanup_current_button = ttk.Button(right, text="Cleanup tags from current image", command=self.start_cleanup_current)
        self.cleanup_current_button.grid(row=6, column=0, sticky="w", pady=(4, 6))
        Tooltip(self.cleanup_current_button, "Apply inpainting and save only the currently displayed image. If the output file already exists you will be asked whether to overwrite it.")

        ttk.Label(right, text="Detected tags in current image", font=("Segoe UI", 11, "bold")).grid(row=7, column=0, sticky="w")
        cols = ("family", "id", "decision_margin", "center")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=8)
        for c, w in zip(cols, (120, 80, 120, 220)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="w")
        self.tree.grid(row=8, column=0, sticky="nsew", pady=(8, 0))

        self.on_inpaint_method_change()
        self.on_reduce_res_change()

    def build_detector(self):
        fam = self.family_var.get()
        families = fam
        qd = max(1.0, float(self.quad_decimate_var.get()))
        nthreads = min(4, os.cpu_count() or 1)
        self.detector = Detector(families=families, quad_decimate=qd, nthreads=nthreads)
        self.opencv_detectors = self.build_opencv_detectors(families, qd)

    def build_opencv_detectors(self, families, quad_decimate):
        if not hasattr(cv2, "aruco"):
            return {}

        if hasattr(cv2.aruco, "DetectorParameters"):
            create_parameters = cv2.aruco.DetectorParameters
        elif hasattr(cv2.aruco, "DetectorParameters_create"):
            create_parameters = cv2.aruco.DetectorParameters_create
        else:
            return {}

        detectors = {}
        for family in families.split():
            dictionary_name = ARUCO_APRILTAG_FAMILIES.get(family)
            if not dictionary_name or not hasattr(cv2.aruco, dictionary_name):
                continue

            params = create_parameters()
            params.adaptiveThreshWinSizeMin = 3
            params.adaptiveThreshWinSizeMax = 61
            params.adaptiveThreshWinSizeStep = 8
            params.minMarkerPerimeterRate = 0.01
            params.maxMarkerPerimeterRate = 6.0
            params.polygonalApproxAccuracyRate = 0.08
            params.minCornerDistanceRate = 0.02
            params.minDistanceToBorder = 0
            params.minOtsuStdDev = 2.0
            params.perspectiveRemovePixelPerCell = 8
            params.perspectiveRemoveIgnoredMarginPerCell = 0.05
            params.maxErroneousBitsInBorderRate = 0.5
            params.errorCorrectionRate = 0.8

            if hasattr(params, "cornerRefinementMethod") and hasattr(cv2.aruco, "CORNER_REFINE_APRILTAG"):
                params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
            elif hasattr(params, "cornerRefinementMethod") and hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
                params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

            if hasattr(params, "relativeCornerRefinmentWinSize"):
                params.relativeCornerRefinmentWinSize = 0.5
            if hasattr(params, "aprilTagQuadDecimate"):
                params.aprilTagQuadDecimate = quad_decimate
            if hasattr(params, "aprilTagCriticalRad"):
                params.aprilTagCriticalRad = 0.0
            if hasattr(params, "aprilTagMaxLineFitMse"):
                params.aprilTagMaxLineFitMse = 30.0
            if hasattr(params, "aprilTagMinClusterPixels"):
                params.aprilTagMinClusterPixels = 3
            if hasattr(params, "aprilTagMaxNmaxima"):
                params.aprilTagMaxNmaxima = 20
            if hasattr(params, "aprilTagMinWhiteBlackDiff"):
                params.aprilTagMinWhiteBlackDiff = 0
            if hasattr(params, "aprilTagDeglitch"):
                params.aprilTagDeglitch = 1
            if hasattr(params, "useAruco3Detection"):
                params.useAruco3Detection = False

            dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
            if hasattr(cv2.aruco, "ArucoDetector"):
                detectors[family] = cv2.aruco.ArucoDetector(dictionary, params)
            else:
                detectors[family] = (dictionary, params)

        return detectors

    def on_inpaint_method_change(self, _event=None):
        uses_radius = self.inpaint_method_var.get() != ROBUST_INPAINT_METHOD
        self.radius_spin.configure(state="normal" if uses_radius else "disabled")

    def _toggle_pdf_section(self):
        if self._pdf_section_open:
            self._pdf_frame.grid_remove()
            self._pdf_toggle_btn.configure(text="▶  Print Tags PDF")
            self._pdf_section_open = False
        else:
            self._pdf_frame.grid(row=1, column=0, sticky="ew", pady=(0, 4))
            self._pdf_toggle_btn.configure(text="▼  Print Tags PDF")
            self._pdf_section_open = True

    def _on_pdf_family_changed(self, event=None):
        fam = self.pdf_family_var.get()
        max_id = APRILTAG_FAMILY_MAX_ID.get(fam, 586)
        self.pdf_id_to_var.set(min(int(self.pdf_id_to_var.get()), max_id))
        self.pdf_id_from_var.set(min(int(self.pdf_id_from_var.get()), max_id))

    def _generate_apriltag_pdf(self):
        import tempfile
        try:
            from reportlab.lib.pagesizes import A4, letter
            from reportlab.lib.units import mm
            from reportlab.pdfgen import canvas as rl_canvas
        except ImportError:
            messagebox.showerror("Missing dependency", "reportlab is not installed.\nRun: pip install reportlab")
            return

        family = self.pdf_family_var.get()
        max_id = APRILTAG_FAMILY_MAX_ID.get(family, 586)
        id_from = int(self.pdf_id_from_var.get())
        id_to = int(self.pdf_id_to_var.get())
        if id_from > id_to:
            messagebox.showerror("Invalid range", '"From" ID must be ≤ "To" ID.')
            return
        if id_to > max_id:
            messagebox.showerror("Invalid range", f"{family} only has IDs 0–{max_id}.")
            return

        tag_size_mm = int(self.pdf_tag_size_var.get())
        margin_mm = int(self.pdf_margin_var.get())
        show_label = self.pdf_label_var.get()
        label_size_pct = self.pdf_label_size_var.get() / 100.0
        page_choice = self.pdf_page_var.get()
        page_size = A4 if page_choice == "A4" else letter

        # Check OpenCV aruco availability
        dict_attr = ARUCO_APRILTAG_FAMILIES.get(family, "")
        if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, dict_attr):
            messagebox.showerror("OpenCV", f"Your OpenCV build does not include aruco / {dict_attr}.\nInstall opencv-contrib-python.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save AprilTag PDF",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            initialfile=f"apriltags_{family}_{id_from}-{id_to}.pdf",
        )
        if not save_path:
            return

        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_attr))
        page_w, page_h = page_size

        tag_pt = tag_size_mm * mm          # black marker side length
        border_pt = margin_mm * mm         # white border around each tag
        cell_pt = tag_pt + 2 * border_pt   # full cell = tag + white border on all sides
        page_offset = 5 * mm               # fixed page-edge clearance
        row_h = cell_pt  # rows are flush so the adjacent cut marks share one line

        # Auto-compute columns so cells fill the page width
        cols = max(1, int((page_w - 2 * page_offset) / cell_pt))

        tag_px = 300  # render each tag at 300×300 px for crisp output

        # Cut-mark parameters (drawn outside the white border)
        cm_arm = 6    # arm length in pt (≈ 2.1 mm)
        cm_gap = 2    # gap between cell edge and start of line in pt

        def draw_cut_marks(canvas, tx, ty, size):
            """Corner cut marks at the outer edge of the white border (tx,ty = bottom-left)."""
            canvas.saveState()
            canvas.setStrokeColorRGB(0, 0, 0)
            canvas.setLineWidth(0.3)
            corners = [
                (tx,        ty),         # bottom-left
                (tx + size, ty),         # bottom-right
                (tx,        ty + size),  # top-left
                (tx + size, ty + size),  # top-right
            ]
            for cx, cy in corners:
                h_dir = -1 if cx == tx else 1
                canvas.line(cx + h_dir * cm_gap, cy,
                            cx + h_dir * (cm_gap + cm_arm), cy)
                v_dir = -1 if cy == ty else 1
                canvas.line(cx, cy + v_dir * cm_gap,
                            cx, cy + v_dir * (cm_gap + cm_arm))
            canvas.restoreState()

        tmp_dir = tempfile.mkdtemp()
        try:
            c = rl_canvas.Canvas(save_path, pagesize=page_size)
            tag_ids = list(range(id_from, id_to + 1))
            y_cursor = page_h - page_offset  # top of first row

            for i, tag_id in enumerate(tag_ids):
                col = i % cols
                if col == 0 and i != 0:
                    y_cursor -= row_h
                    if y_cursor - row_h < page_offset:
                        c.showPage()
                        y_cursor = page_h - page_offset

                img = cv2.aruco.generateImageMarker(dictionary, tag_id, tag_px)
                tmp_path = os.path.join(tmp_dir, f"tag_{tag_id}.png")
                cv2.imwrite(tmp_path, img)

                # x_cell / y_cell = outer corners of the white border
                x_cell = page_offset + col * cell_pt
                y_cell = y_cursor - cell_pt

                # draw black marker inset by border_pt inside the cell
                c.drawImage(tmp_path,
                            x_cell + border_pt, y_cell + border_pt,
                            width=tag_pt, height=tag_pt)

                # cut marks go around the full cell (white border edges)
                draw_cut_marks(c, x_cell, y_cell, cell_pt)

                if show_label and border_pt >= 4:
                    font_size = border_pt * label_size_pct
                    c.setFont("Helvetica", font_size)
                    label_y = y_cell + border_pt / 2 - font_size * 0.35
                    c.drawCentredString(x_cell + cell_pt / 2, label_y, f"{family}:{tag_id:03d}")

            c.save()
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

        messagebox.showinfo("PDF saved", f"PDF saved to:\n{save_path}")

    def browse_output_folder(self):
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self.output_folder_var.set(folder)

    def on_reduce_res_change(self):
        state = "normal" if self.reduce_res_var.get() else "disabled"
        self.res_scale_spin.configure(state=state)

    def focused_widget_consumes_arrows(self, widget):
        if widget is None:
            return False
        return widget.winfo_class() in {"Entry", "TEntry", "Spinbox", "TSpinbox", "TCombobox", "Text"}

    # ── Key handlers ──────────────────────────────────────────────────────

    def on_key_add_mask(self, event):
        if self.focused_widget_consumes_arrows(event.widget):
            return None
        if not self.worker_running and self.results:
            self.start_polygon_mode("add")
        return "break"

    def on_key_erase_mask(self, event):
        if self.focused_widget_consumes_arrows(event.widget):
            return None
        if not self.worker_running and self.results:
            self.start_polygon_mode("erase")
        return "break"

    def on_key_enter(self, event):
        if self.focused_widget_consumes_arrows(event.widget):
            return None
        if self.edit_mode in {"add", "erase"} and len(self.edit_points) >= 3:
            self.apply_polygon_edit()
            return "break"
        return None

    # ── Right-click canvas handlers ────────────────────────────────────────

    def on_right_button_press(self, event):
        """Start tracking right-click for pan vs single-click detection."""
        if not self.results or self.worker_running:
            return "break"
        self._right_pan_start = (event.x, event.y)
        self._right_drag_occurred = False
        self.preview_canvas.focus_set()
        return "break"

    def on_right_drag(self, event):
        """Pan while right mouse button is held and dragged."""
        if self._right_pan_start is None or self.current_index < 0:
            return "break"
        dx = event.x - self._right_pan_start[0]
        dy = event.y - self._right_pan_start[1]
        if abs(dx) + abs(dy) > 4:
            self._right_drag_occurred = True
        self._right_pan_start = (event.x, event.y)
        self.preview_pan_x += dx
        self.preview_pan_y += dy
        image = self.get_result_image(self.current_index)
        if image is not None:
            image_h, image_w = image.shape[:2]
            self.clamp_preview_pan(image_w, image_h)
        self.render_preview()
        return "break"

    def on_right_button_release(self, event):
        """End right-click: if no drag occurred treat as a click action."""
        was_drag = self._right_drag_occurred
        self._right_pan_start = None
        self._right_drag_occurred = False
        if was_drag:
            return "break"
        # Right-click without drag
        if self.edit_mode in {"add", "erase"}:
            if len(self.edit_points) >= 3:
                self.apply_polygon_edit()
        elif not self.worker_running and self.results and self.last_edit_mode:
            # Recall last polygon mode
            self.start_polygon_mode(self.last_edit_mode)
        return "break"

    def on_middle_double_click(self, _event=None):
        """Zoom fit on double-click of middle mouse button."""
        self.reset_preview_view()
        return "break"

    # ── Cache helpers ─────────────────────────────────────────────────────

    def _get_masks_dir(self, ref_path=None):
        """Return the masks cache subdirectory based on the current output folder setting."""
        custom_out = self.output_folder_var.get().strip()
        if custom_out:
            base_dir = Path(custom_out)
        elif ref_path:
            base_dir = Path(ref_path).parent / "apriltag_cleaned"
        elif self.input_files:
            base_dir = Path(self.input_files[0]).parent / "apriltag_cleaned"
        else:
            return None
        return base_dir / "masks"

    def _load_scan_index(self, masks_dir):
        """Load scan_index.json from masks_dir. Returns {} on any error."""
        index_path = Path(masks_dir) / SCAN_INDEX_FILENAME
        if not index_path.exists():
            return {}
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_scan_index(self, masks_dir, index):
        """Write scan_index.json atomically (write to .tmp then rename) to prevent corruption on crash."""
        index_path = Path(masks_dir) / SCAN_INDEX_FILENAME
        Path(masks_dir).mkdir(parents=True, exist_ok=True)
        tmp_path = index_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
        tmp_path.replace(index_path)

    def _check_scan_cache(self):
        """Return (index_dict, set_of_cached_input_paths) for the current input list."""
        if not self.input_files:
            return {}, set()
        masks_dir = self._get_masks_dir(self.input_files[0])
        if masks_dir is None:
            return {}, set()
        index = self._load_scan_index(masks_dir)
        cached_paths = set()
        for path in self.input_files:
            entry = index.get(str(path))
            if entry:
                mask_path = Path(entry.get("mask_path", ""))
                if mask_path.exists():
                    cached_paths.add(path)
        return index, cached_paths

    def _tags_to_json(self, tags):
        """Serialize a tag list to JSON-safe dicts including polygon corners."""
        result = []
        for t in tags:
            corners = t.get("corners", [])
            if hasattr(corners, "tolist"):
                corners = corners.tolist()
            expanded = t.get("expanded_corners", [])
            if hasattr(expanded, "tolist"):
                expanded = expanded.tolist()
            center = t.get("center", [0, 0])
            if not isinstance(center, list):
                center = list(center)
            result.append({
                "family": t.get("family", ""),
                "id": int(t.get("id", 0)),
                "decision_margin": float(t.get("decision_margin", 0.0)),
                "center": center,
                "corners": corners,
                "expanded_corners": expanded,
            })
        return result

    def _save_mask_to_cache(self, result):
        """Save a scan mask PNG and update scan_index.json. Called from background thread."""
        mask = result.get("mask")
        if mask is None:
            return
        masks_dir = self._get_masks_dir(result["path"])
        if masks_dir is None:
            return
        masks_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(result["path"]).stem
        mask_path = masks_dir / f"{stem}_scan_mask.png"
        cv2.imwrite(str(mask_path), mask)
        with self._scan_index_lock:
            index = self._load_scan_index(masks_dir)
            index[str(result["path"])] = {
                "mask_path": str(mask_path),
                "tags": self._tags_to_json(result.get("tags", [])),
            }
            self._save_scan_index(masks_dir, index)

    def _load_result_from_cache(self, path, file_index, cache_entry):
        """Reconstruct a scan result from a cache entry. Raises on failure."""
        mask_path = Path(cache_entry.get("mask_path", ""))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not load cached mask: {mask_path}")
        det_rows = []
        for t in cache_entry.get("tags", []):
            center = t.get("center", [0, 0])
            if isinstance(center, list):
                center = (float(center[0]), float(center[1]))
            det_rows.append({
                "family": t.get("family", ""),
                "id": int(t.get("id", 0)),
                "decision_margin": float(t.get("decision_margin", 0.0)),
                "center": center,
                "corners": t.get("corners", []),
                "expanded_corners": t.get("expanded_corners", []),
            })
        return {
            "path": path,
            "file_index": file_index,
            "mask": mask,
            "tags": det_rows,
            "tag_count": len(det_rows),
            "output_path": None,
            "overlay_path": None,
            "mask_path": None,
            "error": None,
            "save_error": None,
            "from_cache": True,
        }

    # ── Auto-save ─────────────────────────────────────────────────────────

    def _autosave_current_mask(self, idx):
        """Trigger a background save of the mask for results[idx]."""
        if idx < 0 or idx >= len(self.results):
            return
        result = self.results[idx]
        mask = result.get("mask")
        if mask is None:
            return
        data = {
            "path": result["path"],
            "tags": list(result.get("tags", [])),
            "file_index": result.get("file_index", 0),
        }
        threading.Thread(
            target=self._do_autosave_mask,
            args=(data, mask.copy()),
            daemon=True,
        ).start()

    def _do_autosave_mask(self, data, mask):
        """Background worker: write mask PNG and update the scan index."""
        try:
            masks_dir = self._get_masks_dir(data["path"])
            if masks_dir is None:
                return
            masks_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(data["path"]).stem
            mask_path = masks_dir / f"{stem}_scan_mask.png"
            cv2.imwrite(str(mask_path), mask)
            with self._scan_index_lock:
                index = self._load_scan_index(masks_dir)
                index[str(data["path"])] = {
                    "mask_path": str(mask_path),
                    "tags": self._tags_to_json(data["tags"]),
                }
                self._save_scan_index(masks_dir, index)
            self.root.after(0, lambda n=Path(data["path"]).name: self.status_var.set(
                f"Auto-saved mask for {n}."
            ))
        except Exception as exc:
            self.root.after(0, lambda msg=str(exc): self.status_var.set(
                f"Auto-save warning: {msg}"
            ))

    def build_detection_variants(self, gray):
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)
        sharpened = cv2.addWeighted(contrast, 1.35, cv2.GaussianBlur(contrast, (0, 0), 1.2), -0.35, 0)

        variants = [
            (gray, 1.0),
            (contrast, 1.0),
            (sharpened, 1.0),
        ]

        # Only upscale very small images where tags may be too tiny to detect
        if max(gray.shape[:2]) < 800:
            upscaled = cv2.resize(sharpened, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            variants.append((upscaled, 2.0))

        return variants

    def merge_detections(self, candidates):
        merged = []
        for candidate in candidates:
            candidate_center = np.array(candidate["center"], dtype=np.float32)
            replaced = False
            for idx, existing in enumerate(merged):
                if existing["family"] != candidate["family"] or existing["id"] != candidate["id"]:
                    continue

                existing_center = np.array(existing["center"], dtype=np.float32)
                distance = float(np.linalg.norm(existing_center - candidate_center))
                proximity = max(12.0, max(existing["size"], candidate["size"]) * 0.45)
                if distance > proximity:
                    continue

                if candidate["decision_margin"] > existing["decision_margin"]:
                    merged[idx] = candidate
                replaced = True
                break

            if not replaced:
                merged.append(candidate)

        merged.sort(key=lambda item: (item["family"], item["id"], item["center"][1], item["center"][0]))
        return merged

    def _extract_pupil_detections(self, detections, scale=1.0):
        candidates = []
        for det in detections:
            corners = np.array(det.corners, dtype=np.float32) / scale
            center = np.array(det.center, dtype=np.float32) / scale
            edge_vectors = np.roll(corners, -1, axis=0) - corners
            tag_size = float(np.linalg.norm(edge_vectors, axis=1).mean()) if len(edge_vectors) else 0.0
            family = det.tag_family.decode("utf-8") if isinstance(det.tag_family, (bytes, bytearray)) else str(det.tag_family)
            candidates.append({
                "family": family,
                "id": int(det.tag_id),
                "decision_margin": float(getattr(det, "decision_margin", 0.0)),
                "center": (float(center[0]), float(center[1])),
                "corners": corners,
                "size": tag_size,
            })
        return candidates

    def detect_tags(self, gray):
        # Fast path: try raw gray only — zero preprocessing cost.
        # If any tags are found, skip the expensive multi-variant pipeline entirely.
        candidates = self._extract_pupil_detections(self.detector.detect(gray))
        if candidates:
            return self.merge_detections(candidates)

        # Slow path: only reached when gray finds nothing.
        # Build enhanced variants (CLAHE, sharpened) and run full pipeline.
        variants = self.build_detection_variants(gray)
        for variant, scale in variants[1:]:  # variants[0] is raw gray, already tried
            candidates.extend(self._extract_pupil_detections(self.detector.detect(variant), scale))

        # Run OpenCV aruco on the contrast-enhanced variant
        opencv_variant, opencv_scale = variants[1] if len(variants) > 1 else variants[0]
        candidates.extend(self.detect_tags_with_opencv(opencv_variant, opencv_scale))

        return self.merge_detections(candidates)

    def detect_tags_with_opencv(self, image, scale):
        if not self.opencv_detectors:
            return []

        candidates = []
        for family, detector in self.opencv_detectors.items():
            if hasattr(detector, "detectMarkers"):
                corners_list, ids, _rejected = detector.detectMarkers(image)
            else:
                dictionary, params = detector
                corners_list, ids, _rejected = cv2.aruco.detectMarkers(image, dictionary, parameters=params)

            if ids is None or len(ids) == 0:
                continue

            for corners, tag_id in zip(corners_list, np.array(ids).reshape(-1)):
                corners = np.array(corners, dtype=np.float32).reshape(-1, 2) / scale
                center = corners.mean(axis=0)
                edge_vectors = np.roll(corners, -1, axis=0) - corners
                tag_size = float(np.linalg.norm(edge_vectors, axis=1).mean()) if len(edge_vectors) else 0.0
                candidates.append({
                    "family": family,
                    "id": int(tag_id),
                    "decision_margin": 0.0,
                    "center": (float(center[0]), float(center[1])),
                    "corners": corners,
                    "size": tag_size,
                })

        return candidates

    def expand_tag_corners(self, corners, expansion_pct, image_shape):
        expanded = corners.copy()
        if expansion_pct > 0:
            edge_vectors = np.roll(corners, -1, axis=0) - corners
            tag_size = float(np.linalg.norm(edge_vectors, axis=1).mean()) if len(edge_vectors) else 0.0
            pad_px = tag_size * (expansion_pct / 100.0)
            center = corners.mean(axis=0)
            grown = []
            for pt in corners:
                vec = pt - center
                norm = np.linalg.norm(vec)
                if norm < 1e-6:
                    grown.append(pt)
                else:
                    grown.append(pt + (vec / norm) * pad_px)
            expanded = np.array(grown, dtype=np.float32)

        expanded = np.rint(expanded).astype(np.int32)
        expanded[:, 0] = np.clip(expanded[:, 0], 0, image_shape[1] - 1)
        expanded[:, 1] = np.clip(expanded[:, 1], 0, image_shape[0] - 1)
        return expanded

    def apply_inpaint(self, image, mask):
        method_name = self.inpaint_method_var.get()
        if method_name == ROBUST_INPAINT_METHOD:
            if not hasattr(cv2, "xphoto") or not hasattr(cv2.xphoto, "inpaint"):
                raise RuntimeError("Shift-Map requires opencv-contrib-python. Reinstall dependencies from requirements.txt.")

            # xphoto uses 0 for missing pixels and non-zero for known pixels.
            valid_mask = np.where(mask > 0, 0, 255).astype(np.uint8)
            lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
            restored_lab = lab_image.copy()
            cv2.xphoto.inpaint(lab_image, valid_mask, restored_lab, cv2.xphoto.INPAINT_SHIFTMAP)
            return cv2.cvtColor(restored_lab, cv2.COLOR_Lab2BGR)

        method = cv2.INPAINT_TELEA if method_name == "Telea" else cv2.INPAINT_NS
        return cv2.inpaint(image, mask, float(self.radius_var.get()), method)

    def build_mask_overlay(self, image, mask):
        overlay = image.copy()
        if mask is not None and np.count_nonzero(mask) > 0:
            color_layer = image.copy()
            color_layer[mask > 0] = (0, 0, 255)
            overlay = cv2.addWeighted(image, 0.72, color_layer, 0.28, 0)
        return overlay

    def resolve_output_target(self, path, file_index):
        custom_out = self.output_folder_var.get().strip()
        out_dir = Path(custom_out) if custom_out else Path(path).parent / "apriltag_cleaned"
        out_dir.mkdir(exist_ok=True, parents=True)

        stem = Path(path).stem
        suffix = Path(path).suffix
        out_stem = f"{file_index:05d}" if self.rename_sequential_var.get() else stem
        return out_dir, out_stem, suffix

    def select_files(self):
        paths = filedialog.askopenfilenames(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp")])
        if paths:
            self.add_paths(paths)

    def select_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        paths = [str(p) for p in sorted(Path(folder).iterdir()) if p.suffix.lower() in IMAGE_EXTS]
        self.add_paths(paths)

    def add_paths(self, paths):
        existing = set(self.input_files)
        for p in paths:
            if p not in existing:
                self.input_files.append(p)
                self.file_list.insert(tk.END, p)
        self.status_var.set(f"Loaded {len(self.input_files)} image(s).")

    def reset_edit_state(self):
        self.edit_mode = None
        self.edit_points = []
        self._rect_drag_start = None
        self._rect_drag_current = None
        self._rect_dragging = False

    def update_preview_zoom_label(self):
        self.preview_zoom_var.set(f"{self.preview_zoom:.1f}x")

    def reset_preview_view(self, render=True):
        self.preview_zoom = 1.0
        self.preview_pan_x = 0.0
        self.preview_pan_y = 0.0
        self.preview_drag_start = None
        self.update_preview_zoom_label()
        if render and self.results:
            self.render_preview()

    def clear_preview_cache(self):
        self.preview_cache_index = None
        self.preview_source_image = None

    def clear_preview(self):
        self.preview_canvas.delete("all")
        self.preview_photo = None
        self.preview_transform = None
        self.preview_drag_start = None
        self.update_preview_zoom_label()
        self.preview_index_var.set("No scan results yet.")
        width = max(self.preview_canvas.winfo_width(), 900)
        height = max(self.preview_canvas.winfo_height(), 520)
        self.preview_canvas.create_text(width // 2, height // 2, text="No preview available", fill="#ddd")

    def has_pending_polygon(self):
        return bool(self.edit_points)

    def ensure_no_pending_polygon(self):
        if not self.has_pending_polygon():
            return True
        messagebox.showinfo("Polygon in progress", "Apply or cancel the current polygon before changing image or starting processing.")
        return False

    def update_controls(self):
        has_results = bool(self.results)
        is_editing = self.edit_mode in {"add", "erase"}

        has_current = has_results and 0 <= self.current_index < len(self.results)
        self.scan_button.configure(state="disabled" if self.worker_running else "normal")
        self.apply_cleanup_button.configure(state="normal" if has_results and not self.worker_running else "disabled")
        self.cleanup_current_button.configure(state="normal" if has_current and not self.worker_running else "disabled")
        self.cancel_button.configure(state="normal" if self.worker_running else "disabled")

        nav_state = "normal" if has_results and not self.worker_running else "disabled"
        self.prev_button.configure(state=nav_state)
        self.next_button.configure(state=nav_state)

        edit_state = "normal" if has_results and not self.worker_running else "disabled"
        self.add_mask_button.configure(state=edit_state, text="Add new mask [active]" if self.edit_mode == "add" else "Add new mask")
        self.erase_mask_button.configure(state=edit_state, text="Erase existing mask [active]" if self.edit_mode == "erase" else "Erase existing mask")
        self.apply_polygon_button.configure(state="normal" if is_editing and len(self.edit_points) >= 3 and not self.worker_running else "disabled")
        zoom_state = "normal" if has_results and not self.worker_running else "disabled"
        self.zoom_in_button.configure(state=zoom_state)
        self.zoom_out_button.configure(state=zoom_state)
        self.zoom_fit_button.configure(state=zoom_state)
        self.preview_canvas.configure(cursor="crosshair" if is_editing and not self.worker_running else "arrow")

        if self.worker_running:
            self.preview_hint_var.set("Processing in progress. Use Cancel to stop after the current image.")
        elif not has_results:
            self.preview_hint_var.set("Run a scan to review masks. Use Left/Right arrows to move between images.")
        elif self.edit_mode == "add":
            self.preview_hint_var.set("Add polygon mask  (M): click to place points → right-click or Enter to apply  |  click+drag → rectangle  |  Escape to cancel. Right-drag or Shift+drag to pan.")
        elif self.edit_mode == "erase":
            self.preview_hint_var.set("Erase mask area  (E): click to place points → right-click or Enter to apply  |  click+drag → rectangle  |  Escape to cancel. Right-drag or Shift+drag to pan.")
        else:
            self.preview_hint_var.set(
                "← → to navigate  |  M: add mask  |  E: erase mask  |  Right-click: recall last tool  |  Wheel: zoom  |  Right-drag / Shift+drag: pan  |  Double-middle-click: fit"
            )

    def clear_files(self):
        self.cancel_requested = False
        self.input_files.clear()
        self.results.clear()
        self.current_index = -1
        self.mask_dirty = False
        self.reset_edit_state()
        self.reset_preview_view(render=False)
        self.clear_preview_cache()
        self.file_list.delete(0, tk.END)
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.progress.configure(value=0)
        self.clear_preview()
        self.status_var.set("Input list cleared.")
        self.update_controls()

    def cancel_processing(self):
        if not self.worker_running:
            return
        self.cancel_requested = True
        self.status_var.set("Cancellation requested. Finishing current image...")

    def open_output_folder(self):
        custom_out = self.output_folder_var.get().strip()
        if custom_out:
            out_dir = Path(custom_out)
        elif self.input_files:
            out_dir = Path(self.input_files[0]).parent / "apriltag_cleaned"
        else:
            messagebox.showinfo("Output", "No output folder configured and no input images selected.")
            return
        out_dir.mkdir(exist_ok=True, parents=True)
        if sys.platform.startswith("win"):
            os.startfile(str(out_dir))
        else:
            messagebox.showinfo("Output folder", str(out_dir))

    def start_scan(self):
        if not self.input_files:
            messagebox.showwarning("No images", "Add some images or a folder first.")
            return
        if self.worker_running or not self.ensure_no_pending_polygon():
            return

        # Ask user what to do when cached masks already exist
        _, cached_set = self._check_scan_cache()
        use_cache = True
        if cached_set:
            answer = messagebox.askyesnocancel(
                "Existing masks found",
                f"Cached masks were found for {len(cached_set)} of {len(self.input_files)} image(s).\n\n"
                "Yes  \u2192  Load existing masks (skip re-scanning those images)\n"
                "No   \u2192  Recalculate and overwrite all masks\n"
                "Cancel  \u2192  Abort",
            )
            if answer is None:
                return
            use_cache = bool(answer)
        self._scan_use_cache = use_cache

        self.cancel_requested = False
        self.results = []
        self.current_index = -1
        self.mask_dirty = False
        self.reset_edit_state()
        self.reset_preview_view(render=False)
        self.clear_preview_cache()
        self.clear_preview()
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.build_detector()
        self.worker_running = True
        self.update_controls()
        self.status_var.set("Scanning images and building masks...")
        threading.Thread(target=self.scan_images, daemon=True).start()

    def scan_images(self):
        total = len(self.input_files)
        self.root.after(0, lambda: self.progress.configure(maximum=total, value=0))

        # Load cache; user already chose whether to use it in start_scan
        cache_index, cached_set = self._check_scan_cache()
        if not getattr(self, '_scan_use_cache', True):
            cached_set = set()  # user chose to recalculate all
        cached_count = 0

        for idx, path in enumerate(self.input_files, start=1):
            if self.cancel_requested:
                break

            result = None
            if path in cached_set:
                try:
                    result = self._load_result_from_cache(path, idx - 1, cache_index[str(path)])
                    cached_count += 1
                except Exception:
                    result = None

            if result is None:
                try:
                    result = self.scan_single_image(path, file_index=idx - 1)
                    try:
                        self._save_mask_to_cache(result)
                    except Exception as save_exc:
                        err_msg = str(save_exc)
                        self.root.after(0, lambda m=err_msg: self.status_var.set(
                            f"Warning: mask cache save failed \u2014 {m}"
                        ))
                except Exception as exc:
                    result = {
                        "path": path,
                        "file_index": idx - 1,
                        "mask": None,
                        "tags": [],
                        "tag_count": 0,
                        "output_path": None,
                        "overlay_path": None,
                        "mask_path": None,
                        "error": str(exc),
                        "save_error": None,
                        "from_cache": False,
                    }

            self.results.append(result)
            label = "[Cache]" if result.get("from_cache") else "Scanned"
            status_msg = f"{label} {idx}/{total}: {Path(path).name}"
            self.root.after(0, lambda i=idx: self.progress.configure(value=i))
            self.root.after(0, lambda m=status_msg: self.status_var.set(m))

        self.root.after(0, lambda cc=cached_count: self.refresh_after_scan(cc))

    def scan_single_image(self, path, file_index=0):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Could not read image")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        error = None
        try:
            detections = self.detect_tags(gray)
        except Exception as exc:
            detections = []
            error = str(exc)

        mask = np.zeros(gray.shape, dtype=np.uint8)
        expansion_pct = max(0.0, float(self.expansion_pct_var.get()))

        det_rows = []
        for det in detections:
            corners = np.array(det["corners"], dtype=np.float32)
            expanded = self.expand_tag_corners(corners, expansion_pct, mask.shape)
            cv2.fillConvexPoly(mask, expanded, 255)
            det_rows.append({
                "family": det["family"],
                "id": det["id"],
                "decision_margin": det["decision_margin"],
                "center": det["center"],
                "corners": corners.tolist(),
                "expanded_corners": expanded.tolist(),
            })

        return {
            "path": path,
            "file_index": file_index,
            "mask": mask,
            "tags": det_rows,
            "tag_count": len(det_rows),
            "output_path": None,
            "overlay_path": None,
            "mask_path": None,
            "error": error,
            "save_error": None,
            "from_cache": False,
        }

    def refresh_after_scan(self, cached_count=0):
        self.worker_running = False
        self.cancel_requested = False
        self.mask_dirty = False
        if self.results:
            self.show_result(0)
        else:
            self.clear_preview()

        scanned = len(self.results)
        total_tags = sum(r.get("tag_count", 0) for r in self.results)
        warnings = sum(1 for r in self.results if r.get("error"))
        warning_text = f", with {warnings} scan warning(s)" if warnings else ""
        cache_text = f", {cached_count} loaded from cache" if cached_count > 0 else ""
        self.status_var.set(
            f"Scan complete. Analyzed {scanned} image(s), found {total_tags} detected tag(s){cache_text}{warning_text}. "
            "Review or edit masks, then click Cleanup tags."
        )
        self.update_controls()

    def start_cleanup(self):
        if not self.results:
            messagebox.showwarning("No scan results", "Run Scan listed images first.")
            return
        if self.worker_running or not self.ensure_no_pending_polygon():
            return

        # Check how many output files already exist
        existing_count = 0
        for result in self.results:
            out_dir, out_stem, suffix = self.resolve_output_target(result["path"], result.get("file_index", 0))
            if (out_dir / f"{out_stem}{suffix}").exists():
                existing_count += 1

        skip_existing = False
        if existing_count > 0:
            answer = messagebox.askyesnocancel(
                "Existing output files",
                f"{existing_count} output file(s) already exist in the output folder.\n\n"
                "Yes  \u2192  Overwrite existing files\n"
                "No   \u2192  Skip existing files (only save new ones)\n"
                "Cancel  \u2192  Abort",
            )
            if answer is None:
                return
            skip_existing = not bool(answer)  # No = skip existing
        self._cleanup_skip_existing = skip_existing

        self.cancel_requested = False
        self.reset_edit_state()
        self.worker_running = True
        self.update_controls()
        self.status_var.set("Applying cleanup and saving outputs...")
        threading.Thread(target=self.apply_cleanup_results, daemon=True).start()

    def start_cleanup_current(self):
        if self.worker_running or self.current_index < 0 or self.current_index >= len(self.results):
            return
        result = self.results[self.current_index]
        mask = result.get("mask")
        if mask is None:
            messagebox.showinfo("No mask", "This image has no mask yet. Run Scan first.")
            return

        # Resolve output path in the main thread so the dialog can show it
        out_dir, out_stem, suffix = self.resolve_output_target(result["path"], result.get("file_index", 0))
        output_path = out_dir / f"{out_stem}{suffix}"

        if output_path.exists():
            if not messagebox.askyesno(
                "Output file already exists",
                f"A cleaned version of this image already exists:\n{output_path}\n\nOverwrite it?",
            ):
                return

        # Snapshot all Tkinter-var values here in the main thread — never read
        # Tkinter variables from the background worker thread.
        try:
            reduce_res_scale = max(10, min(90, int(self.reduce_res_scale_var.get()))) / 100.0
        except Exception:
            reduce_res_scale = 1.0
        job = {
            "idx":              self.current_index,
            "src_path":         result["path"],
            "file_index":       result.get("file_index", 0),
            "mask":             mask.copy(),        # copy: worker must not share the live array
            "tags":             list(result.get("tags", [])),
            "out_dir":          out_dir,
            "out_stem":         out_stem,
            "suffix":           suffix,
            "output_path":      output_path,
            "inpaint_method":   self.inpaint_method_var.get(),
            "radius":           float(self.radius_var.get()),
            "reduce_res":       self.reduce_res_var.get(),
            "reduce_res_scale": reduce_res_scale,
            "save_overlays":    self.save_overlays_var.get(),
        }
        self.worker_running = True
        self.update_controls()
        self.status_var.set(f"Cleaning up {Path(result['path']).name}…")
        threading.Thread(target=self._do_cleanup_current, args=(job,), daemon=True).start()

    def _do_cleanup_current(self, job):
        idx = job["idx"]
        try:
            image = cv2.imread(job["src_path"], cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError("Could not read source image.")

            mask = job["mask"]
            if mask.shape != image.shape[:2]:
                mask = np.zeros(image.shape[:2], dtype=np.uint8)

            # Inpaint
            method_name = job["inpaint_method"]
            if np.count_nonzero(mask) > 0:
                if method_name == ROBUST_INPAINT_METHOD:
                    if not hasattr(cv2, "xphoto") or not hasattr(cv2.xphoto, "inpaint"):
                        raise RuntimeError("Shift-Map requires opencv-contrib-python.")
                    valid_mask = np.where(mask > 0, 0, 255).astype(np.uint8)
                    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
                    out_lab = lab.copy()
                    cv2.xphoto.inpaint(lab, valid_mask, out_lab, cv2.xphoto.INPAINT_SHIFTMAP)
                    cleaned = cv2.cvtColor(out_lab, cv2.COLOR_Lab2BGR)
                else:
                    method = cv2.INPAINT_TELEA if method_name == "Telea" else cv2.INPAINT_NS
                    cleaned = cv2.inpaint(image, mask, job["radius"], method)
            else:
                cleaned = image.copy()

            # Optional resize
            if job["reduce_res"]:
                h, w = cleaned.shape[:2]
                sc = job["reduce_res_scale"]
                cleaned = cv2.resize(
                    cleaned,
                    (max(1, int(w * sc)), max(1, int(h * sc))),
                    interpolation=cv2.INTER_AREA,
                )

            out_dir   = job["out_dir"]
            out_stem  = job["out_stem"]
            suffix    = job["suffix"]
            output_path = job["output_path"]

            # Write cleaned image
            cv2.imwrite(str(output_path), cleaned)

            # Write overlay if requested
            overlay_path = None
            if job["save_overlays"]:
                overlay = self.build_mask_overlay(image, mask)
                overlays_dir = out_dir / "overlays"
                overlays_dir.mkdir(parents=True, exist_ok=True)
                overlay_path = overlays_dir / f"{out_stem}_mask_overlay{suffix}"
                cv2.imwrite(str(overlay_path), overlay)

            # Save binary mask alongside the cleaned image, named consistently
            masks_out_dir = out_dir / "masks"
            masks_out_dir.mkdir(parents=True, exist_ok=True)
            mask_out_path = masks_out_dir / f"{out_stem}_mask.png"
            cv2.imwrite(str(mask_out_path), mask)

            # Update internal scan cache so it reflects manual edits
            self._save_mask_to_cache({
                "path": job["src_path"],
                "mask": mask,
                "tags": job["tags"],
            })

            op  = str(output_path)
            ovp = str(overlay_path) if overlay_path else None
            mop = str(mask_out_path)
            self.root.after(0, lambda: self._finish_cleanup_current(idx, op, ovp, mop, None))

        except Exception as exc:
            err = str(exc)
            self.root.after(0, lambda e=err: self._finish_cleanup_current(idx, None, None, None, e))

    def _finish_cleanup_current(self, idx, output_path, overlay_path, mask_path, error):
        self.worker_running = False
        if 0 <= idx < len(self.results):
            result = self.results[idx]
            if error:
                result["save_error"] = error
            else:
                result["output_path"] = output_path
                result["overlay_path"] = overlay_path
                result["mask_path"] = mask_path
                result["save_error"] = None
        if error:
            self.status_var.set(f"Error: {error}")
            messagebox.showerror("Cleanup failed", f"Could not clean up the current image:\n\n{error}")
        else:
            name = Path(self.results[idx]["path"]).name if 0 <= idx < len(self.results) else ""
            self.status_var.set(f"Saved: {name}")
        if self.results and 0 <= self.current_index < len(self.results):
            self.show_result(self.current_index)
        self.update_controls()

    def apply_cleanup_results(self):
        total = len(self.results)
        self.root.after(0, lambda: self.progress.configure(maximum=total, value=0))

        for idx, result in enumerate(self.results, start=1):
            if self.cancel_requested:
                break
            try:
                output_path, overlay_path, mask_path = self.save_result(result)
                result["output_path"] = output_path
                result["overlay_path"] = overlay_path
                result["mask_path"] = mask_path
                result["save_error"] = None
            except Exception as exc:
                result["output_path"] = None
                result["overlay_path"] = None
                result["mask_path"] = None
                result["save_error"] = str(exc)

            self.root.after(0, lambda i=idx: self.progress.configure(value=i))
            self.root.after(0, lambda msg=f"Saved {idx}/{total}: {Path(result['path']).name}": self.status_var.set(msg))

        self.root.after(0, self.refresh_after_cleanup)

    def save_result(self, result):
        image = cv2.imread(result["path"], cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Could not read image")

        mask = result.get("mask")
        if mask is None or mask.shape != image.shape[:2]:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            result["mask"] = mask

        if np.count_nonzero(mask) > 0:
            cleaned = self.apply_inpaint(image, mask)
        else:
            cleaned = image.copy()

        if self.reduce_res_var.get():
            scale = max(10, min(90, int(self.reduce_res_scale_var.get()))) / 100.0
            h, w = cleaned.shape[:2]
            cleaned_save = cv2.resize(
                cleaned,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            cleaned_save = cleaned

        out_dir, out_stem, suffix = self.resolve_output_target(result["path"], result.get("file_index", 0))

        output_path = out_dir / f"{out_stem}{suffix}"
        if getattr(self, '_cleanup_skip_existing', False) and output_path.exists():
            return str(output_path), None, None
        cv2.imwrite(str(output_path), cleaned_save)

        overlay_path = None
        if self.save_overlays_var.get():
            overlay = self.build_mask_overlay(image, mask)
            overlays_dir = out_dir / "overlays"
            overlays_dir.mkdir(parents=True, exist_ok=True)
            overlay_path = overlays_dir / f"{out_stem}_mask_overlay{suffix}"
            cv2.imwrite(str(overlay_path), overlay)

        # Save binary mask alongside the cleaned image, named consistently
        masks_out_dir = out_dir / "masks"
        masks_out_dir.mkdir(parents=True, exist_ok=True)
        mask_out_path = masks_out_dir / f"{out_stem}_mask.png"
        cv2.imwrite(str(mask_out_path), mask)

        return str(output_path), str(overlay_path) if overlay_path else None, str(mask_out_path)

    def refresh_after_cleanup(self):
        self.worker_running = False
        self.cancel_requested = False
        if self.results:
            self.show_result(self.current_index if self.current_index >= 0 else 0)
        else:
            self.clear_preview()

        saved = sum(1 for r in self.results if r.get("output_path"))
        warnings = sum(1 for r in self.results if r.get("error") or r.get("save_error"))
        warning_text = f", with {warnings} warning(s)" if warnings else ""
        self.status_var.set(
            f"Cleanup complete. Saved {saved} cleaned image(s){warning_text}. The saved files include both detected masks and your manual edits."
        )
        self.update_controls()

    def on_left_key(self, event):
        if self.focused_widget_consumes_arrows(event.widget):
            return None
        self.show_previous_result()
        return "break"

    def on_right_key(self, event):
        if self.focused_widget_consumes_arrows(event.widget):
            return None
        self.show_next_result()
        return "break"

    def start_polygon_mode(self, mode):
        if not self.results or self.worker_running:
            return
        if self.edit_mode == mode and not self.edit_points:
            self.edit_mode = None
        else:
            self.edit_mode = mode
            self.last_edit_mode = mode      # remember for right-click recall
            self.edit_points = []
            self.preview_canvas.focus_set()
        self.update_controls()
        self.render_preview()

    def cancel_polygon(self, _event=None):
        if self.edit_mode is None and not self.edit_points:
            return None
        self.reset_edit_state()
        self.update_controls()
        self.render_preview()
        return "break"

    def canvas_to_image_point(self, x, y):
        if not self.preview_transform:
            return None

        scale = self.preview_transform["scale"]
        offset_x = self.preview_transform["offset_x"]
        offset_y = self.preview_transform["offset_y"]
        image_w = self.preview_transform["image_width"]
        image_h = self.preview_transform["image_height"]
        display_w = self.preview_transform["display_width"]
        display_h = self.preview_transform["display_height"]

        # Clamp to image boundary — clicks outside the image project to the nearest edge point
        image_x = min(max((x - offset_x) / scale, 0.0), image_w - 1.0)
        image_y = min(max((y - offset_y) / scale, 0.0), image_h - 1.0)
        return (image_x, image_y)

    def image_to_canvas_point(self, point):
        if not self.preview_transform:
            return None
        return (
            self.preview_transform["offset_x"] + point[0] * self.preview_transform["scale"],
            self.preview_transform["offset_y"] + point[1] * self.preview_transform["scale"],
        )

    def on_preview_press(self, event):
        if event.state & 0x0001:  # Shift held → pan mode handles it
            return
        if self.worker_running or self.edit_mode not in {"add", "erase"}:
            return
        self._rect_drag_start = (event.x, event.y)
        self._rect_drag_current = (event.x, event.y)
        self._rect_dragging = False

    def on_preview_b1_motion(self, event):
        if self.worker_running or self.edit_mode not in {"add", "erase"}:
            return
        if self._rect_drag_start is None:
            return
        if abs(event.x - self._rect_drag_start[0]) > 5 or abs(event.y - self._rect_drag_start[1]) > 5:
            self._rect_dragging = True
        self._rect_drag_current = (event.x, event.y)
        if self._rect_dragging:
            self.render_preview()

    def on_preview_b1_release(self, event):
        if self.worker_running or self.edit_mode not in {"add", "erase"}:
            self._rect_drag_start = None
            self._rect_drag_current = None
            self._rect_dragging = False
            return
        start = self._rect_drag_start
        was_dragging = self._rect_dragging
        self._rect_drag_start = None
        self._rect_drag_current = None
        self._rect_dragging = False

        if start is None:
            return

        if was_dragging:
            # Apply the rectangle as a 4-point polygon
            p1 = self.canvas_to_image_point(*start)
            p2 = self.canvas_to_image_point(event.x, event.y)
            if p1 is None or p2 is None:
                self.render_preview()
                return
            x1, y1 = p1
            x2, y2 = p2
            if abs(x2 - x1) < 2 or abs(y2 - y1) < 2:
                self.render_preview()
                return
            self.edit_points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            self.apply_polygon_edit()
        else:
            # Normal click → place a polygon point
            point = self.canvas_to_image_point(event.x, event.y)
            if point is None:
                return
            self.edit_points.append(point)
            self.update_controls()
            self.render_preview()

    def on_preview_complete_polygon(self, _event=None):
        if self.edit_mode in {"add", "erase"} and len(self.edit_points) >= 3:
            self.apply_polygon_edit()
            return "break"
        return None

    def apply_polygon_edit(self):
        if self.current_index < 0 or self.current_index >= len(self.results):
            return
        if self.edit_mode not in {"add", "erase"}:
            return
        if len(self.edit_points) < 3:
            messagebox.showinfo("Polygon", "Add at least three points to define a polygon.")
            return

        result = self.results[self.current_index]
        image = self.get_result_image(self.current_index)
        if image is None:
            messagebox.showerror("Preview", "Could not load the current image for editing.")
            return

        mask = result.get("mask")
        if mask is None or mask.shape != image.shape[:2]:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            result["mask"] = mask

        polygon = np.rint(np.array(self.edit_points, dtype=np.float32)).astype(np.int32)
        fill_value = 255 if self.edit_mode == "add" else 0
        cv2.fillPoly(mask, [polygon], fill_value)

        self.mask_dirty = True     # mark as needing auto-save
        action = "added to" if self.edit_mode == "add" else "removed from"
        self.reset_edit_state()
        self.update_controls()
        self.render_preview()
        self.status_var.set(f"Polygon {action} the mask for {Path(result['path']).name}.")

    def on_preview_resize(self, _event=None):
        if self.results:
            self.render_preview()
        else:
            self.clear_preview()

    def get_preview_metrics(self, image_w, image_h):
        canvas_w = max(self.preview_canvas.winfo_width(), 900)
        canvas_h = max(self.preview_canvas.winfo_height(), 520)
        fit_scale = min(canvas_w / image_w, canvas_h / image_h)
        scale = fit_scale * self.preview_zoom
        display_w = max(1, int(round(image_w * scale)))
        display_h = max(1, int(round(image_h * scale)))
        scale = display_w / image_w
        return {
            "canvas_width": float(canvas_w),
            "canvas_height": float(canvas_h),
            "scale": float(scale),
            "display_width": float(display_w),
            "display_height": float(display_h),
            "base_offset_x": (canvas_w - display_w) / 2.0,
            "base_offset_y": (canvas_h - display_h) / 2.0,
        }

    def clamp_preview_pan(self, image_w, image_h):
        metrics = self.get_preview_metrics(image_w, image_h)

        if metrics["display_width"] <= metrics["canvas_width"]:
            self.preview_pan_x = 0.0
        else:
            min_pan_x = metrics["canvas_width"] - metrics["display_width"] - metrics["base_offset_x"]
            max_pan_x = -metrics["base_offset_x"]
            self.preview_pan_x = min(max_pan_x, max(min_pan_x, self.preview_pan_x))

        if metrics["display_height"] <= metrics["canvas_height"]:
            self.preview_pan_y = 0.0
        else:
            min_pan_y = metrics["canvas_height"] - metrics["display_height"] - metrics["base_offset_y"]
            max_pan_y = -metrics["base_offset_y"]
            self.preview_pan_y = min(max_pan_y, max(min_pan_y, self.preview_pan_y))

        return metrics

    def set_preview_zoom(self, new_zoom, anchor_canvas=None):
        if not self.results:
            return

        new_zoom = max(1.0, min(8.0, float(new_zoom)))
        if abs(new_zoom - self.preview_zoom) < 1e-6:
            return

        anchor_point = None
        if anchor_canvas and self.preview_transform:
            anchor_point = self.canvas_to_image_point(anchor_canvas[0], anchor_canvas[1])

        if anchor_point is None and self.preview_transform:
            anchor_canvas = (
                self.preview_canvas.winfo_width() / 2.0,
                self.preview_canvas.winfo_height() / 2.0,
            )
            anchor_point = self.canvas_to_image_point(anchor_canvas[0], anchor_canvas[1])

        self.preview_zoom = new_zoom
        self.update_preview_zoom_label()

        image = self.get_result_image(self.current_index)
        if image is None:
            self.render_preview()
            return

        image_h, image_w = image.shape[:2]
        metrics = self.get_preview_metrics(image_w, image_h)
        if anchor_point is not None and anchor_canvas is not None:
            desired_offset_x = anchor_canvas[0] - anchor_point[0] * metrics["scale"]
            desired_offset_y = anchor_canvas[1] - anchor_point[1] * metrics["scale"]
            self.preview_pan_x = desired_offset_x - metrics["base_offset_x"]
            self.preview_pan_y = desired_offset_y - metrics["base_offset_y"]

        self.clamp_preview_pan(image_w, image_h)
        self.render_preview()

    def zoom_in(self):
        self.set_preview_zoom(self.preview_zoom * 1.25)

    def zoom_out(self):
        self.set_preview_zoom(self.preview_zoom / 1.25)

    def on_preview_mousewheel(self, event):
        if not self.results or self.worker_running:
            return "break"

        if hasattr(event, "delta") and event.delta:
            direction = 1 if event.delta > 0 else -1
        elif getattr(event, "num", None) == 4:
            direction = 1
        elif getattr(event, "num", None) == 5:
            direction = -1
        else:
            return "break"

        factor = 1.25 if direction > 0 else 1 / 1.25
        self.set_preview_zoom(self.preview_zoom * factor, anchor_canvas=(event.x, event.y))
        return "break"

    def start_preview_pan(self, event):
        if not self.results or self.worker_running:
            return "break"
        self.preview_drag_start = (event.x, event.y)
        self.preview_canvas.focus_set()
        return "break"

    def on_preview_pan(self, event):
        if self.preview_drag_start is None or self.current_index < 0:
            return "break"

        dx = event.x - self.preview_drag_start[0]
        dy = event.y - self.preview_drag_start[1]
        self.preview_drag_start = (event.x, event.y)
        self.preview_pan_x += dx
        self.preview_pan_y += dy

        image = self.get_result_image(self.current_index)
        if image is not None:
            image_h, image_w = image.shape[:2]
            self.clamp_preview_pan(image_w, image_h)
        self.render_preview()
        return "break"

    def end_preview_pan(self, _event=None):
        self.preview_drag_start = None
        return "break"

    def get_result_image(self, idx):
        if idx < 0 or idx >= len(self.results):
            return None
        if self.preview_cache_index != idx or self.preview_source_image is None:
            image = cv2.imread(self.results[idx]["path"], cv2.IMREAD_COLOR)
            if image is None:
                self.clear_preview_cache()
                return None
            self.preview_cache_index = idx
            self.preview_source_image = image
        return self.preview_source_image

    def draw_edit_polygon(self):
        if not self.edit_points:
            return

        color = "#5dff6c" if self.edit_mode == "add" else "#ffb347"
        canvas_points = [self.image_to_canvas_point(point) for point in self.edit_points]
        canvas_points = [point for point in canvas_points if point is not None]
        if len(canvas_points) < 1:
            return

        for point in canvas_points:
            self.preview_canvas.create_oval(point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3, fill=color, outline=color)

        if len(canvas_points) >= 2:
            flat_points = [coord for point in canvas_points for coord in point]
            self.preview_canvas.create_line(*flat_points, fill=color, width=2)
        if len(canvas_points) >= 3:
            first = canvas_points[0]
            last = canvas_points[-1]
            self.preview_canvas.create_line(last[0], last[1], first[0], first[1], fill=color, width=1, dash=(4, 3))

    def draw_rect_preview(self):
        """Draw the live rectangle while the user is dragging to create a rect mask."""
        if not self._rect_dragging or self._rect_drag_start is None or self._rect_drag_current is None:
            return
        if self.edit_mode not in {"add", "erase"}:
            return
        color = "#5dff6c" if self.edit_mode == "add" else "#ffb347"
        x1, y1 = self._rect_drag_start
        x2, y2 = self._rect_drag_current
        self.preview_canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)
        for cx, cy in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
            self.preview_canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill=color, outline=color)

    def render_preview(self):
        self.preview_canvas.delete("all")
        result = self.results[self.current_index] if 0 <= self.current_index < len(self.results) else None
        if result is None:
            self.clear_preview()
            return

        image = self.get_result_image(self.current_index)
        if image is None:
            self.clear_preview()
            return

        overlay = self.build_mask_overlay(image, result.get("mask"))
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        image_h, image_w = overlay_rgb.shape[:2]

        metrics = self.clamp_preview_pan(image_w, image_h)
        display_w = int(metrics["display_width"])
        display_h = int(metrics["display_height"])
        offset_x = metrics["base_offset_x"] + self.preview_pan_x
        offset_y = metrics["base_offset_y"] + self.preview_pan_y
        interpolation = cv2.INTER_AREA if metrics["scale"] <= 1.0 else cv2.INTER_CUBIC
        resized = cv2.resize(overlay_rgb, (display_w, display_h), interpolation=interpolation)

        self.preview_transform = {
            "scale": metrics["scale"],
            "offset_x": offset_x,
            "offset_y": offset_y,
            "image_width": float(image_w),
            "image_height": float(image_h),
            "display_width": float(display_w),
            "display_height": float(display_h),
        }

        self.preview_photo = ImageTk.PhotoImage(Image.fromarray(resized))
        self.preview_canvas.create_image(
            offset_x,
            offset_y,
            image=self.preview_photo,
            anchor="nw",
        )
        self.preview_canvas.create_rectangle(
            offset_x,
            offset_y,
            offset_x + display_w,
            offset_y + display_h,
            outline="#666",
        )

        if result.get("error"):
            self.preview_canvas.create_text(
                14,
                14,
                anchor="nw",
                width=max(120, metrics["canvas_width"] - 28),
                text=f"Scan note: {result['error']}",
                fill="#ffcf7a",
            )

        self.draw_edit_polygon()
        self.draw_rect_preview()

    def sync_file_selection(self, idx):
        self.ignore_file_selection = True
        try:
            self.file_list.selection_clear(0, tk.END)
            if idx >= 0:
                self.file_list.selection_set(idx)
                self.file_list.activate(idx)
                self.file_list.see(idx)
        finally:
            self.ignore_file_selection = False

    def on_file_select(self, _event=None):
        if self.ignore_file_selection or not self.results:
            return
        sel = self.file_list.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx == self.current_index:
            return
        if not self.ensure_no_pending_polygon():
            self.sync_file_selection(self.current_index)
            return
        if idx < len(self.results):
            self.show_result(idx)

    def show_previous_result(self):
        self.navigate_result(-1)

    def show_next_result(self):
        self.navigate_result(1)

    def navigate_result(self, step):
        if not self.results or self.worker_running or not self.ensure_no_pending_polygon():
            return
        if self.current_index < 0:
            target = 0
        else:
            target = max(0, min(len(self.results) - 1, self.current_index + step))
        if target != self.current_index:
            self.show_result(target)

    def show_result(self, idx):
        if idx < 0 or idx >= len(self.results):
            return

        # Auto-save previous image's mask if it was manually edited
        if (
            self.mask_dirty
            and 0 <= self.current_index < len(self.results)
            and self.current_index != idx
        ):
            self._autosave_current_mask(self.current_index)

        self.mask_dirty = False
        self.current_index = idx
        self.sync_file_selection(idx)
        self.reset_preview_view(render=False)
        self.clear_preview_cache()

        result = self.results[idx]
        for item in self.tree.get_children():
            self.tree.delete(item)
        for tag in result.get("tags", []):
            self.tree.insert("", tk.END, values=(
                tag["family"], tag["id"], f"{tag['decision_margin']:.2f}", f"({tag['center'][0]:.1f}, {tag['center'][1]:.1f})"
            ))

        note = " [scan note]" if result.get("error") else ""
        cache_note = " [cached]" if result.get("from_cache") else ""
        self.preview_index_var.set(f"{idx + 1}/{len(self.results)} - {Path(result['path']).name}{cache_note}{note}")
        self.render_preview()
        self.update_controls()


def main():
    root = tk.Tk()
    try:
        from tkinter import TclError
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except Exception:
        pass
    app = AprilTagCleanerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()