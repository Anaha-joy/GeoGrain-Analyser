"""
GeoGrain Analyzer Professional v8.0
Professional PDF Report Generator
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    PageBreak
)

from reportlab.lib.styles import getSampleStyleSheet


# =====================================================
# FREQUENCY CURVE (BASEGRAIN STYLE)
# =====================================================

def create_frequency_curve(grain_sizes):

    fig = plt.figure()

    if len(grain_sizes) > 0:

        sizes = np.array(grain_sizes)

        sizes = sizes[sizes > 0]

        sizes_sorted = np.sort(sizes)

        cumulative = (
            np.arange(1, len(sizes_sorted) + 1)
            / len(sizes_sorted)
            * 100
        )

        plt.plot(sizes_sorted, cumulative, linewidth=2)

        plt.xscale("log")
        plt.ylim(0, 100)

    # ✅ RENAMED
    plt.title("Number-Weighted Grain Size Distribution")
    plt.xlabel("Grain Size (mm) [log scale]")
    plt.ylabel("Percent Finer (%)")
    plt.grid()

    return fig


# =====================================================
# LOG SCALE (PHI) CURVE WITH VOLUME SUPPORT
# =====================================================

def create_log_curve(grain_sizes, grain_volumes=None):

    fig = plt.figure()

    if len(grain_sizes) > 0:

        sizes = np.array(grain_sizes)

        valid = sizes > 0
        sizes = sizes[valid]

        if grain_volumes is not None:
            volumes = np.array(grain_volumes)[valid]
        else:
            volumes = np.ones_like(sizes)

        sort_idx = np.argsort(sizes)
        sizes_sorted = sizes[sort_idx]
        volumes_sorted = volumes[sort_idx]

        volumes_sorted = np.log1p(volumes_sorted)
        volumes_sorted = volumes_sorted / np.sum(volumes_sorted)

        cumulative = np.cumsum(volumes_sorted) * 100

        plt.plot(sizes_sorted, cumulative, 'o-', linewidth=2)

        plt.xscale("log")
        plt.ylim(0, 100)

    # ✅ RENAMED
    plt.title("Volume-Weighted Grain Size Distribution")
    plt.xlabel("Grain Size (mm) [log scale]")
    plt.ylabel("Percent Finer (%)")
    plt.grid()

    return fig


# =====================================================
# D-CURVE
# =====================================================

def create_dcurve(stats):

    fig = plt.figure()

    labels = ["D10", "D50", "D84", "D90"]

    values = [
        stats.get("d10", 0),
        stats.get("d50", 0),
        stats.get("d84", 0),
        stats.get("d90", 0)
    ]

    plt.bar(labels, values)

    # ✅ RENAMED
    plt.title("Grain Size Percentile Distribution")
    plt.ylabel("Size (mm)")
    plt.grid()

    return fig


# =====================================================
# SAVE PDF REPORT
# =====================================================

def save_pdf(grain_sizes, stats, overlay_path, csv_path, grain_volumes=None):

    os.makedirs("output", exist_ok=True)

    pdf_path = os.path.abspath("output/grain_report.pdf")

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(pdf_path)

    elements = []

    elements.append(
        Paragraph(
            "GeoGrain Analyzer Professional Report",
            styles["Heading1"]
        )
    )

    elements.append(Spacer(1, 12))

    try:
        df = pd.read_csv(csv_path)
        mean_a = df["a_axis_mm"].mean()
        mean_b = df["b_axis_mm"].mean()
    except:
        mean_a = 0
        mean_b = 0

    elements.append(Paragraph(f"Total Grains: {stats.get('count',0)}", styles["Normal"]))
    elements.append(Paragraph(f"Mean Size: {stats.get('mean',0):.2f} mm", styles["Normal"]))
    elements.append(Paragraph(f"D10: {stats.get('d10',0):.2f} mm", styles["Normal"]))
    elements.append(Paragraph(f"D50: {stats.get('d50',0):.2f} mm", styles["Normal"]))
    elements.append(Paragraph(f"D84: {stats.get('d84',0):.2f} mm", styles["Normal"]))
    elements.append(Paragraph(f"D90: {stats.get('d90',0):.2f} mm", styles["Normal"]))
    elements.append(Paragraph(f"Mean a-axis: {mean_a:.2f} mm", styles["Normal"]))
    elements.append(Paragraph(f"Mean b-axis: {mean_b:.2f} mm", styles["Normal"]))

    elements.append(Spacer(1, 20))

    elements.append(Paragraph("Detected Grains Overlay", styles["Heading2"]))
    elements.append(Spacer(1, 10))
    elements.append(Image(overlay_path, width=500, height=350))

    elements.append(PageBreak())

    # =================================================
    # FREQUENCY CURVE
    # =================================================

    freq_path = "output/frequency_curve.png"
    fig = create_frequency_curve(grain_sizes)
    fig.savefig(freq_path)
    plt.close(fig)

    # ✅ RENAMED
    elements.append(Paragraph("Number-Weighted Grain Size Distribution", styles["Heading2"]))
    elements.append(Spacer(1, 10))
    elements.append(Image(freq_path, width=500, height=350))

    elements.append(PageBreak())

    # =================================================
    # LOG CURVE (VOLUME)
    # =================================================

    log_path = "output/log_curve.png"
    fig = create_log_curve(grain_sizes, grain_volumes)
    fig.savefig(log_path)
    plt.close(fig)

    # ✅ RENAMED
    elements.append(Paragraph("Volume-Weighted Grain Size Distribution", styles["Heading2"]))
    elements.append(Spacer(1, 10))
    elements.append(Image(log_path, width=500, height=350))

    elements.append(PageBreak())

    # =================================================
    # D CURVE
    # =================================================

    dcurve_path = "output/dcurve.png"
    fig = create_dcurve(stats)
    fig.savefig(dcurve_path)
    plt.close(fig)

    # ✅ RENAMED
    elements.append(Paragraph("Grain Size Percentile Distribution", styles["Heading2"]))
    elements.append(Spacer(1, 10))
    elements.append(Image(dcurve_path, width=500, height=350))

    doc.build(elements)

    print("PDF saved:", pdf_path)

    return pdf_path