"""
Professional Splash Screen with Image
GeoGrain Analyzer Professional v2.0
"""

import tkinter as tk
from PIL import Image, ImageTk
import os


def show_splash(root, duration=3000):

    splash = tk.Toplevel(root)

    splash.overrideredirect(True)

    width = 600
    height = 400

    screen_width = splash.winfo_screenwidth()
    screen_height = splash.winfo_screenheight()

    x = int((screen_width - width) / 2)
    y = int((screen_height - height) / 2)

    splash.geometry(f"{width}x{height}+{x}+{y}")

    splash.configure(bg="black")

    # SPLASH IMAGE PATH
    image_path = os.path.join(
        os.getcwd(),
        "GeoGrain_Analyser_Wallpaper.png"
    )

    if os.path.exists(image_path):

        img = Image.open(image_path)
        img = img.resize((600, 400), Image.LANCZOS)

        photo = ImageTk.PhotoImage(img)

        label = tk.Label(splash, image=photo, borderwidth=0)
        label.image = photo
        label.pack()

    else:

        label = tk.Label(
            splash,
            text="GeoGrain Analyzer Professional v2.0",
            font=("Segoe UI", 20, "bold"),
            fg="white",
            bg="black"
        )

        label.pack(expand=True)

    splash.update()

    root.after(duration, splash.destroy)
