"""
GeoGrain Analyzer Professional v7.0
Main launcher with splash screen
SAM AI compatible
Thread-safe startup
"""

import tkinter as tk
import traceback
import sys

from modules.gui import GeoGrainGUI
from modules.splash import show_splash


# =====================================================
# GLOBAL ERROR HANDLER
# =====================================================
def handle_exception(exc_type, exc_value, exc_traceback):

    error = "".join(
        traceback.format_exception(
            exc_type,
            exc_value,
            exc_traceback
        )
    )

    print("\n========== ERROR OCCURRED ==========\n")
    print(error)


# Activate global error handler
sys.excepthook = handle_exception


# =====================================================
# START MAIN GUI
# =====================================================
def start_main(root):

    try:
        GeoGrainGUI(root)

    except Exception as e:
        print("GUI Startup Error:", str(e))


# =====================================================
# MAIN FUNCTION
# =====================================================
def main():

    root = tk.Tk()

    # Hide main window during splash
    root.withdraw()

    # Show splash screen
    show_splash(
        root,
        duration=3000
    )

    # Launch GUI after splash
    root.after(
        3000,
        lambda: [
            root.deiconify(),
            start_main(root)
        ]
    )

    root.mainloop()


# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":

    main()
