import os
import csv
from modules.grain_detection import detect_grains


def process_batch(input_folder, pixel_to_mm):

    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    summary_file = os.path.join(output_folder, "summary.csv")

    # open csv file
    with open(summary_file, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "Image",
            "Grain count",
            "Average size (px)",
            "Average size (mm)"
        ])

        # loop through images
        for filename in os.listdir(input_folder):

            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):

                image_path = os.path.join(input_folder, filename)

                print(f"Processing: {filename}")

                result = detect_grains(
                    image_path=image_path,
                    pixel_to_mm=pixel_to_mm
                )

                if result:

                    writer.writerow([
                        filename,
                        result["count"],
                        result["avg_px"],
                        result["avg_mm"]
                    ])

    print("Batch processing complete.")
