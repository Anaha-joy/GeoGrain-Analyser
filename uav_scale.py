"""
UAV scale calculator using EXIF metadata
Thesis-grade accurate scale estimation
"""

from PIL import Image, ExifTags


def get_pixel_to_mm(image_path):

    try:

        img = Image.open(image_path)

        exif = img._getexif()

        if exif is None:
            return default_scale()

        exif_data = {}

        for tag, value in exif.items():
            decoded = ExifTags.TAGS.get(tag, tag)
            exif_data[decoded] = value


        altitude = exif_data.get("GPSAltitude", None)
        focal_length = exif_data.get("FocalLength", None)


        if altitude is None or focal_length is None:
            return default_scale()


        altitude = float(altitude)
        focal_length = float(focal_length)


        # Typical DJI sensor pixel size
        sensor_pixel_size_mm = 0.0024


        pixel_to_mm = (altitude * sensor_pixel_size_mm) / focal_length


        pixel_to_mm *= 1000


        if pixel_to_mm < 1 or pixel_to_mm > 100:
            return default_scale()


        print("Auto scale detected:", pixel_to_mm, "mm/pixel")

        return pixel_to_mm


    except:

        return default_scale()


def default_scale():

    print("Using default UAV scale: 12 mm/pixel")

    return 12.0
