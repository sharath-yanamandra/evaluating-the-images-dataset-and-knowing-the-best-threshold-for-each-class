#!/usr/bin/env python3
"""
EXIF Data Viewer
Shows all EXIF metadata in an image file
"""

import sys
import os
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def view_exif_data(image_path):
    """
    Display all EXIF data from an image
    
    Args:
        image_path: Path to the image file
    """
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' does not exist.")
        return
    
    filename = os.path.basename(image_path)
    print(f"EXIF Data for: {filename}")
    print("=" * 60)
    
    try:
        with Image.open(image_path) as image:
            # Get basic image info
            print(f"Image Size: {image.size}")
            print(f"Image Mode: {image.mode}")
            print(f"Image Format: {image.format}")
            print()
            
            # Get EXIF data
            exif_data = image.getexif()
            
            if not exif_data:
                print("❌ No EXIF data found in this image.")
                return
            
            print("📋 EXIF Data:")
            print("-" * 40)
            
            # Display regular EXIF tags
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, tag_id)
                
                # Handle GPS data separately
                if tag_name == 'GPSInfo':
                    print(f"{tag_name}:")
                    if isinstance(value, dict):
                        for gps_tag_id, gps_value in value.items():
                            gps_tag_name = GPSTAGS.get(gps_tag_id, gps_tag_id)
                            print(f"  {gps_tag_name}: {gps_value}")
                    else:
                        print(f"  {value}")
                else:
                    # Truncate very long values
                    if isinstance(value, (str, bytes)) and len(str(value)) > 100:
                        display_value = str(value)[:100] + "..."
                    else:
                        display_value = value
                    
                    print(f"{tag_name}: {display_value}")
            
            # Check specifically for GPS data
            print("\n🌍 GPS Information:")
            print("-" * 40)
            
            gps_info = exif_data.get(34853)  # GPS IFD tag
            if gps_info:
                print("✅ GPS data found!")
                for gps_tag_id, gps_value in gps_info.items():
                    gps_tag_name = GPSTAGS.get(gps_tag_id, gps_tag_id)
                    print(f"  {gps_tag_name}: {gps_value}")
            else:
                print("❌ No GPS data found.")
                
    except Exception as e:
        print(f"Error reading image: {str(e)}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 exif_viewer.py <image_path>")
        print("Example: python3 exif_viewer.py \"path/to/image.jpg\"")
        sys.exit(1)
    
    image_path = sys.argv[1]
    view_exif_data(image_path)

if __name__ == "__main__":
    # Check if PIL is available
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS, GPSTAGS
    except ImportError:
        print("Error: Pillow library is required. Install it with:")
        print("pip install Pillow")
        sys.exit(1)
    
    main()