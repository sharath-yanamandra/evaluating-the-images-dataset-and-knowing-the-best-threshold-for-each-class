#!/usr/bin/env python3
"""
DJI GPS Coordinates Editor
Extracts GPS coordinates from DJI thermal images and updates them in decimal format
"""

import os
import sys
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import argparse
from fractions import Fraction
import exifread
from io import BytesIO

def dms_to_decimal(degrees, minutes, seconds, direction):
    """Convert DMS to decimal degrees"""
    decimal = float(degrees) + float(minutes)/60 + float(seconds)/3600
    if direction in ['S', 'W']:
        decimal = -decimal
    return decimal

def decimal_to_dms(decimal_degrees):
    """Convert decimal degrees to DMS format for EXIF"""
    is_negative = decimal_degrees < 0
    decimal_degrees = abs(decimal_degrees)
    
    degrees = int(decimal_degrees)
    minutes_float = (decimal_degrees - degrees) * 60
    minutes = int(minutes_float)
    seconds = (minutes_float - minutes) * 60
    
    # Convert to fractions for EXIF format
    degrees_frac = Fraction(degrees, 1)
    minutes_frac = Fraction(minutes, 1)
    seconds_frac = Fraction(seconds).limit_denominator(10000)
    
    return (degrees_frac, minutes_frac, seconds_frac), is_negative

def extract_gps_with_exifread(image_path):
    """Extract GPS coordinates using exifread library"""
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f)
            
            # Look for GPS tags
            gps_tags = {key: tags[key] for key in tags.keys() if key.startswith('GPS')}
            
            if not gps_tags:
                return None, None
            
            print("GPS tags found:")
            for key, value in gps_tags.items():
                print(f"  {key}: {value}")
            
            # Extract coordinates
            if 'GPS GPSLatitude' in gps_tags and 'GPS GPSLongitude' in gps_tags:
                # Parse latitude
                lat_dms_str = str(gps_tags['GPS GPSLatitude']).replace('[', '').replace(']', '')
                lat_parts = lat_dms_str.split(', ')
                
                lat_degrees = float(lat_parts[0])
                lat_minutes = float(lat_parts[1])
                
                # Handle fractional seconds
                if '/' in lat_parts[2]:
                    num, den = lat_parts[2].split('/')
                    lat_seconds = float(num) / float(den)
                else:
                    lat_seconds = float(lat_parts[2])
                
                lat_ref = str(gps_tags.get('GPS GPSLatitudeRef', 'N'))
                
                # Parse longitude
                lon_dms_str = str(gps_tags['GPS GPSLongitude']).replace('[', '').replace(']', '')
                lon_parts = lon_dms_str.split(', ')
                
                lon_degrees = float(lon_parts[0])
                lon_minutes = float(lon_parts[1])
                
                if '/' in lon_parts[2]:
                    num, den = lon_parts[2].split('/')
                    lon_seconds = float(num) / float(den)
                else:
                    lon_seconds = float(lon_parts[2])
                
                lon_ref = str(gps_tags.get('GPS GPSLongitudeRef', 'E'))
                
                # Convert to decimal
                lat_decimal = dms_to_decimal(lat_degrees, lat_minutes, lat_seconds, lat_ref)
                lon_decimal = dms_to_decimal(lon_degrees, lon_minutes, lon_seconds, lon_ref)
                
                return lat_decimal, lon_decimal
                
        return None, None
        
    except Exception as e:
        print(f"Error extracting GPS with exifread: {e}")
        return None, None

def update_gps_coordinates(image_path, new_lat, new_lon, backup=True):
    """
    Update GPS coordinates in the image file
    """
    try:
        # Create backup if requested
        if backup:
            backup_path = image_path + '.backup'
            if not os.path.exists(backup_path):
                import shutil
                shutil.copy2(image_path, backup_path)
                print(f"✅ Backup created: {backup_path}")
        
        # Open the image
        with Image.open(image_path) as image:
            # Get existing EXIF data
            exif_dict = image.getexif()
            
            # Convert new coordinates to DMS format
            lat_dms, lat_negative = decimal_to_dms(new_lat)
            lon_dms, lon_negative = decimal_to_dms(new_lon)
            
            print(f"Converting coordinates:")
            print(f"  Decimal: {new_lat:.6f}, {new_lon:.6f}")
            print(f"  DMS: {lat_dms[0]}°{lat_dms[1]}'{lat_dms[2]:.4f}\" {'S' if lat_negative else 'N'}, "
                  f"{lon_dms[0]}°{lon_dms[1]}'{lon_dms[2]:.4f}\" {'W' if lon_negative else 'E'}")
            
            # Create or update GPS info dictionary
            gps_ifd = {
                1: 'N' if not lat_negative else 'S',  # GPSLatitudeRef
                2: lat_dms,                            # GPSLatitude
                3: 'E' if not lon_negative else 'W',  # GPSLongitudeRef
                4: lon_dms,                            # GPSLongitude
            }
            
            # Preserve existing GPS data if available
            if 34853 in exif_dict:
                existing_gps = exif_dict[34853]
                if isinstance(existing_gps, dict):
                    # Keep other GPS fields like altitude, timestamp, etc.
                    for key, value in existing_gps.items():
                        if key not in [1, 2, 3, 4]:  # Don't overwrite lat/lon fields
                            gps_ifd[key] = value
            
            # Update EXIF with new GPS info
            exif_dict[34853] = gps_ifd
            
            # Save the image with updated EXIF
            # For MPO files, we need to be careful about the format
            if image.format == 'MPO':
                # Try to save as JPEG with EXIF
                output_path = image_path.replace('.JPG', '_updated.JPG')
                image.save(output_path, 'JPEG', exif=exif_dict, quality=95)
                print(f"✅ Updated image saved as: {output_path}")
                print("Note: MPO format converted to JPEG to preserve GPS changes")
            else:
                # Save in original format
                image.save(image_path, exif=exif_dict, quality=95)
                print(f"✅ GPS coordinates updated in: {image_path}")
                
        return True
        
    except Exception as e:
        print(f"❌ Error updating GPS coordinates: {e}")
        return False

def manual_coordinate_input():
    """Allow manual input of GPS coordinates"""
    print("\n📍 Manual GPS Coordinate Input")
    print("-" * 40)
    
    try:
        lat = float(input("Enter latitude (decimal degrees, e.g., 27.514534): "))
        lon = float(input("Enter longitude (decimal degrees, e.g., 72.416506): "))
        
        # Validate ranges
        if not (-90 <= lat <= 90):
            print("❌ Invalid latitude. Must be between -90 and 90.")
            return None, None
        
        if not (-180 <= lon <= 180):
            print("❌ Invalid longitude. Must be between -180 and 180.")
            return None, None
        
        return lat, lon
        
    except ValueError:
        print("❌ Invalid input. Please enter numeric values.")
        return None, None

def process_image(image_path, manual_coords=False, new_lat=None, new_lon=None):
    """
    Main function to process a DJI image
    """
    filename = os.path.basename(image_path)
    print(f"Processing: {filename}")
    print("=" * 60)
    
    # Step 1: Extract existing coordinates
    if not manual_coords:
        print("📍 Extracting existing GPS coordinates...")
        lat, lon = extract_gps_with_exifread(image_path)
        
        if lat is None or lon is None:
            print("❌ No GPS coordinates found in image.")
            print("Would you like to add coordinates manually? (y/n): ", end="")
            choice = input().lower().strip()
            if choice == 'y':
                lat, lon = manual_coordinate_input()
                if lat is None:
                    return False
            else:
                return False
        else:
            print(f"✅ Current GPS coordinates: {lat:.6f}, {lon:.6f}")
    else:
        # Use provided coordinates
        lat, lon = new_lat, new_lon
        print(f"📍 Using provided coordinates: {lat:.6f}, {lon:.6f}")
    
    # Step 2: Option to modify coordinates
    if not manual_coords:
        print(f"\nCurrent coordinates: {lat:.6f}, {lon:.6f}")
        print("Options:")
        print("1. Keep current coordinates and update format")
        print("2. Enter new coordinates")
        print("3. Cancel")
        
        choice = input("Choose option (1-3): ").strip()
        
        if choice == '2':
            new_coords = manual_coordinate_input()
            if new_coords[0] is not None:
                lat, lon = new_coords
        elif choice == '3':
            print("Operation cancelled.")
            return False
    
    # Step 3: Update the image
    print(f"\n🔄 Updating GPS coordinates to: {lat:.6f}, {lon:.6f}")
    success = update_gps_coordinates(image_path, lat, lon)
    
    if success:
        print("\n🎉 GPS coordinates successfully updated!")
        print(f"Location: https://www.google.com/maps?q={lat},{lon}")
    
    return success

def main():
    parser = argparse.ArgumentParser(description='Edit GPS coordinates in DJI thermal images')
    parser.add_argument('image_path', help='Path to DJI image file')
    parser.add_argument('--lat', type=float, help='New latitude (decimal degrees)')
    parser.add_argument('--lon', type=float, help='New longitude (decimal degrees)')
    parser.add_argument('--no-backup', action='store_true', help='Skip creating backup file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: File '{args.image_path}' does not exist.")
        return
    
    # Check if coordinates provided via command line
    manual_coords = args.lat is not None and args.lon is not None
    
    process_image(args.image_path, manual_coords, args.lat, args.lon)

if __name__ == "__main__":
    # Check dependencies
    try:
        import exifread
    except ImportError:
        print("Installing required dependency...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "exifread"])
        import exifread
    
    main()