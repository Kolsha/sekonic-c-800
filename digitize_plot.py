import cv2
import numpy as np
import pytesseract
import os
import csv
import re
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass


@dataclass
class PlotData:
    roi: np.ndarray
    y_max: int

# --- Configuration ---
IMAGE_SIZE = (1940, 1366)

# You might need to adjust these pixel coordinates if your images have a different layout or resolution.
# These values are (y1, y2, x1, x2) for cropping.
CROP_COORDS = {
    "title": (0, 50, 0, IMAGE_SIZE[0]),
    # (120, 40, 160, 1220)
    "y_axis_labels": (40, 1220, 120, 160),
    "plot_area": (60, 1205, 180, 1880), # The core area with the colored graph
    "cct_footer": (1320, IMAGE_SIZE[1], 0, IMAGE_SIZE[0]),
}


# The known data range for the X-axis (Wavelength)
X_AXIS_RANGE = (380, 780) 

# --- End of Configuration ---

def verify_image_size(image, image_path, strict=True):
    """
    Verifies that the image matches the expected dimensions.
    
    Args:
        image: The loaded OpenCV image
        image_path: Path to the image file (for error reporting)
        strict: If True, raises an error on size mismatch. If False, prints warning.
    
    Returns:
        bool: True if size matches or strict=False, False if size mismatch and strict=True
    """
    actual_height, actual_width = image.shape[:2]
    expected_width, expected_height = IMAGE_SIZE
    
    if (actual_width, actual_height) != IMAGE_SIZE:
        message = (f"Image size mismatch for {os.path.basename(image_path)}\n"
                  f"  Expected: {expected_width}x{expected_height} (WxH)\n"
                  f"  Actual:   {actual_width}x{actual_height} (WxH)")
        
        if strict:
            print(f"Error: {message}")
            print(f"  Please verify the image resolution or update IMAGE_SIZE constant")
            return False
        else:
            print(f"Warning: {message}")
            print(f"  Proceeding with processing, but results may be inaccurate")
            return True
    
    return True

def extract_text_from_roi(image, roi_key):
    """Crops the image to a given ROI and extracts text using Tesseract OCR."""
    coords = CROP_COORDS[roi_key]
    roi = image[coords[0]:coords[1], coords[2]:coords[3]]
    # show_image(roi, roi_key)

    # Pre-processing for better OCR: convert to grayscale and threshold
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh_roi = cv2.threshold(gray_roi, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Use pytesseract to extract text
    text = pytesseract.image_to_string(thresh_roi, config='--psm 6').strip()
    return text

def show_image(image, title):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_y_axis_max(image):
    """Extracts the maximum value from the Y-axis labels."""
    y_axis_text = extract_text_from_roi(image, 'y_axis_labels')
    # Find all numbers in the extracted text
    numbers = [int(n) for n in re.findall(r'\d+', y_axis_text)]
    if not numbers:
        raise ValueError("Could not read Y-axis labels. Check CROP_COORDS['y_axis_labels'].")
    return max(numbers)

def get_cct_value(image):
    """Extracts the CCT value from the image footer."""
    footer_text = extract_text_from_roi(image, 'cct_footer')
    match = re.search(r'CCT\s*=\s*(\d+)', footer_text)
    if match:
        return int(match.group(1))
    return None

def digitize_plot(image_path, strict_size_check=True):
    """
    Main function to process a single plot image and return its data.
    
    Args:
        image_path: Path to the image file to process
        strict_size_check: If True, requires exact size match. If False, allows different sizes with warning.
    """
    print(f"Processing {os.path.basename(image_path)}...")
    
    # 1. Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not open image {image_path}")
        return None, None, None
    
    # Verify image size matches expected dimensions
    actual_height, actual_width = image.shape[:2]
    print(f"  - Image size: {actual_width}x{actual_height} (WxH)")
    
    if not verify_image_size(image, image_path, strict=strict_size_check):
        return None, None, None

    # 2. Extract metadata
    try:
        
        y_max = get_y_axis_max(image)
        cct = get_cct_value(image)
        print(f"  - Detected Y-axis max: {y_max}")
        print(f"  - Detected CCT: {cct}K")
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        return None, None, None

    # 3. Isolate and process the plot area
    coords = CROP_COORDS['plot_area']
    plot_roi = image[coords[0]:coords[1], coords[2]:coords[3]]
    
    
    plot_height, plot_width, _ = plot_roi.shape
    
    # The background is white/near-white. We can define a threshold.
    # Pixels with B, G, or R values below this are considered part of the plot.
    background_threshold = 240
    
    # Replace black pixels with white before processing
    plot_roi_gray = cv2.cvtColor(plot_roi, cv2.COLOR_BGR2GRAY)
    # Replace very dark pixels (black) with white
    plot_roi_gray[plot_roi_gray < 10] = 255
    
    plot_roi_filtered = cv2.threshold(plot_roi_gray, background_threshold, 255, cv2.THRESH_BINARY)[1]
    # show_image(plot_roi, 'plot_area')
    # exit(0)

    
    spectral_data = []

    # 4. Iterate through each pixel column (X-axis)
    for x_pixel in range(plot_width):
        column = plot_roi_filtered[:, x_pixel]
        
        # Find all pixels in the column that are not background
        # np.where returns a tuple of arrays; we want the first one for row indices
        non_background_indices = np.where(column < background_threshold)[0]
        # print(non_background_indices)
        
        y_pixel = None
        if non_background_indices.size > 0:
            # The top-most non-background pixel is the peak of our curve for this x.
            # In image coordinates, a smaller y means higher up.
            y_pixel = np.min(non_background_indices)
            
        # 5. Convert pixel coordinates to data values
        # Wavelength (X-axis)
        wavelength = X_AXIS_RANGE[0] + (x_pixel / plot_width) * (X_AXIS_RANGE[1] - X_AXIS_RANGE[0])
        
        # Intensity (Y-axis)
        intensity = 0.0
        if y_pixel is not None:
            # Y-axis is inverted: y_pixel=0 is at the top (max value)
            intensity = (plot_height - y_pixel) / plot_height * y_max
        
        spectral_data.append((wavelength, intensity))

    return spectral_data, cct, PlotData(plot_roi, y_max)

def save_data_to_csv(data, cct, output_path):
    """Saves the extracted spectral data to a CSV file."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write metadata as a commented header
        writer.writerow([f'# Spectral Distribution Data'])
        if cct:
            writer.writerow([f'# CCT = {cct}K'])
        
        # Write data header
        writer.writerow(['Wavelength[nm]', 'Intensity[mW·m⁻²·nm⁻¹]'])
        
        # Write data rows
        for wavelength, intensity in data:
            writer.writerow([f'{wavelength:.2f}', f'{intensity:.4f}'])
    print(f"  - Data successfully saved to {output_path}")

def load_spectral_data(csv_path):
    """
    Load spectral data from a CSV file.
    
    Args:
        csv_path: Path to the CSV file containing spectral data
        
    Returns:
        tuple: (wavelengths, intensities, cct_value, filename)
               Returns (None, None, None, None) if file cannot be loaded
    """
    try:
        wavelengths = []
        intensities = []
        cct_value = None
        filename = Path(csv_path).stem
        
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            
        # Parse metadata from comments
        for line in lines:
            if line.startswith('# CCT = '):
                cct_match = re.search(r'CCT = (\d+)', line)
                if cct_match:
                    cct_value = int(cct_match.group(1))
                    
        # Read CSV data
        csv_reader = csv.reader(lines)
        header_found = False
        
        for row in csv_reader:
            if not row or row[0].startswith('#'):
                continue
            if not header_found and 'Wavelength' in row[0]:
                header_found = True
                continue
            if header_found:
                try:
                    wavelength = float(row[0])
                    intensity = float(row[1])
                    wavelengths.append(wavelength)
                    intensities.append(intensity)
                except (ValueError, IndexError):
                    continue
                    
        return np.array(wavelengths), np.array(intensities), cct_value, filename
        
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None, None, None, None

def display_spectral_data(csv_path, save_plot=False, output_dir=None, plot_data=None):
    """
    Display a single spectral distribution plot with source image as background.
    
    Args:
        csv_path: Path to the CSV file containing spectral data
        save_plot: If True, save the plot as PNG file
        output_dir: Directory to save plot (default: same as CSV file)
        plot_data: Plot data to use as background (default: None)
    """
    wavelengths, intensities, cct_value, filename = load_spectral_data(csv_path)
    
    if wavelengths is None:
        print(f"Could not load data from {csv_path}")
        return

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Try to load and display source image as background

    if plot_data is not None :
        try:
            import matplotlib.image as mpimg
            # background_image = mpimg.imread(str())
            
            # Display the image as background with reduced opacity
            ax.imshow(plot_data.roi, aspect='auto', alpha=0.3, 
                     extent=[X_AXIS_RANGE[0], X_AXIS_RANGE[1], 0, plot_data.y_max])
        except Exception as e:
            print(f"Could not load background image: {e}")
    
    # Plot the spectral data with enhanced visibility for overlay
    ax.plot(wavelengths, intensities, linewidth=3, color='red', 
            label=f'{filename} (CCT: {cct_value}K)', alpha=0.9)
    
    # Add a subtle white outline to make the line more visible against the background
    ax.plot(wavelengths, intensities, linewidth=5, color='white', alpha=0.6, zorder=1)
    ax.plot(wavelengths, intensities, linewidth=3, color='red', alpha=0.9, zorder=2)
    
    # Customize the plot
    ax.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Intensity (mW·m⁻²·nm⁻¹)', fontsize=12, fontweight='bold')
    ax.set_title(f'Spectral Distribution - {filename}', fontsize=14, fontweight='bold')
    
    # Enhanced grid for better visibility over image background
    ax.grid(True, alpha=0.5, color='white', linewidth=1)
    ax.grid(True, alpha=0.3, color='black', linewidth=0.5)
    
    # Enhanced legend with background
    legend = ax.legend(fancybox=True, shadow=True, framealpha=0.9)
    legend.get_frame().set_facecolor('white')
    
    # Set wavelength range to standard visible spectrum
    ax.set_xlim(X_AXIS_RANGE[0], X_AXIS_RANGE[1])
    
    # Set y-axis to start from 0 and add some headroom
    y_max = plot_data.y_max if plot_data is not None else max(intensities) * 1.1
    ax.set_ylim(0, y_max)
    
    # Add some styling
    plt.tight_layout()
    
    if save_plot:
        if output_dir is None:
            output_dir = Path(csv_path).parent
        output_path = Path(output_dir) / f'{filename}_spectral_plot_with_background.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    plt.show()


def main():
    """
    Finds all image files in the input directory, processes them, 
    and saves the output to CSV files in the output directory.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Digitize spectral plots from images')
    parser.add_argument('--input-dir', '-i', default='images', 
                       help='Input directory containing images (default: current directory)')
    parser.add_argument('--output-dir', '-o', default='output_csv',
                       help='Output directory for CSV files (default: output_csv)')
    parser.add_argument('--allow-size-mismatch', action='store_true',
                       help='Allow processing images with different sizes (with warning)')
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    # Validate input directory
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_files = []
    
    # Find all image files in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(supported_extensions):
            image_files.append(filename)
    
    if not image_files:
        print(f"No image files found in '{input_dir}' with supported extensions: {supported_extensions}")
        return
    
    print(f"Found {len(image_files)} image file(s) in '{input_dir}'")
    
    # Process each image file
    for filename in image_files:
        image_path = os.path.join(input_dir, filename)
        
        spectral_data, cct, plot_data = digitize_plot(image_path, strict_size_check=not args.allow_size_mismatch)
        
        if spectral_data:
            # Create a meaningful output filename
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f'{base_name}.csv')
            save_data_to_csv(spectral_data, cct, output_path)
            display_spectral_data(output_path, plot_data=plot_data)
            

if __name__ == '__main__':
    # On Windows, you might need to point pytesseract to the tesseract.exe
    # Example: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    main()
