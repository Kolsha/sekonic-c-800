# Spectral Distribution Plot Digitizer

This Python script automates the extraction of spectral distribution data from plot images. It uses OCR (Optical Character Recognition) and image processing to convert graph images into CSV data files.

## Features

- **Automated OCR**: Extracts metadata like CCT (Correlated Color Temperature) and Y-axis scale from images
- **Plot digitization**: Converts colored curves in graphs to numerical data points
- **Batch processing**: Processes all supported image files in the directory
- **CSV export**: Saves extracted data in CSV format with proper headers and metadata

## Prerequisites

### 1. Python Dependencies

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install opencv-python numpy pytesseract
```

### 2. Tesseract OCR Engine

The script requires Google's Tesseract OCR engine to be installed on your system:

#### macOS
```bash
brew install tesseract
```

#### Windows
Download and install from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki). Make sure to add it to your system's PATH during installation.

#### Linux (Debian/Ubuntu)
```bash
sudo apt-get install tesseract-ocr
```

## Usage

### Basic Usage

1. Place your spectral distribution images in the same directory as `digitize_plot.py`
2. Run the script:
   ```bash
   python digitize_plot.py
   ```
3. Check the `output_csv` folder for the generated CSV files

### Supported Image Formats

- PNG (.png)
- JPEG (.jpg, .jpeg)
- BMP (.bmp)
- TIFF (.tiff)

## Configuration

The script includes a configuration section at the top that defines crop coordinates for different regions of the plot:

```python
CROP_COORDS = {
    "title": (0, 40, 0, 780),
    "y_axis_labels": (40, 380, 0, 40),
    "plot_area": (41, 378, 55, 765),
    "cct_footer": (380, 420, 0, 780),
}
```

### Adjusting for Different Image Layouts

If your images have a different layout or resolution, you may need to adjust these coordinates:

1. Open an image in an image editor (Paint, GIMP, etc.)
2. Find the pixel coordinates of each region
3. Update the `CROP_COORDS` dictionary accordingly
4. Coordinates are in format: `(y1, y2, x1, x2)`

## Output Format

The generated CSV files include:

1. **Metadata header**: Comments with extraction information and CCT value
2. **Data columns**: Wavelength (nm) and Intensity (mW·m⁻²·nm⁻¹)
3. **Data points**: One row per wavelength measurement

### Example Output

```csv
# Spectral Distribution Data
# CCT = 4719K
Wavelength[nm],Intensity[mW·m⁻²·nm⁻¹]
380.00,0.0000
380.56,0.0000
458.23,8.8754
458.79,8.9112
535.11,8.4521
535.67,8.3995
620.45,10.2111
621.01,10.1553
779.44,0.0000
780.00,0.0000
```

## Troubleshooting

### OCR Issues

If the script fails to read Y-axis labels or CCT values:

1. Check image quality and resolution
2. Verify the `CROP_COORDS` are correctly set for your images
3. Ensure text in the image is clear and not rotated

### Curve Detection Issues

The script assumes:
- White or light gray background (RGB values > 240)
- Colored curves that contrast with the background
- Standard spectral distribution plot layout

### Windows Tesseract Path

If you're on Windows and get a Tesseract error, uncomment and modify this line in the script:

```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

## Technical Details

### Processing Steps

1. **Image loading**: Reads image files using OpenCV
2. **Region extraction**: Crops specific areas for OCR and plot processing
3. **OCR processing**: Extracts text using Tesseract with preprocessing
4. **Plot digitization**: Scans each pixel column to find curve points
5. **Coordinate transformation**: Converts pixel coordinates to real data values
6. **Data export**: Saves results to CSV with proper formatting

### Algorithm Notes

- **Y-axis scaling**: Automatically detected from OCR of axis labels
- **X-axis range**: Fixed to 380-780 nm (standard visible spectrum)
- **Curve detection**: Uses color thresholding to identify non-background pixels
- **Peak finding**: Selects topmost colored pixel in each column as the curve point

## License

This script is provided as-is for educational and research purposes.
