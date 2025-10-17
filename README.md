# Sekonic C-800 Spectrometer Tools

A comprehensive Python toolkit for parsing Sekonic C-800 spectrometer memory backup files and digitizing spectral plots from images.

## Features

### 📊 Memory Backup Parser
- **Parse C-800 backup files** (.hex or binary format)
- **Extract measurement data**: CCT, illuminance, CRI values, CIE coordinates
- **Spectral Power Distribution (SPD)** data extraction
- **Export to CSV** with customizable naming
- **Visualization** with matplotlib plots

### 🖼️ Plot Digitization
- **OCR-based text extraction** from spectral plot images
- **Automatic plot area detection** and data extraction
- **Background image overlay** for validation
- **CSV export** with metadata (CCT values, etc.)

## Installation

```bash
git clone https://github.com/yourusername/sekonic-c-800.git
cd sekonic-c-800
pip install -r requirements.txt
```

### Dependencies
- `opencv-python>=4.8.0` - Image processing
- `numpy>=1.21.0` - Numerical operations
- `pytesseract>=0.3.10` - OCR text extraction
- `matplotlib>=3.6.0` - Plotting and visualization
- `pandas>=1.5.0` - Data manipulation

## Usage

### Memory Backup Parser

```python
from sekonic_c800_parser import SekonicC800Parser

# Parse backup file
parser = SekonicC800Parser('backup.hex')
measurements = parser.parse()

# Access measurement data
for measurement in measurements:
    print(f"CCT: {measurement.cct}K")
    print(f"Illuminance: {measurement.illuminance_lx} lx")
    print(f"CRI Ra: {measurement.cri_ra}")
    print(f"CIE (x,y): ({measurement.cie_x}, {measurement.cie_y})")
```

### Command Line Interface

```bash
# Basic parsing
python sekonic_c800_parser.py backup.hex

# Export SPD data to CSV
python sekonic_c800_parser.py backup.hex -o ./csv_exports

# Generate plots
python sekonic_c800_parser.py backup.hex --plot

# Combined export and plotting
python sekonic_c800_parser.py backup.hex -o ./data --plot
```

### Plot Digitization

```python
from utils.digitize_plot import digitize_plot

# Process single image
spectral_data, cct, plot_data = digitize_plot('spectral_plot.png')

# Save to CSV
save_data_to_csv(spectral_data, cct, 'output.csv')
```

```bash
# Batch process images
python utils/digitize_plot.py --input-dir images --output-dir output_csv

# Allow different image sizes
python utils/digitize_plot.py -i images -o output --allow-size-mismatch
```

## File Structure

```
sekonic-c-800/
├── sekonic_c800_parser.py    # Main parser module
├── utils/
│   └── digitize_plot.py      # Plot digitization utilities
├── files/                    # Sample data files
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
└── README.md                # This file
```

## Data Formats

### Input Files
- **Memory backups**: `.hex` or binary `.mem` files from Sekonic C-800
- **Plot images**: PNG, JPG, JPEG, BMP, TIFF formats

### Output Files
- **CSV exports**: Spectral data with wavelength and intensity columns
- **Metadata**: CCT values, timestamps, measurement IDs

## Example Output

### Parsed Measurement Data
```
Measurement 1 (ID: 1):
  Profile: iPhone-13-Pro
  Timestamp: 2024-01-15 14:30:25
  CCT: 6500.0K
  Illuminance: 13.2 lx
  CRI Ra: 95.2
  CIE (x,y): (0.3127, 0.3290)
  SPD points: 400
```

### CSV Export Format
```csv
Wavelength (nm),Spectral Power Density (mW·m⁻²·nm⁻¹)
380.0,0.000123
381.0,0.000145
...
```

## Configuration

### Plot Digitization Settings
Edit `utils/digitize_plot.py` to adjust:
- `IMAGE_SIZE`: Expected image dimensions
- `CROP_COORDS`: ROI coordinates for different plot areas
- `X_AXIS_RANGE`: Wavelength range (default: 380-780 nm)

## Requirements

- Python 3.7+
- Tesseract OCR (for plot digitization)
- OpenCV for image processing

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues and questions, please open an issue on GitHub.

---

**Note**: This toolkit is designed for Sekonic C-800 spectrometer data. For other spectrometer models, the file format parsing may need adaptation.

Also take a look at [skreader-go](https://github.com/akares/skreader-go) project.

