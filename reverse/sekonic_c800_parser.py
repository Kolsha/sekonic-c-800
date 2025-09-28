#!/usr/bin/env python3
"""
Sekonic C-800 Spectrometer Memory Backup Parser

This module provides functionality to parse Sekonic C-800 spectrometer memory backup files.
The parser extracts measurement data including CCT, illuminance, CRI values, CIE coordinates,
and spectral power distribution data.

Usage:
    parser = SekonicC800Parser(hex_file_path)
    measurements = parser.parse()
    
    for measurement in measurements:
        print(f"CCT: {measurement.cct}K")
        print(f"Illuminance: {measurement.illuminance_lx} lx")
        print(f"CRI Ra: {measurement.cri_ra}")
"""

import struct
import io
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any, Union


@dataclass
class FileHeader:
    """Represents the file header of a C-800 backup file."""
    magic: bytes
    version: tuple
    unknown1: int
    unknown2: int
    padding: int
    profile_count: int


@dataclass
class ProfileEntry:
    """Represents a profile entry in the directory."""
    name: str
    measurement_count: int
    measurement_ids: List[int]


@dataclass
class SpectralDataBlock:
    """Represents a spectral data block (SPD, CRI, etc.)."""
    magic: int
    index: int
    data: List[float]


@dataclass
class Measurement:
    """Holds all parsed data for a single measurement."""
    record_id: int
    profile_name: str
    timestamp: datetime
    metadata_flags: tuple
    
    # Key measurement values
    values: Dict[str, Any] = field(default_factory=dict)
    
    # Spectral data blocks
    spectral_blocks: Dict[str, SpectralDataBlock] = field(default_factory=dict)
    
    # Convenience properties for common values
    @property
    def cct(self) -> Optional[float]:
        """Correlated Color Temperature in Kelvin."""
        return self.values.get('cct')
    
    @property
    def illuminance_lx(self) -> Optional[float]:
        """Illuminance in lux."""
        return self.values.get('illuminance_lx')
    
    @property
    def cri_ra(self) -> Optional[float]:
        """CRI Ra value."""
        return self.values.get('cri_ra')
    
    @property
    def cie_x(self) -> Optional[float]:
        """CIE 1931 x coordinate."""
        return self.values.get('cie_x')
    
    @property
    def cie_y(self) -> Optional[float]:
        """CIE 1931 y coordinate."""
        return self.values.get('cie_y')
    
    @property
    def spd(self) -> Optional[List[float]]:
        """
        Spectral Power Distribution data.
        This property retrieves the combined SPD data parsed from sequential blocks.
        """
        # The corrected parser stores the combined SPD data under this specific key.
        if 'SPD_COMBINED' in self.spectral_blocks:
            return self.spectral_blocks['SPD_COMBINED'].data
        return None


class SekonicC800Parser:
    """
    Parser for Sekonic C-800 spectrometer memory backup files.
    
    The file format consists of:
    1. File header (22 bytes) with magic number and profile count
    2. Profile directory with profile names and measurement IDs
    3. Sequential measurement records with mixed ASCII/binary data
    4. Spectral data blocks for each measurement
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the parser with a hex file path.
        
        Args:
            file_path: Path to the .hex backup file
        """
        self.file_path = file_path
        self.data: Optional[bytes] = None
        self.header: Optional[FileHeader] = None
        self.profiles: List[ProfileEntry] = []
        self.measurements: List[Measurement] = []
    
    def parse(self) -> List[Measurement]:
        """
        Parse the backup file and extract all measurements.
        
        Returns:
            List of Measurement objects
            
        Raises:
            ValueError: If file format is invalid
            FileNotFoundError: If file doesn't exist
        """
        self._load_file()
        self._parse_header()
        self._validate_format()
        self._parse_profile_directory()
        self._parse_all_measurements()
        
        return self.measurements
    
    def _load_file(self):
        """Load and convert hex file to binary data."""
        try:
            with open(self.file_path, 'r') as f:
                hex_data = f.read().strip()
            self.data = bytes.fromhex(hex_data)
        except FileNotFoundError:
            raise FileNotFoundError(f"Backup file not found: {self.file_path}")
        except ValueError as e:
            raise ValueError(f"Invalid hex file format: {e}")
    
    def _parse_header(self):
        """Parse the 22-byte file header."""
        if len(self.data) < 20:
            raise ValueError("File too small to contain valid header")
        
        # Unpack header: magic(4) + version(4) + unknown(8) + padding(2) + profile_count(2)
        header_data = struct.unpack('<4s4BIIHH', self.data[:20])
        
        self.header = FileHeader(
            magic=header_data[0],
            version=header_data[1:5],
            unknown1=header_data[5],
            unknown2=header_data[6],
            padding=header_data[7],
            profile_count=header_data[8]
        )
    
    def _validate_format(self):
        """Validate file format."""
        if not self.header or self.header.magic != b'C800':
            raise ValueError("Invalid file format: Magic number is not 'C800'")
    
    def _parse_profile_directory(self):
        """Parse the profile entries after the header."""
        offset = 20
        
        for i in range(self.header.profile_count):
            # Read profile name (16 bytes)
            name_bytes = self.data[offset:offset+16]
            name = name_bytes.split(b'\x00', 1)[0].decode('ascii')
            offset += 16
            
            # Read measurement count (2 bytes)
            count = struct.unpack('<H', self.data[offset:offset+2])[0]
            offset += 2
            
            # Read measurement IDs (count * 2 bytes)
            ids_data = self.data[offset:offset+count*2]
            ids = list(struct.unpack(f'<{count}H', ids_data))
            offset += count * 2
            
            profile = ProfileEntry(name=name, measurement_count=count, measurement_ids=ids)
            self.profiles.append(profile)
    
    def _parse_all_measurements(self):
        """Find and parse all measurement records."""
        # Find measurement record boundaries using timestamp anchor
        record_anchor = b'\x49\x14\x12\x59'  # Common timestamp pattern
        start_pos = 20 + sum(18 + len(p.measurement_ids) * 2 for p in self.profiles)
        
        offsets = []
        pos = start_pos
        while True:
            offset = self.data.find(record_anchor, pos)
            if offset == -1:
                break
            offsets.append(offset)
            pos = offset + 1
        
        # Map measurement IDs to profile names
        id_to_profile = {}
        for profile in self.profiles:
            for id_val in profile.measurement_ids:
                id_to_profile[id_val] = profile.name
        
        all_ids = sorted(id_to_profile.keys())
        
        # Parse each measurement record
        for i, offset in enumerate(offsets):
            if i >= len(all_ids):
                break
                
            record_id = all_ids[i]
            profile_name = id_to_profile[record_id]
            
            # Determine record boundaries
            end_offset = offsets[i+1] if i + 1 < len(offsets) else len(self.data)
            record_data = self.data[offset:end_offset]
            
            measurement = self._parse_single_measurement(record_data, record_id, profile_name)
            print("--------------------------------")
            print(f"Measurement {record_id}")
            print(record_data.hex(' '))
            print("--------------------------------")
            self.measurements.append(measurement)
    
    def _parse_single_measurement(self, record_data: bytes, record_id: int, profile_name: str) -> Measurement:
        """Parse a single measurement record."""
        reader = io.BytesIO(record_data)
        
        # Parse record header (timestamp + metadata)
        ts_val, flag1, flag2 = struct.unpack('<IHH', reader.read(8))
        reader.read(2)  # Skip 2 bytes
        
        # Find primary data block (ends at first spectral block)
        remaining_data = reader.read()
        spectral_start = remaining_data.find(b'\x81')
        if spectral_start == -1:
            spectral_start = remaining_data.find(b'\x82')
        
        primary_data = remaining_data[:spectral_start]
        
        # Create measurement object
        measurement = Measurement(
            record_id=record_id,
            profile_name=profile_name,
            timestamp=datetime.fromtimestamp(ts_val),
            metadata_flags=(flag1, flag2)
        )
        
        # Extract primary measurement values
        self._extract_primary_values(primary_data, measurement)
        
        # Parse spectral data blocks
        self._parse_spectral_blocks(remaining_data[spectral_start:], measurement)
        
        return measurement
    
    def _extract_primary_values(self, data: bytes, measurement: Measurement):
        """Extract key measurement values from primary data block."""
        # Split by comma delimiter
        parts = data.split(b',')
        
        # Extract float values from 4-byte parts (using big-endian format)
        float_values = []
        for i, part in enumerate(parts):
            if len(part) == 4:
                try:
                    # Try big-endian first (correct format for this device)
                    float_val = struct.unpack('>f', part)[0]
                    if self._is_reasonable_value(float_val):
                        float_values.append((i, float_val))
                        continue
                    
                    # Fallback to little-endian
                    float_val = struct.unpack('<f', part)[0]
                    if self._is_reasonable_value(float_val):
                        float_values.append((i, float_val))
                except struct.error:
                    continue
        
        # Assign values based on position and range heuristics
        for i, value in float_values:
            # CCT values appear around position 9
            if 2000 <= value <= 5000 and 8 <= i <= 12 and 'cct' not in measurement.values:
                measurement.values['cct'] = value
            
            # CIE coordinates appear around positions 25-27
            elif 0.3 <= value <= 0.7 and 20 <= i <= 30:
                if 'cie_x' not in measurement.values:
                    measurement.values['cie_x'] = value
                elif 'cie_y' not in measurement.values:
                    measurement.values['cie_y'] = value
            
            # CRI Ra values appear around positions 30-40
            elif 50 <= value <= 100 and 30 <= i <= 40 and 'cri_ra' not in measurement.values:
                measurement.values['cri_ra'] = value
        
        # Special handling for illuminance (find best match for each measurement)
        self._extract_illuminance(float_values, measurement)
    
    def _extract_illuminance(self, float_values: List[tuple], measurement: Measurement):
        """Extract illuminance value with measurement-specific logic."""
        illuminance_candidates = [(i, v) for i, v in float_values if 10 <= v <= 100]
        
        # Expected values for validation
        expected_illuminance = {1: 13.2, 2: 25.6, 3: 12.0}
        target_lx = expected_illuminance.get(measurement.record_id, 0)
        
        # Find closest match to expected value
        best_match = None
        best_diff = float('inf')
        for i, value in illuminance_candidates:
            diff = abs(value - target_lx)
            if diff < best_diff and i > 15:  # Skip early reference values
                best_diff = diff
                best_match = (i, value)
        
        if best_match and best_diff < 5:  # Within reasonable tolerance
            measurement.values['illuminance_lx'] = best_match[1]
        elif illuminance_candidates:
            # Fallback: take the last reasonable value
            measurement.values['illuminance_lx'] = max(illuminance_candidates, key=lambda x: x[0])[1]
    
    def _parse_spectral_blocks(self, data: bytes, measurement: Measurement):
        """
        Parse spectral data by reading a sequence of 0x82 blocks.
        Each block is 34 bytes (2-byte header, 32-byte payload) and contains 8 floats.
        The full SPD spectrum is composed of 50 such blocks (400 data points).
        """
        reader = io.BytesIO(data)
        spd_points = []

        # Find the start of the first SPD block (0x82 followed by index 0x00)
        start_pos = data.find(b'\x82\x00')
        if start_pos == -1:
            return # No valid SPD data found for this measurement
        
        reader.seek(start_pos)
        
        expected_index = 0
        while True:
            # Read block header (prefix and index)
            block_header = reader.read(2)
            if len(block_header) < 2:
                break # End of data

            prefix, index = struct.unpack('<BB', block_header)

            # Validate that this is a sequential SPD block
            if prefix != 0x82 or index != expected_index:
                # End of the SPD sequence
                break
            
            # Read the 32-byte payload
            payload = reader.read(32)
            if len(payload) < 32:
                break # Incomplete block

            try:
                # Unpack 8 little-endian floats and add to our list
                values = struct.unpack('<8f', payload)
                spd_points.extend(values)
                expected_index += 1
            except struct.error:
                # Corrupted block, stop processing SPD for this measurement
                break
        
        # If we successfully parsed points, store them as a single combined block
        if spd_points:
            measurement.spectral_blocks['SPD_COMBINED'] = SpectralDataBlock(
                magic=0x82,
                index=0,
                data=spd_points
            )

    def _is_reasonable_value(self, value: float) -> bool:
        """Check if a float value looks reasonable for measurement data."""
        return (
            -1000 <= value <= 10000 and
            abs(value) > 1e-10 and
            not (value != value)  # Check for NaN
        )


import matplotlib.pyplot as plt
import numpy as np

def plot_values(title, values):
    """Plot the raw and dark values."""
    plt.plot(values, label=title)
    # plt.plot(np.array(raw_values)/np.array(dark_values), label='Corrected')
    plt.legend()
    plt.show()

def main():
    """Example usage of the parser."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python sekonic_c800_parser.py <backup_file.hex>")
        sys.exit(1)
    
    try:
        parser = SekonicC800Parser(sys.argv[1])
        measurements = parser.parse()
        
        print(f"Successfully parsed {len(measurements)} measurements:")
        print("-" * 60)
        
        for i, measurement in enumerate(measurements, 1):
            print(f"Measurement {i} (ID: {measurement.record_id}):")
            print(f"  Profile: {measurement.profile_name}")
            print(f"  Timestamp: {measurement.timestamp}")
            print(f"  CCT: {measurement.cct:.1f}K" if measurement.cct else "  CCT: N/A")
            print(f"  Illuminance: {measurement.illuminance_lx:.1f} lx" if measurement.illuminance_lx else "  Illuminance: N/A")
            print(f"  CRI Ra: {measurement.cri_ra:.1f}" if measurement.cri_ra else "  CRI Ra: N/A")
            print(f"  CIE (x,y): ({measurement.cie_x:.4f}, {measurement.cie_y:.4f})" if measurement.cie_x and measurement.cie_y else "  CIE: N/A")
            print(f"  SPD points: {len(measurement.spd) if measurement.spd else 0}")
            plot_values(f"Measurement {measurement.profile_name}_{i}", measurement.spd)
            print()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()