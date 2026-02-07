#!/usr/bin/env python3
# -*-coding:utf-8-*-

"""
CSI Data Preprocessing Pipeline for People Counting
====================================================

This module implements a complete preprocessing pipeline for CSI data
with strict MIMO synchronization and feature extraction.

Features:
- Loads CSI data from multiple MIMO links (4 links: rx1_mac1, rx2_mac1, rx1_mac2, rx2_mac2)
- Enforces strict timestamp synchronization across all 4 links
- Extracts amplitude and phase features from complex CSI values
- Applies signal filtering (Hampel filter, optional Savitzky-Golay)
- Discards non-synchronized samples
- Outputs preprocessed dataset ready for modeling
"""

import pandas as pd
import numpy as np
import json
import os
from typing import List, Dict, Tuple, Optional
from scipy import signal
from scipy.signal import savgol_filter


class CSIPreprocessor:
    """
    Preprocessor for CSI data with MIMO synchronization.
    """
    
    def __init__(self, time_window_ms: int = 50, apply_hampel: bool = True, 
                 apply_savgol: bool = False, hampel_window: int = 5, hampel_n_sigma: float = 3.0):
        """
        Initialize the CSI preprocessor.
        
        Args:
            time_window_ms: Maximum time difference (ms) for considering samples synchronized
            apply_hampel: Whether to apply Hampel filter for outlier detection
            apply_savgol: Whether to apply Savitzky-Golay filter for smoothing
            hampel_window: Window size for Hampel filter
            hampel_n_sigma: Number of standard deviations for Hampel outlier detection
        """
        self.time_window_ms = time_window_ms
        self.apply_hampel = apply_hampel
        self.apply_savgol = apply_savgol
        self.hampel_window = hampel_window
        self.hampel_n_sigma = hampel_n_sigma
        
    def load_csi_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load CSI data from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with CSI data
        """
        print(f"Loading CSI data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows")
        return df
    
    def parse_csi_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse CSI data from JSON string to complex array.
        
        Args:
            df: DataFrame with 'data' column containing JSON strings
            
        Returns:
            DataFrame with parsed CSI complex values
        """
        print("Parsing CSI data strings to complex arrays...")
        
        def parse_csi_string(csi_str):
            """Parse CSI JSON string to complex array."""
            try:
                csi_raw = json.loads(csi_str)
                # Convert I/Q pairs to complex numbers
                # Format: [I0, Q0, I1, Q1, ..., In, Qn]
                csi_complex = []
                for i in range(0, len(csi_raw), 2):
                    real = csi_raw[i+1]  # I component
                    imag = csi_raw[i]     # Q component
                    csi_complex.append(complex(real, imag))
                return csi_complex
            except (json.JSONDecodeError, IndexError):
                return None
        
        # Parse CSI data
        df['csi_complex'] = df['data'].apply(parse_csi_string)
        
        # Remove rows with parsing errors
        initial_count = len(df)
        df = df[df['csi_complex'].notna()].copy()
        removed = initial_count - len(df)
        if removed > 0:
            print(f"Removed {removed} rows with parsing errors")
        
        return df
    
    def extract_amplitude_phase(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract amplitude and phase from complex CSI values.
        
        Args:
            df: DataFrame with 'csi_complex' column
            
        Returns:
            DataFrame with amplitude and phase features
        """
        print("Extracting amplitude and phase features...")
        
        def get_amplitude(csi_complex):
            """Calculate amplitude (magnitude) of complex CSI."""
            return [abs(c) for c in csi_complex]
        
        def get_phase(csi_complex):
            """Calculate phase of complex CSI."""
            return [np.angle(c) for c in csi_complex]
        
        df['csi_amplitude'] = df['csi_complex'].apply(get_amplitude)
        df['csi_phase'] = df['csi_complex'].apply(get_phase)
        
        return df
    
    def apply_hampel_filter(self, series: np.ndarray) -> np.ndarray:
        """
        Apply Hampel filter to detect and remove outliers.
        
        Args:
            series: Input signal
            
        Returns:
            Filtered signal
        """
        n = len(series)
        filtered = series.copy()
        k = self.hampel_window // 2
        
        for i in range(k, n - k):
            window = series[i - k:i + k + 1]
            median = np.median(window)
            mad = np.median(np.abs(window - median))
            threshold = self.hampel_n_sigma * 1.4826 * mad  # Scale factor for MAD
            
            if np.abs(series[i] - median) > threshold:
                filtered[i] = median
        
        return filtered
    
    def filter_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply signal filtering to CSI amplitude data.
        
        Args:
            df: DataFrame with 'csi_amplitude' column
            
        Returns:
            DataFrame with filtered amplitude
        """
        print("Applying signal filters...")
        
        def filter_amplitude(amp_array):
            """Apply filters to amplitude array."""
            amp = np.array(amp_array)
            
            # Apply Hampel filter
            if self.apply_hampel:
                amp = self.apply_hampel_filter(amp)
            
            # Apply Savitzky-Golay filter
            if self.apply_savgol and len(amp) > 5:
                try:
                    amp = savgol_filter(amp, window_length=5, polyorder=2)
                except:
                    pass  # Skip if not enough points
            
            return amp.tolist()
        
        df['csi_amplitude_filtered'] = df['csi_amplitude'].apply(filter_amplitude)
        
        return df
    
    def synchronize_mimo_links(self, link_data: Dict[str, pd.DataFrame], 
                                timestamp_col: str = 'local_timestamp') -> pd.DataFrame:
        """
        Synchronize 4 MIMO links using timestamps.
        Only keep samples where all 4 links have data within the time window.
        
        Args:
            link_data: Dictionary mapping link names to DataFrames
                      Expected keys: 'rx1_mac1', 'rx2_mac1', 'rx1_mac2', 'rx2_mac2'
            timestamp_col: Column name containing timestamps
            
        Returns:
            Synchronized DataFrame with data from all 4 links
        """
        print("\n=== Strict MIMO Synchronization ===")
        print(f"Time window tolerance: {self.time_window_ms} ms")
        
        required_links = ['rx1_mac1', 'rx2_mac1', 'rx1_mac2', 'rx2_mac2']
        
        # Check all required links are present
        missing_links = [link for link in required_links if link not in link_data]
        if missing_links:
            raise ValueError(f"Missing required MIMO links: {missing_links}")
        
        # Print initial sample counts
        print("\nInitial sample counts per link:")
        for link in required_links:
            print(f"  {link}: {len(link_data[link])} samples")
        
        # Sort each link by timestamp
        for link in required_links:
            link_data[link] = link_data[link].sort_values(timestamp_col).reset_index(drop=True)
        
        # Use the link with fewest samples as reference
        ref_link = min(required_links, key=lambda x: len(link_data[x]))
        print(f"\nUsing {ref_link} as reference link ({len(link_data[ref_link])} samples)")
        
        synchronized_data = []
        discarded_count = 0
        
        # For each timestamp in reference link, find matching timestamps in other links
        for idx, row in link_data[ref_link].iterrows():
            ref_timestamp = row[timestamp_col]
            
            # Find matching samples in other links within time window
            matched_samples = {ref_link: row}
            
            for link in required_links:
                if link == ref_link:
                    continue
                
                # Find samples within time window
                time_diffs = np.abs(link_data[link][timestamp_col] - ref_timestamp)
                within_window = time_diffs <= self.time_window_ms
                
                if within_window.any():
                    # Get closest match
                    closest_idx = time_diffs[within_window].idxmin()
                    matched_samples[link] = link_data[link].loc[closest_idx]
                else:
                    # No match found for this link
                    matched_samples = None
                    break
            
            # Only keep if all 4 links have matching samples
            if matched_samples is not None and len(matched_samples) == 4:
                # Combine data from all links
                combined_sample = {
                    'timestamp': ref_timestamp
                }
                
                for link, sample in matched_samples.items():
                    # Add link-specific features with prefix
                    for col in sample.index:
                        if col not in ['timestamp', timestamp_col]:
                            combined_sample[f'{link}_{col}'] = sample[col]
                
                synchronized_data.append(combined_sample)
            else:
                discarded_count += 1
        
        print(f"\n✓ Synchronized samples: {len(synchronized_data)}")
        print(f"✗ Discarded samples (not synchronized): {discarded_count}")
        print(f"Synchronization rate: {len(synchronized_data) / len(link_data[ref_link]) * 100:.1f}%")
        
        if len(synchronized_data) == 0:
            raise ValueError("No synchronized samples found! Check timestamp alignment.")
        
        return pd.DataFrame(synchronized_data)
    
    def preprocess_pipeline(self, link_files: Dict[str, str], 
                           output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Complete preprocessing pipeline for CSI data.
        
        Args:
            link_files: Dictionary mapping link names to file paths
                       Expected keys: 'rx1_mac1', 'rx2_mac1', 'rx1_mac2', 'rx2_mac2'
            output_file: Optional path to save preprocessed data
            
        Returns:
            Preprocessed and synchronized DataFrame
        """
        print("\n" + "="*60)
        print("CSI PREPROCESSING PIPELINE FOR PEOPLE COUNTING")
        print("="*60)
        
        # Load and preprocess each link
        link_data = {}
        
        for link_name, filepath in link_files.items():
            print(f"\n--- Processing {link_name} ---")
            
            # Load data
            df = self.load_csi_csv(filepath)
            
            # Parse CSI strings
            df = self.parse_csi_data(df)
            
            # Extract features
            df = self.extract_amplitude_phase(df)
            
            # Apply filters
            df = self.filter_signals(df)
            
            link_data[link_name] = df
        
        # Synchronize all links
        synchronized_df = self.synchronize_mimo_links(link_data)
        
        # Save if requested
        if output_file:
            print(f"\nSaving preprocessed data to {output_file}...")
            synchronized_df.to_csv(output_file, index=False)
            print(f"✓ Saved {len(synchronized_df)} synchronized samples")
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60 + "\n")
        
        return synchronized_df


def main():
    """
    Example usage of the CSI preprocessing pipeline.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Preprocess CSI data for people counting with MIMO synchronization"
    )
    parser.add_argument('--rx1_mac1', required=True, help='CSV file for rx1_mac1 link')
    parser.add_argument('--rx2_mac1', required=True, help='CSV file for rx2_mac1 link')
    parser.add_argument('--rx1_mac2', required=True, help='CSV file for rx1_mac2 link')
    parser.add_argument('--rx2_mac2', required=True, help='CSV file for rx2_mac2 link')
    parser.add_argument('--output', required=True, help='Output CSV file for preprocessed data')
    parser.add_argument('--time_window', type=int, default=50, 
                       help='Time window (ms) for synchronization (default: 50)')
    parser.add_argument('--no_hampel', action='store_true', help='Disable Hampel filter')
    parser.add_argument('--savgol', action='store_true', help='Enable Savitzky-Golay filter')
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = CSIPreprocessor(
        time_window_ms=args.time_window,
        apply_hampel=not args.no_hampel,
        apply_savgol=args.savgol
    )
    
    # Define link files
    link_files = {
        'rx1_mac1': args.rx1_mac1,
        'rx2_mac1': args.rx2_mac1,
        'rx1_mac2': args.rx1_mac2,
        'rx2_mac2': args.rx2_mac2
    }
    
    # Run preprocessing pipeline
    preprocessed_data = preprocessor.preprocess_pipeline(link_files, args.output)
    
    print(f"\nPreprocessed data shape: {preprocessed_data.shape}")
    print(f"Columns: {list(preprocessed_data.columns)[:10]}... ({len(preprocessed_data.columns)} total)")


if __name__ == '__main__':
    main()
