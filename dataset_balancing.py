#!/usr/bin/env python3
# -*-coding:utf-8-*-

"""
Dataset Balancing for People Counting
======================================

This module implements:
1. Aggregation of classes by number of people (not by modality)
2. Exclusion of class 8D (insufficient samples)
3. SMOTE for rare classes (<50 samples)
4. Documentation and visualization of class distributions

Class mapping:
- 0 → 0 people (absence)
- 1D, 1S → 1 person
- 2D → 2 people
- 3D → 3 people
- 4D → 4 people
- 5D → 5 people
- 6D → 6 people
- 8D → excluded (insufficient samples)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import Tuple, Dict, Optional
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


class DatasetBalancer:
    """
    Dataset balancing for people counting with class aggregation and SMOTE.
    """
    
    def __init__(self, smote_threshold: int = 50, k_neighbors: int = 3):
        """
        Initialize the dataset balancer.
        
        Args:
            smote_threshold: Minimum samples per class before applying SMOTE
            k_neighbors: Number of neighbors for SMOTE (use 3 for rare classes)
        """
        self.smote_threshold = smote_threshold
        self.k_neighbors = k_neighbors
        self.class_mapping = {
            '0': 0,      # No person
            '1D': 1,     # 1 person (dynamic)
            '1S': 1,     # 1 person (static)
            '2D': 2,     # 2 people
            '3D': 3,     # 3 people
            '4D': 4,     # 4 people
            '5D': 5,     # 5 people
            '6D': 6,     # 6 people
            # '8D' is excluded
        }
        self.distribution_history = []
        
    def aggregate_by_person_count(self, X: np.ndarray, y: np.ndarray, 
                                   class_labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aggregate classes by number of people (not by modality).
        
        Args:
            X: Feature matrix
            y: Original class labels (e.g., '0', '1D', '1S', '2D', etc.)
            class_labels: Optional array of class label strings
            
        Returns:
            X_agg, y_agg: Features and aggregated labels (0-6 for person count)
        """
        print("\n=== Class Aggregation by Person Count ===")
        
        # Convert labels to strings if needed
        if class_labels is not None:
            y_str = class_labels
        else:
            y_str = y.astype(str) if isinstance(y[0], (int, float)) else y
        
        # Apply mapping
        y_aggregated = []
        X_filtered = []
        discarded_count = 0
        
        for i, label in enumerate(y_str):
            if label in self.class_mapping:
                y_aggregated.append(self.class_mapping[label])
                X_filtered.append(X[i])
            else:
                # Discard 8D and other unknown classes
                discarded_count += 1
        
        X_agg = np.array(X_filtered)
        y_agg = np.array(y_aggregated)
        
        print(f"Original classes: {sorted(set(y_str))}")
        print(f"Aggregated to person counts: {sorted(set(y_agg))}")
        print(f"Discarded samples (8D and unknown): {discarded_count}")
        
        # Print distribution
        self._print_distribution("After Aggregation", y_agg)
        self.distribution_history.append(('After Aggregation', Counter(y_agg)))
        
        return X_agg, y_agg
    
    def apply_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE to rare classes (<50 samples).
        
        Args:
            X: Feature matrix
            y: Class labels
            
        Returns:
            X_balanced, y_balanced: Features and labels after SMOTE
        """
        print("\n=== SMOTE for Rare Classes ===")
        
        # Count samples per class
        class_counts = Counter(y)
        print(f"Class distribution before SMOTE:")
        for cls in sorted(class_counts.keys()):
            print(f"  Class {cls}: {class_counts[cls]} samples")
        
        # Identify rare classes
        rare_classes = {cls: count for cls, count in class_counts.items() 
                       if count < self.smote_threshold}
        
        if not rare_classes:
            print(f"✓ No rare classes (<{self.smote_threshold} samples) found. Skipping SMOTE.")
            return X, y
        
        print(f"\nRare classes (<{self.smote_threshold} samples): {list(rare_classes.keys())}")
        
        # Calculate target sample counts
        sampling_strategy = {}
        for cls, count in rare_classes.items():
            target_count = min(self.smote_threshold, count * 3)  # Don't oversample too much
            sampling_strategy[cls] = target_count
            print(f"  Class {cls}: {count} → {target_count} samples")
        
        # Determine k_neighbors (must be less than smallest class size)
        min_samples = min(class_counts.values())
        k_neighbors = min(self.k_neighbors, min_samples - 1)
        
        if k_neighbors < 1:
            print(f"⚠ Warning: Not enough samples for SMOTE (min={min_samples}). Skipping.")
            return X, y
        
        print(f"\nApplying SMOTE with k_neighbors={k_neighbors}...")
        
        try:
            smote = SMOTE(sampling_strategy=sampling_strategy, 
                         k_neighbors=k_neighbors, 
                         random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            
            print(f"✓ SMOTE applied successfully")
            self._print_distribution("After SMOTE", y_balanced)
            self.distribution_history.append(('After SMOTE', Counter(y_balanced)))
            
            return X_balanced, y_balanced
            
        except Exception as e:
            print(f"⚠ Warning: SMOTE failed ({str(e)}). Returning original data.")
            return X, y
    
    def balance_dataset(self, X: np.ndarray, y: np.ndarray, 
                        class_labels: Optional[np.ndarray] = None,
                        use_class_weight: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Complete dataset balancing pipeline.
        
        Args:
            X: Feature matrix
            y: Original class labels
            class_labels: Optional array of class label strings
            use_class_weight: If True, compute class weights for model training
            
        Returns:
            X_balanced, y_balanced, class_weights_dict
        """
        print("\n" + "="*60)
        print("DATASET BALANCING PIPELINE")
        print("="*60)
        
        # Store original distribution
        print("\n=== Original Distribution ===")
        if class_labels is not None:
            self._print_distribution("Original", class_labels)
            self.distribution_history.append(('Original', Counter(class_labels)))
        else:
            self._print_distribution("Original", y)
            self.distribution_history.append(('Original', Counter(y)))
        
        # Step 1: Aggregate by person count
        X_agg, y_agg = self.aggregate_by_person_count(X, y, class_labels)
        
        # Step 2: Apply SMOTE to rare classes
        X_balanced, y_balanced = self.apply_smote(X_agg, y_agg)
        
        # Step 3: Calculate class weights
        class_weights = None
        if use_class_weight:
            class_weights = self._calculate_class_weights(y_balanced)
        
        print("\n" + "="*60)
        print("BALANCING COMPLETE")
        print("="*60 + "\n")
        
        return X_balanced, y_balanced, class_weights
    
    def _calculate_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Calculate class weights for handling imbalanced data in models.
        
        Args:
            y: Class labels
            
        Returns:
            Dictionary mapping class to weight
        """
        print("\n=== Class Weights Calculation ===")
        
        class_counts = Counter(y)
        total_samples = len(y)
        n_classes = len(class_counts)
        
        class_weights = {}
        for cls, count in sorted(class_counts.items()):
            weight = total_samples / (n_classes * count)
            class_weights[cls] = weight
            print(f"  Class {cls}: weight = {weight:.3f}")
        
        return class_weights
    
    def _print_distribution(self, stage: str, y: np.ndarray):
        """Print class distribution statistics."""
        class_counts = Counter(y)
        total = len(y)
        
        print(f"\n{stage}:")
        for cls in sorted(class_counts.keys()):
            count = class_counts[cls]
            percentage = count / total * 100
            print(f"  Class {cls}: {count:6d} samples ({percentage:5.1f}%)")
        print(f"  Total: {total} samples")
    
    def visualize_distribution(self, output_file: str = 'class_distribution.png'):
        """
        Visualize class distribution before and after balancing.
        
        Args:
            output_file: Path to save the visualization
        """
        print(f"\nGenerating distribution visualization...")
        
        n_stages = len(self.distribution_history)
        fig, axes = plt.subplots(1, n_stages, figsize=(6*n_stages, 5))
        
        if n_stages == 1:
            axes = [axes]
        
        for idx, (stage_name, distribution) in enumerate(self.distribution_history):
            ax = axes[idx]
            
            # Sort by class number
            classes = sorted(distribution.keys())
            counts = [distribution[cls] for cls in classes]
            
            # Create bar plot
            bars = ax.bar(classes, counts, color='steelblue', edgecolor='black', alpha=0.7)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=10)
            
            ax.set_xlabel('Number of People', fontsize=12)
            ax.set_ylabel('Sample Count', fontsize=12)
            ax.set_title(stage_name, fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.set_xticks(classes)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved distribution plot to {output_file}")
        plt.close()
    
    def generate_report(self, output_file: str = 'balancing_report.txt'):
        """
        Generate a text report documenting the balancing process.
        
        Args:
            output_file: Path to save the report
        """
        print(f"\nGenerating balancing report...")
        
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DATASET BALANCING REPORT FOR PEOPLE COUNTING\n")
            f.write("="*80 + "\n\n")
            
            f.write("METHODOLOGY\n")
            f.write("-" * 80 + "\n")
            f.write("1. Class Aggregation: Merged classes by person count\n")
            f.write("   - 1D, 1S → 1 person\n")
            f.write("   - 2D → 2 people, 3D → 3 people, etc.\n")
            f.write("   - Excluded 8D (insufficient samples)\n\n")
            
            f.write(f"2. SMOTE for Rare Classes:\n")
            f.write(f"   - Threshold: {self.smote_threshold} samples\n")
            f.write(f"   - K-neighbors: {self.k_neighbors}\n")
            f.write(f"   - Target: Generate synthetic samples up to {self.smote_threshold} per class\n\n")
            
            f.write("3. Class Weights: Calculated for natural distribution handling in models\n\n")
            
            f.write("CLASS DISTRIBUTION EVOLUTION\n")
            f.write("-" * 80 + "\n\n")
            
            for stage_name, distribution in self.distribution_history:
                f.write(f"{stage_name}:\n")
                total = sum(distribution.values())
                
                for cls in sorted(distribution.keys()):
                    count = distribution[cls]
                    percentage = count / total * 100
                    f.write(f"  Class {cls}: {count:6d} samples ({percentage:5.1f}%)\n")
                
                f.write(f"  Total: {total} samples\n\n")
            
            f.write("="*80 + "\n")
        
        print(f"✓ Saved report to {output_file}")


def main():
    """
    Example usage of the dataset balancing pipeline.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Balance dataset for people counting with aggregation and SMOTE"
    )
    parser.add_argument('--input', required=True, help='Input CSV file with features and labels')
    parser.add_argument('--output', required=True, help='Output CSV file for balanced dataset')
    parser.add_argument('--label_column', default='label', help='Column name for class labels')
    parser.add_argument('--smote_threshold', type=int, default=50, 
                       help='Minimum samples before applying SMOTE (default: 50)')
    parser.add_argument('--k_neighbors', type=int, default=3,
                       help='K-neighbors for SMOTE (default: 3)')
    parser.add_argument('--viz_output', default='class_distribution.png',
                       help='Output file for distribution visualization')
    parser.add_argument('--report_output', default='balancing_report.txt',
                       help='Output file for balancing report')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Separate features and labels
    if args.label_column not in df.columns:
        raise ValueError(f"Label column '{args.label_column}' not found in data")
    
    y = df[args.label_column].values
    X = df.drop(columns=[args.label_column]).values
    
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Create balancer
    balancer = DatasetBalancer(
        smote_threshold=args.smote_threshold,
        k_neighbors=args.k_neighbors
    )
    
    # Balance dataset
    X_balanced, y_balanced, class_weights = balancer.balance_dataset(
        X, y, class_labels=y, use_class_weight=True
    )
    
    # Save balanced dataset
    print(f"\nSaving balanced dataset to {args.output}...")
    df_balanced = pd.DataFrame(X_balanced, columns=df.drop(columns=[args.label_column]).columns)
    df_balanced[args.label_column] = y_balanced
    df_balanced.to_csv(args.output, index=False)
    print(f"✓ Saved {len(df_balanced)} samples")
    
    # Generate visualizations and report
    balancer.visualize_distribution(args.viz_output)
    balancer.generate_report(args.report_output)
    
    print("\n✓ Balancing pipeline complete!")


if __name__ == '__main__':
    main()
