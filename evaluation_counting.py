#!/usr/bin/env python3
# -*-coding:utf-8-*-

"""
Systematic Error Analysis for People Counting
==============================================

This module implements:
1. Per-class error analysis
2. MAE calculation for each class
3. Error distribution analysis
4. Systematic error patterns detection
5. Comprehensive visualization and reporting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from sklearn.metrics import mean_absolute_error, confusion_matrix


class CountingEvaluator:
    """
    Evaluator for systematic error analysis in people counting.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.error_data = []
        
    def analyze_per_class_errors(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Analyze errors for each class separately.
        
        Args:
            y_true: True labels (person counts)
            y_pred: Predicted labels (person counts)
            
        Returns:
            Dictionary with per-class error analysis
        """
        print("\n=== Per-Class Error Analysis ===\n")
        
        unique_classes = sorted(set(y_true))
        per_class_results = {}
        
        for cls in unique_classes:
            # Get indices for this class
            class_mask = y_true == cls
            y_true_cls = y_true[class_mask]
            y_pred_cls = y_pred[class_mask]
            
            if len(y_true_cls) == 0:
                continue
            
            # Calculate metrics for this class
            mae_cls = mean_absolute_error(y_true_cls, y_pred_cls)
            
            # Exact accuracy for this class
            exact_acc = np.mean(y_true_cls == y_pred_cls)
            
            # Accuracy within ±1 for this class
            within_1 = np.mean(np.abs(y_true_cls - y_pred_cls) <= 1)
            
            # Calculate error distribution
            errors = y_pred_cls - y_true_cls
            
            # Bias (mean error)
            bias = np.mean(errors)
            
            # Over/under estimation
            overestimation = np.sum(errors > 0)
            underestimation = np.sum(errors < 0)
            correct = np.sum(errors == 0)
            
            per_class_results[cls] = {
                'n_samples': len(y_true_cls),
                'mae': mae_cls,
                'exact_accuracy': exact_acc,
                'accuracy_within_1': within_1,
                'bias': bias,
                'overestimation_count': overestimation,
                'underestimation_count': underestimation,
                'correct_count': correct,
                'error_distribution': errors
            }
            
            # Print results
            print(f"Class {cls} ({cls} person{'s' if cls != 1 else ''}) - {len(y_true_cls)} samples:")
            print(f"  MAE: {mae_cls:.3f}")
            print(f"  Exact Accuracy: {exact_acc:.3f}")
            print(f"  Accuracy ±1: {within_1:.3f}")
            print(f"  Bias (mean error): {bias:.3f}")
            print(f"  Overestimations: {overestimation} ({overestimation/len(y_true_cls)*100:.1f}%)")
            print(f"  Underestimations: {underestimation} ({underestimation/len(y_true_cls)*100:.1f}%)")
            print(f"  Correct predictions: {correct} ({correct/len(y_true_cls)*100:.1f}%)")
            print()
        
        return per_class_results
    
    def detect_systematic_patterns(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Detect systematic error patterns.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with detected patterns
        """
        print("\n=== Systematic Error Pattern Detection ===\n")
        
        patterns = {}
        
        # 1. Confusion between adjacent classes
        cm = confusion_matrix(y_true, y_pred)
        n_classes = cm.shape[0]
        
        # Check for off-diagonal patterns
        adjacent_confusion = 0
        total_errors = 0
        
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j:
                    total_errors += cm[i, j]
                    if abs(i - j) == 1:  # Adjacent classes
                        adjacent_confusion += cm[i, j]
        
        if total_errors > 0:
            adjacent_ratio = adjacent_confusion / total_errors
            patterns['adjacent_confusion_ratio'] = adjacent_ratio
            print(f"Adjacent class confusion: {adjacent_ratio*100:.1f}% of all errors")
        
        # 2. Consistent over/underestimation
        errors = y_pred - y_true
        mean_error = np.mean(errors)
        
        if abs(mean_error) > 0.1:
            if mean_error > 0:
                patterns['systematic_bias'] = 'overestimation'
                print(f"Systematic overestimation detected (mean error: +{mean_error:.3f})")
            else:
                patterns['systematic_bias'] = 'underestimation'
                print(f"Systematic underestimation detected (mean error: {mean_error:.3f})")
        else:
            patterns['systematic_bias'] = 'none'
            print("No systematic bias detected")
        
        # 3. Class-specific patterns
        unique_classes = sorted(set(y_true))
        problematic_classes = []
        
        for cls in unique_classes:
            class_mask = y_true == cls
            class_accuracy = np.mean(y_true[class_mask] == y_pred[class_mask])
            
            if class_accuracy < 0.5:
                problematic_classes.append(cls)
        
        if problematic_classes:
            patterns['problematic_classes'] = problematic_classes
            print(f"\nProblematic classes (accuracy < 50%): {problematic_classes}")
        else:
            print("\nNo problematic classes detected")
        
        return patterns
    
    def visualize_error_distribution(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                     output_file: str = 'error_distribution.png'):
        """
        Visualize error distribution.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            output_file: Path to save the visualization
        """
        print(f"\nGenerating error distribution visualization...")
        
        errors = y_pred - y_true
        unique_classes = sorted(set(y_true))
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Overall error histogram
        ax = axes[0, 0]
        ax.hist(errors, bins=range(int(errors.min())-1, int(errors.max())+2), 
               edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
        ax.set_xlabel('Prediction Error (predicted - true)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Overall Error Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 2. Error distribution by class
        ax = axes[0, 1]
        error_by_class = []
        labels = []
        
        for cls in unique_classes:
            class_mask = y_true == cls
            class_errors = errors[class_mask]
            error_by_class.append(class_errors)
            labels.append(f'{cls}')
        
        bp = ax.boxplot(error_by_class, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax.axhline(0, color='red', linestyle='--', linewidth=1)
        ax.set_xlabel('True Person Count', fontsize=11)
        ax.set_ylabel('Prediction Error', fontsize=11)
        ax.set_title('Error Distribution by Class', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # 3. Confusion matrix
        ax = axes[1, 0]
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                   cbar=True, square=True)
        ax.set_xlabel('Predicted Count', fontsize=11)
        ax.set_ylabel('True Count', fontsize=11)
        ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
        
        # 4. MAE by class
        ax = axes[1, 1]
        mae_by_class = []
        
        for cls in unique_classes:
            class_mask = y_true == cls
            if np.sum(class_mask) > 0:
                mae_cls = mean_absolute_error(y_true[class_mask], y_pred[class_mask])
                mae_by_class.append(mae_cls)
            else:
                mae_by_class.append(0)
        
        bars = ax.bar(range(len(unique_classes)), mae_by_class, 
                     color='coral', edgecolor='black', alpha=0.7)
        ax.set_xticks(range(len(unique_classes)))
        ax.set_xticklabels([f'{cls}' for cls in unique_classes])
        ax.set_xlabel('True Person Count', fontsize=11)
        ax.set_ylabel('MAE', fontsize=11)
        ax.set_title('Mean Absolute Error by Class', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved error distribution plot to {output_file}")
        plt.close()
    
    def generate_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   per_class_results: Dict, patterns: Dict,
                                   output_file: str = 'evaluation_report.txt'):
        """
        Generate comprehensive evaluation report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            per_class_results: Results from per-class analysis
            patterns: Detected systematic patterns
            output_file: Path to save the report
        """
        print(f"\nGenerating evaluation report...")
        
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SYSTEMATIC ERROR ANALYSIS FOR PEOPLE COUNTING\n")
            f.write("="*80 + "\n\n")
            
            # Overall metrics
            f.write("OVERALL METRICS\n")
            f.write("-"*80 + "\n")
            mae_overall = mean_absolute_error(y_true, y_pred)
            exact_acc_overall = np.mean(y_true == y_pred)
            within_1_overall = np.mean(np.abs(y_true - y_pred) <= 1)
            within_2_overall = np.mean(np.abs(y_true - y_pred) <= 2)
            
            f.write(f"Total samples: {len(y_true)}\n")
            f.write(f"MAE: {mae_overall:.3f}\n")
            f.write(f"Exact Accuracy: {exact_acc_overall:.3f}\n")
            f.write(f"Accuracy ±1 person: {within_1_overall:.3f}\n")
            f.write(f"Accuracy ±2 persons: {within_2_overall:.3f}\n\n")
            
            # Per-class metrics
            f.write("PER-CLASS ERROR ANALYSIS\n")
            f.write("-"*80 + "\n\n")
            
            for cls in sorted(per_class_results.keys()):
                result = per_class_results[cls]
                f.write(f"Class {cls} ({cls} person{'s' if cls != 1 else ''}):\n")
                f.write(f"  Samples: {result['n_samples']}\n")
                f.write(f"  MAE: {result['mae']:.3f}\n")
                f.write(f"  Exact Accuracy: {result['exact_accuracy']:.3f}\n")
                f.write(f"  Accuracy ±1: {result['accuracy_within_1']:.3f}\n")
                f.write(f"  Bias: {result['bias']:.3f}\n")
                f.write(f"  Overestimations: {result['overestimation_count']} " +
                       f"({result['overestimation_count']/result['n_samples']*100:.1f}%)\n")
                f.write(f"  Underestimations: {result['underestimation_count']} " +
                       f"({result['underestimation_count']/result['n_samples']*100:.1f}%)\n")
                f.write(f"  Correct: {result['correct_count']} " +
                       f"({result['correct_count']/result['n_samples']*100:.1f}%)\n\n")
            
            # Systematic patterns
            f.write("SYSTEMATIC ERROR PATTERNS\n")
            f.write("-"*80 + "\n")
            
            if 'adjacent_confusion_ratio' in patterns:
                f.write(f"Adjacent class confusion: {patterns['adjacent_confusion_ratio']*100:.1f}% of all errors\n")
            
            if 'systematic_bias' in patterns:
                f.write(f"Systematic bias: {patterns['systematic_bias']}\n")
            
            if 'problematic_classes' in patterns:
                f.write(f"Problematic classes (accuracy < 50%): {patterns['problematic_classes']}\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"✓ Saved evaluation report to {output_file}")
    
    def evaluate_complete(self, y_true: np.ndarray, y_pred: np.ndarray,
                         output_prefix: str = 'evaluation') -> Dict:
        """
        Complete evaluation pipeline.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            output_prefix: Prefix for output files
            
        Returns:
            Dictionary with all evaluation results
        """
        print("\n" + "="*70)
        print("SYSTEMATIC ERROR ANALYSIS")
        print("="*70)
        
        # Per-class analysis
        per_class_results = self.analyze_per_class_errors(y_true, y_pred)
        
        # Pattern detection
        patterns = self.detect_systematic_patterns(y_true, y_pred)
        
        # Generate visualizations
        self.visualize_error_distribution(y_true, y_pred, 
                                          f'{output_prefix}_error_distribution.png')
        
        # Generate report
        self.generate_evaluation_report(y_true, y_pred, per_class_results, patterns,
                                       f'{output_prefix}_report.txt')
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70 + "\n")
        
        return {
            'per_class_results': per_class_results,
            'patterns': patterns
        }


def main():
    """
    Example usage of the evaluation pipeline.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Systematic error analysis for people counting model"
    )
    parser.add_argument('--predictions', required=True, 
                       help='CSV file with true and predicted labels')
    parser.add_argument('--true_column', default='y_true', 
                       help='Column name for true labels')
    parser.add_argument('--pred_column', default='y_pred', 
                       help='Column name for predicted labels')
    parser.add_argument('--output_prefix', default='evaluation',
                       help='Prefix for output files')
    
    args = parser.parse_args()
    
    # Load predictions
    print(f"Loading predictions from {args.predictions}...")
    df = pd.read_csv(args.predictions)
    
    y_true = df[args.true_column].values
    y_pred = df[args.pred_column].values
    
    print(f"Loaded {len(y_true)} predictions")
    
    # Create evaluator
    evaluator = CountingEvaluator()
    
    # Run evaluation
    results = evaluator.evaluate_complete(y_true, y_pred, args.output_prefix)
    
    print("\n✓ Evaluation pipeline complete!")


if __name__ == '__main__':
    main()
