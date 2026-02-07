#!/usr/bin/env python3
# -*-coding:utf-8-*-

"""
Modeling for People Counting with 5-Fold Cross-Validation
==========================================================

This module implements:
1. 5-Fold Stratified Cross-Validation (replaces single 80/20 split)
2. Multiple models: XGBoost, LightGBM, RandomForest
3. Per-class metrics: precision, recall, F1-score
4. Counting-specific metrics: MAE, accuracy within ±1 and ±2 persons
5. Confusion matrices for each fold
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    mean_absolute_error, accuracy_score
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class PeopleCountingModel:
    """
    Model training and evaluation for people counting with 5-Fold CV.
    """
    
    def __init__(self, n_folds: int = 5, random_state: int = 42):
        """
        Initialize the modeling pipeline.
        
        Args:
            n_folds: Number of folds for cross-validation
            random_state: Random seed for reproducibility
        """
        self.n_folds = n_folds
        self.random_state = random_state
        self.results = defaultdict(list)
        self.confusion_matrices = defaultdict(list)
        
    def create_model(self, model_type: str, class_weights: Optional[Dict] = None):
        """
        Create a classification model.
        
        Args:
            model_type: Type of model ('xgboost', 'lightgbm', 'randomforest')
            class_weights: Optional class weights for imbalanced data
            
        Returns:
            Initialized model
        """
        if model_type == 'xgboost':
            # Convert class_weights to sample_weight format for XGBoost
            model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='mlogloss'
            )
        elif model_type == 'lightgbm':
            # LightGBM supports class_weight parameter
            model = LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight=class_weights if class_weights else None,
                verbose=-1
            )
        elif model_type == 'randomforest':
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight=class_weights if class_weights else None
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model
    
    def calculate_counting_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate counting-specific metrics.
        
        Args:
            y_true: True labels (person counts)
            y_pred: Predicted labels (person counts)
            
        Returns:
            Dictionary of counting metrics
        """
        # Mean Absolute Error
        mae = mean_absolute_error(y_true, y_pred)
        
        # Exact accuracy
        exact_acc = accuracy_score(y_true, y_pred)
        
        # Accuracy within ±1 person
        within_1 = np.abs(y_true - y_pred) <= 1
        acc_within_1 = np.mean(within_1)
        
        # Accuracy within ±2 persons
        within_2 = np.abs(y_true - y_pred) <= 2
        acc_within_2 = np.mean(within_2)
        
        return {
            'mae': mae,
            'exact_accuracy': exact_acc,
            'accuracy_within_1': acc_within_1,
            'accuracy_within_2': acc_within_2
        }
    
    def train_with_cv(self, X: np.ndarray, y: np.ndarray, 
                      model_type: str = 'xgboost',
                      class_weights: Optional[Dict] = None) -> Dict:
        """
        Train model using 5-Fold Stratified Cross-Validation.
        
        Args:
            X: Feature matrix
            y: Labels (person counts 0-6)
            model_type: Type of model to train
            class_weights: Optional class weights
            
        Returns:
            Dictionary with CV results
        """
        print(f"\n{'='*70}")
        print(f"Training {model_type.upper()} with {self.n_folds}-Fold Cross-Validation")
        print(f"{'='*70}\n")
        
        # Initialize stratified k-fold
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        fold_results = []
        
        # Iterate through folds
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            print(f"\n--- Fold {fold_idx}/{self.n_folds} ---")
            
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            print(f"Training samples: {len(X_train)}")
            print(f"Test samples: {len(X_test)}")
            
            # Create and train model
            model = self.create_model(model_type, class_weights)
            
            # Fit model
            if model_type == 'xgboost' and class_weights:
                # Apply sample weights for XGBoost
                sample_weights = np.array([class_weights.get(label, 1.0) for label in y_train])
                model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
            else:
                model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            counting_metrics = self.calculate_counting_metrics(y_test, y_pred)
            
            # Classification report (per-class metrics)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Store results
            fold_result = {
                'fold': fold_idx,
                'counting_metrics': counting_metrics,
                'classification_report': report,
                'confusion_matrix': cm,
                'y_true': y_test,
                'y_pred': y_pred
            }
            fold_results.append(fold_result)
            
            # Store confusion matrix
            self.confusion_matrices[model_type].append((fold_idx, cm))
            
            # Print fold results
            print(f"\nCounting Metrics:")
            print(f"  MAE: {counting_metrics['mae']:.3f}")
            print(f"  Exact Accuracy: {counting_metrics['exact_accuracy']:.3f}")
            print(f"  Accuracy ±1 person: {counting_metrics['accuracy_within_1']:.3f}")
            print(f"  Accuracy ±2 persons: {counting_metrics['accuracy_within_2']:.3f}")
        
        # Aggregate results across folds
        aggregated_results = self._aggregate_cv_results(fold_results, model_type)
        
        return aggregated_results
    
    def _aggregate_cv_results(self, fold_results: List[Dict], model_type: str) -> Dict:
        """
        Aggregate results across all CV folds.
        
        Args:
            fold_results: List of results from each fold
            model_type: Type of model
            
        Returns:
            Aggregated results dictionary
        """
        print(f"\n{'='*70}")
        print(f"Aggregated Results for {model_type.upper()} ({self.n_folds}-Fold CV)")
        print(f"{'='*70}\n")
        
        # Aggregate counting metrics
        mae_scores = [fr['counting_metrics']['mae'] for fr in fold_results]
        exact_acc_scores = [fr['counting_metrics']['exact_accuracy'] for fr in fold_results]
        within_1_scores = [fr['counting_metrics']['accuracy_within_1'] for fr in fold_results]
        within_2_scores = [fr['counting_metrics']['accuracy_within_2'] for fr in fold_results]
        
        print("Counting Metrics (Mean ± Std):")
        print(f"  MAE: {np.mean(mae_scores):.3f} ± {np.std(mae_scores):.3f}")
        print(f"  Exact Accuracy: {np.mean(exact_acc_scores):.3f} ± {np.std(exact_acc_scores):.3f}")
        print(f"  Accuracy ±1 person: {np.mean(within_1_scores):.3f} ± {np.std(within_1_scores):.3f}")
        print(f"  Accuracy ±2 persons: {np.mean(within_2_scores):.3f} ± {np.std(within_2_scores):.3f}")
        
        # Per-class metrics (average across folds)
        print("\nPer-Class Metrics (averaged across folds):")
        
        # Get all unique classes
        all_classes = set()
        for fr in fold_results:
            all_classes.update([k for k in fr['classification_report'].keys() 
                              if k not in ['accuracy', 'macro avg', 'weighted avg']])
        
        per_class_metrics = {}
        for cls in sorted(all_classes):
            precisions = []
            recalls = []
            f1_scores = []
            
            for fr in fold_results:
                if cls in fr['classification_report']:
                    precisions.append(fr['classification_report'][cls]['precision'])
                    recalls.append(fr['classification_report'][cls]['recall'])
                    f1_scores.append(fr['classification_report'][cls]['f1-score'])
            
            if precisions:  # Only if class appeared in at least one fold
                per_class_metrics[cls] = {
                    'precision': np.mean(precisions),
                    'recall': np.mean(recalls),
                    'f1-score': np.mean(f1_scores)
                }
                
                print(f"  Class {cls}:")
                print(f"    Precision: {np.mean(precisions):.3f} ± {np.std(precisions):.3f}")
                print(f"    Recall: {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")
                print(f"    F1-Score: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
        
        return {
            'model_type': model_type,
            'n_folds': self.n_folds,
            'fold_results': fold_results,
            'aggregated_counting_metrics': {
                'mae_mean': np.mean(mae_scores),
                'mae_std': np.std(mae_scores),
                'exact_accuracy_mean': np.mean(exact_acc_scores),
                'exact_accuracy_std': np.std(exact_acc_scores),
                'accuracy_within_1_mean': np.mean(within_1_scores),
                'accuracy_within_1_std': np.std(within_1_scores),
                'accuracy_within_2_mean': np.mean(within_2_scores),
                'accuracy_within_2_std': np.std(within_2_scores),
            },
            'per_class_metrics': per_class_metrics
        }
    
    def visualize_confusion_matrices(self, model_type: str, output_file: str):
        """
        Visualize confusion matrices for all CV folds.
        
        Args:
            model_type: Type of model
            output_file: Path to save the visualization
        """
        if model_type not in self.confusion_matrices:
            print(f"No confusion matrices found for {model_type}")
            return
        
        cms = self.confusion_matrices[model_type]
        n_folds = len(cms)
        
        # Create subplots
        fig, axes = plt.subplots(1, n_folds, figsize=(5*n_folds, 4))
        if n_folds == 1:
            axes = [axes]
        
        for idx, (fold_num, cm) in enumerate(cms):
            ax = axes[idx]
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       cbar=True, square=True)
            
            ax.set_title(f'Fold {fold_num}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicted Count', fontsize=10)
            ax.set_ylabel('True Count', fontsize=10)
        
        plt.suptitle(f'{model_type.upper()} - Confusion Matrices ({n_folds}-Fold CV)', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved confusion matrices to {output_file}")
        plt.close()
    
    def save_results_report(self, results: Dict, output_file: str):
        """
        Save detailed results report to file.
        
        Args:
            results: Results dictionary from train_with_cv
            output_file: Path to save the report
        """
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"PEOPLE COUNTING MODEL EVALUATION - {results['model_type'].upper()}\n")
            f.write(f"{results['n_folds']}-Fold Stratified Cross-Validation\n")
            f.write("="*80 + "\n\n")
            
            # Aggregated counting metrics
            f.write("COUNTING METRICS (Mean ± Std across folds)\n")
            f.write("-"*80 + "\n")
            metrics = results['aggregated_counting_metrics']
            f.write(f"MAE: {metrics['mae_mean']:.3f} ± {metrics['mae_std']:.3f}\n")
            f.write(f"Exact Accuracy: {metrics['exact_accuracy_mean']:.3f} ± {metrics['exact_accuracy_std']:.3f}\n")
            f.write(f"Accuracy ±1 person: {metrics['accuracy_within_1_mean']:.3f} ± {metrics['accuracy_within_1_std']:.3f}\n")
            f.write(f"Accuracy ±2 persons: {metrics['accuracy_within_2_mean']:.3f} ± {metrics['accuracy_within_2_std']:.3f}\n\n")
            
            # Per-class metrics
            f.write("PER-CLASS METRICS (averaged across folds)\n")
            f.write("-"*80 + "\n")
            for cls in sorted(results['per_class_metrics'].keys()):
                metrics = results['per_class_metrics'][cls]
                f.write(f"\nClass {cls} ({cls} person{'s' if int(cls) != 1 else ''}):\n")
                f.write(f"  Precision: {metrics['precision']:.3f}\n")
                f.write(f"  Recall: {metrics['recall']:.3f}\n")
                f.write(f"  F1-Score: {metrics['f1-score']:.3f}\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"✓ Saved results report to {output_file}")


def main():
    """
    Example usage of the people counting modeling pipeline.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train people counting model with 5-Fold Cross-Validation"
    )
    parser.add_argument('--input', required=True, help='Input CSV file with balanced dataset')
    parser.add_argument('--label_column', default='label', help='Column name for class labels')
    parser.add_argument('--model', default='xgboost', 
                       choices=['xgboost', 'lightgbm', 'randomforest'],
                       help='Model type to train')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--output_dir', default='./results', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Separate features and labels
    y = df[args.label_column].values
    X = df.drop(columns=[args.label_column]).values
    
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Create model trainer
    trainer = PeopleCountingModel(n_folds=args.n_folds)
    
    # Train with CV
    results = trainer.train_with_cv(X, y, model_type=args.model)
    
    # Save results
    report_file = os.path.join(args.output_dir, f'{args.model}_cv_report.txt')
    trainer.save_results_report(results, report_file)
    
    # Visualize confusion matrices
    cm_file = os.path.join(args.output_dir, f'{args.model}_confusion_matrices.png')
    trainer.visualize_confusion_matrices(args.model, cm_file)
    
    print("\n✓ Modeling pipeline complete!")


if __name__ == '__main__':
    main()
