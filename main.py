---

## 📄 **FILE 6: `main.py`**
```python
"""
Main Execution Script
Orchestrates the complete pipeline: data generation, training, and XAI analysis
"""

import os
import sys
import numpy as np
from config import MODELS_DIR, OUTPUT_CONFIG
from data_generator import generate_dataset
from trainer import run_training_pipeline
from xai_analysis import run_xai_analysis


def print_header():
    """Print project header"""
    print("\n" + "="*70)
    print(" "*5 + "ADVANCED TIME SERIES FORECASTING WITH XAI")
    print(" "*10 + "LSTM + SHAP Explainability Analysis")
    print("="*70)
    print("\nProject Components:")
    print("  1. Multivariate Time Series Generation")
    print("  2. LSTM Model Training with Attention")
    print("  3. Performance Evaluation (RMSE, MAE, R²)")
    print("  4. SHAP-based Explainability Analysis")
    print("="*70 + "\n")


def main():
    """Execute complete project pipeline"""
    print_header()
    
    try:
        # Step 1: Generate Dataset
        print("\n" + "█"*70)
        print("STEP 1/3: Data Generation")
        print("█"*70)
        df = generate_dataset()
        
        # Step 2: Train Model
        print("\n" + "█"*70)
        print("STEP 2/3: Model Training & Evaluation")
        print("█"*70)
        model, preprocessor, test_data, metrics, predictions = run_training_pipeline()
        
        # Step 3: XAI Analysis
        print("\n" + "█"*70)
        print("STEP 3/3: Explainability Analysis")
        print("█"*70)
        X_test, y_test = test_data
        feature_importance, temporal_importance, insights = run_xai_analysis(
            model, X_test, preprocessor
        )
        
        # Summary
        print("\n" + "="*70)
        print(" "*20 + "PROJECT COMPLETE")
        print("="*70)
        print("\nDeliverables Generated:")
        print("  ✓ Dataset: data/multivariate_timeseries.csv")
        print(f"  ✓ Trained Model: models/{OUTPUT_CONFIG['model_name']}")
        print(f"  ✓ Performance Metrics: results/{OUTPUT_CONFIG['metrics_file']}")
        print(f"  ✓ XAI Analysis Report: results/{OUTPUT_CONFIG['report_file']}")
        
        print("\nKey Results:")
        print(f"  • Overall RMSE: {metrics['overall']['RMSE']:.6f}")
        print(f"  • Overall MAE:  {metrics['overall']['MAE']:.6f}")
        print(f"  • Overall R²:   {metrics['overall']['R2']:.6f}")
        
        print(f"\n  • Top Feature: {max(feature_importance.items(), key=lambda x: x[1])[0]}")
        print(f"    Contribution: {max(feature_importance.values()):.1f}%")
        
        print("\n" + "="*70)
        print("\n✓ All deliverables successfully generated!")
        print("✓ Check the 'results/' directory for detailed reports.\n")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
```

---

## 📄 **FILE 7: `requirements.txt`**
