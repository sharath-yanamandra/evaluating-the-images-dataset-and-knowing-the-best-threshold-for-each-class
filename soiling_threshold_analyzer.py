import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import sys
import json
import numpy as np
from pathlib import Path
import argparse
import subprocess
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import pandas as pd

class SoilingModelInference:
    def __init__(self, model_path, device='auto'):
        """
        Initialize the soiling detection inference class
        
        Args:
            model_path (str): Path to the .pt model file
            device (str): Device to run inference on ('auto', 'cuda', 'cpu')
        """
        self.device = self._get_device(device)
        self.model = self._load_model(model_path)
        self.transform = self._get_default_transform()
        
        # Soiling class mapping
        self.class_mapping = {
            0: 'clean',
            1: 'medium_soiled', 
            2: 'fully_soiled'
        }
        
    def _get_device(self, device):
        """Determine the best device to use"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def _load_model(self, model_path):
        """Load the PyTorch model"""
        try:
            # Try to detect if it's a YOLO/Ultralytics model
            if 'ultralytics' in str(model_path).lower() or self._is_yolo_model(model_path):
                print("Detected YOLO/Ultralytics soiling model. Loading with ultralytics...")
                return self._load_yolo_model(model_path)
            
            # Try loading with weights_only=False for trusted models
            try:
                model = torch.load(model_path, map_location=self.device, weights_only=False)
                print("✓ Soiling model loaded successfully (weights_only=False)")
            except Exception as first_error:
                print(f"First attempt failed: {first_error}")
                print("Trying with weights_only=True...")
                
                try:
                    torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
                    model = torch.load(model_path, map_location=self.device, weights_only=True)
                    print("✓ Soiling model loaded successfully (weights_only=True)")
                except Exception as second_error:
                    print(f"Second attempt failed: {second_error}")
                    raise first_error
            
            if isinstance(model, dict):
                print("Warning: Loaded state dict. You may need to define model architecture.")
                return model
            
            model.eval()
            return model
            
        except Exception as e:
            print(f"Error loading soiling model: {e}")
            raise
    
    def _is_yolo_model(self, model_path):
        """Check if the model is a YOLO model by examining the file"""
        try:
            with open(model_path, 'rb') as f:
                content = f.read(1000)
                return b'ultralytics' in content or b'DetectionModel' in content
        except:
            return False
    
    def _load_yolo_model(self, model_path):
        """Load YOLO model using ultralytics"""
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            print("✓ YOLO soiling model loaded successfully")
            return model
        except ImportError:
            print("ultralytics not installed. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
            from ultralytics import YOLO
            model = YOLO(model_path)
            print("✓ YOLO soiling model loaded successfully")
            return model
    
    def _get_default_transform(self):
        """Default image preprocessing transform"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict_single(self, image_path):
        """Run inference on a single image with memory optimization"""
        if hasattr(self.model, 'predict'):  # YOLO model
            try:
                # Clear GPU cache before prediction
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                results = self.model.predict(image_path, verbose=False, save=False)
                if results:
                    result = results[0]
                    prediction = {
                        'boxes': result.boxes.xyxy.cpu().numpy() if result.boxes is not None else [],
                        'scores': result.boxes.conf.cpu().numpy() if result.boxes is not None else [],
                        'classes': result.boxes.cls.cpu().numpy() if result.boxes is not None else [],
                        'names': result.names if hasattr(result, 'names') else {}
                    }
                    
                    del result, results
                    return prediction
                return None
            except Exception as e:
                print(f"Error with YOLO prediction for {os.path.basename(image_path)}: {e}")
                return None
        
        # Handle regular PyTorch models
        preprocessed = self.preprocess_image(image_path)
        if preprocessed is None:
            return None
            
        with torch.no_grad():
            output = self.model(preprocessed)
            
            if isinstance(output, torch.Tensor):
                return output.cpu().numpy()
            elif isinstance(output, (list, tuple)):
                return [o.cpu().numpy() if isinstance(o, torch.Tensor) else o for o in output]
            else:
                return output
    
    def predict_batch(self, image_paths, batch_size=1):
        """Run inference on multiple images with memory optimization"""
        results = {}
        
        if hasattr(self.model, 'predict'):  # YOLO model
            print(f"Using memory-optimized YOLO prediction (batch_size={batch_size})...")
            
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i+batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(image_paths)-1)//batch_size + 1}: {len(batch_paths)} images")
                
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    yolo_results = self.model.predict(batch_paths, verbose=False, save=False, stream=True)
                    
                    for path, result in zip(batch_paths, yolo_results):
                        prediction = {
                            'boxes': result.boxes.xyxy.cpu().numpy() if result.boxes is not None else [],
                            'scores': result.boxes.conf.cpu().numpy() if result.boxes is not None else [],
                            'classes': result.boxes.cls.cpu().numpy() if result.boxes is not None else [],
                            'names': result.names if hasattr(result, 'names') else {}
                        }
                        results[path] = prediction
                        del result
                    
                    import gc
                    gc.collect()
                    
                except Exception as e:
                    print(f"Batch prediction failed: {e}")
                    for path in batch_paths:
                        try:
                            result = self.predict_single(path)
                            if result is not None:
                                results[path] = result
                        except Exception as single_error:
                            print(f"Failed to process {os.path.basename(path)}: {single_error}")
                            continue
            
            return results
        
        # Regular PyTorch model batch processing
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            for path in batch_paths:
                result = self.predict_single(path)
                if result is not None:
                    results[path] = result
        
        return results

class SoilingThresholdAnalyzer:
    def __init__(self, results_dir='./soiling_analysis'):
        """
        Initialize the threshold analyzer for soiling detection
        
        Args:
            results_dir (str): Directory to save analysis results
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        self.class_mapping = {
            0: 'clean',
            1: 'medium_soiled', 
            2: 'fully_soiled'
        }
        
        # Collect all detection data for analysis
        self.all_detections = defaultdict(list)
        
    def analyze_detections(self, results):
        """
        Analyze detection results to understand confidence distributions
        
        Args:
            results (dict): Detection results from model inference
        """
        print("\n📊 Analyzing soiling detection confidence distributions...")
        
        total_detections = 0
        
        # Collect all detections by class
        for image_path, detections in results.items():
            scores = np.array(detections.get('scores', []))
            classes = np.array(detections.get('classes', []))
            class_names = detections.get('names', {})
            
            for score, class_id in zip(scores, classes):
                class_name = self.get_class_name(class_id, class_names)
                self.all_detections[class_name].append(score)
                total_detections += 1
        
        print(f"✓ Analyzed {total_detections} total detections")
        
        # Calculate statistics for each class
        self.class_stats = {}
        for class_name, scores in self.all_detections.items():
            if scores:
                scores_array = np.array(scores)
                self.class_stats[class_name] = {
                    'count': len(scores),
                    'mean': np.mean(scores_array),
                    'std': np.std(scores_array),
                    'min': np.min(scores_array),
                    'max': np.max(scores_array),
                    'q25': np.percentile(scores_array, 25),
                    'median': np.percentile(scores_array, 50),
                    'q75': np.percentile(scores_array, 75),
                    'q90': np.percentile(scores_array, 90),
                    'q95': np.percentile(scores_array, 95)
                }
        
        return self.class_stats
    
    def get_class_name(self, class_id, class_names):
        """Get class name from class ID"""
        if isinstance(class_names, dict) and class_names:
            key_options = [str(int(class_id)), int(class_id), str(class_id)]
            for key in key_options:
                if key in class_names:
                    return class_names[key]
        
        return self.class_mapping.get(int(class_id), f"Class_{int(class_id)}")
    
    def recommend_thresholds(self):
        """
        Recommend optimal thresholds based on confidence distributions
        """
        print("\n🎯 SOILING DETECTION THRESHOLD RECOMMENDATIONS:")
        print("=" * 70)
        
        recommended_thresholds = {}
        
        for class_name, stats in self.class_stats.items():
            if stats['count'] == 0:
                continue
                
            # Threshold recommendations based on soiling detection requirements
            if class_name == 'clean':
                # For clean panels, we want high confidence to avoid false positives
                # Recommend 75th percentile to keep only high-confidence clean detections
                recommended = max(0.6, stats['q75'])
                reasoning = "High threshold to ensure truly clean panels"
                
            elif class_name == 'medium_soiled':
                # Medium soiling is often harder to detect, use median
                # Balanced approach to catch moderate soiling
                recommended = max(0.4, stats['median'])
                reasoning = "Balanced threshold for moderate soiling detection"
                
            elif class_name == 'fully_soiled':
                # Fully soiled should be easier to detect, but critical to catch
                # Use 25th percentile to catch more cases of heavy soiling
                recommended = max(0.45, stats['q25'])
                reasoning = "Lower threshold to catch critical heavy soiling"
                
            else:
                # Generic recommendation
                recommended = stats['median']
                reasoning = "Median-based threshold"
            
            # Ensure threshold is reasonable (between 0.1 and 0.9)
            recommended = max(0.1, min(0.9, recommended))
            recommended_thresholds[class_name] = round(recommended, 2)
            
            print(f"\n{class_name.upper()}:")
            print(f"  Detections: {stats['count']}")
            print(f"  Confidence range: {stats['min']:.3f} - {stats['max']:.3f}")
            print(f"  Mean ± Std: {stats['mean']:.3f} ± {stats['std']:.3f}")
            print(f"  Quartiles: Q25={stats['q25']:.3f}, Q50={stats['median']:.3f}, Q75={stats['q75']:.3f}")
            print(f"  🎯 RECOMMENDED THRESHOLD: {recommended:.2f}")
            print(f"  💡 Reasoning: {reasoning}")
        
        return recommended_thresholds
    
    def create_visualizations(self):
        """Create confidence distribution visualizations"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Soiling Detection Confidence Analysis', fontsize=16, fontweight='bold')
            
            # Colors for each class
            colors = {'clean': 'green', 'medium_soiled': 'orange', 'fully_soiled': 'red'}
            
            # Plot 1: Histograms
            ax1 = axes[0, 0]
            for class_name, scores in self.all_detections.items():
                if scores:
                    ax1.hist(scores, bins=20, alpha=0.7, label=class_name, 
                            color=colors.get(class_name, 'blue'), density=True)
            ax1.set_xlabel('Confidence Score')
            ax1.set_ylabel('Density')
            ax1.set_title('Confidence Score Distributions')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Box plots
            ax2 = axes[0, 1]
            box_data = []
            box_labels = []
            for class_name in ['clean', 'medium_soiled', 'fully_soiled']:
                if class_name in self.all_detections and self.all_detections[class_name]:
                    box_data.append(self.all_detections[class_name])
                    box_labels.append(class_name)
            
            if box_data:
                bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
                for patch, class_name in zip(bp['boxes'], box_labels):
                    patch.set_facecolor(colors.get(class_name, 'blue'))
                    patch.set_alpha(0.7)
            ax2.set_ylabel('Confidence Score')
            ax2.set_title('Confidence Score Box Plots')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Detection counts
            ax3 = axes[1, 0]
            class_names = list(self.class_stats.keys())
            counts = [self.class_stats[name]['count'] for name in class_names]
            bars = ax3.bar(class_names, counts, color=[colors.get(name, 'blue') for name in class_names], alpha=0.7)
            ax3.set_ylabel('Number of Detections')
            ax3.set_title('Detection Counts by Soiling Level')
            ax3.grid(True, alpha=0.3)
            
            # Add count labels on bars
            for bar, count in zip(bars, counts):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        str(count), ha='center', va='bottom')
            
            # Plot 4: Cumulative distributions
            ax4 = axes[1, 1]
            for class_name, scores in self.all_detections.items():
                if scores:
                    sorted_scores = np.sort(scores)
                    y = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
                    ax4.plot(sorted_scores, y, label=class_name, 
                            color=colors.get(class_name, 'blue'), linewidth=2)
            ax4.set_xlabel('Confidence Score')
            ax4.set_ylabel('Cumulative Probability')
            ax4.set_title('Cumulative Distribution Functions')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'soiling_confidence_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Visualization saved to: {os.path.join(self.results_dir, 'soiling_confidence_analysis.png')}")
            
        except ImportError:
            print("⚠️  Matplotlib not available. Install with: pip install matplotlib")
        except Exception as e:
            print(f"⚠️  Could not create visualizations: {e}")
    
    def save_analysis_report(self, recommended_thresholds):
        """Save detailed analysis report"""
        report_path = os.path.join(self.results_dir, 'soiling_threshold_analysis.txt')
        
        with open(report_path, 'w') as f:
            f.write("SOILING DETECTION THRESHOLD ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Analysis Date: {pd.Timestamp.now()}\n")
            f.write(f"Total Images Analyzed: {len(self.all_detections)}\n\n")
            
            f.write("CONFIDENCE STATISTICS BY SOILING LEVEL:\n")
            f.write("-" * 50 + "\n")
            for class_name, stats in self.class_stats.items():
                f.write(f"\n{class_name.upper()}:\n")
                f.write(f"  Total detections: {stats['count']}\n")
                f.write(f"  Confidence range: {stats['min']:.3f} - {stats['max']:.3f}\n")
                f.write(f"  Mean: {stats['mean']:.3f}\n")
                f.write(f"  Standard deviation: {stats['std']:.3f}\n")
                f.write(f"  25th percentile: {stats['q25']:.3f}\n")
                f.write(f"  50th percentile (median): {stats['median']:.3f}\n")
                f.write(f"  75th percentile: {stats['q75']:.3f}\n")
                f.write(f"  90th percentile: {stats['q90']:.3f}\n")
                f.write(f"  95th percentile: {stats['q95']:.3f}\n")
            
            f.write(f"\nRECOMMENDED THRESHOLDS:\n")
            f.write("-" * 30 + "\n")
            for class_name, threshold in recommended_thresholds.items():
                f.write(f"{class_name}: {threshold}\n")
            
            f.write(f"\nPYTHON DICTIONARY FORMAT:\n")
            f.write("-" * 30 + "\n")
            f.write("balanced_thresholds = {\n")
            for class_name, threshold in recommended_thresholds.items():
                f.write(f"    '{class_name}': {threshold},\n")
            f.write("}\n")
        
        print(f"✓ Analysis report saved to: {report_path}")

def get_image_files(directory, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
    """Get all image files from a directory"""
    image_files = []
    seen_files = set()
    
    directory_path = Path(directory)
    for ext in extensions:
        for file_path in directory_path.glob(f'*{ext}'):
            file_key = file_path.name.lower()
            if file_key not in seen_files:
                image_files.append(str(file_path))
                seen_files.add(file_key)
        
        for file_path in directory_path.glob(f'*{ext.upper()}'):
            file_key = file_path.name.lower()
            if file_key not in seen_files:
                image_files.append(str(file_path))
                seen_files.add(file_key)
    
    return sorted(image_files)

def main():
    print("🧹 Soiling Detection Threshold Analyzer")
    print("=" * 60)
    print("Analyzing: clean, medium_soiled, fully_soiled")
    
    # Configuration - Update these paths for soiling analysis
    MODEL_PATH = 'C:/Users/DELL/Desktop/ACME/thermal and soiling/soiling_best.pt'
    IMAGES_DIR = 'C:/Users/DELL/Desktop/ACME/thermal and soiling/rgb'
    ANALYSIS_DIR = 'C:/Users/DELL/Desktop/ACME/thermal and soiling/rgb/soiling_analysis'
    BATCH_SIZE = 1
    
    # Check if files exist
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Soiling model file not found: {MODEL_PATH}")
        return
    
    if not os.path.exists(IMAGES_DIR):
        print(f"❌ Images directory not found: {IMAGES_DIR}")
        return
    
    try:
        # Memory optimization setup
        print("\n🧠 Setting up memory optimization...")
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.8)
            print(f"✓ GPU memory optimized")
        
        # Step 1: Load soiling model and run inference
        print("\n📋 Step 1: Loading soiling detection model...")
        inference = SoilingModelInference(MODEL_PATH)
        print("✓ Soiling model loaded successfully")
        
        # Get image files
        image_files = get_image_files(IMAGES_DIR)
        total_images = len(image_files)
        print(f"✓ Found {total_images} images for soiling analysis")
        
        if not image_files:
            print("❌ No images found!")
            return
        
        if total_images > 100:
            print(f"⚠️  Large dataset ({total_images} images) - this may take time")
            print("💡 Consider using a smaller sample for initial threshold analysis")
        
        # Step 2: Run inference to collect confidence data
        print("🔄 Running soiling detection to analyze confidence distributions...")
        results = inference.predict_batch(image_files, batch_size=BATCH_SIZE)
        print(f"✓ Soiling detection complete on {len(results)} images")
        
        # Clear model from memory
        del inference
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Step 3: Analyze confidence distributions
        print(f"\n📋 Step 3: Analyzing confidence distributions...")
        analyzer = SoilingThresholdAnalyzer(ANALYSIS_DIR)
        class_stats = analyzer.analyze_detections(results)
        
        # Step 4: Generate threshold recommendations
        print(f"\n📋 Step 4: Generating threshold recommendations...")
        recommended_thresholds = analyzer.recommend_thresholds()
        
        # Step 5: Create visualizations and save report
        print(f"\n📋 Step 5: Creating analysis visualizations...")
        analyzer.create_visualizations()
        analyzer.save_analysis_report(recommended_thresholds)
        
        # Final summary
        print(f"\n🎉 Soiling Threshold Analysis Complete!")
        print(f"📁 Analysis results saved in: {ANALYSIS_DIR}")
        print(f"\n🎯 RECOMMENDED THRESHOLDS FOR YOUR SOILING MODEL:")
        print("=" * 60)
        print("balanced_thresholds = {")
        for class_name, threshold in recommended_thresholds.items():
            print(f"    '{class_name}': {threshold},")
        print("}")
        
        print(f"\n💡 Next steps:")
        print(f"1. Review the confidence analysis charts")
        print(f"2. Test these thresholds on your validation images")
        print(f"3. Adjust thresholds based on your specific requirements")
        print(f"4. Use these thresholds in your soiling detection pipeline")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Final memory cleanup
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()