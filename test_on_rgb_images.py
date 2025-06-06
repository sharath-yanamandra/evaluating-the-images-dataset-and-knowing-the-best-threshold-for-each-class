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
        
    def _get_device(self, device):
        """Determine the best device to use"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def _load_model(self, model_path):
        """Load the PyTorch model"""
        try:
            # First, try to detect if it's a YOLO/Ultralytics model
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

class SoilingDetectionVisualizer:
    def __init__(self, output_dir='./soiling_detected_images', optimized_thresholds=None):
        """
        Initialize the soiling visualizer with data-driven optimized thresholds
        
        Args:
            output_dir (str): Directory to save visualized images
            optimized_thresholds (dict): Per-class confidence thresholds based on your analysis
        """
        self.output_dir = output_dir
        
        # Data-driven optimized thresholds from your analysis
        self.optimized_thresholds = optimized_thresholds or {
            'fully_soiled': 0.55,      # Q25 - catch critical heavy soiling cases
            'medium_soiled': 0.48,     # Median - balanced detection for moderate soiling
            'clean': 0.76              # Q75 - high confidence for truly clean panels
        }
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Color palette for soiling levels (BGR format for OpenCV)
        self.colors = {
            'clean': (0, 255, 0),           # Green - clean panels
            'medium_soiled': (0, 165, 255), # Orange - moderate soiling
            'fully_soiled': (0, 0, 255),    # Red - heavy soiling (critical)
            'default': (255, 255, 255)      # White - unknown
        }
        
        # Backup color list
        self.backup_colors = [
            (0, 255, 0), (0, 165, 255), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 255), (128, 128, 0)
        ]
        
        print(f"🧹 Using data-driven optimized thresholds:")
        for class_name, threshold in self.optimized_thresholds.items():
            print(f"  - {class_name}: {threshold}")
    
    def get_class_name(self, class_id, class_names):
        """Get class name from class ID for soiling detection"""
        if isinstance(class_names, dict) and class_names:
            key_options = [str(int(class_id)), int(class_id), str(class_id)]
            for key in key_options:
                if key in class_names:
                    return class_names[key]
        
        # Soiling class mapping
        soiling_classes = {
            0: 'clean',
            1: 'medium_soiled',
            2: 'fully_soiled'
        }
        
        return soiling_classes.get(int(class_id), f"Unknown_Class_{int(class_id)}")
    
    def apply_optimized_filtering(self, detections):
        """Apply data-driven optimized thresholds to filter detections"""
        boxes = np.array(detections.get('boxes', []))
        scores = np.array(detections.get('scores', []))
        classes = np.array(detections.get('classes', []))
        class_names = detections.get('names', {})
        
        if len(boxes) == 0:
            return boxes, scores, classes, {}
        
        # Apply per-class filtering with optimized thresholds
        keep_indices = []
        class_stats = {}
        
        for i, (score, class_id) in enumerate(zip(scores, classes)):
            class_name = self.get_class_name(class_id, class_names)
            threshold = self.optimized_thresholds.get(class_name, 0.5)
            
            # Track statistics
            if class_name not in class_stats:
                class_stats[class_name] = {'total': 0, 'kept': 0, 'threshold': threshold}
            class_stats[class_name]['total'] += 1
            
            if score >= threshold:
                keep_indices.append(i)
                class_stats[class_name]['kept'] += 1
        
        # Filter detections
        if keep_indices:
            filtered_boxes = boxes[keep_indices]
            filtered_scores = scores[keep_indices]
            filtered_classes = classes[keep_indices]
        else:
            filtered_boxes = np.array([])
            filtered_scores = np.array([])
            filtered_classes = np.array([])
        
        return filtered_boxes, filtered_scores, filtered_classes, class_stats
    
    def draw_soiling_detections(self, image_path, detections):
        """Draw soiling detections on image with optimized filtering"""
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Error loading image: {image_path}")
            return False
        
        height, width = img.shape[:2]
        
        # Apply optimized filtering
        filtered_boxes, filtered_scores, filtered_classes, class_stats = self.apply_optimized_filtering(detections)
        class_names = detections.get('names', {})
        
        total_original = len(detections.get('boxes', []))
        total_filtered = len(filtered_boxes)
        
        print(f"🧹 {os.path.basename(image_path)}: {total_original} → {total_filtered} detections (optimized filtering)")
        
        if len(filtered_boxes) == 0:
            print(f"   No detections pass optimized thresholds")
            cv2.putText(img, "No detections pass optimized thresholds", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            save_path = os.path.join(self.output_dir, f"{base_name}_soiling.jpg")
            cv2.imwrite(save_path, img)
            return True
        
        # Draw each detection
        class_counts = {}
        for i, (box, score, class_id) in enumerate(zip(filtered_boxes, filtered_scores, filtered_classes)):
            if len(box) != 4:
                continue
                
            x1, y1, x2, y2 = map(int, box)
            
            # Get color for this soiling level
            class_name = self.get_class_name(class_id, class_names)
            if class_name in self.colors:
                color = self.colors[class_name]
            else:
                color = self.backup_colors[int(class_id) % len(self.backup_colors)]
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Count detections per class
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # Prepare label with threshold info
            threshold_used = self.optimized_thresholds.get(class_name, 0.5)
            label = f"{class_name}: {score:.2f} (T:{threshold_used})"
            
            # Draw label background
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            cv2.rectangle(
                img, 
                (x1, y1 - label_height - 10), 
                (x1 + label_width, y1), 
                color, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                img, 
                label, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                2
            )
        
        # Add summary text
        summary = f"Soiling Detection: {total_filtered} panels (from {total_original})"
        cv2.putText(img, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add per-class soiling stats
        y_offset = 60
        for class_name in ['clean', 'medium_soiled', 'fully_soiled']:
            if class_name in class_counts:
                count = class_counts[class_name]
                threshold = self.optimized_thresholds.get(class_name, 0.5)
                class_text = f"{class_name}: {count} (T:{threshold})"
                color = self.colors.get(class_name, (255, 255, 255))
                cv2.putText(img, class_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                y_offset += 20
        
        # Save image
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(self.output_dir, f"{base_name}_soiling.jpg")
        cv2.imwrite(save_path, img)
        
        # Print detailed stats
        for class_name, stats in class_stats.items():
            if stats['total'] > 0:
                kept_pct = (stats['kept'] / stats['total']) * 100
                print(f"   {class_name}: {stats['kept']}/{stats['total']} kept ({kept_pct:.1f}%) at T:{stats['threshold']}")
        
        return True

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

def save_soiling_results(results, output_dir, format='json'):
    """Save soiling detection results"""
    os.makedirs(output_dir, exist_ok=True)
    
    if format == 'json':
        json_results = {}
        for path, result in results.items():
            filename = os.path.basename(path)
            if isinstance(result, dict):
                json_result = {}
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        json_result[key] = value.tolist()
                    else:
                        json_result[key] = value
                json_results[filename] = json_result
            elif isinstance(result, np.ndarray):
                json_results[filename] = result.tolist()
            elif isinstance(result, list):
                json_results[filename] = [r.tolist() if isinstance(r, np.ndarray) else r for r in result]
            else:
                json_results[filename] = result
        
        with open(os.path.join(output_dir, 'soiling_results.json'), 'w') as f:
            json.dump(json_results, f, indent=2)

def apply_soiling_filtering_to_results(results, optimized_thresholds):
    """Apply optimized thresholds to filter soiling results and create summary"""
    class_mapping = {
        0: 'clean',
        1: 'medium_soiled',
        2: 'fully_soiled'
    }
    
    filtered_results = {}
    class_counts_before = {'clean': 0, 'medium_soiled': 0, 'fully_soiled': 0}
    class_counts_after = {'clean': 0, 'medium_soiled': 0, 'fully_soiled': 0}
    
    for image_name, detections in results.items():
        boxes = np.array(detections.get('boxes', []))
        scores = np.array(detections.get('scores', []))
        classes = np.array(detections.get('classes', []))
        names = detections.get('names', {})
        
        # Count original detections
        for class_id in classes:
            class_name = class_mapping.get(int(class_id), f"Class_{int(class_id)}")
            if class_name in class_counts_before:
                class_counts_before[class_name] += 1
        
        # Apply per-class filtering
        keep_indices = []
        
        for i, (score, class_id) in enumerate(zip(scores, classes)):
            class_name = class_mapping.get(int(class_id), f"Class_{int(class_id)}")
            threshold = optimized_thresholds.get(class_name, 0.5)
            
            if score >= threshold:
                keep_indices.append(i)
                if class_name in class_counts_after:
                    class_counts_after[class_name] += 1
        
        # Create filtered detections
        if keep_indices:
            filtered_detections = {
                'boxes': boxes[keep_indices].tolist(),
                'scores': scores[keep_indices].tolist(),
                'classes': classes[keep_indices].tolist(),
                'names': names
            }
        else:
            filtered_detections = {
                'boxes': [],
                'scores': [],
                'classes': [],
                'names': names
            }
        
        filtered_results[image_name] = filtered_detections
    
    return filtered_results, class_counts_before, class_counts_after

def main():
    print("🧹 Solar Panel Soiling Detection with Data-Driven Thresholds")
    print("=" * 70)
    print("Detecting: clean, medium_soiled, fully_soiled")
    
    # Data-driven optimized thresholds from your analysis
    optimized_thresholds = {
        'fully_soiled': 0.55,      # Q25 - catch critical heavy soiling
        'medium_soiled': 0.48,     # Median - balanced moderate detection
        'clean': 0.76              # Q75 - high confidence for clean panels
    }
    
    print("\n🎯 Using data-driven optimized thresholds from your analysis:")
    for class_name, threshold in optimized_thresholds.items():
        print(f"  - {class_name}: {threshold}")
    print("=" * 70)
    
    # Configuration - Update these paths
    MODEL_PATH = 'C:/Users/DELL/Desktop/ACME/thermal and soiling/soiling_best.pt'
    IMAGES_DIR = 'C:/Users/DELL/Desktop/ACME/thermal and soiling/rgb'
    RESULTS_DIR = 'C:/Users/DELL/Desktop/ACME/thermal and soiling/rgb/result json'
    DETECTED_IMAGES_DIR = 'C:/Users/DELL/Desktop/ACME/thermal and soiling/rgb/result images'
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
        print(f"✓ Found {total_images} images for soiling detection")
        
        if not image_files:
            print("❌ No images found!")
            return
        
        if total_images > 50:
            print(f"⚠️  Processing {total_images} images - memory-optimized processing")
        
        # Step 2: Run soiling detection
        print("🔄 Running soiling detection...")
        results = inference.predict_batch(image_files, batch_size=BATCH_SIZE)
        print(f"✓ Soiling detection complete on {len(results)} images")
        
        # Clear model from memory
        del inference
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Step 3: Apply optimized filtering and save results
        print(f"\n📋 Step 3: Applying optimized filtering and saving results...")
        filtered_results, counts_before, counts_after = apply_soiling_filtering_to_results(results, optimized_thresholds)
        
        # Save filtered results
        save_soiling_results(filtered_results, RESULTS_DIR, format='json')
        print(f"✓ Optimized soiling results saved to '{RESULTS_DIR}/soiling_results.json'")
        
        # Step 4: Print soiling analysis statistics
        print(f"\n📊 SOILING DETECTION RESULTS (Optimized Thresholds):")
        print("=" * 70)
        total_before = sum(counts_before.values())
        total_after = sum(counts_after.values())
        
        for class_name in ['clean', 'medium_soiled', 'fully_soiled']:
            before = counts_before[class_name]
            after = counts_after[class_name]
            threshold = optimized_thresholds[class_name]
            if before > 0:
                kept_pct = (after / before) * 100
                print(f"{class_name.upper()}:")
                print(f"  Before: {before:4d} → After: {after:4d} ({kept_pct:5.1f}% kept)")
                print(f"  Threshold: T:{threshold}")
        
        print(f"\nTOTAL SUMMARY:")
        print(f"  Before: {total_before:4d} → After: {total_after:4d}")
        reduction = total_before - total_after
        reduction_pct = (reduction / total_before) * 100 if total_before > 0 else 0
        print(f"  Reduction: {reduction:4d} detections ({reduction_pct:.1f}%)")
        
        # Step 5: Create soiling visualizations
        print(f"\n📋 Step 5: Creating soiling detection visualizations...")
        visualizer = SoilingDetectionVisualizer(DETECTED_IMAGES_DIR, optimized_thresholds)
        
        processed_count = 0
        clean_panels = counts_after['clean']
        medium_soiled_panels = counts_after['medium_soiled']
        heavy_soiled_panels = counts_after['fully_soiled']
        
        # Process visualizations in chunks
        chunk_size = 10
        image_items = list(results.items())
        
        for i in range(0, len(image_items), chunk_size):
            chunk = image_items[i:i+chunk_size]
            print(f"Creating soiling visualizations for chunk {i//chunk_size + 1}/{(len(image_items)-1)//chunk_size + 1}")
            
            for image_path, detections in chunk:
                try:
                    success = visualizer.draw_soiling_detections(image_path, detections)
                    if success:
                        processed_count += 1
                except Exception as e:
                    print(f"Error processing visualization for {os.path.basename(image_path)}: {e}")
                    continue
            
            # Memory cleanup after each chunk
            gc.collect()
        
        print(f"\n🎉 Solar Panel Soiling Analysis Complete!")
        print(f"✓ Processed {processed_count} solar panel images")
        print(f"📁 Results saved in:")
        print(f"  📄 {RESULTS_DIR}/soiling_results.json - Optimized filtered soiling data")
        print(f"  🖼️  {DETECTED_IMAGES_DIR}/ - Images with soiling level annotations")
        
        # Final soiling assessment
        total_panels = clean_panels + medium_soiled_panels + heavy_soiled_panels
        if total_panels > 0:
            clean_pct = (clean_panels / total_panels) * 100
            medium_pct = (medium_soiled_panels / total_panels) * 100
            heavy_pct = (heavy_soiled_panels / total_panels) * 100
            
            print(f"\n📊 SOILING ASSESSMENT SUMMARY:")
            print(f"🟢 Clean panels: {clean_panels} ({clean_pct:.1f}%)")
            print(f"🟡 Medium soiled: {medium_soiled_panels} ({medium_pct:.1f}%)")
            print(f"🔴 Heavily soiled: {heavy_soiled_panels} ({heavy_pct:.1f}%)")
            
            if heavy_pct > 60:
                print(f"\n🚨 CRITICAL: {heavy_pct:.1f}% heavily soiled - immediate cleaning required!")
            elif heavy_pct > 30:
                print(f"\n⚠️  WARNING: {heavy_pct:.1f}% heavily soiled - cleaning recommended")
            elif clean_pct > 70:
                print(f"\n✅ GOOD: {clean_pct:.1f}% clean panels - system well maintained")
            else:
                print(f"\n📈 MODERATE: Mixed soiling levels - monitor and plan maintenance")
        
        print(f"\n💡 Data-driven soiling detection provides accurate maintenance insights!")
       
    except Exception as e:
        print(f"❌ Error during soiling detection: {e}")
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