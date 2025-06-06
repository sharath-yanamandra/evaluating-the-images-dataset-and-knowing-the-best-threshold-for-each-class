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

class ModelInference:
    def __init__(self, model_path, device='auto'):
        """
        Initialize the inference class
        
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
                print("Detected YOLO/Ultralytics model. Loading with ultralytics...")
                return self._load_yolo_model(model_path)
            
            # Try loading with weights_only=False for trusted models
            try:
                model = torch.load(model_path, map_location=self.device, weights_only=False)
                print("✓ Model loaded successfully (weights_only=False)")
            except Exception as first_error:
                print(f"First attempt failed: {first_error}")
                print("Trying with weights_only=True...")
                
                # Try with weights_only=True and safe globals
                try:
                    torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
                    model = torch.load(model_path, map_location=self.device, weights_only=True)
                    print("✓ Model loaded successfully (weights_only=True)")
                except Exception as second_error:
                    print(f"Second attempt failed: {second_error}")
                    raise first_error
            
            # If it's a state dict, you'll need to define your model architecture
            if isinstance(model, dict):
                print("Warning: Loaded state dict. You may need to define model architecture.")
                print("Available keys:", list(model.keys())[:5], "..." if len(model.keys()) > 5 else "")
                return model
            
            model.eval()  # Set to evaluation mode
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("If this is a state dict, please modify the script to include your model architecture.")
            raise
    
    def _is_yolo_model(self, model_path):
        """Check if the model is a YOLO model by examining the file"""
        try:
            # Try to peek at the model without fully loading it
            with open(model_path, 'rb') as f:
                content = f.read(1000)  # Read first 1000 bytes
                return b'ultralytics' in content or b'DetectionModel' in content
        except:
            return False
    
    def _load_yolo_model(self, model_path):
        """Load YOLO model using ultralytics"""
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            print("✓ YOLO model loaded successfully")
            return model
        except ImportError:
            print("ultralytics not installed. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
            from ultralytics import YOLO
            model = YOLO(model_path)
            print("✓ YOLO model loaded successfully")
            return model
    
    def _get_default_transform(self):
        """Default image preprocessing transform"""
        return transforms.Compose([
            transforms.Resize((224, 224)),  # Adjust size based on your model
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
    
    def set_custom_transform(self, transform):
        """Set a custom preprocessing transform"""
        self.transform = transform
    
    def preprocess_image(self, image_path):
        """Preprocess a single image"""
        try:
            image = Image.open(image_path).convert('RGB')
            tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
            return tensor.to(self.device)
        except Exception as e:
            print(f"Error preprocessing {image_path}: {e}")
            return None
    
    def predict_single(self, image_path):
        """Run inference on a single image with memory optimization"""
        # Handle YOLO models differently
        if hasattr(self.model, 'predict'):  # YOLO model
            try:
                # Clear GPU cache before prediction
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                results = self.model.predict(image_path, verbose=False, save=False)
                # Extract predictions from YOLO results
                if results:
                    result = results[0]
                    prediction = {
                        'boxes': result.boxes.xyxy.cpu().numpy() if result.boxes is not None else [],
                        'scores': result.boxes.conf.cpu().numpy() if result.boxes is not None else [],
                        'classes': result.boxes.cls.cpu().numpy() if result.boxes is not None else [],
                        'names': result.names if hasattr(result, 'names') else {}
                    }
                    
                    # Clear result from memory
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
            
            # Handle different output types
            if isinstance(output, torch.Tensor):
                return output.cpu().numpy()
            elif isinstance(output, (list, tuple)):
                return [o.cpu().numpy() if isinstance(o, torch.Tensor) else o for o in output]
            else:
                return output
    
    def predict_batch(self, image_paths, batch_size=1):
        """Run inference on multiple images with memory optimization"""
        results = {}
        
        # Handle YOLO models with memory-efficient processing
        if hasattr(self.model, 'predict'):  # YOLO model
            print(f"Using memory-optimized YOLO prediction (batch_size={batch_size})...")
            
            # Process in smaller batches to avoid memory overflow
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i+batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(image_paths)-1)//batch_size + 1}: {len(batch_paths)} images")
                
                try:
                    # Clear GPU cache before each batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Process batch with YOLO
                    yolo_results = self.model.predict(batch_paths, verbose=False, save=False, stream=True)
                    
                    for path, result in zip(batch_paths, yolo_results):
                        prediction = {
                            'boxes': result.boxes.xyxy.cpu().numpy() if result.boxes is not None else [],
                            'scores': result.boxes.conf.cpu().numpy() if result.boxes is not None else [],
                            'classes': result.boxes.cls.cpu().numpy() if result.boxes is not None else [],
                            'names': result.names if hasattr(result, 'names') else {}
                        }
                        results[path] = prediction
                        
                        # Clear result from memory immediately
                        del result
                    
                    # Force garbage collection
                    import gc
                    gc.collect()
                    
                except Exception as e:
                    print(f"Batch prediction failed: {e}")
                    print("Falling back to single image predictions...")
                    # Fall back to single predictions for this batch
                    for path in batch_paths:
                        try:
                            result = self.predict_single(path)
                            if result is not None:
                                results[path] = result
                        except Exception as single_error:
                            print(f"Failed to process {os.path.basename(path)}: {single_error}")
                            continue
            
            return results
        
        # Handle regular PyTorch models
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_tensors = []
            valid_paths = []
            
            # Preprocess batch
            for path in batch_paths:
                tensor = self.preprocess_image(path)
                if tensor is not None:
                    batch_tensors.append(tensor.squeeze(0))  # Remove batch dim for stacking
                    valid_paths.append(path)
            
            if not batch_tensors:
                continue
                
            # Stack tensors and run inference
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                
                # Store results
                if isinstance(outputs, torch.Tensor):
                    outputs = outputs.cpu().numpy()
                    for j, path in enumerate(valid_paths):
                        results[path] = outputs[j]
                elif isinstance(outputs, (list, tuple)):
                    outputs = [o.cpu().numpy() if isinstance(o, torch.Tensor) else o for o in outputs]
                    for j, path in enumerate(valid_paths):
                        results[path] = [o[j] for o in outputs]
        
        return results

class DetectionVisualizer:
    def __init__(self, output_dir='./detected_images', balanced_thresholds=None):
        """
        Initialize the visualizer with balanced thresholds for 3 classes
        
        Args:
            output_dir (str): Directory to save visualized images
            balanced_thresholds (dict): Per-class confidence thresholds
        """
        self.output_dir = output_dir
        
        # Updated balanced thresholds for 3 classes only
        self.balanced_thresholds = balanced_thresholds or {
            'hotspot': 0.5,                        # Critical defects - balanced threshold
            'string_reverse_polarity': 0.35,       # Lower threshold - preserve rare detections
            'diode_failure': 0.45                  # Balanced approach for component failures
        }
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Updated color palette for 3 solar defect classes (BGR format for OpenCV)
        self.colors = {
            'hotspot': (0, 0, 255),                 # Red - for hotspots (critical)
            'string_reverse_polarity': (0, 165, 255), # Orange - for polarity issues  
            'diode_failure': (128, 0, 128),         # Purple - for diode failures
            'default': (0, 255, 0)                  # Green - for unknown classes
        }
        
        # Backup color list for any additional classes
        self.backup_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 255), (128, 128, 0)
        ]
        
        print(f"🎯 Using balanced thresholds for 3 classes:")
        for class_name, threshold in self.balanced_thresholds.items():
            print(f"  - {class_name}: {threshold}")
    
    def get_class_name(self, class_id, class_names):
        """Get class name from class ID for the updated 3-class model"""
        if isinstance(class_names, dict) and class_names:
            # Try different key formats
            key_options = [str(int(class_id)), int(class_id), str(class_id)]
            for key in key_options:
                if key in class_names:
                    return class_names[key]
        
        # Updated custom classes for the new 3-class model
        custom_classes = {
            0: 'hotspot',
            1: 'string_reverse_polarity',
            2: 'diode_failure'
        }
        
        return custom_classes.get(int(class_id), f"Unknown_Class_{int(class_id)}")
    
    def apply_balanced_filtering(self, detections):
        """Apply balanced thresholds to filter detections per class"""
        boxes = np.array(detections.get('boxes', []))
        scores = np.array(detections.get('scores', []))
        classes = np.array(detections.get('classes', []))
        class_names = detections.get('names', {})
        
        if len(boxes) == 0:
            return boxes, scores, classes, {}
        
        # Apply per-class filtering
        keep_indices = []
        class_stats = {}
        
        for i, (score, class_id) in enumerate(zip(scores, classes)):
            class_name = self.get_class_name(class_id, class_names)
            threshold = self.balanced_thresholds.get(class_name, 0.5)
            
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
    
    def draw_detections(self, image_path, detections):
        """Draw detections on image and save with balanced filtering"""
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Error loading image: {image_path}")
            return False
        
        height, width = img.shape[:2]
        
        # Apply balanced filtering
        filtered_boxes, filtered_scores, filtered_classes, class_stats = self.apply_balanced_filtering(detections)
        class_names = detections.get('names', {})
        
        total_original = len(detections.get('boxes', []))
        total_filtered = len(filtered_boxes)
        
        print(f"🔍 {os.path.basename(image_path)}: {total_original} → {total_filtered} detections (balanced filtering)")
        
        if len(filtered_boxes) == 0:
            print(f"   No detections pass balanced thresholds")
            cv2.putText(img, "No detections pass balanced thresholds", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            save_path = os.path.join(self.output_dir, f"{base_name}_balanced.jpg")
            cv2.imwrite(save_path, img)
            return True
        
        # Draw each detection
        class_counts = {}
        for i, (box, score, class_id) in enumerate(zip(filtered_boxes, filtered_scores, filtered_classes)):
            if len(box) != 4:
                continue
                
            x1, y1, x2, y2 = map(int, box)
            
            # Get color for this class
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
            threshold_used = self.balanced_thresholds.get(class_name, 0.5)
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
        summary = f"Balanced Filtering: {total_filtered} detections (from {total_original})"
        cv2.putText(img, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add per-class filtering stats
        y_offset = 60
        for class_name, count in class_counts.items():
            threshold = self.balanced_thresholds.get(class_name, 0.5)
            class_text = f"{class_name}: {count} (T:{threshold})"
            cv2.putText(img, class_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            y_offset += 20
        
        # Save image
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(self.output_dir, f"{base_name}_balanced.jpg")
        cv2.imwrite(save_path, img)
        
        # Print detailed stats
        for class_name, stats in class_stats.items():
            if stats['total'] > 0:
                kept_pct = (stats['kept'] / stats['total']) * 100
                print(f"   {class_name}: {stats['kept']}/{stats['total']} kept ({kept_pct:.1f}%) at ≥{stats['threshold']}")
        
        return True

def get_image_files(directory, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
    """Get all image files from a directory (avoiding duplicates)"""
    image_files = []
    seen_files = set()
    
    directory_path = Path(directory)
    for ext in extensions:
        # Search for lowercase extensions
        for file_path in directory_path.glob(f'*{ext}'):
            file_key = file_path.name.lower()
            if file_key not in seen_files:
                image_files.append(str(file_path))
                seen_files.add(file_key)
        
        # Search for uppercase extensions
        for file_path in directory_path.glob(f'*{ext.upper()}'):
            file_key = file_path.name.lower()
            if file_key not in seen_files:
                image_files.append(str(file_path))
                seen_files.add(file_key)
    
    print(f"📁 Image file breakdown:")
    ext_counts = {}
    for file_path in image_files:
        ext = Path(file_path).suffix.lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
    
    for ext, count in ext_counts.items():
        print(f"  {ext}: {count} files")
    
    return sorted(image_files)  # Sort for consistent processing order

def save_results(results, output_dir, format='json'):
    """Save inference results"""
    os.makedirs(output_dir, exist_ok=True)
    
    if format == 'json':
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for path, result in results.items():
            filename = os.path.basename(path)
            if isinstance(result, dict):  # YOLO format
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
        
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(json_results, f, indent=2)

def apply_balanced_filtering_to_results(results, balanced_thresholds):
    """Apply balanced thresholds to filter results and create summary for 3 classes"""
    # Updated class mapping for the new 3-class model
    class_mapping = {
        0: 'hotspot',
        1: 'string_reverse_polarity',
        2: 'diode_failure'
    }
    
    filtered_results = {}
    class_counts_before = {'hotspot': 0, 'string_reverse_polarity': 0, 'diode_failure': 0}
    class_counts_after = {'hotspot': 0, 'string_reverse_polarity': 0, 'diode_failure': 0}
    
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
            threshold = balanced_thresholds.get(class_name, 0.5)
            
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
    print("🌞 Solar Panel Defect Detection - Updated 3-Class Model")
    print("=" * 60)
    print("Detecting: hotspot, string_reverse_polarity, diode_failure")
    
    # Updated balanced thresholds for 3 classes
    balanced_thresholds = {
        'hotspot': 0.5,                        # Critical defects - balanced threshold
        'string_reverse_polarity': 0.35,       # Lower threshold - preserve rare detections
        'diode_failure': 0.45                  # Balanced approach for component failures
    }
    
    print("\n🎯 Using balanced thresholds for 3 classes:")
    for class_name, threshold in balanced_thresholds.items():
        print(f"  - {class_name}: {threshold}")
    print("=" * 60)
    
    # Configuration - Update these paths
    MODEL_PATH = 'C:/Users/DELL/Desktop/ACME/thermal and soiling/thermal_best.pt'
    IMAGES_DIR = 'C:/Users/DELL/Desktop/ACME/thermal and soiling/thermal'
    RESULTS_DIR = 'C:/Users/DELL/Desktop/ACME/thermal and soiling/result images'
    DETECTED_IMAGES_DIR = 'C:/Users/DELL/Desktop/ACME/thermal and soiling/result json'
    BATCH_SIZE = 1  # Reduced batch size to prevent memory issues
    
    # Check if files exist
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
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
            print(f"✓ GPU memory cleared")
            # Set memory fraction to prevent OOM
            torch.cuda.set_per_process_memory_fraction(0.8)
        
        # Step 1: Load model and run inference
        print("\n📋 Step 1: Loading solar defect detection model...")
        inference = ModelInference(MODEL_PATH)
        print("✓ Model loaded successfully")
        
        # Get image files with memory consideration
        image_files = get_image_files(IMAGES_DIR)
        total_images = len(image_files)
        print(f"✓ Found {total_images} solar panel images")
        
        if not image_files:
            print("❌ No images found!")
            return
        
        # Memory warning for large datasets
        if total_images > 50:
            print(f"⚠️  Processing {total_images} images - this may take time to prevent memory overflow")
            print("💡 Processing with batch_size=1 for memory safety")
        
        # Run batch inference with memory optimization
        print("🔄 Running defect detection with memory optimization...")
        results = inference.predict_batch(image_files, batch_size=BATCH_SIZE)
        print(f"✓ Defect detection complete on {len(results)} images")
        
        # Clear inference model from memory
        del inference
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Step 2: Apply balanced filtering and save results
        print(f"\n📋 Step 2: Applying balanced filtering and saving results...")
        filtered_results, counts_before, counts_after = apply_balanced_filtering_to_results(results, balanced_thresholds)
        
        # Save filtered results
        save_results(filtered_results, RESULTS_DIR, format='json')
        print(f"✓ Balanced filtered results saved to '{RESULTS_DIR}/results.json'")
        
        # Step 3: Print comparison statistics
        print(f"\n📊 BALANCED FILTERING RESULTS (3 Classes):")
        print("=" * 60)
        total_before = sum(counts_before.values())
        total_after = sum(counts_after.values())
        
        for class_name in ['hotspot', 'string_reverse_polarity', 'diode_failure']:
            before = counts_before[class_name]
            after = counts_after[class_name]
            threshold = balanced_thresholds[class_name]
            if before > 0:
                kept_pct = (after / before) * 100
                print(f"{class_name}:")
                print(f"  Before: {before:4d} → After: {after:4d} ({kept_pct:5.1f}% kept)")
                print(f"  Threshold: ≥{threshold}")
        
        print(f"\nTOTAL SUMMARY:")
        print(f"  Before: {total_before:4d} → After: {total_after:4d}")
        reduction = total_before - total_after
        reduction_pct = (reduction / total_before) * 100 if total_before > 0 else 0
        print(f"  Reduction: {reduction:4d} detections ({reduction_pct:.1f}%)")
        
        # Step 4: Create balanced visualizations with memory management
        print(f"\n📋 Step 4: Creating balanced visualization images...")
        visualizer = DetectionVisualizer(DETECTED_IMAGES_DIR, balanced_thresholds)
        
        processed_count = 0
        critical_panels = 0
        
        # Process visualizations in smaller chunks to manage memory
        chunk_size = 10  # Process 10 images at a time for visualization
        image_items = list(results.items())
        
        for i in range(0, len(image_items), chunk_size):
            chunk = image_items[i:i+chunk_size]
            print(f"Creating visualizations for chunk {i//chunk_size + 1}/{(len(image_items)-1)//chunk_size + 1}")
            
            for image_path, detections in chunk:
                try:
                    success = visualizer.draw_detections(image_path, detections)
                    if success:
                        processed_count += 1
                        
                        # Check for critical defects (hotspots) after filtering
                        filtered_boxes, filtered_scores, filtered_classes, _ = visualizer.apply_balanced_filtering(detections)
                        class_names = detections.get('names', {})
                        
                        for class_id in filtered_classes:
                            class_name = visualizer.get_class_name(class_id, class_names)
                            if class_name == 'hotspot':
                                critical_panels += 1
                                break
                except Exception as e:
                    print(f"Error processing visualization for {os.path.basename(image_path)}: {e}")
                    continue
            
            # Memory cleanup after each chunk
            gc.collect()
        
        print(f"\n🎉 Solar Defect Analysis Complete (3-Class Model)!")
        print(f"✓ Processed {processed_count} solar panel images")
        print(f"🔴 Critical panels with hotspots (after balanced filtering): {critical_panels}")
        print(f"📁 Results saved in:")
        print(f"  📄 {RESULTS_DIR}/results.json - Balanced filtered detection data")
        print(f"  🖼️  {DETECTED_IMAGES_DIR}/ - Images with balanced threshold annotations")
        
        # Final recommendations
        hotspot_count = counts_after['hotspot']
        diode_failure_count = counts_after['diode_failure']
        polarity_count = counts_after['string_reverse_polarity']
        images_with_defects = len([r for r in filtered_results.values() if len(r['boxes']) > 0])
        defect_rate = (images_with_defects / len(results)) * 100
        
        print(f"\n📊 DEFECT SUMMARY:")
        if hotspot_count > 0:
            print(f"🚨 URGENT: {hotspot_count} high-confidence hotspots detected - immediate inspection required!")
        
        if diode_failure_count > 0:
            print(f"⚠️  WARNING: {diode_failure_count} diode failures detected - component replacement needed")
        
        if polarity_count > 0:
            print(f"🔧 MAINTENANCE: {polarity_count} string reverse polarity issues - wiring inspection required")
        
        if defect_rate > 20:
            print(f"⚠️  WARNING: High defect rate ({defect_rate:.1f}%) - system maintenance recommended")
        elif defect_rate < 5:
            print(f"✅ GOOD: Low defect rate ({defect_rate:.1f}%) - system appears healthy")
        else:
            print(f"📈 MODERATE: Defect rate ({defect_rate:.1f}%) - monitor system performance")
        
        print(f"\n💡 3-Class model provides focused defect detection for critical issues!")
        print(f"🧠 Memory-optimized processing completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
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


'''
 Solar Defect Analysis Complete (3-Class Model)!
✓ Processed 3153 solar panel images
🔴 Critical panels with hotspots (after balanced filtering): 1247
📁 Results saved in:
  📄 C:/Users/DELL/Desktop/ACME/thermal and soiling/result images/results.json - Balanced filtered detection data
  🖼️  C:/Users/DELL/Desktop/ACME/thermal and soiling/result json/ - Images with balanced threshold annotations

📊 DEFECT SUMMARY:
🚨 URGENT: 2790 high-confidence hotspots detected - immediate inspection required!
⚠️  WARNING: 1003 diode failures detected - component replacement needed
🔧 MAINTENANCE: 2 string reverse polarity issues - wiring inspection required
⚠️  WARNING: High defect rate (50.7%) - system maintenance recommended

💡 3-Class model provides focused defect detection for critical issues!
🧠 Memory-optimized processing completed successfully!
'''