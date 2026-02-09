import os
import json
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from typing import List, Tuple, Dict, Optional
import torch
import warnings
warnings.filterwarnings('ignore')

class PseudoBoxGenerator:
    """Optimized heuristic-based pseudo bounding box generator for lecture content"""
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize the pseudo box generator
        
        Args:
            device: 'cpu' or 'cuda' for face detection
        """
        self.device = device
        self.face_cascade = None
        self.text_detector = None
        self._init_detectors()
        
    def _init_detectors(self):
        """Initialize OpenCV detectors"""
        try:
            # Load face cascade for instructor detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            # Initialize text detector (MSER for equations)
            self.text_detector = cv2.MSER_create(
                _delta=5,
                _min_area=50,
                _max_area=20000,
                _max_variation=0.25
            )
        except Exception as e:
            print(f"Warning: Could not initialize detectors: {e}")
    
    def detect_equations(self, image: np.ndarray) -> List[List[int]]:
        """
        Detect equation regions using text detection and contour analysis
        
        Equations typically have:
        - High contrast text
        - Rectangular boundaries
        - Dense text regions
        """
        boxes = []
        
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Enhance contrast for better detection
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Method 1: MSER for text regions
            regions, _ = self.text_detector.detectRegions(enhanced)
            
            for region in regions:
                if len(region) > 10:  # Filter small regions
                    x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
                    
                    # Filter by aspect ratio (equations are often wider than tall)
                    aspect_ratio = w / max(h, 1)
                    if 0.5 < aspect_ratio < 8 and w > 30 and h > 20:
                        boxes.append([x, y, w, h])
            
            # Method 2: Adaptive thresholding for dark text on light background
            binary = cv2.adaptiveThreshold(enhanced, 255, 
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 2)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 100:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    # Check if this box overlaps significantly with existing boxes
                    overlapping = False
                    for existing in boxes:
                        iou = self._calculate_iou([x, y, w, h], existing)
                        if iou > 0.5:
                            overlapping = True
                            break
                    
                    if not overlapping:
                        boxes.append([x, y, w, h])
            
            # Merge overlapping boxes
            boxes = self._merge_overlapping_boxes(boxes, threshold=0.3)
            
            # If no boxes found, fall back to entire image
            if not boxes:
                h, w = image.shape[:2]
                boxes = [[0, 0, w, h]]
            
        except Exception as e:
            print(f"Equation detection failed: {e}")
            h, w = image.shape[:2]
            boxes = [[0, 0, w, h]]
        
        return boxes
    
    def detect_diagrams(self, image: np.ndarray) -> List[List[int]]:
        """
        Detect diagram regions using edge detection and shape analysis
        
        Diagrams typically have:
        - Strong edges
        - Geometric shapes
        - Connected components
        """
        boxes = []
        
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Dilate edges to connect nearby edges
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 200:  # Minimum area for diagrams
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    # Filter by shape complexity (diagrams often have complex shapes)
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity < 0.8:  # Not too circular
                            boxes.append([x, y, w, h])
            
            # If no boxes found, look for color transitions
            if not boxes:
                boxes = self._detect_color_regions(image)
            
            # Merge overlapping boxes
            boxes = self._merge_overlapping_boxes(boxes, threshold=0.4)
            
            if not boxes:
                h, w = image.shape[:2]
                boxes = [[0, 0, w, h]]
                
        except Exception as e:
            print(f"Diagram detection failed: {e}")
            h, w = image.shape[:2]
            boxes = [[0, 0, w, h]]
        
        return boxes
    
    def detect_instructor(self, image: np.ndarray) -> List[List[int]]:
        """
        Detect instructor using face and upper body detection
        """
        boxes = []
        
        try:
            if self.face_cascade is None:
                # Fallback: look for human-shaped contours
                return self._detect_human_shaped_regions(image)
            
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            for (x, y, w, h) in faces:
                # Expand box to include upper body (assuming face is in upper part)
                body_h = int(h * 2.5)  # Approximate body height
                body_y = max(0, y - int(h * 0.5))
                body_h = min(image.shape[0] - body_y, body_h)
                
                boxes.append([x, body_y, w, body_h])
            
            # If no faces found, try full body detection via contours
            if not boxes:
                boxes = self._detect_human_shaped_regions(image)
            
        except Exception as e:
            print(f"Instructor detection failed: {e}")
            # Fallback to center region
            h, w = image.shape[:2]
            boxes = [[int(w*0.3), int(h*0.2), int(w*0.4), int(h*0.6)]]
        
        return boxes
    
    def detect_code_snippets(self, image: np.ndarray) -> List[List[int]]:
        """
        Detect code snippet regions (monospaced text, often in boxes)
        """
        boxes = []
        
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Look for rectangular regions with uniform texture
            binary = cv2.adaptiveThreshold(gray, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 2)
            
            # Find contours and look for rectangular shapes
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 500 < area < 50000:  # Reasonable size for code blocks
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    # Check rectangularity
                    rect_area = w * h
                    if area / rect_area > 0.7:  # Good rectangle
                        boxes.append([x, y, w, h])
            
            if not boxes:
                # Try finding monospaced text regions
                boxes = self.detect_equations(image)
            
        except Exception as e:
            print(f"Code snippet detection failed: {e}")
            h, w = image.shape[:2]
            boxes = [[int(w*0.1), int(h*0.1), int(w*0.8), int(h*0.8)]]
        
        return boxes
    
    def _detect_color_regions(self, image: np.ndarray) -> List[List[int]]:
        """Detect regions with distinct color characteristics"""
        boxes = []
        
        if len(image.shape) != 3:
            return boxes
        
        # Convert to LAB color space for better color difference
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Calculate color variance per pixel
        l, a, b = cv2.split(lab)
        color_variance = np.std([l, a, b], axis=0)
        
        # Threshold on color variance
        _, binary = cv2.threshold(color_variance.astype(np.uint8), 
                                  20, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                boxes.append([x, y, w, h])
        
        return boxes
    
    def _detect_human_shaped_regions(self, image: np.ndarray) -> List[List[int]]:
        """Fallback method for human detection using shape analysis"""
        boxes = []
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Use background subtraction (simple threshold)
        _, binary = cv2.threshold(gray, 0, 255, 
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find large foreground regions
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:  # Minimum size for person
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Check aspect ratio (human-like)
                aspect_ratio = h / max(w, 1)
                if 1.5 < aspect_ratio < 4:  # Typical human aspect ratio
                    boxes.append([x, y, w, h])
        
        return boxes
    
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Calculate Intersection over Union for two boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate coordinates
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = w1 * h1
        box2_area = w2 * h2
        
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0
    
    def _merge_overlapping_boxes(self, boxes: List[List[int]], 
                                threshold: float = 0.3) -> List[List[int]]:
        """Merge overlapping boxes"""
        if not boxes:
            return []
        
        boxes = sorted(boxes, key=lambda x: x[2]*x[3], reverse=True)
        merged = []
        used = [False] * len(boxes)
        
        for i in range(len(boxes)):
            if used[i]:
                continue
            
            current = boxes[i]
            for j in range(i+1, len(boxes)):
                if used[j]:
                    continue
                
                if self._calculate_iou(current, boxes[j]) > threshold:
                    # Merge boxes
                    x1 = min(current[0], boxes[j][0])
                    y1 = min(current[1], boxes[j][1])
                    x2 = max(current[0] + current[2], boxes[j][0] + boxes[j][2])
                    y2 = max(current[1] + current[3], boxes[j][1] + boxes[j][3])
                    
                    current = [x1, y1, x2 - x1, y2 - y1]
                    used[j] = True
            
            merged.append(current)
            used[i] = True
        
        return merged
    
    def generate_pseudo_boxes(self, image_path: str, 
                            concept: str) -> Tuple[List[List[int]], List[int]]:
        """
        Generate pseudo boxes for an image based on concept
        
        Returns:
            Tuple of (list_of_boxes, fallback_box)
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            # Generate boxes based on concept
            concept_lower = concept.lower()
            
            if 'equation' in concept_lower or 'math' in concept_lower:
                boxes = self.detect_equations(image)
            elif 'diagram' in concept_lower or 'chart' in concept_lower:
                boxes = self.detect_diagrams(image)
            elif 'instructor' in concept_lower or 'person' in concept_lower:
                boxes = self.detect_instructor(image)
            elif 'code' in concept_lower or 'snippet' in concept_lower:
                boxes = self.detect_code_snippets(image)
            elif 'flow' in concept_lower:
                boxes = self.detect_diagrams(image)  # Similar to diagrams
            elif 'table' in concept_lower:
                # Tables often have grid-like structure
                boxes = self._detect_table_regions(image)
            else:
                # Default: detect any text-like regions
                boxes = self.detect_equations(image)
            
            # Ensure we have at least one box
            if not boxes:
                boxes = [[0, 0, w, h]]
            
            # Fallback box is the entire image
            fallback_box = [0, 0, w, h]
            
            return boxes, fallback_box
            
        except Exception as e:
            print(f"Error generating boxes for {image_path}: {e}")
            h, w = 640, 480  # Default dimensions
            return [[0, 0, w, h]], [0, 0, w, h]
    
    def _detect_table_regions(self, image: np.ndarray) -> List[List[int]]:
        """Detect table-like regions using line detection"""
        boxes = []
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Detect lines using Hough transform
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                               minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            # Find bounding box of lines
            all_points = lines.reshape(-1, 2)
            x_min, y_min = all_points.min(axis=0)
            x_max, y_max = all_points.max(axis=0)
            
            w, h = x_max - x_min, y_max - y_min
            if w > 50 and h > 50:  # Reasonable table size
                boxes.append([x_min, y_min, w, h])
        
        return boxes

def create_coco_annotation(data_root: str, output_dir: str, 
                         splits: List[str] = ['train', 'val', 'test'],
                         max_images: Optional[int] = None,
                         min_box_size: int = 20) -> Dict[str, str]:
    """
    Main function to create COCO-format JSON files
    
    Args:
        data_root: Root directory containing concept folders
        output_dir: Where to save annotation files
        splits: List of splits to process
        max_images: Maximum number of images per split (None for all)
        min_box_size: Minimum box dimension to keep
    
    Returns:
        Dictionary of output file paths
    """
    
    # Define lecture-specific categories
    categories = [
        {"id": 1, "name": "equation"},
        {"id": 2, "name": "diagram"},
        {"id": 3, "name": "flow_chart"},
        {"id": 4, "name": "code_snippet"},
        {"id": 5, "name": "instructor"},
        {"id": 6, "name": "table"},
        {"id": 7, "name": "graph_chart"},
        {"id": 8, "name": "text_slide"},
    ]
    
    category_name_to_id = {cat["name"]: cat["id"] for cat in categories}
    category_id_to_name = {cat["id"]: cat["name"] for cat in categories}
    
    # Initialize pseudo box generator
    box_generator = PseudoBoxGenerator()
    
    output_files = {}
    
    for split in splits:
        print(f"\n{'='*60}")
        print(f"Processing {split} split...")
        print(f"{'='*60}")
        
        coco_data = {
            "info": {
                "description": f"Lecture Frames - {split}",
                "version": "1.0",
                "year": 2024,
                "contributor": "Pseudo-box Generator",
                "date_created": str(np.datetime64('now'))
            },
            "licenses": [{"id": 1, "name": "Academic Use"}],
            "categories": categories,
            "images": [],
            "annotations": []
        }
        
        image_id = 1
        annotation_id = 1
        processed_count = 0
        total_boxes = 0
        
        # Walk through directory structure
        concept_folders = [f for f in os.listdir(data_root) 
                          if os.path.isdir(os.path.join(data_root, f))]
        
        for concept_folder in concept_folders:
            concept_path = os.path.join(data_root, concept_folder)
            
            split_path = os.path.join(concept_path, split)
            if not os.path.exists(split_path):
                print(f"  Skipping {concept_folder}/{split} - directory not found")
                continue
            
            # Get list of image files
            image_files = []
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                image_files.extend(list(Path(split_path).glob(f'*{ext}')))
                image_files.extend(list(Path(split_path).glob(f'*{ext.upper()}')))
            
            if max_images:
                image_files = image_files[:max_images]
            
            print(f"  Processing {concept_folder}: {len(image_files)} images")
            
            for image_file in image_files:
                if max_images and processed_count >= max_images:
                    break
                
                try:
                    # Get image dimensions
                    with Image.open(image_file) as img:
                        width, height = img.size
                    
                    # Add image info
                    image_info = {
                        "id": image_id,
                        "file_name": str(Path(concept_folder) / split / image_file.name),
                        "width": width,
                        "height": height,
                        "license": 1,
                        "concept": concept_folder
                    }
                    coco_data["images"].append(image_info)
                    
                    # Generate pseudo-bounding boxes
                    boxes, fallback_box = box_generator.generate_pseudo_boxes(
                        str(image_file), concept_folder
                    )
                    
                    # Filter small boxes
                    valid_boxes = []
                    for box in boxes:
                        x, y, w, h = box
                        if w >= min_box_size and h >= min_box_size:
                            # Ensure box is within image bounds
                            x = max(0, min(x, width - 1))
                            y = max(0, min(y, height - 1))
                            w = min(w, width - x)
                            h = min(h, height - y)
                            valid_boxes.append([x, y, w, h])
                    
                    # If no valid boxes, use fallback
                    if not valid_boxes:
                        valid_boxes = [fallback_box]
                    
                    # Add annotations for each box
                    category_id = category_name_to_id.get(
                        concept_folder.lower().replace(' ', '_'), 1
                    )
                    
                    for box in valid_boxes:
                        x, y, w, h = box
                        area = w * h
                        
                        annotation = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_id,
                            "bbox": [float(x), float(y), float(w), float(h)],
                            "area": float(area),
                            "segmentation": [],
                            "iscrowd": 0,
                            "attributes": {
                                "generated_by": "pseudo_box_generator",
                                "concept": concept_folder
                            }
                        }
                        coco_data["annotations"].append(annotation)
                        
                        annotation_id += 1
                        total_boxes += 1
                    
                    image_id += 1
                    processed_count += 1
                    
                    # Progress indicator
                    if processed_count % 50 == 0:
                        print(f"    Processed {processed_count} images...")
                    
                except Exception as e:
                    print(f"    Error processing {image_file}: {e}")
                    continue
        
        # Save the COCO JSON file
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{split}.json")
        
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2, default=str)
        
        output_files[split] = output_file
        
        print(f"\n  Created {output_file}")
        print(f"  Images: {len(coco_data['images'])}")
        print(f"  Annotations: {len(coco_data['annotations'])}")
        print(f"  Average boxes per image: {len(coco_data['annotations'])/max(len(coco_data['images']),1):.2f}")
        
        # Save a summary
        summary_file = os.path.join(output_dir, f"{split}_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Split: {split}\n")
            f.write(f"Total images: {len(coco_data['images'])}\n")
            f.write(f"Total annotations: {len(coco_data['annotations'])}\n")
            f.write(f"Categories:\n")
            for cat in categories:
                count = sum(1 for ann in coco_data["annotations"] 
                           if ann["category_id"] == cat["id"])
                f.write(f"  {cat['name']}: {count}\n")
    
    # Create visualization of box generation
    visualize_sample_boxes(data_root, output_dir, box_generator)
    
    print(f"\n{'='*60}")
    print("Pseudo Box Generation Complete!")
    print(f"{'='*60}")
    print("\n⚠️  IMPORTANT NOTES:")
    print("1. These are PSEUDO boxes generated by heuristics")
    print("2. Quality varies by concept and image complexity")
    print("3. For production, manually verify and correct boxes")
    print("4. Consider using annotation_tool.py for corrections")
    
    return output_files

def visualize_sample_boxes(data_root: str, output_dir: str, 
                         box_generator: PseudoBoxGenerator, num_samples: int = 5):
    """Visualize generated boxes for quality checking"""
    import matplotlib.pyplot as plt
    
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    
    # Find sample images from each concept
    concept_folders = [f for f in os.listdir(data_root) 
                      if os.path.isdir(os.path.join(data_root, f))]
    
    for concept in concept_folders[:3]:  # First 3 concepts
        concept_path = os.path.join(data_root, concept, "train")
        if not os.path.exists(concept_path):
            continue
        
        image_files = list(Path(concept_path).glob("*.jpg")) + \
                     list(Path(concept_path).glob("*.png"))
        
        if not image_files:
            continue
        
        for i, image_file in enumerate(image_files[:num_samples]):
            try:
                image = cv2.imread(str(image_file))
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                boxes, _ = box_generator.generate_pseudo_boxes(str(image_file), concept)
                
                # Draw boxes on image
                for box in boxes:
                    x, y, w, h = box
                    cv2.rectangle(image_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(image_rgb, concept[:15], (x, max(y-5, 10)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                # Save visualization
                plt.figure(figsize=(10, 8))
                plt.imshow(image_rgb)
                plt.title(f"{concept} - Generated Boxes: {len(boxes)}")
                plt.axis('off')
                
                viz_file = os.path.join(output_dir, "visualizations", 
                                       f"{concept}_{i+1}.png")
                plt.savefig(viz_file, bbox_inches='tight', dpi=100)
                plt.close()
                
            except Exception as e:
                print(f"  Could not visualize {image_file}: {e}")
                continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pseudo bounding boxes in COCO format")
    parser.add_argument("--data_root", default="data/frames", 
                       help="Root directory of your frames organized by concept")
    parser.add_argument("--output_dir", default="data/annotations_detectron", 
                       help="Output directory for JSON files")
    parser.add_argument("--max_images", type=int, default=None,
                       help="Maximum images per split (for testing)")
    parser.add_argument("--min_box_size", type=int, default=20,
                       help="Minimum box dimension to keep")
    parser.add_argument("--splits", nargs='+', default=['train', 'val', 'test'],
                       help="Splits to process")
    
    args = parser.parse_args()
    
    print("Starting Pseudo Box Generation...")
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {args.output_dir}")
    print(f"Splits: {args.splits}")
    print(f"Max images per split: {args.max_images or 'All'}")
    
    create_coco_annotation(args.data_root, args.output_dir, 
                          args.splits, args.max_images, args.min_box_size)