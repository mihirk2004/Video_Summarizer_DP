import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import json
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

class LectureObjectDetector:
    """Inference wrapper for lecture object detection"""
    
    def __init__(self, config_path: str, weights_path: str, threshold: float = 0.5):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(config_path)
        self.cfg.MODEL.WEIGHTS = weights_path
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.predictor = DefaultPredictor(self.cfg)
        self.metadata = MetadataCatalog.get(
            self.cfg.DATASETS.TEST[0] if len(self.cfg.DATASETS.TEST) > 0 else "__unused"
        )
    
    def detect(self, image_path: str) -> Dict[str, Any]:
        """Detect objects in a single image"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        outputs = self.predictor(image)
        
        # Format results
        instances = outputs["instances"]
        results = {
            "image_path": image_path,
            "image_size": image.shape[:2],
            "detections": []
        }
        
        if len(instances) > 0:
            for i in range(len(instances)):
                detection = {
                    "bbox": instances.pred_boxes[i].tensor[0].cpu().numpy().tolist(),
                    "score": float(instances.scores[i].cpu().numpy()),
                    "class_id": int(instances.pred_classes[i].cpu().numpy()),
                    "class_name": self.metadata.thing_classes[int(instances.pred_classes[i])]
                }
                results["detections"].append(detection)
        
        return results
    
    def batch_detect(self, image_paths: List[str]) -> Dict[str, Any]:
        """Detect objects in multiple images"""
        results = {"images": []}
        
        for img_path in image_paths:
            try:
                result = self.detect(img_path)
                results["images"].append(result)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                results["images"].append({
                    "image_path": img_path,
                    "error": str(e)
                })
        
        return results
    
    def export_detections(self, image_paths: List[str], output_file: str):
        """Export detections to JSON file"""
        results = self.batch_detect(image_paths)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Detections exported to {output_file}")
        
        # Print summary
        total_detections = sum(len(img["detections"]) for img in results["images"] 
                              if "detections" in img)
        print(f"Total images: {len(results['images'])}")
        print(f"Total detections: {total_detections}")
        
        return results