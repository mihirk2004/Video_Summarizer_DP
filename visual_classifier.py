#!/usr/bin/env python3
"""
Hybrid Visual Frame Classifier — ResNet50 + CLIP Stem Classifier
Uses both trained models for robust frame classification via hybrid decision strategy.

Categories: Graph_Chart, Computer_Code, Equation, Diagrams
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Default model paths
DEFAULT_MODEL_DIR = "models/visual"

# Categories (must match training)
CATEGORIES = ['Graph_Chart', 'Computer_Code', 'Equation', 'Diagrams']
NUM_CLASSES = len(CATEGORIES)


# ──────────────────────────────────────────────
#  Model Architectures (matching training)
# ──────────────────────────────────────────────

class FineTunedCLIP(nn.Module):
    """Fine-tuned CLIP model with classification head (matches notebook)"""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super(FineTunedCLIP, self).__init__()
        from transformers import CLIPModel, CLIPProcessor

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.embed_dim = self.clip_model.config.projection_dim

        # Freeze CLIP backbone
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Classifier head (matches training notebook)
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def get_features(self, images):
        """Extract 512-dim L2-normalized CLIP features"""
        device = next(self.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        pixel_values = inputs['pixel_values'].to(device)
        vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values)
        image_features = vision_outputs[1]  # pooled output
        image_features = self.clip_model.visual_projection(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def forward(self, images):
        """Forward pass — expects list of PIL images"""
        image_features = self.get_features(images)
        logits = self.classifier(image_features)
        return logits


def _create_resnet_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """Create ResNet50 architecture matching training notebook"""
    resnet = models.resnet50(weights=None)
    num_features = resnet.fc.in_features
    resnet.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return resnet


# ──────────────────────────────────────────────
#  Transforms
# ──────────────────────────────────────────────

RESNET_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ──────────────────────────────────────────────
#  Hybrid Classifier
# ──────────────────────────────────────────────

class HybridVisualClassifier:
    """
    Hybrid visual frame classifier using ResNet50 + CLIP stem classifier.
    Uses max_confidence or agreement_threshold strategies for fusion.
    """

    def __init__(
        self,
        model_dir: str = DEFAULT_MODEL_DIR,
        strategy: str = 'max_confidence',
        threshold: float = 0.7,
        device: Optional[str] = None
    ):
        self.model_dir = Path(model_dir)
        self.strategy = strategy
        self.threshold = threshold
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.categories = CATEGORIES

        self._resnet_model = None
        self._clip_model = None

    def load_models(self):
        """Load both models into memory"""
        self._load_resnet()
        self._load_clip()

    def unload_models(self):
        """Free GPU memory"""
        del self._resnet_model, self._clip_model
        self._resnet_model = None
        self._clip_model = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_resnet(self):
        """Load trained ResNet50 model"""
        if self._resnet_model is not None:
            return

        resnet_path = self.model_dir / "resnet50_best_model.pth"
        if not resnet_path.exists():
            alt_path = self.model_dir / "resnet50_final_model.pth"
            if alt_path.exists():
                resnet_path = alt_path
            else:
                raise FileNotFoundError(
                    f"ResNet model not found at {resnet_path} or {alt_path}"
                )

        model = _create_resnet_model()
        checkpoint = torch.load(str(resnet_path), map_location=self.device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        model.eval()
        self._resnet_model = model
        print(f"✓ ResNet50 loaded from {resnet_path}")

    def _load_clip(self):
        """Load fine-tuned CLIP stem classifier"""
        if self._clip_model is not None:
            return

        clip_path = self.model_dir / "clip_stem_classifier.pth"
        if not clip_path.exists():
            raise FileNotFoundError(f"CLIP model not found at {clip_path}")

        model = FineTunedCLIP(num_classes=NUM_CLASSES)
        checkpoint = torch.load(str(clip_path), map_location=self.device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        model.eval()
        self._clip_model = model
        print(f"✓ CLIP stem classifier loaded from {clip_path}")

    @torch.no_grad()
    def _predict_resnet(self, image: Image.Image) -> Tuple[int, float]:
        """Get ResNet prediction and confidence"""
        img_tensor = RESNET_TRANSFORM(image).unsqueeze(0).to(self.device)
        outputs = self._resnet_model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
        return pred.item(), conf.item()

    @torch.no_grad()
    def _predict_clip(self, image: Image.Image) -> Tuple[int, float]:
        """Get CLIP prediction and confidence"""
        logits = self._clip_model([image])
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, 1)
        return pred.item(), conf.item()

    @torch.no_grad()
    def get_clip_embedding(self, image: Image.Image) -> torch.Tensor:
        """Extract the 512-dimensional CLIP embedding for a frame (for multimodal fusion)"""
        self._load_clip()
        return self._clip_model.get_features([image]).squeeze(0)

    def _hybrid_decision(
        self,
        resnet_pred: int,
        resnet_conf: float,
        clip_pred: int,
        clip_conf: float
    ) -> Tuple[int, float, str]:
        """
        Combine predictions from both models.
        Strategies:
            'max_confidence': choose the prediction with higher confidence.
            'agreement_threshold': if they agree, accept; if disagree, choose
                higher confidence only if > threshold, else reject.
        """
        if self.strategy == 'max_confidence':
            if resnet_conf >= clip_conf:
                return resnet_pred, resnet_conf, 'resnet'
            else:
                return clip_pred, clip_conf, 'clip'

        elif self.strategy == 'agreement_threshold':
            if resnet_pred == clip_pred:
                conf = max(resnet_conf, clip_conf)
                return resnet_pred, conf, 'both'
            else:
                if resnet_conf >= clip_conf and resnet_conf > self.threshold:
                    return resnet_pred, resnet_conf, 'resnet'
                elif clip_conf > resnet_conf and clip_conf > self.threshold:
                    return clip_pred, clip_conf, 'clip'
                else:
                    return -1, 0.0, 'reject'
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def classify_frame(self, frame_path: str) -> Dict:
        """
        Classify a single frame using hybrid approach.

        Returns:
            {
                'category': str,
                'confidence': float,
                'decided_by': str,   # 'resnet', 'clip', 'both', or 'reject'
                'resnet_pred': str,
                'resnet_conf': float,
                'clip_pred': str,
                'clip_conf': float,
            }
        """
        self._load_resnet()
        self._load_clip()

        try:
            image = Image.open(frame_path).convert('RGB')
        except Exception as e:
            return {
                'category': 'Unknown',
                'confidence': 0.0,
                'decided_by': 'error',
                'error': str(e)
            }

        resnet_pred, resnet_conf = self._predict_resnet(image)
        clip_pred, clip_conf = self._predict_clip(image)

        final_pred, final_conf, decided_by = self._hybrid_decision(
            resnet_pred, resnet_conf, clip_pred, clip_conf
        )

        if final_pred == -1:
            category = 'Unknown'
        else:
            category = self.categories[final_pred]

        return {
            'category': category,
            'confidence': round(final_conf, 4),
            'decided_by': decided_by,
            'resnet_pred': self.categories[resnet_pred],
            'resnet_conf': round(resnet_conf, 4),
            'clip_pred': self.categories[clip_pred],
            'clip_conf': round(clip_conf, 4),
        }

    def classify_frames_batch(
        self,
        frame_paths: List[str],
        progress_callback=None
    ) -> List[Dict]:
        """Classify multiple frames with optional progress callback"""
        self.load_models()
        results = []

        for i, path in enumerate(frame_paths):
            result = self.classify_frame(path)
            result['path'] = path
            results.append(result)

            if progress_callback and (i + 1) % 5 == 0:
                progress_callback(i + 1, len(frame_paths))

        return results
