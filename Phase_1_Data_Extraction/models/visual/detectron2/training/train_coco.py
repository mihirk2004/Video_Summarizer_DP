import os
import sys
import torch
import argparse
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data import MetadataCatalog
import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger

class CocoTrainer(DefaultTrainer):
    """Custom trainer for COCO fine-tuning"""
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
    
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg)
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

def setup_coco_datasets(cfg):
    """Setup COCO datasets"""
    from detectron2.data.datasets import register_coco_instances
    
    # COCO dataset paths - update these based on your local paths
    coco_root = "data/coco"  # Update this path
    
    if not os.path.exists(coco_root):
        print(f"Warning: COCO directory not found at {coco_root}")
        print("Please download COCO dataset or update the path.")
        return False
    
    # Register COCO datasets
    register_coco_instances(
        "coco_2017_train",
        {},
        f"{coco_root}/annotations/instances_train2017.json",
        f"{coco_root}/train2017"
    )
    
    register_coco_instances(
        "coco_2017_val",
        {},
        f"{coco_root}/annotations/instances_val2017.json",
        f"{coco_root}/val2017"
    )
    
    return True

def train_coco_model(config_file: str, output_dir: str, 
                    resume: bool = False, gpu_id: int = 0):
    """
    Fine-tune on COCO dataset
    
    Args:
        config_file: Path to config YAML
        output_dir: Output directory for checkpoints
        resume: Whether to resume from checkpoint
        gpu_id: GPU device ID
    """
    
    # Setup
    setup_logger()
    torch.cuda.set_device(gpu_id)
    
    # Load configuration
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.OUTPUT_DIR = output_dir
    cfg.MODEL.DEVICE = f"cuda:{gpu_id}"
    
    # Create output directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Setup COCO datasets
    if not setup_coco_datasets(cfg):
        print("COCO dataset setup failed. Please check paths.")
        return
    
    # Print configuration
    print("\n" + "="*60)
    print("COCO Fine-tuning Configuration")
    print("="*60)
    print(f"Config file: {config_file}")
    print(f"Output dir: {cfg.OUTPUT_DIR}")
    print(f"GPU: {gpu_id}")
    print(f"Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"Max iterations: {cfg.SOLVER.MAX_ITER}")
    print("="*60 + "\n")
    
    # Initialize trainer
    trainer = CocoTrainer(cfg)
    
    # Save configuration
    with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
        f.write(cfg.dump())
    
    # Train
    print("Starting COCO fine-tuning...")
    trainer.resume_or_load(resume=resume)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise
    
    print(f"\nTraining complete! Model saved in {cfg.OUTPUT_DIR}")
    
    # Evaluate final model
    print("\nEvaluating final model on COCO validation set...")
    evaluator = COCOEvaluator("coco_2017_val", cfg, True, cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "coco_2017_val")
    
    results = trainer.test(cfg, trainer.model, evaluators=[evaluator])
    
    # Save results
    results_file = os.path.join(cfg.OUTPUT_DIR, "coco_evaluation_results.json")
    import json
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_file}")
    
    return cfg.OUTPUT_DIR

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Detectron2 on COCO")
    parser.add_argument("--config", default="configs/coco_pretrain.yaml",
                       help="Path to config file")
    parser.add_argument("--output", default=None,
                       help="Output directory (default: auto-generated)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint")
    parser.add_argument("--gpu", type=int, default=0,
                       help="GPU ID to use")
    parser.add_argument("--iterations", type=int, default=None,
                       help="Override max iterations")
    
    args = parser.parse_args()
    
    # Create output directory with timestamp
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"output/coco_finetune_{timestamp}"
    
    # Override iterations if specified
    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    if args.iterations:
        cfg.SOLVER.MAX_ITER = args.iterations
    
    # Train
    train_coco_model(args.config, args.output, args.resume, args.gpu)

if __name__ == "__main__":
    main()