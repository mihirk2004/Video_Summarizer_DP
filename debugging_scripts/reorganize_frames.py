import os
import shutil
import random
from pathlib import Path
import argparse

def reorganize_frames(source_dir, target_dir, splits=[0.7, 0.15, 0.15], seed=42):
    """
    Reorganize frames from source_dir to target_dir with train/val/test splits
    
    Source structure: data/organized_frames_fixed/Concept/image.jpg
    Target structure: data/frames/Concept/train/, val/, test/
    """
    
    random.seed(seed)
    
    # Define split names and ratios
    split_names = ['train', 'val', 'test']
    
    # Create target directory
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all concept folders
    source_path = Path(source_dir)
    concepts = [f for f in source_path.iterdir() if f.is_dir()]
    
    print(f"Found {len(concepts)} concepts:")
    for concept in concepts:
        print(f"  - {concept.name}")
    
    # Process each concept
    for concept in concepts:
        concept_name = concept.name
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(concept.glob(ext))
        
        if not image_files:
            print(f"Warning: No images found in {concept_name}")
            continue
        
        print(f"\nProcessing {concept_name}: {len(image_files)} images")
        
        # Shuffle images
        random.shuffle(image_files)
        
        # Calculate split indices
        n_total = len(image_files)
        n_train = int(n_total * splits[0])
        n_val = int(n_total * splits[1])
        # n_test = n_total - n_train - n_val
        
        # Split files
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]
        
        # Create concept directory structure
        concept_target = target_dir / concept_name
        for split in split_names:
            (concept_target / split).mkdir(parents=True, exist_ok=True)
        
        # Function to copy files
        def copy_files(files, split_name):
            for file_path in files:
                target_path = concept_target / split_name / file_path.name
                # Handle duplicate filenames
                counter = 1
                while target_path.exists():
                    stem = file_path.stem
                    suffix = file_path.suffix
                    new_name = f"{stem}_{counter}{suffix}"
                    target_path = concept_target / split_name / new_name
                    counter += 1
                
                # Copy file
                shutil.copy2(file_path, target_path)
        
        # Copy files to respective splits
        copy_files(train_files, 'train')
        copy_files(val_files, 'val')
        copy_files(test_files, 'test')
        
        # Print summary
        print(f"  Train: {len(train_files)} images")
        print(f"  Val: {len(val_files)} images")
        print(f"  Test: {len(test_files)} images")
        
        # Verify
        verify_path = concept_target / 'train'
        copied_count = len(list(verify_path.glob('*.*')))
        if copied_count != len(train_files):
            print(f"  Warning: Expected {len(train_files)} train images, found {copied_count}")
    
    # Create a summary file
    create_summary(target_dir)

def create_summary(target_dir):
    """Create a summary of the reorganized data"""
    summary = {}
    target_path = Path(target_dir)
    
    for concept in target_path.iterdir():
        if concept.is_dir():
            concept_summary = {}
            for split in ['train', 'val', 'test']:
                split_path = concept / split
                if split_path.exists():
                    image_count = len(list(split_path.glob('*.*')))
                    concept_summary[split] = image_count
            
            summary[concept.name] = concept_summary
    
    # Save summary
    summary_file = target_path / "data_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Data Reorganization Summary\n")
        f.write("=" * 50 + "\n\n")
        
        total_images = 0
        for concept, splits in summary.items():
            f.write(f"{concept}:\n")
            concept_total = 0
            for split, count in splits.items():
                f.write(f"  {split}: {count} images\n")
                concept_total += count
            f.write(f"  Total: {concept_total} images\n\n")
            total_images += concept_total
        
        f.write("=" * 50 + "\n")
        f.write(f"Grand Total: {total_images} images\n")
        f.write(f"Concepts: {len(summary)}\n")
    
    print(f"\n{'='*60}")
    print("Data Reorganization Complete!")
    print(f"Summary saved to: {summary_file}")
    
    # Display summary
    print("\nSummary:")
    print("-" * 40)
    for concept, splits in summary.items():
        print(f"{concept}:")
        for split, count in splits.items():
            print(f"  {split}: {count} images")
    print("-" * 40)

def create_symlinks(source_dir, target_dir, splits=[0.7, 0.15, 0.15], seed=42):
    """
    Alternative: Create symbolic links instead of copying files
    Useful for large datasets to save disk space
    """
    random.seed(seed)
    
    # Define split names and ratios
    split_names = ['train', 'val', 'test']
    
    # Create target directory
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all concept folders
    source_path = Path(source_dir)
    concepts = [f for f in source_path.iterdir() if f.is_dir()]
    
    for concept in concepts:
        concept_name = concept.name
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
            image_files.extend(concept.glob(ext))
        
        if not image_files:
            continue
        
        # Shuffle and split
        random.shuffle(image_files)
        n_total = len(image_files)
        n_train = int(n_total * splits[0])
        n_val = int(n_total * splits[1])
        
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]
        
        # Create symlinks
        for split_name, files in zip(split_names, [train_files, val_files, test_files]):
            split_dir = target_dir / concept_name / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            for file_path in files:
                target_path = split_dir / file_path.name
                if not target_path.exists():
                    # Create relative symlink
                    relative_source = os.path.relpath(file_path, split_dir)
                    try:
                        os.symlink(relative_source, target_path)
                    except (OSError, NotImplementedError):
                        # Fallback to copying if symlinks not supported (e.g., Windows without admin)
                        shutil.copy2(file_path, target_path)
        
        print(f"Created {concept_name}: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")
    
    create_summary(target_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorganize frame data into train/val/test splits")
    parser.add_argument("--source", default="data/organized_frames_fixed", 
                       help="Source directory with concept folders")
    parser.add_argument("--target", default="data/frames", 
                       help="Target directory for organized data")
    parser.add_argument("--train_ratio", type=float, default=0.7,
                       help="Ratio for training set (default: 0.7)")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                       help="Ratio for validation set (default: 0.15)")
    parser.add_argument("--test_ratio", type=float, default=0.15,
                       help="Ratio for test set (default: 0.15)")
    parser.add_argument("--symlink", action="store_true",
                       help="Create symbolic links instead of copying (saves disk space)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Normalize ratios
    total = args.train_ratio + args.val_ratio + args.test_ratio
    train_ratio = args.train_ratio / total
    val_ratio = args.val_ratio / total
    test_ratio = args.test_ratio / total
    
    splits = [train_ratio, val_ratio, test_ratio]
    
    print(f"Source: {args.source}")
    print(f"Target: {args.target}")
    print(f"Splits: Train={train_ratio:.1%}, Val={val_ratio:.1%}, Test={test_ratio:.1%}")
    print(f"Using symlinks: {args.symlink}")
    
    if args.symlink:
        create_symlinks(args.source, args.target, splits, args.seed)
    else:
        reorganize_frames(args.source, args.target, splits, args.seed)
    
    print(f"\nNext steps:")
    print(f"1. Check the organized data at: {args.target}")
    print(f"2. Generate pseudo boxes using: python models/visual/detectron2/dataset/pseudo_box_generator.py")
    print(f"3. Start training with: jupyter notebook Train_Detectron2_Complete_Pipeline.ipynb")