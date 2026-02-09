import os
import sys
import zipfile
import tarfile
import requests
from tqdm import tqdm
from pathlib import Path
import argparse
import subprocess
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class COCODownloader:
    """Download and prepare COCO dataset for Detectron2"""
    
    def __init__(self, download_dir="data/coco"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # COCO 2017 URLs
        self.urls = {
            "train2017": "https://images.cocodataset.org/zips/train2017.zip",
            "val2017": "https://images.cocodataset.org/zips/val2017.zip",
            "annotations": "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        }
        
        # File sizes for progress tracking (in MB)
        self.file_sizes = {
            "train2017": 18000,  # ~18 GB
            "val2017": 800,      # ~800 MB
            "test2017": 6000,    # ~6 GB (optional)
            "annotations": 250   # ~250 MB
        }

    def download_file(self, url, filename, description):
        filepath = self.download_dir / filename

        if filepath.exists():
            print(f"{description} already exists at {filepath}")
            return filepath

        print(f"Downloading {description}...")

        # Retry strategy
        session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[500, 502, 503, 504]
        )
        session.mount("http://", HTTPAdapter(max_retries=retries))
        session.mount("https://", HTTPAdapter(max_retries=retries))

        try:
            response = session.get(url, stream=True, timeout=60)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"❌ Download failed: {e}")
            return None

        total_size = int(response.headers.get("content-length", 0))

        with open(filepath, "wb") as f, tqdm(
            desc=description,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"Downloaded {description}")
        return filepath

    
    def extract_zip(self, zip_path, extract_dir=None):
        """Extract ZIP file"""
        if extract_dir is None:
            extract_dir = self.download_dir
        
        print(f"Extracting {zip_path.name}...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get total files for progress bar
            total_files = len(zip_ref.namelist())
            
            # Extract with progress
            for i, file in enumerate(zip_ref.namelist()):
                zip_ref.extract(file, extract_dir)
                if i % 1000 == 0:
                    print(f"  Extracted {i}/{total_files} files...")
        
        print(f"Extracted to {extract_dir}")
    
    def download_minimal_coco(self):
        """Download minimal COCO dataset (images + annotations for train/val)"""
        print("Downloading minimal COCO dataset (train2017 + val2017 + annotations)...")
        
        # Download annotations
        ann_zip = self.download_file(
            self.urls["annotations"],
            "annotations_trainval2017.zip",
            "COCO annotations"
        )
        self.extract_zip(ann_zip)
        
        # Download train images
        train_zip = self.download_file(
            self.urls["train2017"],
            "train2017.zip",
            "COCO train2017 images"
        )
        self.extract_zip(train_zip)
        
        # Download val images
        val_zip = self.download_file(
            self.urls["val2017"],
            "val2017.zip",
            "COCO val2017 images"
        )
        self.extract_zip(val_zip)
        
        print("\nMinimal COCO dataset downloaded successfully!")
    
    def download_full_coco(self):
        """Download full COCO dataset including test images"""
        self.download_minimal_coco()
        
        # Download test images (optional)
        test_zip = self.download_file(
            self.urls["test2017"],
            "test2017.zip",
            "COCO test2017 images"
        )
        self.extract_zip(test_zip)
        
        print("\nFull COCO dataset downloaded successfully!")
    
    def verify_dataset(self):
        """Verify COCO dataset structure"""
        required_dirs = [
            self.download_dir / "annotations",
            self.download_dir / "train2017",
            self.download_dir / "val2017"
        ]
        
        required_files = [
            self.download_dir / "annotations" / "instances_train2017.json",
            self.download_dir / "annotations" / "instances_val2017.json"
        ]
        
        print("Verifying COCO dataset structure...")
        
        all_ok = True
        for dir_path in required_dirs:
            if dir_path.exists():
                print(f"✓ Directory exists: {dir_path}")
            else:
                print(f"✗ Missing directory: {dir_path}")
                all_ok = False
        
        for file_path in required_files:
            if file_path.exists():
                print(f"✓ File exists: {file_path}")
            else:
                print(f"✗ Missing file: {file_path}")
                all_ok = False
        
        # Count images
        if (self.download_dir / "train2017").exists():
            train_images = len(list((self.download_dir / "train2017").glob("*.jpg")))
            print(f"✓ Train images: {train_images:,}")
        
        if (self.download_dir / "val2017").exists():
            val_images = len(list((self.download_dir / "val2017").glob("*.jpg")))
            print(f"✓ Val images: {val_images:,}")
        
        return all_ok
    
    
    def get_disk_usage(self):
        """Calculate disk usage of COCO dataset"""
        total_size = 0
        
        for path in self.download_dir.rglob('*'):
            if path.is_file():
                total_size += path.stat().st_size
        
        # Convert to GB
        total_gb = total_size / (1024**3)
        
        print(f"\nCOCO dataset disk usage: {total_gb:.2f} GB")
        return total_gb

def check_disk_space(required_gb=25):
    """Check if enough disk space is available"""
    import shutil
    
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024**3)
    
    print(f"Available disk space: {free_gb:.2f} GB")
    print(f"Required (approx): {required_gb} GB")
    
    if free_gb < required_gb:
        print(f"⚠️ Warning: Insufficient disk space!")
        print(f"Consider downloading only the minimal dataset.")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Download COCO dataset for Detectron2")
    parser.add_argument("--download_dir", default="data/coco", 
                       help="Directory to download COCO dataset")
    parser.add_argument("--minimal", action="store_true",
                       help="Download only minimal dataset (train+val+annotations, ~19GB)")
    parser.add_argument("--full", action="store_true",
                       help="Download full dataset including test images (~25GB)")
    parser.add_argument("--verify", action="store_true",
                       help="Verify downloaded dataset")
    parser.add_argument("--skip_space_check", action="store_true",
                       help="Skip disk space check")
    
    args = parser.parse_args()
    
    # Check disk space
    if not args.skip_space_check:
        required_space = 19 if args.minimal else 25
        if not check_disk_space(required_space):
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Download cancelled.")
                return
    
    # Initialize downloader
    downloader = COCODownloader(args.download_dir)
    
    if args.verify:
        downloader.verify_dataset()
        downloader.get_disk_usage()
        return
    
    # Download based on option
    if args.minimal:
        downloader.download_minimal_coco()
    elif args.full:
        downloader.download_full_coco()
    else:
        # Default: minimal
        print("No option specified. Downloading minimal COCO dataset...")
        downloader.download_minimal_coco()
    
    # Verify download
    if downloader.verify_dataset():
        print("\n✅ COCO dataset downloaded and verified successfully!")
    else:
        print("\n⚠️ Some files might be missing. Check the warnings above.")
    
    # Create instructions
    #downloader.create_symlinks_for_detectron2()
    
    # Show disk usage
    downloader.get_disk_usage()
    
    print("\n" + "="*60)
    print("COCO Dataset Setup Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Reorganize your frames data (if not done):")
    print("   python reorganize_frames.py")
    print("\n2. Generate pseudo bounding boxes:")
    print("   python models/visual/detectron2/dataset/pseudo_box_generator.py")
    print("\n3. Start training:")
    print("   jupyter notebook Train_Detectron2_Complete_Pipeline.ipynb")

if __name__ == "__main__":
    # Install required packages if missing
    try:
        from tqdm import tqdm
    except ImportError:
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm", "requests"])
        from tqdm import tqdm
    
    main()