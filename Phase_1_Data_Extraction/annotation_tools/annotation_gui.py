# annotation_tools/annotation_gui.py
import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
from PIL import Image, ImageTk
import cv2

class VideoAnnotationGUI:
    def __init__(self, json_path: str):
        self.json_path = json_path
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.current_frame = 0
        self.annotations = []
        
        # Load frames
        self.frames = self.data['processing']['frames']
        
        # Create main window
        self.root = tk.Tk()
        self.root.title(f"Annotate: {self.data['video_id']}")
        self.root.geometry("1200x800")
        
        self.create_widgets()
        self.load_frame(0)
        
    def create_widgets(self):
        # Left panel: Frame display
        self.left_panel = tk.Frame(self.root)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Image display
        self.img_label = tk.Label(self.left_panel)
        self.img_label.pack()
        
        # Frame navigation
        nav_frame = tk.Frame(self.left_panel)
        nav_frame.pack(fill=tk.X, pady=10)
        
        tk.Button(nav_frame, text="← Previous", 
                 command=self.prev_frame).pack(side=tk.LEFT, padx=5)
        tk.Button(nav_frame, text="Next →", 
                 command=self.next_frame).pack(side=tk.LEFT, padx=5)
        
        # Frame info
        self.info_label = tk.Label(self.left_panel, text="", font=("Arial", 10))
        self.info_label.pack()
        
        # Right panel: Annotation controls
        self.right_panel = tk.Frame(self.root, width=400)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        # Transcript section
        transcript_frame = tk.LabelFrame(self.right_panel, text="Current Transcript", padx=10, pady=10)
        transcript_frame.pack(fill=tk.X, pady=5)
        
        self.transcript_text = tk.Text(transcript_frame, height=8, width=40)
        self.transcript_text.pack()
        
        # Annotation controls
        anno_frame = tk.LabelFrame(self.right_panel, text="Annotations", padx=10, pady=10)
        anno_frame.pack(fill=tk.X, pady=5)
        
        # Concept checkboxes
        self.concept_vars = {}
        concepts = [
            "Mathematical Equation",
            "Scientific Diagram", 
            "Computer Code",
            "Instructor Pointing",
            "Instructor Writing",
            "Slide Presentation",
            "Graph/Chart"
        ]
        
        for concept in concepts:
            var = tk.BooleanVar()
            cb = tk.Checkbutton(anno_frame, text=concept, variable=var)
            cb.pack(anchor=tk.W)
            self.concept_vars[concept] = var
        
        # Quality rating
        tk.Label(anno_frame, text="Frame Quality:").pack(anchor=tk.W)
        self.quality_var = tk.StringVar(value="Good")
        qualities = ["Good", "Fair", "Poor", "Blurry", "Dark"]
        for quality in qualities:
            rb = tk.Radiobutton(anno_frame, text=quality, 
                               variable=self.quality_var, value=quality)
            rb.pack(anchor=tk.W)
        
        # Notes
        tk.Label(anno_frame, text="Notes:").pack(anchor=tk.W)
        self.notes_text = tk.Text(anno_frame, height=3, width=30)
        self.notes_text.pack(fill=tk.X)
        
        # Save button
        tk.Button(anno_frame, text="Save Annotation", 
                 command=self.save_annotation, bg="green", fg="white").pack(pady=10)
        
        # Progress
        self.progress_label = tk.Label(self.right_panel, text="", font=("Arial", 9))
        self.progress_label.pack()
    
    def load_frame(self, frame_idx: int):
        """Load and display a frame"""
        if 0 <= frame_idx < len(self.frames):
            self.current_frame = frame_idx
            frame_info = self.frames[frame_idx]
            
            # Load and display image
            img = cv2.imread(frame_info['path'])
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            # Resize for display
            display_width = 600
            ratio = display_width / img_pil.width
            display_height = int(img_pil.height * ratio)
            img_pil = img_pil.resize((display_width, display_height), Image.Resampling.LANCZOS)
            
            img_tk = ImageTk.PhotoImage(img_pil)
            self.img_label.config(image=img_tk)
            self.img_label.image = img_tk
            
            # Update info
            info_text = f"Frame {frame_idx+1}/{len(self.frames)}\n"
            info_text += f"Time: {frame_info['timestamp']:.1f}s\n"
            info_text += f"Instructor: {'Yes' if frame_info.get('instructor') else 'No'}"
            self.info_label.config(text=info_text)
            
            # Update transcript
            self.update_transcript(frame_info['timestamp'])
            
            # Update progress
            progress = f"Progress: {frame_idx+1}/{len(self.frames)} frames"
            self.progress_label.config(text=progress)
            
            # Load existing annotation if any
            self.load_existing_annotation()
    
    def update_transcript(self, timestamp: float):
        """Show transcript segments around current timestamp"""
        transcript_text = ""
        segments = self.data['processing']['transcript']['segments']
        
        for segment in segments:
            if segment['start'] <= timestamp <= segment['end'] + 10:
                transcript_text += f"[{segment['start']:.1f}s]: {segment['text']}\n\n"
        
        self.transcript_text.delete(1.0, tk.END)
        self.transcript_text.insert(1.0, transcript_text)
    
    def save_annotation(self):
        """Save annotation for current frame"""
        annotation = {
            "frame_index": self.current_frame,
            "timestamp": self.frames[self.current_frame]['timestamp'],
            "concepts": [c for c, var in self.concept_vars.items() if var.get()],
            "quality": self.quality_var.get(),
            "notes": self.notes_text.get(1.0, tk.END).strip()
        }
        
        # Save to annotations list
        self.annotations.append(annotation)
        
        # Save to file
        self.save_to_file()
        
        # Move to next frame
        self.next_frame()
        
        messagebox.showinfo("Saved", f"Annotation saved for frame {self.current_frame}")
    
    def save_to_file(self):
        """Save all annotations to JSON file"""
        output_path = self.json_path.replace('.json', '_annotated.json')
        
        # Create copy of data with annotations
        annotated_data = self.data.copy()
        annotated_data['annotations']['frames'] = self.annotations
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(annotated_data, f, indent=2, ensure_ascii=False)
    
    def load_existing_annotation(self):
        """Load existing annotation for current frame"""
        # Clear all inputs
        for var in self.concept_vars.values():
            var.set(False)
        self.notes_text.delete(1.0, tk.END)
        
        # Check if we have annotation for this frame
        for anno in self.annotations:
            if anno['frame_index'] == self.current_frame:
                # Set concepts
                for concept in anno['concepts']:
                    if concept in self.concept_vars:
                        self.concept_vars[concept].set(True)
                
                # Set quality
                self.quality_var.set(anno.get('quality', 'Good'))
                
                # Set notes
                self.notes_text.insert(1.0, anno.get('notes', ''))
                break
    
    def next_frame(self):
        if self.current_frame < len(self.frames) - 1:
            self.load_frame(self.current_frame + 1)
    
    def prev_frame(self):
        if self.current_frame > 0:
            self.load_frame(self.current_frame - 1)
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    # Example usage
    json_file = "data/processed/lecture_002/lecture_002.json"
    if os.path.exists(json_file):
        app = VideoAnnotationGUI(json_file)
        app.run()
    else:
        print(f"File not found: {json_file}")