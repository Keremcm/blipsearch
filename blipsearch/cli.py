import os
import sys
import logging
import warnings

# --- Suppress Noise and Warnings ---
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['OPENCV_LOG_LEVEL'] = 'FATAL'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import cv2
import torch
import sqlite3
import argparse
import subprocess
from pathlib import Path
from PIL import Image

from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

from tqdm import tqdm

DB_DIR = os.path.expanduser("~/.blipsearch")
os.makedirs(DB_DIR, exist_ok=True)
DB_NAME = os.path.join(DB_DIR, "video_index.db")

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS captions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_path TEXT,
            timestamp_ms INTEGER,
            caption TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS indexed_videos (
            video_path TEXT PRIMARY KEY,
            last_modified REAL
        )
    ''')
    conn.commit()
    conn.close()

def find_videos(root_dir):
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'}
    video_paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith('.') and d not in ['node_modules', 'venv', 'env', '__pycache__']]
        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            if ext in video_extensions:
                video_paths.append(os.path.join(dirpath, f))
    return video_paths

class BlipSearch:
    def __init__(self, model_id="Salesforce/blip-image-captioning-base"):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        init_db()

    def _load_model(self):
        if self.model is None:
            # Hide unnecessary HuggingFace logs at stdout level
            self.processor = BlipProcessor.from_pretrained(self.model_id)
            self.model = BlipForConditionalGeneration.from_pretrained(self.model_id).to(self.device)

    def generate_caption(self, image):
        self._load_model()
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        # Using max_new_tokens to prevent warnings
        out = self.model.generate(**inputs, max_new_tokens=50) 
        return self.processor.decode(out[0], skip_special_tokens=True)

    def index_new_videos(self, sample_rate_ms=2000):
        home_dir = os.path.expanduser("~")
        all_videos = find_videos(home_dir)
        
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        new_videos = []
        for vid in all_videos:
            try:
                mtime = os.path.getmtime(vid)
                cursor.execute("SELECT last_modified FROM indexed_videos WHERE video_path = ?", (vid,))
                row = cursor.fetchone()
                
                if row is None or row[0] < mtime:
                    new_videos.append((vid, mtime))
            except OSError:
                continue

        if not new_videos:
            conn.close()
            return  # Exit silently if database is up-to-date.

        print(f"✨ Found {len(new_videos)} new video(s). Indexing...")

        # Show indexing process with a simple progress bar
        for vid_path, mtime in tqdm(new_videos, desc="Processing", unit="video"):
            cursor.execute("DELETE FROM captions WHERE video_path = ?", (vid_path,))
            
            try:
                cap = cv2.VideoCapture(vid_path)
                if not cap.isOpened():
                    continue
                
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps == 0 or fps is None or fps != fps:
                    fps = 25.0
                
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames <= 0:
                    cap.release()
                    continue

                duration_ms = (total_frames / fps) * 1000
                
                for ms in range(0, int(duration_ms), sample_rate_ms):
                    cap.set(cv2.CAP_PROP_POS_MSEC, ms)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)
                    
                    caption = self.generate_caption(image)
                    cursor.execute("INSERT INTO captions (video_path, timestamp_ms, caption) VALUES (?, ?, ?)",
                                   (vid_path, ms, caption))
                
                cursor.execute("INSERT OR REPLACE INTO indexed_videos (video_path, last_modified) VALUES (?, ?)",
                               (vid_path, mtime))
                conn.commit()
                cap.release()
            except Exception:
                pass # Hide error messages to prevent terminal clutter

        conn.close()

    def search(self, query):
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        cursor.execute("SELECT video_path, timestamp_ms, caption FROM captions")
        results = cursor.fetchall()
        
        matches = []
        query_words = query.lower().split()
        
        for video_path, timestamp, caption in results:
            score = sum(1 for word in query_words if word in caption.lower())
            if score > 0:
                matches.append((score, video_path, timestamp, caption))
        
        conn.close()
        
        if not matches:
            print("No matching video found.")
            return

        matches.sort(key=lambda x: x[0], reverse=True)
        best_match = matches[0]
        
        print(f"Video   : {best_match[1]}")
        print(f"Time    : {best_match[2] // 1000} seconds")
        print(f"Caption : {best_match[3]}")
        
        self.open_video(best_match[1])

    def open_video(self, video_path):
        print("Opening video...")
        try:
            # Prevent xdg-open from printing its own output to the terminal
            with open(os.devnull, 'w') as devnull:
                subprocess.Popen(["xdg-open", video_path], stdout=devnull, stderr=devnull)
        except Exception:
            pass

def main():
    parser = argparse.ArgumentParser(description="BLIP-based on-device natural language video search engine")
    parser.add_argument("query", nargs="*", help="Your search query")
    # Hide --rate parameter from help output for simplicity
    parser.add_argument("--rate", type=int, default=2000, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # If user hasn't provided a query, exit silently without printing full help
    if not args.query:
        print("Usage: blipsearch <search query>")
        return

    searcher = BlipSearch()
    searcher.index_new_videos(sample_rate_ms=args.rate)
    
    query_str = " ".join(args.query)
    searcher.search(query_str)

if __name__ == "__main__":
    main()