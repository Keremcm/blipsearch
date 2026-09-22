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

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

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

# Directories never scanned (assets, repos, caches, tool exports)
IGNORED_DIRNAMES = {
    'node_modules', 'venv', 'env', '__pycache__', 'go',
    'repos', 'Mods', 'mods', '.git', 'site-packages',
}

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS captions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            media_path TEXT,
            timestamp_ms INTEGER,
            caption TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS indexed_videos (
            media_path TEXT PRIMARY KEY,
            last_modified REAL
        )
    ''')
    conn.commit()
    conn.close()

def _repair_stale_index(conn, cursor):
    """If files were marked indexed but never got captions, re-index them."""
    cursor.execute("SELECT COUNT(*) FROM indexed_videos")
    indexed = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM captions")
    captioned = cursor.fetchone()[0]
    if indexed > 0 and captioned == 0:
        cursor.execute("DELETE FROM indexed_videos")
        conn.commit()
        return True
    return False

def find_media(root_dir):
    media_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.jpg', '.jpeg', '.png', '.webp'}
    media_paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames
                       if not d.startswith('.') and d not in IGNORED_DIRNAMES]
        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            if ext in media_extensions:
                media_paths.append(os.path.join(dirpath, f))
    return media_paths

class BlipSearch:
    def __init__(self, model_id="Salesforce/blip-image-captioning-base", include_external=False):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        self.include_external = include_external
        init_db()

    def _external_disks_available(self):
        """Check if any external drive is mounted."""
        external_dirs = ["/mnt", "/media"]
        for ext_dir in external_dirs:
            if os.path.exists(ext_dir) and os.listdir(ext_dir):
                return True
        return False

    def _cleanup_deleted_files(self):
        """Remove entries for deleted files from database."""
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        cursor.execute("SELECT DISTINCT media_path FROM indexed_videos")
        all_indexed = cursor.fetchall()
        
        deleted_count = 0
        for (media_path,) in all_indexed:
            if not os.path.exists(media_path):
                cursor.execute("DELETE FROM captions WHERE media_path = ?", (media_path,))
                cursor.execute("DELETE FROM indexed_videos WHERE media_path = ?", (media_path,))
                deleted_count += 1
        
        if deleted_count > 0:
            conn.commit()
        
        conn.close()

    def _cleanup_external_files(self):
        """Remove entries for external files when --external is not used."""
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM captions WHERE media_path LIKE '/mnt/%' OR media_path LIKE '/media/%'")
        cursor.execute("DELETE FROM indexed_videos WHERE media_path LIKE '/mnt/%' OR media_path LIKE '/media/%'")
        
        conn.commit()
        conn.close()

    def _load_model(self):
        if self.model is None:
            # Hide unnecessary HuggingFace logs at stdout level
            try:
                self.processor = BlipProcessor.from_pretrained(self.model_id)
                self.model = BlipForConditionalGeneration.from_pretrained(self.model_id).to(self.device)
            except Exception:
                print("\n❌ BLIP modeli yüklenemedi. İlk çalıştırmada internet bağlantısı gerekir;")
                print("   model otomatik olarak indirilir (~1.4GB). Bir kez indikten sonra tamamen çevrimdışı çalışır.")
                raise

    def generate_caption(self, image):
        self._load_model()
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        # Using max_new_tokens to prevent warnings
        out = self.model.generate(**inputs, max_new_tokens=50) 
        return self.processor.decode(out[0], skip_special_tokens=True)

    def index_new_media(self, sample_rate_ms=2000):
        # Clean up deleted files first
        self._cleanup_deleted_files()
        
        # If not including external, remove external files from DB
        if not self.include_external:
            self._cleanup_external_files()
        
        search_dirs = [os.path.expanduser("~")]
        
        if self.include_external:
            # Add common mount points for external drives
            external_dirs = ["/mnt", "/media", "/run/media"]
            for ext_dir in external_dirs:
                if os.path.exists(ext_dir):
                    search_dirs.append(ext_dir)
        
        all_media = []
        for search_dir in search_dirs:
            try:
                found = find_media(search_dir)
                all_media.extend(found)
                if found and self.include_external and (search_dir.startswith("/mnt") or search_dir.startswith("/media") or search_dir.startswith("/run")):
                    print(f"📁 Found {len(found)} file(s) in {search_dir}")
            except PermissionError:
                continue
        
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        # Repair stale index: files marked indexed but with no captions
        if _repair_stale_index(conn, cursor):
            print("🔧 Eski bozuk index fark edildi, yeniden indexleniyor...")

        new_media = []
        for media_path in all_media:
            try:
                mtime = os.path.getmtime(media_path)
                cursor.execute("SELECT last_modified FROM indexed_videos WHERE media_path = ?", (media_path,))
                row = cursor.fetchone()
                
                if row is None or row[0] < mtime:
                    new_media.append((media_path, mtime))
            except OSError:
                continue

        if not new_media:
            conn.close()
            return  # Exit silently if database is up-to-date.

        print(f"✨ Found {len(new_media)} new media file(s). Indexing...")

        indexed_count = 0
        failed = []
        # Show indexing process with a simple progress bar
        for media_path, mtime in tqdm(new_media, desc="Processing", unit="file"):
            cursor.execute("DELETE FROM captions WHERE media_path = ?", (media_path,))
            
            try:
                ext = os.path.splitext(media_path)[1].lower()
                image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
                
                if ext in image_extensions:
                    with Image.open(media_path) as img:
                        img_rgb = img.convert("RGB")
                        caption = self.generate_caption(img_rgb)
                        cursor.execute("INSERT INTO captions (media_path, timestamp_ms, caption) VALUES (?, ?, ?)",
                                       (media_path, 0, caption))
                else:
                    cap = cv2.VideoCapture(media_path)
                    if not cap.isOpened():
                        raise RuntimeError("video açılamadı")
                    
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if fps == 0 or fps is None or fps != fps:
                        fps = 25.0
                    
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if total_frames <= 0:
                        cap.release()
                        raise RuntimeError("video kare sayısı alınamadı")

                    duration_ms = (total_frames / fps) * 1000
                    frame_count = 0
                    
                    for ms in range(0, int(duration_ms), sample_rate_ms):
                        cap.set(cv2.CAP_PROP_POS_MSEC, ms)
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(frame_rgb)
                        
                        caption = self.generate_caption(image)
                        cursor.execute("INSERT INTO captions (media_path, timestamp_ms, caption) VALUES (?, ?, ?)",
                                       (media_path, ms, caption))
                        frame_count += 1
                    
                    cap.release()
                    if frame_count == 0:
                        raise RuntimeError("video karesi çıkarılamadı")

                cursor.execute("INSERT OR REPLACE INTO indexed_videos (media_path, last_modified) VALUES (?, ?)",
                               (media_path, mtime))
                conn.commit()
                indexed_count += 1
            except Exception as e:
                failed.append((media_path, e))

        conn.close()

        if indexed_count > 0:
            print(f"✅ {indexed_count} medya dosyası indexlendi.")
        if failed:
            print(f"⚠️ {len(failed)} dosya indexlenemedi (bir sonraki çalıştırmada tekrar denenir):")
            for path, err in failed[:5]:
                print(f"   • {os.path.basename(path)}: {err}")
            if len(failed) > 5:
                print(f"   ... ve {len(failed) - 5} dosya daha")

    def search(self, query):
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        cursor.execute("SELECT media_path, timestamp_ms, caption FROM captions")
        results = cursor.fetchall()
        
        matches = []
        query_words = query.lower().split()
        external_available = self._external_disks_available()
        
        for media_path, timestamp, caption in results:
            # Skip results from external drives if not explicitly indexed there
            if not self.include_external and (media_path.startswith("/mnt") or media_path.startswith("/media")):
                continue
            
            # Skip if file doesn't exist and external disk isn't available
            if not os.path.exists(media_path):
                if media_path.startswith("/mnt") or media_path.startswith("/media"):
                    if not external_available:
                        continue
                else:
                    # Internal file deleted, skip it
                    continue
            
            # Match full words only, not substrings
            caption_words = caption.lower().split()
            score = sum(1 for word in query_words if word in caption_words)
            if score > 0:
                matches.append((score, media_path, timestamp, caption))
        
        conn.close()
        
        if not matches:
            print("Eşleşen medya bulunamadı. (Henüz indexlenmemişse önce bir arama çalıştırın.)")
            return

        matches.sort(key=lambda x: x[0], reverse=True)
        best_match = matches[0]
        
        # Final check: does the file exist?
        if not os.path.exists(best_match[1]):
            print("Database'de bulundu ama dosya artık mevcut değil.")
            return
        
        path, timestamp, caption = best_match[1], best_match[2], best_match[3]
        is_image = path.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        
        print(f"Tip     : {'Fotoğraf' if is_image else 'Video'}")
        print(f"File    : {path}")
        if not is_image:
            minutes, seconds = divmod(timestamp // 1000, 60)
            print(f"Time    : {minutes:02d}:{seconds:02d}")
        print(f"Caption : {caption}")
        
        self.open_media(path)

    def open_media(self, media_path):
        print("Opening media...")
        try:
            # Prevent xdg-open from printing its own output to the terminal
            with open(os.devnull, 'w') as devnull:
                subprocess.Popen(["xdg-open", media_path], stdout=devnull, stderr=devnull)
        except Exception:
            pass

def main():
    parser = argparse.ArgumentParser(description="BLIP-based on-device natural language video search engine")
    parser.add_argument("query", nargs="*", help="Your search query")
    parser.add_argument("--external", action="store_true", help="Also search external drives (/mnt, /media)")
    # Hide --rate parameter from help output for simplicity
    parser.add_argument("--rate", type=int, default=2000, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # If user hasn't provided a query, exit silently without printing full help
    if not args.query:
        print("Usage: blipsearch <search query>")
        print("Options: --external (search external drives), --rate (frame sampling rate in ms)")
        return

    searcher = BlipSearch(include_external=args.external)
    searcher.index_new_media(sample_rate_ms=args.rate)
    
    query_str = " ".join(args.query)
    searcher.search(query_str)

if __name__ == "__main__":
    main()