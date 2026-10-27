import os
import re
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
from PIL import Image, ImageOps

from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

from tqdm import tqdm

DB_DIR = os.path.expanduser("~/.blipsearch")
os.makedirs(DB_DIR, exist_ok=True)
DB_NAME = os.path.join(DB_DIR, "video_index.db")

IGNORE_FILE = os.path.join(DB_DIR, "ignore.txt")

def _load_user_ignores():
    """Extra directory names to skip, one per line in ~/.blipsearch/ignore.txt."""
    try:
        with open(IGNORE_FILE, encoding="utf-8") as f:
            return {l.strip() for l in f if l.strip() and not l.startswith('#')}
    except OSError:
        return set()

# Directories never scanned (assets, repos, caches, tool exports, SDKs, games)
IGNORED_DIRNAMES = {
    'node_modules', 'venv', 'env', '__pycache__', 'go',
    'repos', 'Mods', 'mods', '.git', 'site-packages',
    'Android', 'Games', 'Steam', 'llama.cpp',
} | _load_user_ignores()

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
EXTERNAL_PREFIXES = ("/mnt/", "/media/", "/run/media/")
MIN_IMAGE_BYTES = 10 * 1024   # icons/thumbnails are smaller than this
MIN_IMAGE_SIDE = 256          # ... or have a shorter side than this

STOPWORDS = {
    'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'and', 'or', 'with', 'is', 'are',
    'it', 'its', 'for', 'by', 'from', 'next', 'up', 'down', 'into', 'out', 'over',
    'this', 'that', 'there', 'some', 'very',
}

class SkipMedia(Exception):
    """File is intentionally not indexed (e.g. an icon-sized image)."""

def _is_image(path):
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS

def _is_external(path):
    return path.startswith(EXTERNAL_PREFIXES)

def _is_ignored(path):
    """True if the file sits in a directory find_media() would not descend into."""
    home = os.path.expanduser("~") + os.sep
    in_home = path.startswith(home)
    parts = (path[len(home):] if in_home else path).split(os.sep)[:-1]
    return any(p in IGNORED_DIRNAMES or (in_home and p.startswith('.')) for p in parts)

def _mounted_ancestor(path):
    """Closest mounted volume above path, or None if only the root fs is mounted."""
    d = os.path.dirname(path)
    while d and d != os.sep:
        if os.path.ismount(d):
            return d
        d = os.path.dirname(d)
    return None

def _stem(word):
    """Very light English stemmer so that dogs/dog and chasing/chases match."""
    if len(word) > 4 and word.endswith('ies'):
        word = word[:-3] + 'y'
    elif len(word) > 5 and word.endswith('ing'):
        word = word[:-3]
        if len(word) > 2 and word[-1] == word[-2] and word[-1] not in 'lsz':
            word = word[:-1]
    elif len(word) > 4 and word.endswith('ed'):
        word = word[:-2]
    elif len(word) > 3 and word.endswith('s') and not word.endswith(('ss', 'us')):
        word = word[:-1]
    if len(word) > 3 and word.endswith('e'):
        word = word[:-1]
    return word

def _keywords_unfiltered(text):
    return {_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower())}

def _keywords(text):
    """Stemmed, stopword-free set of words in text."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {_stem(w) for w in words if w not in STOPWORDS}

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
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS failed_media (
            media_path TEXT PRIMARY KEY,
            last_modified REAL,
            error TEXT
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
    media_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'} | IMAGE_EXTENSIONS
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

    def _cleanup_index(self):
        """Drop index entries for deleted files and for directories that are now ignored."""
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        cursor.execute("SELECT media_path FROM indexed_videos UNION SELECT media_path FROM failed_media")
        stale = []
        for (media_path,) in cursor.fetchall():
            if _is_ignored(media_path):
                stale.append((media_path,))
            elif not os.path.exists(media_path):
                if not _is_external(media_path):
                    stale.append((media_path,))
                elif self.include_external and _mounted_ancestor(media_path):
                    # Drive is mounted but the file is gone (an unplugged drive keeps its entries)
                    stale.append((media_path,))

        if stale:
            cursor.executemany("DELETE FROM indexed_videos WHERE media_path = ?", stale)
            cursor.executemany("DELETE FROM failed_media WHERE media_path = ?", stale)
        # Captions are only valid next to an indexed_videos row
        cursor.execute("DELETE FROM captions WHERE media_path NOT IN (SELECT media_path FROM indexed_videos)")

        conn.commit()
        conn.close()

    def _load_model(self):
        if self.model is not None:
            return
        # Use the local cache when possible, so no Hub requests are made after the first download
        for local_only in (True, False):
            try:
                self.processor = BlipProcessor.from_pretrained(self.model_id, local_files_only=local_only)
                self.model = BlipForConditionalGeneration.from_pretrained(
                    self.model_id, local_files_only=local_only).to(self.device)
                return
            except Exception:
                if not local_only:
                    print("\n❌ BLIP modeli yüklenemedi. İlk çalıştırmada internet bağlantısı gerekir;")
                    print("   model otomatik olarak indirilir (~1.4GB). Bir kez indikten sonra tamamen çevrimdışı çalışır.")
                    raise

    def generate_caption(self, image):
        self._load_model()
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        # Using max_new_tokens to prevent warnings
        out = self.model.generate(**inputs, max_new_tokens=50) 
        return self.processor.decode(out[0], skip_special_tokens=True)

    def index_new_media(self, sample_rate_ms=2000, retry_failed=False):
        self._cleanup_index()

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
                if found and self.include_external and search_dir != search_dirs[0]:
                    print(f"📁 Found {len(found)} file(s) in {search_dir}")
            except PermissionError:
                continue

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        # Repair stale index: files marked indexed but with no captions
        if _repair_stale_index(conn, cursor):
            print("🔧 Eski bozuk index fark edildi, yeniden indexleniyor...")

        if retry_failed:
            cursor.execute("DELETE FROM failed_media")
            conn.commit()
        indexed = dict(cursor.execute("SELECT media_path, last_modified FROM indexed_videos").fetchall())
        failed_before = dict(cursor.execute("SELECT media_path, last_modified FROM failed_media").fetchall())

        new_media = []
        for media_path in all_media:
            try:
                st = os.stat(media_path)
            except OSError:
                continue
            if media_path in indexed and indexed[media_path] >= st.st_mtime:
                continue
            # Files that failed before are only retried once they change (or with --retry-failed)
            if media_path in failed_before and failed_before[media_path] >= st.st_mtime:
                continue
            if _is_image(media_path) and st.st_size < MIN_IMAGE_BYTES:
                continue
            new_media.append((media_path, st.st_mtime))

        if not new_media:
            conn.close()
            return  # Exit silently if database is up-to-date.

        print(f"✨ Found {len(new_media)} new media file(s). Indexing...")

        # Load up front: a model problem must stop here, not mark every file as failed
        self._load_model()

        indexed_count = 0
        skipped_count = 0
        failed = []
        # Show indexing process with a simple progress bar
        for media_path, mtime in tqdm(new_media, desc="Processing", unit="file"):
            cursor.execute("DELETE FROM captions WHERE media_path = ?", (media_path,))

            try:
                if _is_image(media_path):
                    with Image.open(media_path) as img:
                        img = ImageOps.exif_transpose(img)  # phone photos store rotation in EXIF
                        if min(img.size) < MIN_IMAGE_SIDE:
                            raise SkipMedia("küçük görsel")
                        caption = self.generate_caption(img.convert("RGB"))
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
                cursor.execute("DELETE FROM failed_media WHERE media_path = ?", (media_path,))
                conn.commit()
                indexed_count += 1
            except Exception as e:
                # Remember the failure so the file is not retried (and the model not reloaded) every run
                cursor.execute("DELETE FROM captions WHERE media_path = ?", (media_path,))
                cursor.execute("INSERT OR REPLACE INTO failed_media (media_path, last_modified, error) VALUES (?, ?, ?)",
                               (media_path, mtime, str(e)))
                conn.commit()
                if isinstance(e, SkipMedia):
                    skipped_count += 1
                else:
                    failed.append((media_path, e))

        conn.close()

        if indexed_count > 0:
            print(f"✅ {indexed_count} medya dosyası indexlendi.")
        if skipped_count > 0:
            print(f"ℹ️ {skipped_count} küçük görsel (ikon vb.) atlandı.")
        if failed:
            print(f"⚠️ {len(failed)} dosya indexlenemedi (dosya değişmedikçe veya --retry-failed verilmedikçe tekrar denenmez):")
            for path, err in failed[:5]:
                print(f"   • {os.path.basename(path)}: {err}")
            if len(failed) > 5:
                print(f"   ... ve {len(failed) - 5} dosya daha")

    def search(self, query, max_results=5):
        query_words = _keywords(query) or _keywords_unfiltered(query)
        if not query_words:
            print("Geçerli bir arama kelimesi bulunamadı.")
            return

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        # Best-scoring caption per file
        best = {}
        for media_path, timestamp, caption in cursor.execute("SELECT media_path, timestamp_ms, caption FROM captions"):
            # Skip results from external drives if not explicitly indexed there
            if not self.include_external and _is_external(media_path):
                continue
            score = len(query_words & _keywords(caption))
            if score > 0 and (media_path not in best or score > best[media_path][0]):
                best[media_path] = (score, media_path, timestamp, caption)

        conn.close()

        # Only now touch the filesystem, and only for the top candidates
        ranked = sorted(best.values(), key=lambda x: x[0], reverse=True)
        results = []
        for item in ranked:
            if os.path.exists(item[1]):
                results.append(item)
                if len(results) == max_results:
                    break

        if not results:
            print("Eşleşen medya bulunamadı. (Henüz indexlenmemişse önce bir arama çalıştırın.)")
            return

        for rank, (score, path, timestamp, caption) in enumerate(results, 1):
            is_image = _is_image(path)
            when = "" if is_image else " @ %02d:%02d" % divmod(timestamp // 1000, 60)
            print(f"{rank}. [{'Fotoğraf' if is_image else 'Video'}] {path}{when}")
            print(f"   Caption : {caption}")

        self.open_media(results[0][1])

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
    parser.add_argument("--retry-failed", action="store_true", help="Retry files that failed to index earlier")
    parser.add_argument("--rate", type=int, default=2000, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # If user hasn't provided a query, exit silently without printing full help
    if not args.query:
        print("Usage: blipsearch <search query>")
        print("Options: --external (search external drives), --retry-failed, --rate (frame sampling rate in ms)")
        return

    searcher = BlipSearch(include_external=args.external)
    searcher.index_new_media(sample_rate_ms=args.rate, retry_failed=args.retry_failed)
    
    query_str = " ".join(args.query)
    searcher.search(query_str)

if __name__ == "__main__":
    main()