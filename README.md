



<div align="center">
  
# 🔍 Blipsearch
*Blazing-fast, purely on-device natural language search engine for your local Videos & Photos.*

![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![License MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-AI-ee4c2c?style=flat-square&logo=pytorch)
![Build](https://img.shields.io/badge/Build-Passing-brightgreen?style=flat-square)
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-ff69b4?style=flat-square)
</div>

---

### 🎥 Watch it in Action!

https://github.com/user-attachments/assets/4fc1df69-c1de-4f90-b8b3-fadeb11952d1


---

## ✨ Why Blipsearch?

- 🔒 **100% Offline & Private:** Everything runs strictly on your machine using local PyTorch frameworks. Zero cloud uploads.
- 🧠 **Natural Language Processing:** Built around `Salesforce/blip-image-captioning-base`. Tell the AI what you remember (`"a dog chasing a ball"`), and it will locate the exact media file.
- 🖼️ **Videos & Photos Supported:** Not just videos! Blipsearch flawlessly indexes `.png`, `.jpg`, `.jpeg`, `.webp`, `.mp4`, `.mkv`, and `.webm` files natively across your device.
- ⚡ **Zero-Config Auto-Discovery:** No manual folder configuration required. It automatically discovers new media files across your home directory while intelligently skipping hidden, cached, project and asset (`repos`, `node_modules`, `Mods`, `Android`, `Games`, ...) folders, plus icon-sized images. Add your own folder names (one per line) to `~/.blipsearch/ignore.txt`. Files that fail to index are not retried until they change, or until you run with `--retry-failed`.
- 🔋 **Smart Lazy-Loading:** The heavy AI model is **only** loaded into VRAM when *new* media is found. Standard text searches consume exactly 0% GPU and return in milliseconds!

> ⚠️ **Note:** the first run downloads the ~1.4 GB BLIP model automatically (needs internet once) and indexes every photo/video it finds — on CPU this can take a while. Everything after that is fully offline and instant.

---

## 🛠️ Technology Stack

Blipsearch leverages state-of-the-art open-source libraries:
*   **[Transformers (HuggingFace)](https://huggingface.co/)** - For advanced BLIP Vision-Language inference.
*   **[PyTorch](https://pytorch.org/)** - For extremely fast GPU acceleration (CUDA).
*   **[OpenCV](https://opencv.org/)** - For rapid and reliable video frame extraction.
*   **[SQLite](https://sqlite.org/)** - For efficient local caching of generated captions.
*   **[Tqdm](https://tqdm.github.io/)** - For clean, minimalistic terminal progress logging.

---

## 🚀 Installation & Setup

It's designed to be effortlessly installed. Ensure you have `Python >= 3.8` on your system.

**1. Clone the repository:**
```bash
git clone https://github.com/Keremcm/blipsearch.git
cd blipsearch
```

**2. Run the automated secure installer:**
```bash
chmod +x install.sh
./install.sh
```
> 💡 *Why `install.sh`? It automatically spins up an isolated virtual environment exclusively for Blipsearch and registers the CLI globally without polluting your Linux/macOS system packages!*

---

## 💻 Usage 

Simply type the command anywhere on your terminal:

```bash
blipsearch "car crashed on the highway"
```

### Options

| Flag | Description |
|------|-------------|
| `--captions N` | Number of differently worded captions generated per image/frame while indexing (default `3`). More captions find more phrasings (`man`, `person`, `cyclist`), but indexing takes longer. Use `--captions 1` for the fastest indexing. |
| `--external` | Also index and search external drives (`/mnt`, `/media`, `/run/media`). |
| `--retry-failed` | Retry files that failed to index earlier. |

`--captions` only affects files indexed from now on; delete `~/.blipsearch/video_index.db` to re-index everything with a new value.

### What happens under the hood?
1. 🔍 Scans your `~` (Home) directory instantly for any media that hasn't been indexed yet.
2. 🤖 If new files are found, it safely wakes up the AI, analyzes the content, and saves several differently worded descriptions per image/frame.
3. 🎯 Matches your input sentence against the database and executes your default media player at the **exact second** the event occurred!

---

<div align="center">
<i>Built with ❤️ by Keremcm & Open Source. Licensed under MIT.</i>
</div>
