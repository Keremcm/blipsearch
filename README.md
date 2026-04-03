



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
- 🖼️ **Videos & Photos Supported:** Not just videos! Blipsearch flawlessly indexes `.png`, `.jpg`, `.mp4`, `.mkv`, and `.webm` files natively across your device in seconds!
- ⚡ **Zero-Config Auto-Discovery:** No manual folder configuration required. It automatically discovers new media files across your system while intelligently skipping hidden & cached folders.
- 🔋 **Smart Lazy-Loading:** The heavy AI model is **only** loaded into VRAM when *new* media is found. Standard text searches consume exactly 0% GPU and return in milliseconds!

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

### What happens under the hood?
1. 🔍 Scans your `~` (Home) directory instantly for any media that hasn't been indexed yet.
2. 🤖 If new files are found, it safely wakes up the AI, analyzes the content, and saves descriptions.
3. 🎯 Matches your input sentence against the database and executes your default media player at the **exact second** the event occurred!

---

<div align="center">
<i>Built with ❤️ by Keremcm & Open Source. Licensed under MIT.</i>
</div>
