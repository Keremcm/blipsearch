# 🔍 Blipsearch

**Blipsearch** is a blazing-fast, strictly on-device video search engine. Have you ever remembered a specific scene from a local video but had no idea where the file was or what it was named? Simply describe it in natural language!

Blipsearch automatically indexes your videos in the background and uses AI to find the exact frame you are looking for. Once found, it instantly opens the video stream in your default media player. 🎬

## ✨ Features

- **100% Offline & Private:** Everything runs purely on your local machine using PyTorch and HuggingFace models. Zero internet connection required.
- **Natural Language Search:** Uses the `Salesforce/blip-image-captioning-base` artificial intelligence to watch and understand your videos. Just type what happened (e.g., `"a dog chasing a ball"`).
- **Zero-Config Auto-Discovery:** No need to specify target directories. It automatically scans your device (skipping hidden and system directories for speed) to find and index your latest videos.
- **Smart Lazy-Loading:** The heavy AI models are only loaded into VRAM when *new* videos are detected. Searching existing videos takes milliseconds with zero GPU usage!
- **Clean & Simple CLI:** Minimalistic command-line interface. No annoying standard error outputs, HuggingFace telemetry warning spam, or broken progress bars.

## 🚀 Installation

Ensure you have Python 3.8+ installed on your system.

1. Clone the repository:
```bash
git clone https://github.com/Keremcm/blipsearch.git
cd blipsearch
```

2. Run the automated installer:
```bash
chmod +x install.sh
./install.sh
```
*(This creates an isolated virtual environment automatically without polluting your OS, and universally ties the `blipsearch` command!)*

## 💡 Usage

Whenever you want to find a video, open your terminal anywhere and type:

```bash
blipsearch "a car crashed on the highway"
```

**What it does behind the scenes:**
1. It swiftly scans your `~` (Home) directory for videos.
2. If it encounters a video it hasn’t seen before, it analyzes it extracting natural descriptions.
3. Finds the highest matching caption and opens the winning video with `xdg-open` on Linux.

## 🛠 Prerequisites

- Python >= 3.11
- Storage: Your captions are extremely lightweight and saved efficiently to `~/.blipsearch/video_index.db`.
- CUDA (Optional but Highly Recommended): The program will automatically switch to GPU processing for blazingly fast indexing if an Nvidia GPU is detected on your system.

## License
MIT License. Feel free to modify, distribute, and enhance it.
