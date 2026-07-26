# 🧵 Loom

> *Video is not generated. It is woven.*

**Loom** is a playful, non-parametric video synthesizer. No neural networks. No gradient descent. No millions of weights. Just algorithms, patterns, and a little bit of magic.

Think of it as a **video loom** — you throw in raw footage, it learns the *motion*, *textures*, and *patterns*, then weaves new videos from your text prompts. It runs on your laptop. It runs on a Raspberry Pi. It runs *anywhere*.

## 🎬 What It Does

```bash
python inference.py \
  --vid my-corpus.vid \
  --prompt "ocean waves at sunset with panning camera" \
  --output sunset.mp4
```

And you get a 5-second video loop. Not photorealistic Hollywood VFX — something **weird, beautiful, and alive**.

## 🧶 The Philosophy

Modern AI video treats pixels as math homework. Loom treats video as **woven fabric**:

| Thread | What It Is |
|--------|-----------|
| 🌊 **Warp** | Motion fields — how the camera moves, how things flow |
| 🎨 **Weft** | Texture patches — the visual fabric of the world |
| ✨ **Pattern** | Cellular automata — organic detail that breathes |
| 🧵 **Shuttle** | Your text prompt, parsed into a scene plan |

The intelligence lives in **how we organize and compose**, not in learned parameters.

## 🚀 Quick Start

### 1. Install (No GPU needed!)

```bash
git clone https://github.com/YOURNAME/loom.git
cd loom
pip install -r requirements.txt
```

That's it. No PyTorch. No TensorFlow. No CUDA. Just NumPy, OpenCV, and SciPy.

### 2. Build a Corpus

Feed it some videos. Any videos. Home movies, stock footage, your cat sleeping.

```bash
mkdir my_videos
# Drop some .mp4 files in there

python corpus.py \
  --input_dir my_videos \
  --output my-corpus.vid \
  --target_size 1000
```

This extracts:
- **Motion primitives** (optical flow patterns)
- **Texture patches** (reusable visual fabric)
- **CA rules** (cellular automata for organic detail)

### 3. Generate!

```bash
python inference.py \
  --vid my-corpus.vid \
  --prompt "forest path dolly camera golden hour" \
  --output forest.mp4 \
  --width 854 \
  --height 480 \
  --frames 120
```

## 🎮 Prompt Ideas to Try

```bash
# Chill vibes
python inference.py --vid my-corpus.vid --prompt "ocean waves sunset panning" --output chill.mp4

# Trippy abstract
python inference.py --vid my-corpus.vid --prompt "neon city night orbit camera" --output trippy.mp4

# Nature documentary style
python inference.py --vid my-corpus.vid --prompt "mountain lake sunrise static" --output nature.mp4

# Weird and experimental
python inference.py --vid my-corpus.vid --prompt "abstract dreamy zoom in" --output weird.mp4
```

## 📁 Project Structure

```
loom/
├── loom.py              # 🧠 Core engine (Motion, Texture, CA, SceneGraph)
├── corpus.py            # 🔨 Build .vid corpus from raw videos
├── inference.py         # 🎬 CLI for video generation
├── requirements.txt     # 📦 Just 5 dependencies, zero ML frameworks
├── setup.py             # 📋 Package setup
├── .github/
│   └── workflows/
│       └── build.yml    # 🤖 CI: build → test → release
├── README.md            # 📖 You are here
├── LICENSE              # ⚖️ MIT
└── .gitignore           # 🙈 Ignore generated files
```

## 🧪 How It Actually Works

### Training (Corpus Building)

Instead of training a neural network, we **extract and index patterns** from your videos:

```
Raw Videos
    ↓
Optical Flow (Farneback) → Motion Primitives
    ↓
Patch Extraction + PCA   → Texture Library
    ↓
Pattern Mining           → CA Rulebook
    ↓
Compressed .vid File
```

All classical computer vision. No GPU needed. A few hours on a laptop.

### Inference (Generation)

```
Text Prompt
    ↓
Parse → Scene Graph (background, camera, lighting, style)
    ↓
Retrieve Motion    → Warp texture through time
Retrieve Texture   → Synthesize from exemplars
Apply CA Rules     → Add organic breathing detail
Color Grade        → Match lighting mood
    ↓
Video Frames!
```

## 💾 The .vid Format

Not a model checkpoint. A **compressed pattern library**:

| Section | What's Inside | ~Size |
|---------|---------------|-------|
| Motion | Optical flow archetypes | ~300MB |
| Texture | PCA-compressed patch dictionary | ~400MB |
| CA Rules | Cellular automata parameters | ~50MB |
| Metadata | Scene graph index | ~50MB |
| **Total** | | **~800MB** |

## ⚡ Performance

| Resolution | Duration | CPU Time | Memory |
|-----------|----------|----------|--------|
| 480p | 5s @ 24fps | ~5-15s | ~300MB |
| 720p | 5s @ 24fps | ~15-30s | ~600MB |
| 1080p | 5s @ 24fps | ~40-60s | ~1.2GB |

*No GPU. No cloud. Just your CPU and some patience.*

## 🎯 Why Build This?

| | Diffusion Models | Loom |
|---|----------------|------|
| Parameters | 1B-50B | **Zero** |
| Training | GPU cluster, weeks | **Laptop, hours** |
| Inference | Needs GPU | **CPU only** |
| Size | 10-50GB | **~1GB** |
| Understandable | Black box | **Every step is inspectable** |
| Fun factor | Serious business | **Playful and weird** |

## 🐛 Known Limitations

Loom is **not** trying to be Veo or Sora. It is:
- ✅ Great for abstract, stylized, and experimental video
- ✅ Great for seamless motion loops
- ✅ Great for running on anything with a CPU
- ❌ Not great for photorealistic human faces
- ❌ Not great for complex multi-object physics
- ❌ Not great for Hollywood VFX

**It is great for making weird, beautiful things.**

## 🤖 CI / GitHub Actions

Push a tag and GitHub Actions will:
1. **Build corpus** from your dataset
2. **Test inference** on Linux, macOS, and Windows
3. **Benchmark** performance
4. **Release** the `.vid` file automatically

```bash
git tag v0.1.0
git push origin v0.1.0
# → GitHub Actions builds and releases your corpus!
```

## 🧶 Contributing

This is a playground. Add new motion extractors. Invent new CA rules. Try weird texture synthesis methods. Break things. Fix things. Make it weirder.

## 📜 License

MIT — built for creative coders who believe intelligence lives in algorithms, not just parameters.

---

*"The best models are the ones you can understand, modify, and have fun with."*
