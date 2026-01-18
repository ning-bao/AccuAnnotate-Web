# AccuAnnotate-Web

**A Web-Based Annotation System for High-Fidelity GUI Element Grounding**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/flask-3.1%2B-green.svg)](https://flask.palletsprojects.com/)

---

## Research Context

This system was developed as part of an Honours thesis at the University of Sydney:

**AccuAnnotate: Scalable Labelling of Graphical User Interfaces and Reinforcement Learning for Vision-Language Model Grounding**

- **Author**: Ning Bao
- **Degree**: Bachelor of Advanced Computing (Honours)
- **Supervisor**: Dr. Hazem El-Alfy
- **Institution**: School of Computer Science, The University of Sydney

### Publications

- 📄 **Thesis**: [AccuAnnotate.pdf](AccuAnnotate.pdf) - Complete research documentation
- 📊 **Presentation**: [AccuAnnotate-Presentation.pdf](AccuAnnotate-Presentation.pdf) - Research overview and results

> **Note**: The AccuAnnotate-2B model and training artifacts will be released soon.

---

## Abstract

This Honours thesis addresses pixel-level grounding on desktop GUIs: given a screenshot and a natural-language instruction, the agent must predict an exact on-screen click point. Current datasets under-represent diverse widgets and auto-labels are often imprecise, constraining both supervised learning and RL fine-tuning.

The thesis introduces two contributions:

1. **AccuAnnotate** - An automated, scalable pipeline for high-fidelity GUI annotation
2. **AccuAnnotate-2B** - A compact VLM fine-tuned with reinforcement learning for precise desktop grounding

AccuAnnotate couples OmniParser-v2 element discovery with image pre/post-processing, instruction validation, a crop-level prompt builder, and a web workspace for controllable detail and organisation. Manual verification over **1,740 tasks yields 98.77% correct labels**.

AccuAnnotate-2B, trained based on ShowUI-2B with distance-based reward optimization, demonstrates **AUC gains of +5.29% on ScreenSpot-desktop (N=334)** and **+16.95% on ScreenSpot-Pro (N=1,581)**, with **DTB reduced by 9.08%** on ScreenSpot-desktop.

---

## What This Repository Contains

This repository hosts the **AccuAnnotate web annotation pipeline** described in Chapter 3 of the thesis. The system provides:

- 🎯 **Intelligent Annotation** - GPT-4o/GPT-5 integration with OmniParser-v2 preprocessing
- 🖼️ **Modern Web Interface** - Flask-based application with real-time visualization
- 📊 **Batch Processing** - Concurrent image processing with progress tracking
- 🔧 **Flexible Pipeline** - Configurable preprocessing, cropping, and prompt strategies
- 📤 **Export Options** - ShowUI-Desktop format and custom exporters
- 🎨 **Visual Feedback** - Interactive canvas with bounding boxes and center points
- ⚡ **High Performance** - SQLite-backed metadata for large-scale datasets

---

## Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (for GPT-based annotation)
- Optional: CUDA-capable GPU (for local OmniParser inference)

### Installation

```bash
# Clone the repository
git clone git@github.com:ning-bao/AccuAnnotate-Web.git
cd AccuAnnotate-Web

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp env.example .env
# Edit .env and add your OpenAI API key

# Run the application
python app.py
```

Open your browser to `http://localhost:5000`

---

## System Architecture

The AccuAnnotate pipeline follows a multi-stage architecture:

```
┌──────────────────┐
│  Image Upload    │
└────────┬─────────┘
         │
         ↓
┌──────────────────┐      ┌─────────────────┐
│  Preprocessing   │─────→│  OmniParser v2  │
│  (Element Detect)│      │  Element Hints  │
└────────┬─────────┘      └─────────────────┘
         │
         ↓
┌──────────────────┐      ┌─────────────────┐
│  Crop Generation │      │  Directional    │
│  & Prompt Build  │─────→│  Text Padding   │
└────────┬─────────┘      └─────────────────┘
         │
         ↓
┌──────────────────┐      ┌─────────────────┐
│   GPT-4o/5 API   │─────→│  Instruction    │
│   Inference      │      │  Generation     │
└────────┬─────────┘      └─────────────────┘
         │
         ↓
┌──────────────────┐
│  Post-processing │
│  & Validation    │
└────────┬─────────┘
         │
         ↓
┌──────────────────┐      ┌─────────────────┐
│  JSON Export     │      │  SQLite Metadata│
└──────────────────┘      └─────────────────┘
```

---

## Key Features from the Thesis

### 1. OmniParser-v2 Integration

Leverages Microsoft's state-of-the-art UI element detector for high-precision candidate proposals.

### 2. Crop-Level Prompting

Generates tight and directional crops for each candidate element, enabling the VLM to read labels and context more accurately.

### 3. Instruction Validation

Enforces constraints: 1-5 unique elements per image, absolute pixel coordinates, actionable instructions (≤120 chars), and duplicate removal.

### 4. Configurable Detail Levels

- **Low**: Short instructions (≤10 words)
- **Normal**: Concise with type and label (≤14 words)
- **High**: Rich context with description, state, and surrounding UI elements

### 5. Web Workspace

Provides a modern interface for batch processing, visual verification, manual editing, and export to training formats.

---

## Usage

### Basic Workflow

1. **Upload images** - Via web interface or place in `data/images/`
2. **Generate annotations** - Click "Generate Annotation" or "Annotate All"
3. **Review and edit** - Use interactive canvas and element cards
4. **Export** - Export to ShowUI-Desktop or custom formats

### Configuration

Key environment variables (see `env.example`):

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4o                     # or gpt-5

# Annotation Settings
ANNOTATOR_DETAIL_LEVEL=high             # low, normal, or high
ANNOTATOR_MAX_INSTRUCTIONS=5            # 1-5 elements per image
ANNOTATOR_PREPROCESS_MAX_ELEMENTS=24    # Candidate hints

# OmniParser Configuration
OMNIPARSER_URL=local                    # Use local inference
OMNIPARSER_MIN_CONF=0.3                 # Detection threshold
```

For detailed configuration and advanced usage, see [DOCUMENTATION.md](DOCUMENTATION.md).

---

## Citation

If you use AccuAnnotate-Web in your research, please cite:

```bibtex
@thesis{bao2026accuannotate,
  title={AccuAnnotate: Scalable Labelling of Graphical User Interfaces 
         and Reinforcement Learning for Vision-Language Model Grounding},
  author={Bao, Ning},
  year={2026},
  school={The University of Sydney},
  type={Bachelor of Advanced Computing (Honours) thesis}
}
```

---

## Technology Stack

- **Backend**: Flask 3.1, Python 3.8+
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **ML/AI**: OpenAI GPT-4o/5, Microsoft OmniParser v2
- **Storage**: SQLite, JSON annotations
- **Image Processing**: Pillow, OpenCV, PyTorch (optional)

---

## Project Structure

```
AccuAnnotate-Web/
├── app.py                      # Flask backend and batch job manager
├── db.py                       # SQLite metadata interface
├── utils/
│   ├── annotator.py           # Core annotation logic (crop-first prompting)
│   └── visualizer.py          # Canvas visualization
├── static/
│   ├── css/style.css          # UI styles
│   └── js/main.js             # Frontend logic
├── templates/
│   └── index.html             # Web interface
├── scripts/
│   ├── export_showui_desktop.py   # ShowUI format export
│   ├── import_data.py             # Dataset import utility
│   └── ...                        # Other utilities
├── data/
│   ├── images/                # Input screenshots
│   ├── annotations/           # Generated JSON annotations
│   └── metadata.db            # SQLite index
├── requirements.txt           # Python dependencies
└── env.example               # Environment template
```

---

## Acknowledgments

This work builds upon:

- **ShowUI-2B** - Base model for AccuAnnotate-2B fine-tuning
- **OmniParser v2** - High-precision UI element detection ([Microsoft Research](https://huggingface.co/microsoft/OmniParser-v2.0))
- **OpenAI GPT-4o/5** - Vision-language instruction generation
- Open-source frameworks: PyTorch, Hugging Face Transformers, Flask

Special thanks to Dr. Hazem El-Alfy for supervision and guidance throughout this research.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions, collaboration, or issues:

- **Email**: enquire@baoning.dev
- **GitHub Issues**: [ning-bao/AccuAnnotate-Web/issues](https://github.com/ning-bao/AccuAnnotate-Web/issues)

---

**Last Updated**: January 2026  
**Status**: Research Prototype - Model Release Pending
