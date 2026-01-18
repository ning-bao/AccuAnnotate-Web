# AccuAnnotate-Web Documentation

Comprehensive technical documentation for the AccuAnnotate-Web annotation system.

---

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Core Concepts](#core-concepts)
4. [API Reference](#api-reference)
5. [Annotation Format](#annotation-format)
6. [Preprocessing Backends](#preprocessing-backends)
7. [Export Formats](#export-formats)
8. [Advanced Usage](#advanced-usage)

---

## Installation

### System Requirements

- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 1GB free space
- **GPU**: Optional, for local OmniParser inference

### Dependencies

```bash
pip install -r requirements.txt
```

Core dependencies:
- Flask 3.1.2 - Web framework
- Pillow 12.0.0 - Image processing
- OpenAI 2.6.1 - GPT API client
- python-dotenv 1.1.1 - Environment management

Optional (for local preprocessing):
- torch, torchvision - PyTorch (OmniParser)
- transformers - Hugging Face models
- ultralytics - YOLO detection

---

## Configuration

### Environment Variables

Create a `.env` file from the template:

```bash
cp env.example .env
```

#### Required Settings

```bash
# OpenAI API Configuration
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4o                    # or gpt-5
OPENAI_MAX_COMPLETION_TOKENS=4096
```

#### Optional Settings

```bash
# Service Configuration
OPENAI_SERVICE_TIER=auto               # auto, default, or scale
OPENAI_TIMEOUT_SECONDS=900
OPENAI_ENABLE_CODE_INTERPRETER=false

# Preprocessing
ANNOTATOR_PREPROCESS_ENABLE=true
ANNOTATOR_PREPROCESS_MAX_ELEMENTS=24
ANNOTATOR_MAX_INSTRUCTIONS=5

# OmniParser Settings
OMNIPARSER_URL=local                   # or HTTP endpoint
OMNIPARSER_MIN_CONF=0.3
OMNIPARSER_CONF_THRESHOLD=0.5

# Database
ANNOTATION_DB_PATH=data/metadata.db

# Flask Settings
MAX_FILE_SIZE_MB=16
BATCH_MAX_WORKERS=3
```

#### Detail Levels

Control annotation verbosity:

```bash
ANNOTATOR_DETAIL_LEVEL=high            # low, normal, or high
```

- **low**: Short instructions (≤10 words), no extra fields
- **normal**: Concise instructions (≤14 words), includes type and label
- **high**: Rich context with type, label, description, context, and state

---

## Core Concepts

### Annotation Pipeline

```
1. Image Upload
   ↓
2. Preprocessing (OmniParser/OpenCV)
   → Detects UI element candidates
   → Returns bounding boxes + confidence
   ↓
3. GPT Processing
   → Analyzes image + element crops
   → Generates instructions
   → Returns grounded annotations
   ↓
4. Post-processing
   → Validates coordinates
   → Removes duplicates
   → Saves to JSON
```

### Coordinate System

AccuAnnotate-Web uses **absolute pixel coordinates**:

```json
{
  "img_size": [1920, 1080],
  "element": [{
    "instruction": "Click the submit button",
    "bbox": [100, 200, 300, 250],      // [x1, y1, x2, y2] in pixels
    "point": [200, 225],                // [cx, cy] in pixels
    "source_id": 1
  }]
}
```

**Validation Rules**:
- All coordinates must be integers
- `0 ≤ x1 < x2 ≤ width` and `0 ≤ y1 < y2 ≤ height`
- Point must be strictly inside bbox: `x1 < cx < x2` and `y1 < cy < y2`

### Element Detection

The system uses a two-stage approach:

1. **Preprocessing**: Detect candidate elements
   - OmniParser v2 (recommended): ML-based detection
   - OpenCV (fallback): Edge detection + contours

2. **Filtering**: Rank and select top candidates
   - Confidence scoring
   - Duplicate removal (IoU > 0.6)
   - Ambiguity-based sampling

---

## API Reference

### REST Endpoints

#### GET /api/images
List all images with pagination.

**Query Parameters**:
- `page` (int): Page number (default: 1)
- `page_size` (int): Items per page (default: 500, max: 5000)

**Response**:
```json
{
  "images": [{"filename": "image.png", "has_annotation": true}],
  "total": 100,
  "page": 1,
  "page_size": 500
}
```

#### GET /api/image/<filename>
Retrieve image file.

#### GET /api/annotation/<filename>
Get annotation JSON for an image.

**Response**:
```json
{
  "img_size": [1920, 1080],
  "element": [...]
}
```

#### POST /api/annotate/<filename>
Generate annotation for an image.

**Request Body** (optional):
```json
{
  "detail_level": "high"              // low, normal, or high
}
```

#### PUT /api/annotation/<filename>
Update annotation for an image.

**Request Body**:
```json
{
  "img_size": [1920, 1080],
  "element": [...]
}
```

#### DELETE /api/annotation/<filename>/element/<index>
Delete a specific element from annotation.

#### POST /api/preprocess/<filename>
Run preprocessing only (no GPT).

**Request Body** (optional):
```json
{
  "max_elements": 30
}
```

**Response**:
```json
{
  "img_size": [1920, 1080],
  "element": [
    {"bbox": [100, 200, 300, 250], "point": [200, 225]}
  ]
}
```

#### POST /api/batch-annotate
Start batch annotation job.

**Request Body**:
```json
{
  "filenames": ["image1.png", "image2.png"],  // optional, all if empty
  "force": false                               // re-annotate existing
}
```

**Response**:
```json
{
  "job_id": "uuid",
  "total": 10
}
```

#### GET /api/batch-annotate/stream/<job_id>
SSE stream for batch progress.

#### POST /api/upload
Upload new image.

**Form Data**:
- `file`: Image file
- `relative_path`: Optional path structure

---

## Annotation Format

### Structure

```json
{
  "img_size": [width_px, height_px],
  "element": [
    {
      "instruction": "Natural language action description",
      "bbox": [x1, y1, x2, y2],
      "point": [cx, cy],
      "source_id": 1,
      
      // Optional contextual fields (high detail level):
      "type": "button",
      "label": "Submit",
      "description": "Primary action button for form submission",
      "context": "Login form",
      "state": "enabled"
    }
  ]
}
```

### Field Descriptions

- **instruction**: Actionable description of the element (≤120 chars)
- **bbox**: Bounding box coordinates `[left, top, right, bottom]`
- **point**: Center point `[x, y]` (used as click target)
- **source_id**: Reference to preprocessing hint ID
- **type**: Element category (button, input, link, etc.)
- **label**: Visible text content
- **description**: What the element is/does
- **context**: Surrounding section or menu
- **state**: Visual state (enabled/disabled, selected/unselected)

---

## Preprocessing Backends

### OmniParser v2 (Recommended)

Microsoft's state-of-the-art UI element detector.

**Configuration**:
```bash
OMNIPARSER_URL=local                    # Use local inference
OMNIPARSER_MIN_CONF=0.3                # Detection threshold
OMNIPARSER_CONF_THRESHOLD=0.5          # Filtering threshold
```

**Local Setup**:
```bash
bash setup_omniparser.sh
```

**Performance**:
- RTX 4090: ~0.8s per image
- A100: ~0.6s per image
- CPU: ~5-10s per image

### OpenCV (Fallback)

Edge detection + contour finding.

Automatically used if OmniParser is unavailable.

---

## Export Formats

### ShowUI-Desktop Format

Export to ShowUI training format:

```bash
python scripts/export_showui_desktop.py \
  --images data/images \
  --annotations data/annotations \
  --output export/ \
  --split train
```

Creates:
```
export/
├── images/
│   └── train/
├── annotations/
│   └── train.json
└── metadata.json
```

### Custom Formats

Extend `scripts/export_showui_desktop.py` or create new exporters.

---

## Advanced Usage

### Batch Processing with Custom Settings

```python
from utils.annotator import GPTAnnotator

annotator = GPTAnnotator(
    model="gpt-4o",
    max_tokens=4096,
    service_tier="scale"
)

for image_path in image_list:
    annotation = annotator.annotate(
        image_path,
        detail_level="high"
    )
    # Save annotation...
```

### Custom Preprocessing

```python
# Get preprocessing hints only
hints = annotator.preprocess_only(
    image_path,
    max_elements=30
)

# Use hints with custom processing
annotation = annotator.annotate_with_hints(
    image_path,
    hints=filtered_hints,
    detail_level="normal"
)
```

### Database Operations

```python
import db

# Initialize database
db.init_db()

# Add image
db.upsert_image("folder/image.png", has_annotation=True, size_bytes=12345)

# Query images
images = db.list_images(limit=100, offset=0)
total = db.count_images()

# Folders
folders = db.list_all_folders()
db.upsert_folder("new_folder")
```

### Visualization

```python
from utils.visualizer import visualize_annotations

# Generate visualization
vis_base64 = visualize_annotations(image_path, annotation)
# Returns base64-encoded PNG with overlays
```

---

## Performance Optimization

### Database Indexing

For large datasets (1000+ images), the SQLite backend provides significant speedup:

- Image listing: 10-50x faster
- Pagination: Constant time
- Folder queries: Indexed

### Concurrent Processing

Adjust batch workers:

```bash
BATCH_MAX_WORKERS=5                    # Increase for more parallelism
```

**Recommendation**: Set to number of CPU cores / 2

### GPU Acceleration

For OmniParser:

```bash
# Force CUDA
python app.py  # Auto-detects GPU

# Force CPU
python omniparser_local.py --device cpu --image test.png
```

---

## Troubleshooting

### Common Issues

**"OpenAI API key is required"**
- Set `OPENAI_API_KEY` in `.env` file
- Verify key starts with `sk-` and is valid

**"OmniParser failed"**
- Check GPU memory (needs ~4GB VRAM)
- Try CPU mode: `OMNIPARSER_URL=local` + `--device cpu`
- System auto-falls back to OpenCV

**"Annotation timeout"**
- Increase `OPENAI_TIMEOUT_SECONDS=1200`
- Reduce `ANNOTATOR_PREPROCESS_MAX_ELEMENTS=12`
- Use `detail_level=low` for faster processing

**Database performance**
- Re-index: `rm data/metadata.db && python -c "import db; db.init_db()"`
- Run `python scripts/import_data.py` to rebuild

---

## Development

### Running Tests

```bash
python demo.py                         # Test annotation pipeline
```

### Code Structure

- `app.py`: Flask routes and batch job management
- `db.py`: SQLite operations
- `utils/annotator.py`: Core annotation logic
- `utils/visualizer.py`: Canvas rendering
- `static/js/main.js`: Frontend logic
- `templates/index.html`: UI layout

---

## Support

For technical issues or questions:
- GitHub Issues: [ning-bao/AccuAnnotate-Web/issues](https://github.com/ning-bao/AccuAnnotate-Web/issues)
- Email: enquiry@baoning.dev | ning.bao.syd@gmail.com

---

**Last Updated**: January 2026
