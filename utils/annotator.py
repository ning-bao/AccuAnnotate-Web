#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OpenAI Annotator (crop-first binding)
- Interleaves per-crop text markers "<crop id=K type=...>" immediately before each crop image
  so the model strongly associates a crop with its hint id.
- Adds directional crops for ambiguous/row-like elements to include the likely text side.
- Keeps shard_indices for back-compat, but the binding now primarily uses the text markers.
"""

import os
import json
import base64
import hashlib
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from string import Template

from PIL import Image
from dotenv import load_dotenv, find_dotenv

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    import requests  # type: ignore
    _REQUESTS_AVAILABLE = True
except Exception:  # pragma: no cover
    requests = None  # type: ignore
    _REQUESTS_AVAILABLE = False

_OMNI_PARSER_INSTANCE = None


# ----------------
# Geometry helpers
# ----------------
def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def _center_of(box: List[int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    cx = _clamp(cx, x1 + 1, x2 - 1)
    cy = _clamp(cy, y1 + 1, y2 - 1)
    return int(cx), int(cy)

def _area(box: List[int]) -> int:
    return max(0, box[2]-box[0]) * max(0, box[3]-box[1])

def _iou(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / max(union, 1)

def _min_neighbor_distance(hints: List[Dict[str, Any]], i: int) -> float:
    ci = _center_of(hints[i]["bbox"])
    best = 1e9
    for j, h in enumerate(hints):
        if j == i: 
            continue
        cj = _center_of(h["bbox"])
        dx = ci[0]-cj[0]; dy = ci[1]-cj[1]
        d2 = (dx*dx + dy*dy) ** 0.5
        if d2 < best:
            best = d2
    return best if best < 1e9 else 1e9

def _is_row_like(box: List[int], W: int, H: int) -> bool:
    x1, y1, x2, y2 = box
    w = max(1, x2-x1); h = max(1, y2-y1)
    ar = w/float(h)
    # Heuristic: list rows/menus tend to be wide and not too tall
    return (ar >= 1.8) and (18 <= h <= 120)

# ---------------------
# OmniParser integration
# ---------------------
def _preload_omniparser(min_conf: float = 0.3):
    global _OMNI_PARSER_INSTANCE
    if _OMNI_PARSER_INSTANCE is not None:
        return _OMNI_PARSER_INSTANCE
    try:
        import omniparser_local  # type: ignore
        _OMNI_PARSER_INSTANCE = omniparser_local.OmniParserV2(min_confidence=min_conf)
        print("  OmniParser local models preloaded")
    except Exception as e:  # pragma: no cover
        print(f"  OmniParser local preload failed / not found: {e}")
        _OMNI_PARSER_INSTANCE = None
    return _OMNI_PARSER_INSTANCE

def _normalize_omni_elements(raw_elems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    norm: List[Dict[str, Any]] = []
    for e in raw_elems or []:
        try:
            bbox = [int(e["bbox"][0]), int(e["bbox"][1]), int(e["bbox"][2]), int(e["bbox"][3])]
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
        except Exception:
            continue
        pt = e.get("point")
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            cx, cy = _center_of(bbox)
            point = [cx, cy]
        else:
            point = [int(pt[0]), int(pt[1])]
        conf = float(e.get("confidence", 0.5))
        norm.append({"bbox": bbox, "point": point, "confidence": conf})
    return norm

def _rank_and_limit_hints(elems: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if not elems:
        return []
    elems = sorted(elems, key=lambda x: (-float(x.get("confidence", 0.5)), _area(x["bbox"])))
    kept: List[Dict[str, Any]] = []
    for e in elems:
        b = e["bbox"]; cx, cy = _center_of(b)
        drop = False
        for k in kept:
            if _iou(b, k["bbox"]) > 0.6:
                drop = True; break
            kcx, kcy = _center_of(k["bbox"])
            if abs(cx - kcx) < 4 and abs(cy - kcy) < 4:
                drop = True; break
        if not drop:
            kept.append(e)
    ranked = kept
    if limit and limit > 0:
        ranked = ranked[:limit]
    ranked.sort(key=lambda e: (_center_of(e["bbox"])[1], _center_of(e["bbox"])[0]))
    for i, e in enumerate(ranked, start=1):
        e["id"] = i
    return ranked

# ---------------
# Main annotator
# ---------------
class GPTAnnotator:
    def __init__(self, api_key=None, model=None, max_tokens=None, service_tier=None, timeout_seconds=None, use_code_interpreter=None):
        dotenv_path = find_dotenv(usecwd=True)
        if dotenv_path:
            load_dotenv(dotenv_path=dotenv_path, override=False)
        else:
            script_env = Path(__file__).resolve().parent.parent / '.env'
            if script_env.exists():
                load_dotenv(dotenv_path=str(script_env), override=False)

        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model or os.getenv('OPENAI_MODEL') or 'gpt-5'

        max_ct_env = os.getenv('OPENAI_MAX_COMPLETION_TOKENS')
        legacy_max_env = os.getenv('OPENAI_MAX_TOKENS')
        self.max_completion_tokens = (
            int(max_tokens) if isinstance(max_tokens, int) else
            int(max_ct_env) if max_ct_env else
            int(legacy_max_env) if legacy_max_env else
            4096
        )

        env_service_tier = os.getenv('OPENAI_SERVICE_TIER', '').strip().lower()
        self.service_tier = (service_tier or env_service_tier) or None

        env_timeout = os.getenv('OPENAI_TIMEOUT_SECONDS', '').strip()
        self.timeout_seconds = (
            float(timeout_seconds) if timeout_seconds is not None else (float(env_timeout) if env_timeout else 900.0)
        )

        env_enable_ci = os.getenv('OPENAI_ENABLE_CODE_INTERPRETER', '').strip().lower()
        if use_code_interpreter is not None:
            self.use_code_interpreter = bool(use_code_interpreter)
        else:
            self.use_code_interpreter = env_enable_ci in ('1','true','yes','on')

        env_pre = os.getenv('ANNOTATOR_PREPROCESS_ENABLE', '').strip().lower()
        self.preprocess_enable = True if env_pre == '' else (env_pre in ('1','true','yes','on'))

        try:
            raw_max = os.getenv('ANNOTATOR_PREPROCESS_MAX_ELEMENTS', '24').strip().lower()
            self.preprocess_max_elements = int(raw_max)
        except Exception:
            self.preprocess_max_elements = 24

        try:
            self.max_instructions = int(os.getenv('ANNOTATOR_MAX_INSTRUCTIONS', '5'))
            if self.max_instructions < 1:
                self.max_instructions = 1
        except Exception:
            self.max_instructions = 5

        # shard/crop settings
        try:
            self.max_shards = int(os.getenv('ANNOTATOR_MAX_SHARDS', '15'))
        except Exception:
            self.max_shards = 15
        try:
            self.shard_topk = int(os.getenv('ANNOTATOR_SHARD_TOPK', '6'))
        except Exception:
            self.shard_topk = 6
        try:
            self.dual_crop_topk = int(os.getenv('ANNOTATOR_DUAL_CROP_TOPK', '8'))
        except Exception:
            self.dual_crop_topk = 8
        try:
            self.pad_px = int(os.getenv('ANNOTATOR_CROP_PAD_PX', '8'))
        except Exception:
            self.pad_px = 8
        try:
            self.text_pad_px = int(os.getenv('ANNOTATOR_TEXT_PAD_PX', '48'))
        except Exception:
            self.text_pad_px = 48
        try:
            self.crop_long_side = int(os.getenv('ANNOTATOR_CROP_LONG_SIDE', '160'))
        except Exception:
            self.crop_long_side = 160

        self.shard_seed_raw = os.getenv('ANNOTATOR_SHARD_SEED', 'auto').strip()

        # Detail level configuration (controls verbosity of generated instructions)
        self.detail_level = (os.getenv('ANNOTATOR_DETAIL_LEVEL', 'high') or 'high').strip().lower()

        # Omni
        self.omni_url = os.getenv('OMNIPARSER_URL', '').strip()
        self.omni_api_key = os.getenv('OMNIPARSER_API_KEY', '').strip()
        try:
            self.omni_timeout = float(os.getenv('OMNIPARSER_TIMEOUT', '30'))
        except Exception:
            self.omni_timeout = 30.0
        try:
            self.omni_min_conf = float(os.getenv('OMNIPARSER_MIN_CONF', '0.3'))
        except Exception:
            self.omni_min_conf = 0.3
        try:
            self.omni_conf_thr = float(os.getenv('OMNIPARSER_CONF_THRESHOLD', '0.5'))
        except Exception:
            self.omni_conf_thr = 0.5

        if not self.api_key or str(self.api_key).startswith('your_'):
            raise ValueError('OPENAI_API_KEY is required')
        if OpenAI is None:
            raise ValueError('openai package is required')

        self.client = OpenAI(api_key=self.api_key)
        self.preprocess_backend = 'omni'

        if self.preprocess_enable:
            if not self.omni_url or self.omni_url.lower() in ('local','localhost'):
                _preload_omniparser(self.omni_min_conf)
            else:
                if not _REQUESTS_AVAILABLE:
                    print('requests not available; HTTP OmniParser may fail')

    # --------------- public API ---------------
    def annotate(self, image_path: str, detail_level: Optional[str] = None) -> dict:
        with Image.open(image_path) as img:
            width, height = img.size
        hints = self._compute_preprocess_hints(image_path, max_elements=self.preprocess_max_elements) if self.preprocess_enable else []
        if not hints:
            return {"img_size": [width, height], "element": []}
        return self._call_openai_api(image_path, width, height, hints=hints, detail_level=detail_level)

    def annotate_with_hints(self, image_path: str, hints: List[dict], detail_level: Optional[str] = None) -> dict:
        with Image.open(image_path) as img:
            width, height = img.size
        if not hints:
            return {"img_size": [width, height], "element": []}
        return self._call_openai_api(image_path, width, height, hints=hints, detail_level=detail_level)

    def preprocess_only(self, image_path: str, max_elements: Optional[int] = None) -> dict:
        with Image.open(image_path) as img:
            W, H = img.size
        n = int(max_elements) if isinstance(max_elements, int) else int(self.preprocess_max_elements)
        elements = self._compute_preprocess_hints(image_path, max_elements=n)
        stripped = [{"bbox": e["bbox"], "point": e["point"]} for e in elements]
        return {"img_size": [W, H], "element": stripped}

    # --------------- guts ---------------
    def _compute_preprocess_hints(self, image_path: str, max_elements: int) -> List[dict]:
        elems: List[Dict[str, Any]] = []
        if not self.omni_url or self.omni_url.lower() in ('local','localhost'):
            try:
                global _OMNI_PARSER_INSTANCE
                if _OMNI_PARSER_INSTANCE is None:
                    _preload_omniparser(self.omni_min_conf)
                if _OMNI_PARSER_INSTANCE is not None:
                    result = _OMNI_PARSER_INSTANCE.parse(image_path, conf_threshold=self.omni_conf_thr, with_captions=False)
                    raw = result.get('elements', []) if isinstance(result, dict) else []
                    elems = _normalize_omni_elements(raw)
            except Exception as e:
                print(f"[Annotator] Local OmniParser failed: {e}")
        else:
            if _REQUESTS_AVAILABLE:
                try:
                    with open(image_path, 'rb') as f:
                        img_bytes = f.read()
                    headers = {'Accept': 'application/json'}
                    if self.omni_api_key:
                        headers['Authorization'] = f"Bearer {self.omni_api_key}"
                    files = {'image': (os.path.basename(image_path), img_bytes, 'application/octet-stream')}
                    params = {'return': 'elements', 'format': 'json', 'conf_threshold': str(self.omni_conf_thr)}
                    resp = requests.post(self.omni_url, headers=headers, files=files, data=params, timeout=self.omni_timeout)
                    resp.raise_for_status()
                    data = resp.json() if hasattr(resp, 'json') else json.loads(resp.text)
                    raw = data.get('elements', []) if isinstance(data, dict) else []
                    elems = _normalize_omni_elements(raw)
                except Exception as e:
                    print(f"[Annotator] OmniParser HTTP failed: {e}")
        ranked = _rank_and_limit_hints(elems, limit=max_elements)
        return ranked

    def _call_openai_api(self, image_path: str, width: int, height: int, hints: List[dict], detail_level: Optional[str] = None) -> dict:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        image_data = base64.b64encode(image_bytes).decode('utf-8')
        mime_type = self._mime_for_ext(os.path.splitext(image_path)[1])

        element_crops, crop_tags, shard_index_map = self._build_crops(image_path, hints)
        # Resolve detail level: request override > instance default > fallback
        dl = (detail_level or getattr(self, 'detail_level', 'high') or 'high').strip().lower()
        if dl not in ("low", "normal", "high"):
            dl = "high"
        prompt = self._build_prompt(width, height, hints, shard_index_map, crop_tags, detail_level=dl)

        model_lower = str(self.model).lower()
        uses_completion_tokens = (model_lower.startswith('gpt-5') or model_lower.startswith('o3'))
        token_param_name_chat = 'max_completion_tokens' if uses_completion_tokens else 'max_tokens'
        token_param_name_resp = 'max_output_tokens'
        allow_service_tier = uses_completion_tokens

        # Build content: prompt + full screenshot + interleaved <crop id=...> marker then image for each crop
        def build_chat_kwargs(limit: int):
            content: List[Dict[str, Any]] = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}},
            ]
            for tag, crop_b64 in zip(crop_tags, element_crops):
                content.append({"type": "text", "text": tag})
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{crop_b64}"}})
            kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "response_format": {"type": "json_object"},
            }
            kwargs[token_param_name_chat] = self.max_completion_tokens
            if allow_service_tier and self.service_tier:
                kwargs["service_tier"] = self.service_tier
            if self.timeout_seconds:
                kwargs["timeout"] = self.timeout_seconds
            return kwargs

        def build_resp_kwargs(limit: int, include_tools: bool = True):
            resp_content: List[Dict[str, Any]] = [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": f"data:{mime_type};base64,{image_data}"},
            ]
            for tag, crop_b64 in zip(crop_tags, element_crops):
                resp_content.append({"type": "input_text", "text": tag})
                resp_content.append({"type": "input_image", "image_url": f"data:image/png;base64,{crop_b64}"})
            kwargs = {
                "model": self.model,
                "input": [{"role": "user", "content": resp_content}],
                "tool_choice": "auto",
            }
            if include_tools:
                kwargs["tools"] = [{"type": "code_interpreter", "container": {"type": "auto"}}]
            kwargs[token_param_name_resp] = self.max_completion_tokens
            if allow_service_tier and self.service_tier:
                kwargs["service_tier"] = self.service_tier
            if self.timeout_seconds:
                kwargs["timeout"] = self.timeout_seconds
            return kwargs

        attempts = [int(self.max_completion_tokens), min(max(int(self.max_completion_tokens) * 2, 2048), 8192)]
        response = None
        for attempt_index, max_ct in enumerate(attempts):
            try:
                used_path = "chat"
                if self.use_code_interpreter:
                    try:
                        response = self.client.responses.create(**build_resp_kwargs(max_ct, include_tools=True))
                        used_path = "responses"
                    except Exception as e_ci:
                        msg = str(e_ci).lower()
                        if "tools[0].container" in msg or "missing required parameter" in msg:
                            response = self.client.responses.create(**build_resp_kwargs(max_ct, include_tools=False))
                            used_path = "responses"
                        else:
                            response = self.client.chat.completions.create(**build_chat_kwargs(max_ct))
                            used_path = "chat"
                else:
                    response = self.client.chat.completions.create(**build_chat_kwargs(max_ct))
                    used_path = "chat"
            except Exception as e:
                raise RuntimeError(f"OpenAI request failed: {e}")

            content, finish_reason = None, None
            try:
                if self.use_code_interpreter and used_path == "responses":
                    try:
                        content = getattr(response, "output_text", None)
                        finish_reason = getattr(response, "finish_reason", None)
                    except Exception:
                        content = None
                    if not content:
                        outputs = getattr(response, "output", None) or getattr(response, "outputs", None) or []
                        texts = []
                        for out in outputs:
                            parts = getattr(out, "content", None) or getattr(out, "contents", None) or []
                            for p in parts:
                                if hasattr(p, "text"):
                                    t_obj = getattr(p, "text")
                                    if isinstance(t_obj, str):
                                        texts.append(t_obj)
                                    else:
                                        try:
                                            texts.append(getattr(t_obj, "value", ""))
                                        except Exception:
                                            pass
                        if texts:
                            content = "\\n".join([t for t in texts if t])
                else:
                    choice0 = response.choices[0]
                    finish_reason = getattr(choice0, "finish_reason", None)
                    content = choice0.message.content
            except Exception as e:
                raise RuntimeError(f"Failed to read OpenAI response: {e}")

            if (not content or str(content).strip() == "") and finish_reason == "length" and attempt_index == 0:
                continue

            try:
                model_out = json.loads(content)
            except Exception as e:
                if finish_reason == "length" and attempt_index == 0:
                    continue
                raise RuntimeError(f"Failed to parse JSON: {e}\\n--- RAW ---\\n{content[:2000]}")

            fixed = self._snap_to_hint_boxes(model_out, hints, width, height)
            return fixed

        raise RuntimeError("Annotation failed after retries")

    def _build_prompt(self, width: int, height: int, hints: List[dict],
                      shard_index_map: List[List[int]], crop_tags: List[str], detail_level: str = "high") -> str:
        packed = []
        for idx, h in enumerate(hints):
            obj = {
                "id": int(h.get("id", idx+1)),
                "bbox": [int(v) for v in h["bbox"]],
                "point": [int(v) for v in h["point"]],
            }
            if idx < len(shard_index_map) and shard_index_map[idx]:
                obj["shard_indices"] = shard_index_map[idx]
            packed.append(obj)
        hints_text = json.dumps(packed, ensure_ascii=False)

        # Explain the interleaving crop tags
        crop_instruction = ""
        if crop_tags:
            crop_instruction = """
  <crop_images>
    After the main screenshot, each crop is preceded by a one-line marker:
      <crop id=K type=T>
    where K is the hint id in <hints> and T is one of:
      - tight        (tight bounding around the candidate)
      - padded       (symmetric padding for small controls)
      - dir-left     (directional padding to the LEFT to include nearby label)
      - dir-right    (directional padding to the RIGHT to include nearby label)
    Treat these crops as the PRIMARY evidence when present. If both tight and a directional
    crop exist for the same id, prefer the directional crop to read the visible label.
  </crop_images>"""

        template = Template("""<SYSTEM>
  You are a UI Grounding Labeler for ShowUI RL.

  GOAL:
    From the attached screenshot, select 1-$max_elems elements STRICTLY from <hints> and return precise groundings, use the main screenshot as the context.

  INPUT:
    - <img_size> gives WIDTH_PX, HEIGHT_PX.
    - <hints> is an array of candidates with keys:
        { "id": int, "bbox": [x1,y1,x2,y2], "point": [cx,cy], optional "shard_indices": [int,...] }
    - If a crop marker "<crop id=K type=T>" appears, the following image is a crop for hint id K.
      Use these crops as the PRIMARY evidence to identify the candidate and to craft its instruction.

  OUTPUT (JSON ONLY):
    {
      "img_size": [WIDTH_PX, HEIGHT_PX],
      "element": [
        {
          "instruction": string,              // concise, actionable; include visible label from the crop when present
          "bbox": [x1, y1, x2, y2],           // ABSOLUTE PIXELS
          "point": [cx, cy],                  // ABSOLUTE PIXELS
          "source_id": int,                   // chosen hint id
          // Optional contextual fields (include depending on <detail_level>):
          // "type": string            (e.g., button, link, input, checkbox, tab, icon)
          // "label": string           (exact visible text if any)
          // "description": string     (what it is / what it does)
          // "context": string         (nearby section/menu/contextual clue)
          // "state": string           (e.g., enabled/disabled, selected/unselected)
        }
      ]
    }

  STRICT RULES:
    1) Choose ONLY among <hints>. Do NOT invent boxes.
    2) The output bbox MUST EQUAL the chosen hint's bbox EXACTLY.
    3) Use the chosen hint's point; if it lies on an edge, move 1 px inward.
    4) When crops are provided for a hint, derive the instruction from the crop's visible content.
       Prefer directional crops (dir-left/dir-right) over tight when both exist.
    5) Avoid duplicates (IoU>0.5 or centers within 4 px => keep one).

  COORDINATES & ORDER:
    - Integers only; 0 <= x1 < x2 <= WIDTH_PX, 0 <= y1 < y2 <= HEIGHT_PX.
    - Point strictly inside bbox.
    - Sort outputs by center: top->bottom, then left->right.
    - Return VALID JSON only; no markdown or prose.
</SYSTEM>

<USER>
  <task>Produce ShowUI grounding JSON for the attached screenshot.</task>
  <img_size>[$width, $height]</img_size>
  <max_elements>$max_elems</max_elements>
  <hints>$hints</hints>$crop_instruction
  <detail_level>$detail_level</detail_level>
  <detail_guidance>
    - If detail_level = "low":
        • Keep each "instruction" short (≤ 10 words).
        • Do NOT include optional contextual fields.
    - If detail_level = "normal":
        • Keep instructions concise (≤ 14 words).
        • Include "label" (exact visible text) when obvious.
        • Include "type" when obvious.
    - If detail_level = "high":
        • Include richer context for each element.
        • Provide "type", "label" (if any), and a brief "description" (≤ 20 words) describing role/appearance.
        • Add "context" with nearby section/menu or group when useful.
        • Add "state" when visually apparent (selected/disabled/etc.).
        • Still follow all STRICT RULES about bbox/point equality with hints.
  </detail_guidance>
  <requirements>
    - 1 <= N <= $max_elems based on visual richness.
    - Prioritize actionable controls (buttons, inputs, tabs, menu items, toggles, icons with clear affordance).
    - When crops exist, read labels/icons strictly inside the crop for the instruction.
  </requirements>
  <return>JSON only. No markdown.</return>
</USER>""")
        max_elems = int(os.getenv("ANNOTATOR_MAX_INSTRUCTIONS", "5"))
        return template.substitute(width=width, height=height, hints=hints_text, crop_instruction=crop_instruction, max_elems=str(max_elems), detail_level=detail_level)

    def _build_crops(self, image_path: str, hints: List[dict]) -> Tuple[List[str], List[str], List[List[int]]]:
        """
        Returns:
          flat_crops_b64: List[str]         (all crops as base64 PNGs)
          crop_tags:      List[str]         (same length; each is a marker like "<crop id=7 type=dir-right>")
          shard_index_map: List[List[int]]  (per-hint 1-indexed positions of crops; kept for back-compat)
        """
        from io import BytesIO
        if not hints:
            return [], [], []

        n = len(hints)
        # settings
        max_shards = max(0, int(getattr(self, 'max_shards', 15)))
        topk = max(0, int(getattr(self, 'shard_topk', 6)))
        dual_topk = max(0, int(getattr(self, 'dual_crop_topk', 8)))
        pad = max(0, int(getattr(self, 'pad_px', 8)))
        tpad = max(0, int(getattr(self, 'text_pad_px', 48)))
        long_side = max(32, int(getattr(self, 'crop_long_side', 160)))

        # seed
        seed_raw = getattr(self, 'shard_seed_raw', 'auto')
        if str(seed_raw).lower() in ('', 'auto'):
            try:
                h = hashlib.md5()
                h.update(os.path.basename(image_path).encode('utf-8'))
                with open(image_path, 'rb') as f:
                    h.update(f.read(1024))
                for hint in hints:
                    h.update(json.dumps({"b": hint.get("bbox")}, sort_keys=True).encode('utf-8'))
                seed = int(h.hexdigest(), 16) % (2**31 - 1)
            except Exception:
                seed = None
        else:
            try:
                seed = int(seed_raw)
            except Exception:
                seed = int(hashlib.md5(str(seed_raw).encode('utf-8')).hexdigest(), 16) % (2**31 - 1)
        rng = random.Random(seed)

        # ambiguity scores
        neighbor_d = [(i, _min_neighbor_distance(hints, i)) for i in range(n)]
        neighbor_d.sort(key=lambda t: t[1])
        # deterministic head by confidence
        idx_by_conf = list(range(n))
        idx_by_conf.sort(key=lambda i: -float(hints[i].get("confidence", 0.5)))
        head = idx_by_conf[:min(topk, n)]
        # weighted sampling for the rest
        remaining = [i for i in range(n) if i not in head]
        d_vals = [d for _, d in neighbor_d if d < 1e9]
        d_min = min(d_vals) if d_vals else 0.0
        d_max = max(d_vals) if d_vals else 1.0
        weights = []
        for i in remaining:
            conf = float(hints[i].get("confidence", 0.5))
            if d_max > d_min:
                d = _min_neighbor_distance(hints, i)
                amb = 1.0 - ((d - d_min) / (d_max - d_min))
            else:
                amb = 0.0
            w = 0.8 * conf + 0.7 * amb + 0.1
            weights.append(max(w, 0.01))

        keep_idx = list(head)
        quota = max_shards - len(keep_idx)
        if quota > 0 and remaining:
            population = remaining[:]
            local_weights = weights[:]
            total_w = sum(local_weights)
            while population and quota > 0:
                if total_w <= 0:
                    j = rng.randrange(len(population))
                else:
                    r = rng.random() * total_w
                    acc = 0.0
                    j = 0
                    for j_idx, w in enumerate(local_weights):
                        acc += w
                        if acc >= r:
                            j = j_idx
                            break
                pick = population.pop(j)
                if j < len(local_weights):
                    total_w -= local_weights.pop(j)
                keep_idx.append(pick); quota -= 1

        # choose which hints get a second (directional/padded) crop
        second_crop_candidates = sorted(keep_idx, key=lambda i: _min_neighbor_distance(hints, i))[:min(dual_topk, len(keep_idx))]
        second_set = set(second_crop_candidates)

        flat_crops: List[str] = []
        crop_tags: List[str] = []
        index_map: List[List[int]] = [[] for _ in hints]

        with Image.open(image_path) as img:
            W, H = img.size
            shard_counter = 0
            keep_set = set(keep_idx[:max_shards])

            for i, h in enumerate(hints):
                if i not in keep_set:
                    continue

                x1, y1, x2, y2 = [int(v) for v in h["bbox"]]
                sid = int(h.get('id', i+1))

                # tight crop
                tight = img.crop((x1, y1, x2, y2))
                try:
                    w0, h0 = tight.size
                    if max(w0, h0) > long_side:
                        scale = long_side / float(max(w0, h0))
                        tight = tight.resize((max(1, int(w0*scale)), max(1, int(h0*scale))), Image.LANCZOS)
                except Exception:
                    pass
                buf_t = BytesIO(); tight.save(buf_t, format='PNG')
                flat_crops.append(base64.b64encode(buf_t.getvalue()).decode('utf-8'))
                shard_counter += 1
                crop_tags.append(f"<crop id={sid} type=tight>")
                index_map[i].append(shard_counter)

                # Decide second crop kind
                if i in second_set:
                    # If row-like, prefer directional toward likely text side
                    direction = None
                    if _is_row_like([x1,y1,x2,y2], W, H):
                        direction = 'right' if ( (x1 + x2)/2.0 ) < (W/2.0) else 'left'
                    # Build the second crop
                    if direction == 'right':
                        px1 = x1; py1 = max(0, y1 - pad)
                        px2 = min(W, x2 + tpad); py2 = min(H, y2 + pad)
                        tag = f"<crop id={sid} type=dir-right>"
                    elif direction == 'left':
                        px1 = max(0, x1 - tpad); py1 = max(0, y1 - pad)
                        px2 = x2; py2 = min(H, y2 + pad)
                        tag = f"<crop id={sid} type=dir-left>"
                    else:
                        px1 = max(0, x1 - pad); py1 = max(0, y1 - pad)
                        px2 = min(W, x2 + pad); py2 = min(H, y2 + pad)
                        tag = f"<crop id={sid} type=padded>"

                    padded = img.crop((px1, py1, px2, py2))
                    try:
                        w1, h1 = padded.size
                        if max(w1, h1) > long_side:
                            scale = long_side / float(max(w1, h1))
                            padded = padded.resize((max(1, int(w1*scale)), max(1, int(h1*scale))), Image.LANCZOS)
                    except Exception:
                        pass
                    buf_p = BytesIO(); padded.save(buf_p, format='PNG')
                    flat_crops.append(base64.b64encode(buf_p.getvalue()).decode('utf-8'))
                    shard_counter += 1
                    crop_tags.append(tag)
                    index_map[i].append(shard_counter)

        return flat_crops, crop_tags, index_map

    def _snap_to_hint_boxes(self, model_out: dict, hints: List[dict], width: int, height: int) -> dict:
        id2hint = {int(h.get("id", i+1)): h for i, h in enumerate(hints)}
        fixed = {"img_size": [width, height], "element": []}
        seen_ids: set = set()

        for el in (model_out or {}).get("element", []):
            sid = el.get("source_id")
            if sid is None:
                sid = self._match_best_hint_id(el.get("bbox"), hints)
            hint = id2hint.get(int(sid)) if sid is not None else None
            if not hint:
                continue
            if int(sid) in seen_ids:
                continue
            seen_ids.add(int(sid))

            x1, y1, x2, y2 = [int(v) for v in hint["bbox"]]
            cx, cy = hint.get("point", _center_of([x1, y1, x2, y2]))
            cx = _clamp(int(cx), x1 + 1, x2 - 1)
            cy = _clamp(int(cy), y1 + 1, y2 - 1)

            inst = (el.get("instruction", "") or "").strip()

            # Preserve any additional contextual fields provided by the model output
            merged = dict(el) if isinstance(el, dict) else {}
            merged["instruction"] = inst
            merged["bbox"] = [x1, y1, x2, y2]
            merged["point"] = [int(cx), int(cy)]
            merged["source_id"] = int(sid)

            fixed["element"].append(merged)

        # simple duplicate pruning
        pruned: List[dict] = []
        for e in fixed["element"]:
            drop = False
            for k in pruned:
                if _iou(e["bbox"], k["bbox"]) > 0.5:
                    drop = True; break
            if not drop:
                pruned.append(e)
        fixed["element"] = pruned
        return fixed

    def _match_best_hint_id(self, bbox: Any, hints: List[dict]) -> Optional[int]:
        if not isinstance(bbox, list) or len(bbox) != 4:
            return hints[0].get("id") if hints else None
        best_id, best_iou = None, -1.0
        for h in hints:
            iou = _iou(bbox, h["bbox"])
            if iou > best_iou:
                best_iou, best_id = iou, int(h.get("id", 0))
        return best_id

    @staticmethod
    def _mime_for_ext(ext: str) -> str:
        ext = (ext or '').lower()
        return {
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
            '.gif': 'image/gif', '.bmp': 'image/bmp', '.webp': 'image/webp'
        }.get(ext, 'image/jpeg')
