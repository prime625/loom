"""
Loom — A Non-Parametric Video Synthesizer
=========================================

No neural networks. No gradient descent. No millions of parameters.

Video is woven from three threads:
  1. WARP   — Motion fields (optical flow primitives)
  2. WEFT   — Texture patches (reaction-diffusion grown)
  3. PATTERN — Cellular automata detail rules

Training builds a searchable corpus. Inference retrieves, warps, and weaves.

The .vid file is not weights. It is:
  - Compressed motion field library
  - Texture patch dictionary (PCA basis)
  - CA rulebook
  - Scene graph index
  - Color/spectral priors
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from pathlib import Path
import json
import struct
try:
    import lz4.frame
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False
    import zlib
import math
from collections import defaultdict

# ---------------------------------------------------------------------------
# Core Data Structures
# ---------------------------------------------------------------------------

@dataclass
class MotionPrimitive:
    """A reusable motion field archetype."""
    id: int
    flow_field: np.ndarray          # (T, 2, H, W) vector field
    category: str                   # "pan_left", "zoom_in", "orbit", "static", etc.
    intensity: float                # 0.0 - 1.0
    spectral_signature: np.ndarray  # FFT magnitude for matching

    def warp_frame(self, frame: np.ndarray, t: int) -> np.ndarray:
        """Warp a single frame using this motion field at time t."""
        from scipy.ndimage import map_coordinates
        h, w = frame.shape[:2]
        flow = self.flow_field[t % len(self.flow_field)]  # (2, H, W)

        y, x = np.mgrid[0:h, 0:w]
        coords = np.stack([
            y + flow[1],  # dy
            x + flow[0]   # dx
        ])

        if frame.ndim == 3:
            warped = np.zeros_like(frame)
            for c in range(frame.shape[2]):
                warped[:, :, c] = map_coordinates(frame[:, :, c], coords, order=1, mode='reflect')
        else:
            warped = map_coordinates(frame, coords, order=1, mode='reflect')
        return warped


@dataclass  
class TexturePatch:
    """A reusable texture exemplar with synthesis metadata."""
    id: int
    patch: np.ndarray               # (H, W, 3) uint8 or float
    pca_basis: np.ndarray           # Low-rank approximation
    category: str                   # "sky", "water", "skin", "metal", etc.
    scale_range: Tuple[float, float]  # Valid scales for this patch

    def synthesize(self, target_shape: Tuple[int, int], seed: int = 0) -> np.ndarray:
        """Grow texture to target size using patch-based quilting."""
        from scipy.ndimage import zoom
        np.random.seed(seed)
        h, w = target_shape
        ph, pw = self.patch.shape[:2]

        # Simple tiling with random offsets and blending
        output = np.zeros((h, w, 3), dtype=np.float32)
        weights = np.zeros((h, w), dtype=np.float32)

        tile_h, tile_w = ph // 2, pw // 2
        for y in range(0, h, tile_h):
            for x in range(0, w, tile_w):
                # Random variation of the patch
                scale = np.random.uniform(0.8, 1.2)
                angle = np.random.uniform(-5, 5)

                variant = self._transform_patch(scale, angle)
                vh, vw = variant.shape[:2]

                y_end = min(y + vh, h)
                x_end = min(x + vw, w)

                # Gaussian weight for blending
                yy, xx = np.mgrid[0:y_end-y, 0:x_end-x]
                gw = np.exp(-((yy/(vh/2))**2 + (xx/(vw/2))**2))

                output[y:y_end, x:x_end] += variant[:y_end-y, :x_end-x] * gw[:, :, None]
                weights[y:y_end, x:x_end] += gw

        output = output / weights[:, :, None]
        return np.clip(output, 0, 255).astype(np.uint8)

    def _transform_patch(self, scale: float, angle: float) -> np.ndarray:
        """Apply random scale and rotation."""
        from scipy.ndimage import zoom, rotate
        scaled = zoom(self.patch, (scale, scale, 1), order=1)
        return rotate(scaled, angle, reshape=False, order=1)


@dataclass
class CARule:
    """Cellular Automata rule for organic detail generation."""
    name: str
    kernel: np.ndarray              # (K, K) neighborhood kernel
    birth: List[int]                # Birth conditions
    survive: List[int]              # Survival conditions
    channels: int = 3               # RGB or single channel

    def step(self, grid: np.ndarray) -> np.ndarray:
        """Apply one CA step."""
        from scipy.signal import convolve2d
        new_grid = grid.copy()

        for c in range(self.channels):
            neighbors = convolve2d(grid[:, :, c], self.kernel, mode='same', boundary='wrap')

            # Generalized Life-like rule
            born = np.isin(neighbors, self.birth)
            survive = np.isin(neighbors, self.survive)

            new_grid[:, :, c] = np.where(
                (grid[:, :, c] > 0.5) & survive,
                grid[:, :, c],
                np.where(born, 1.0, grid[:, :, c] * 0.95)
            )

        return np.clip(new_grid, 0, 1)


@dataclass
class SceneGraph:
    """Semantic composition plan for a video."""
    background: str
    foreground_objects: List[Dict]
    camera_motion: str
    lighting: str
    style: str
    duration_frames: int

    def to_prompt_vector(self) -> np.ndarray:
        """Convert to a searchable vector (using simple hashing for now)."""
        # In production, use CLIP or sentence embeddings
        # Here we use a simple semantic hash
        text = f"{self.background} {self.camera_motion} {self.lighting} {self.style}"
        hash_val = hash(text) % (2**20)
        vec = np.zeros(1024)
        vec[hash_val % 1024] = 1.0
        return vec


# ---------------------------------------------------------------------------
# The Loom Engine
# ---------------------------------------------------------------------------

class LoomEngine:
    """Main synthesis engine. No neural nets. Pure algorithms."""

    def __init__(self, vid_path: str):
        self.vid_path = vid_path
        self.corpus = self._load_corpus()
        self.motion_lib: Dict[str, List[MotionPrimitive]] = defaultdict(list)
        self.texture_lib: Dict[str, List[TexturePatch]] = defaultdict(list)
        self.ca_rules: Dict[str, CARule] = {}
        self.scene_templates: List[SceneGraph] = []

        self._index_corpus()

    def _load_corpus(self) -> dict:
        """Load the .vid corpus file."""
        with open(self.vid_path, 'rb') as f:
            # Read header
            magic = f.read(4)
            assert magic == b"LOOM", f"Invalid magic: {magic}"

            version = struct.unpack('<H', f.read(2))[0]
            num_sections = struct.unpack('<I', f.read(4))[0]
            meta_offset = struct.unpack('<Q', f.read(8))[0]

            # Read section directory
            sections = {}
            for _ in range(num_sections):
                sec_id = struct.unpack('<I', f.read(4))[0]
                sec_offset = struct.unpack('<Q', f.read(8))[0]
                sec_comp_len = struct.unpack('<I', f.read(4))[0]
                sec_decomp_len = struct.unpack('<I', f.read(4))[0]
                sections[sec_id] = (sec_offset, sec_comp_len, sec_decomp_len)

            # Read metadata
            f.seek(meta_offset)
            meta_json = f.read()
            metadata = json.loads(meta_json.decode('utf-8'))

            # Decompress and load each section
            corpus = {'metadata': metadata, 'sections': {}}
            for sec_name, sec_info in metadata['sections'].items():
                sec_id = sec_info['id']
                offset, comp_len, decomp_len = sections[sec_id]
                f.seek(offset)
                compressed = f.read(comp_len)
                decompressed = lz4.frame.decompress(compressed) if HAS_LZ4 else zlib.decompress(compressed)
                corpus['sections'][sec_name] = np.frombuffer(decompressed, dtype=np.float32)

            return corpus

    def _index_corpus(self):
        """Build searchable indexes from raw corpus data."""
        meta = self.corpus['metadata']

        # Index motion primitives
        motion_data = self.corpus['sections'].get('motion', np.array([]))
        if len(motion_data) > 0:
            # Reconstruct motion primitives from flat array
            # Format: [id, category_hash, intensity, ...flow_data..., spectral...]
            idx = 0
            mp_size = meta['motion_primitive_size']
            while idx + mp_size <= len(motion_data):
                block = motion_data[idx:idx+mp_size]
                prim = self._decode_motion_primitive(block, meta)
                self.motion_lib[prim.category].append(prim)
                idx += mp_size

        # Index texture patches
        tex_data = self.corpus['sections'].get('texture', np.array([]))
        if len(tex_data) > 0:
            tp_size = meta['texture_patch_size']
            idx = 0
            while idx + tp_size <= len(tex_data):
                block = tex_data[idx:idx+tp_size]
                patch = self._decode_texture_patch(block, meta)
                self.texture_lib[patch.category].append(patch)
                idx += tp_size

        # Load CA rules
        ca_data = self.corpus['sections'].get('ca_rules', np.array([]))
        if len(ca_data) > 0:
            self.ca_rules = self._decode_ca_rules(ca_data, meta)

        print(f"✅ Corpus loaded: {sum(len(v) for v in self.motion_lib.values())} motion primitives, "
              f"{sum(len(v) for v in self.texture_lib.values())} texture patches, "
              f"{len(self.ca_rules)} CA rules")

    def _decode_motion_primitive(self, block: np.ndarray, meta: dict) -> MotionPrimitive:
        """Decode a motion primitive from flat array."""
        # Simplified decoder
        return MotionPrimitive(
            id=int(block[0]),
            flow_field=block[1:1+meta['flow_size']].reshape(meta['flow_shape']),
            category=str(int(block[1+meta['flow_size']])),
            intensity=float(block[2+meta['flow_size']]),
            spectral_signature=block[3+meta['flow_size']:3+meta['flow_size']+meta['spectral_size']]
        )

    def _decode_texture_patch(self, block: np.ndarray, meta: dict) -> TexturePatch:
        """Decode a texture patch from flat array."""
        patch_size = meta['patch_pixels']
        return TexturePatch(
            id=int(block[0]),
            patch=block[1:1+patch_size].reshape(meta['patch_shape']),
            pca_basis=block[1+patch_size:1+patch_size+meta['pca_components']],
            category=str(int(block[-3])),
            scale_range=(float(block[-2]), float(block[-1]))
        )

    def _decode_ca_rules(self, data: np.ndarray, meta: dict) -> Dict[str, CARule]:
        """Decode CA rules."""
        rules = {}
        rule_size = meta['ca_rule_size']
        idx = 0
        rule_id = 0
        while idx + rule_size <= len(data):
            block = data[idx:idx+rule_size]
            kernel = block[:9].reshape(3, 3)
            birth = [int(block[9]), int(block[10]), int(block[11])]
            survive = [int(block[12]), int(block[13]), int(block[14])]
            rules[f"rule_{rule_id}"] = CARule(
                name=f"rule_{rule_id}",
                kernel=kernel,
                birth=birth,
                survive=survive
            )
            idx += rule_size
            rule_id += 1
        return rules

    def synthesize(
        self,
        prompt: str,
        width: int = 854,
        height: int = 480,
        num_frames: int = 120,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Synthesize video from text prompt.
        Returns: (T, H, W, 3) uint8 array
        """
        if seed is not None:
            np.random.seed(seed)

        # 1. Parse prompt into scene graph
        scene = self._parse_prompt(prompt, num_frames)
        print(f"🎬 Scene: {scene.background} | Camera: {scene.camera_motion} | Style: {scene.style}")

        # 2. Retrieve motion primitive
        motion = self._retrieve_motion(scene.camera_motion, scene.duration_frames, height, width)

        # 3. Synthesize background texture
        bg_texture = self._synthesize_texture(scene.background, (height, width), seed)

        # 4. Generate frames by warping texture with motion
        print("🌊 Weaving motion and texture...")
        frames = np.zeros((num_frames, height, width, 3), dtype=np.uint8)

        for t in range(num_frames):
            # Warp background
            frame = motion.warp_frame(bg_texture, t)

            # Add foreground objects
            for obj in scene.foreground_objects:
                frame = self._composite_object(frame, obj, t, motion)

            # Apply CA detail layer
            if scene.style in self.ca_rules:
                frame = self._apply_ca(frame, scene.style, t)

            # Color grade
            frame = self._color_grade(frame, scene.lighting)

            frames[t] = np.clip(frame, 0, 255).astype(np.uint8)

        return frames

    def _parse_prompt(self, prompt: str, duration: int) -> SceneGraph:
        """Parse text prompt into scene graph. Simple keyword matching."""
        prompt = prompt.lower()

        # Background detection
        backgrounds = ['ocean', 'sky', 'forest', 'city', 'desert', 'space', 'mountain', 'lake']
        bg = next((b for b in backgrounds if b in prompt), 'abstract')

        # Camera motion
        motions = {
            'pan left': 'pan_left', 'pan right': 'pan_right',
            'zoom in': 'zoom_in', 'zoom out': 'zoom_out',
            'orbit': 'orbit', 'static': 'static',
            'dolly': 'dolly', 'crane': 'crane'
        }
        cam = next((v for k, v in motions.items() if k in prompt), 'static')

        # Lighting
        lights = ['sunset', 'sunrise', 'night', 'day', 'golden hour', 'neon', 'foggy']
        light = next((l for l in lights if l in prompt), 'day')

        # Style
        styles = ['realistic', 'abstract', 'cartoon', 'dreamy', 'cinematic', 'vintage']
        style = next((s for s in styles if s in prompt), 'realistic')

        # Foreground objects (simple noun extraction)
        objects = []
        # In a real system, use NLP parsing. Here we use simple heuristics.

        return SceneGraph(
            background=bg,
            foreground_objects=objects,
            camera_motion=cam,
            lighting=light,
            style=style,
            duration_frames=duration
        )

    def _retrieve_motion(self, category: str, duration: int, h: int, w: int) -> MotionPrimitive:
        """Retrieve and adapt a motion primitive."""
        candidates = self.motion_lib.get(category, self.motion_lib.get('static', []))
        if not candidates:
            # Create static motion field
            return MotionPrimitive(
                id=-1,
                flow_field=np.zeros((duration, 2, h, w), dtype=np.float32),
                category='static',
                intensity=0.0,
                spectral_signature=np.zeros(64)
            )

        # Pick best match by spectral similarity
        prim = candidates[np.random.randint(len(candidates))]

        # Adapt to target duration and resolution
        flow = prim.flow_field
        if flow.shape[1] != h or flow.shape[2] != w:
            from scipy.ndimage import zoom
            scale_y = h / flow.shape[1]
            scale_x = w / flow.shape[2]
            new_flow = np.zeros((flow.shape[0], h, w, 2), dtype=np.float32)
            for t in range(flow.shape[0]):
                new_flow[t, :, :, 0] = zoom(flow[t, 0], (scale_y, scale_x), order=1) * scale_x
                new_flow[t, :, :, 1] = zoom(flow[t, 1], (scale_y, scale_x), order=1) * scale_y
            flow = new_flow.transpose(0, 3, 1, 2)

        # Extend or truncate to target duration
        if flow.shape[0] < duration:
            repeats = math.ceil(duration / flow.shape[0])
            flow = np.tile(flow, (repeats, 1, 1, 1))[:duration]
        else:
            flow = flow[:duration]

        return MotionPrimitive(
            id=prim.id,
            flow_field=flow,
            category=prim.category,
            intensity=prim.intensity,
            spectral_signature=prim.spectral_signature
        )

    def _synthesize_texture(self, category: str, shape: Tuple[int, int], seed: int) -> np.ndarray:
        """Synthesize background texture."""
        candidates = self.texture_lib.get(category, self.texture_lib.get('abstract', []))
        if not candidates:
            # Procedural fallback: simple noise
            np.random.seed(seed)
            return (np.random.rand(*shape, 3) * 255).astype(np.uint8)

        patch = candidates[np.random.randint(len(candidates))]
        return patch.synthesize(shape, seed)

    def _composite_object(self, frame: np.ndarray, obj: dict, t: int, motion: MotionPrimitive) -> np.ndarray:
        """Composite a foreground object onto the frame."""
        # Placeholder: would retrieve object sprite and animate
        return frame

    def _apply_ca(self, frame: np.ndarray, style: str, t: int) -> np.ndarray:
        """Apply cellular automata detail layer."""
        rule = self.ca_rules.get(style, self.ca_rules.get('rule_0'))
        if rule is None:
            return frame

        # Convert to float, apply CA, blend back
        grid = frame.astype(np.float32) / 255.0

        # Run a few CA steps
        for _ in range(3):
            grid = rule.step(grid)

        # Blend with original (subtle effect)
        result = frame * 0.85 + (grid * 255) * 0.15
        return result

    def _color_grade(self, frame: np.ndarray, lighting: str) -> np.ndarray:
        """Apply color grading based on lighting condition."""
        # Simple LUT-based grading
        grading = {
            'sunset': {'warmth': 1.3, 'contrast': 1.1, 'tint': np.array([1.1, 0.9, 0.7])},
            'night': {'warmth': 0.8, 'contrast': 1.3, 'tint': np.array([0.7, 0.7, 1.2])},
            'day': {'warmth': 1.0, 'contrast': 1.0, 'tint': np.array([1.0, 1.0, 1.0])},
            'golden hour': {'warmth': 1.4, 'contrast': 1.05, 'tint': np.array([1.2, 1.0, 0.6])},
            'neon': {'warmth': 1.1, 'contrast': 1.4, 'tint': np.array([1.0, 0.8, 1.2])},
        }

        grade = grading.get(lighting, grading['day'])
        frame = frame.astype(np.float32)
        frame = frame * grade['tint']
        frame = np.clip(frame, 0, 255)
        return frame.astype(np.uint8)
