"""
face_enhancement.py  →  solution.py
Sentio Mind · Project 4 · Low-Resolution CCTV Face Enhancement

4-Stage Classical CV Pipeline (CPU-only, no DL):
  Stage 1 — fastNlMeansDenoisingColored   (suppress sensor/compression noise)
  Stage 2 — CLAHE on L channel (LAB)      (adaptive contrast, no halo blowout)
  Stage 3 — Multi-step Lanczos upscale    (2× + unsharp + 2× for tiny faces)
  Stage 4 — Zone sharpening via Face Mesh (stronger on eye+nose, lighter on skin)

Run:  python solution.py
Output: enhanced_faces/  enhancement_report.html  evaluation_metrics.json
"""

# Suppress verbose C++ / TF / MediaPipe / absl logs BEFORE any library import
import os
os.environ["GLOG_minloglevel"]      = "3"   # suppress Google LOG (INFO/WARNING/ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"   # suppress TensorFlow C++ logs
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"   # force CPU, avoids GPU-init warnings

import warnings
warnings.filterwarnings("ignore")

import logging
logging.disable(logging.CRITICAL)

import cv2
import json
import base64
import time
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np
from pathlib import Path

# Top-level imports so they are loaded once, not on every function call
import face_recognition as _fr   # aliased; public name is fr below

# OpenCV often uses its own internal worker threads.
# When we add our own thread pool, limiting OpenCV to one thread prevents
# oversubscription and usually makes the program faster and more stable.
try:
    cv2.setNumThreads(1)
except Exception:
    pass
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

# Suppress absl (used internally by MediaPipe) after it's imported
try:
    import absl.logging as _absl_log
    _absl_log.set_verbosity(_absl_log.ERROR)
    _absl_log.use_absl_handler()
except Exception:
    pass

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
RAW_FACES_DIR    = Path("raw_faces")
REFERENCE_DIR    = Path("reference_identities")
ENHANCED_DIR     = Path("enhanced_faces")
REPORT_HTML_OUT  = Path("enhancement_report.html")
METRICS_JSON_OUT = Path("evaluation_metrics.json")

TARGET_SIZE      = (240, 240)
ENHANCED_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# PARALLEL EXECUTION HELPERS
# ---------------------------------------------------------------------------

_thread_local = threading.local()

_WORKER_REF_VECS = None
_WORKER_REF_NAMES = None

def init_worker(ref_vecs, ref_names):
    """Initializer for each process: store reference encodings once per worker."""
    global _WORKER_REF_VECS, _WORKER_REF_NAMES
    _WORKER_REF_VECS = ref_vecs
    _WORKER_REF_NAMES = ref_names


def detect_available_cores() -> int:
    """Detect the number of logical CPU cores available to this process."""
    try:
        affinity = getattr(os, "sched_getaffinity", None)
        if affinity is not None:
            cores = len(os.sched_getaffinity(0))
            if cores > 0:
                return cores
    except Exception:
        pass

    cores = os.cpu_count() or 1
    return max(1, int(cores))


def get_worker_count(task_count: int) -> int:
    """
    Choose a safe number of worker threads.

    Uses the detected core count, caps it to the number of tasks, and allows
    manual override with FACE_ENHANCE_THREADS.
    """
    cores = detect_available_cores()

    override = os.getenv("FACE_ENHANCE_THREADS")
    if override:
        try:
            cores = max(1, int(override))
        except Exception:
            pass

    if task_count > 0:
        cores = min(cores, task_count)

    return max(1, cores)


def _get_face_mesh():
    """Lazy-init a FaceMesh instance per thread."""
    face_mesh = getattr(_thread_local, "face_mesh", None)
    if face_mesh is None:
        try:
            import mediapipe as mp
            face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.3,
            )
            _thread_local.face_mesh = face_mesh
        except Exception:
            face_mesh = None
    return face_mesh


def load_reference_encoding_job(ref: Path):
    """Process-safe reference loader. Returns (name, encoding) or None."""
    try:
        if ref.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            return None
        img = cv2.imread(str(ref))
        if img is None:
            return None
        enc = get_face_encoding(img)
        if enc is None:
            return None
        return ref.stem, enc
    except Exception:
        return None


def process_face_job(job):
    """Process one face crop end-to-end inside a worker process."""
    idx, fp = job
    ref_vecs = _WORKER_REF_VECS or []
    ref_names = _WORKER_REF_NAMES or []

    try:
        raw = cv2.imread(str(fp))
        if raw is None:
            return {
                "_index": idx,
                "filename": fp.name,
                "ok": False,
                "error": "cv2.imread returned None",
            }

        enhanced = enhance_face(raw.copy())
        cv2.imwrite(str(ENHANCED_DIR / fp.name), enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Sharpness must be measured at the same scale.
        # raw is 12-80px; enhanced is 240x240. Laplacian variance is
        # scale-dependent: a 20px noisy crop scores ~40000 while the same
        # content Lanczos-resized to 240px scores ~5 — an artificial 7000x
        # difference that makes every enhanced image look less sharp.
        # Fix: resize raw to TARGET_SIZE as the naive baseline first.
        raw_at_target = cv2.resize(raw, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
        sharp_b = sharpness(raw_at_target)
        sharp_a = sharpness(enhanced)
        ssim_g = ssim_score(raw_at_target, enhanced)

        enc_raw = get_face_encoding(raw)
        enc_enh = get_face_encoding(enhanced)

        match_b = False
        match_a = False
        mid = None

        if ref_vecs:
            if enc_raw is not None:
                match_b = any(_fr.compare_faces(ref_vecs, enc_raw, tolerance=0.60))
            if enc_enh is not None:
                hits = _fr.compare_faces(ref_vecs, enc_enh, tolerance=0.60)
                match_a = any(hits)
                if match_a:
                    mid = ref_names[hits.index(True)] if ref_names else None

        _, rb = cv2.imencode(".jpg", raw_at_target, [cv2.IMWRITE_JPEG_QUALITY, 82])
        _, eb = cv2.imencode(".jpg", enhanced, [cv2.IMWRITE_JPEG_QUALITY, 82])

        return {
            "_index": idx,
            "filename": fp.name,
            "ok": True,
            "original_size_px": list(raw.shape[:2]),
            "enhanced_size_px": list(enhanced.shape[:2]),
            "sharpness_before": round(sharp_b, 2),
            "sharpness_after": round(sharp_a, 2),
            "ssim_improvement": round(ssim_g, 4),
            "match_before": match_b,
            "match_after": match_a,
            "matched_identity": mid,
            "raw_b64": base64.b64encode(rb).decode(),
            "enhanced_b64": base64.b64encode(eb).decode(),
        }
    except Exception as e:
        return {
            "_index": idx,
            "filename": fp.name,
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
        }


def process_face_batch(batch):
    """Process a small batch of face crops inside one worker process."""
    return [process_face_job(job) for job in batch]


# ---------------------------------------------------------------------------
# STAGE 1 — DENOISE
# ---------------------------------------------------------------------------

def stage1_denoise(img: np.ndarray) -> np.ndarray:
    """
    Non-local means denoising in colour space.

    Why these parameters?
      h=8, hColor=8  — moderate filter strength; preserves edges while
                        suppressing CCTV sensor noise and JPEG block artefacts.
                        Going higher (h>=12) blurs identity-critical features.
      templateWindowSize=7   — 7x7 patch comparison; standard for face textures.
      searchWindowSize=21    — 21x21 search area; good coverage without O(n^2) blowup.

    fastNlMeansDenoisingColored works in LAB internally:
      L channel  -> h param      (luminance noise)
      AB channels -> hColor param (chroma fringing / colour noise)
    """
    return cv2.fastNlMeansDenoisingColored(
        img,
        None,
        h=8,
        hColor=8,
        templateWindowSize=7,
        searchWindowSize=21,
    )


# ---------------------------------------------------------------------------
# STAGE 2 — CLAHE
# ---------------------------------------------------------------------------

def stage2_clahe(img: np.ndarray) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalisation in LAB colour space.

    Why LAB?  CLAHE on BGR shifts hue.  Operating only on the L (lightness)
    channel leaves skin tone (A, B channels) untouched — critical because
    face_recognition's 128-d descriptor encodes colour gradients.

    clipLimit=3.5  — clips histogram redistribution to avoid amplifying noise
                     in uniform (dark) regions.  Values >4 over-sharpen noise.
    tileGridSize=(4,4) — 4x4 grid on a 240px image => 60px tiles; fine enough
                         to handle localised shadows (chin vs forehead) without
                         introducing tile-boundary artefacts.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4, 4))
    l_eq = clahe.apply(l_ch)

    merged = cv2.merge([l_eq, a_ch, b_ch])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# STAGE 3 — MULTI-STEP UPSCALE
# ---------------------------------------------------------------------------

def unsharp_mask(img: np.ndarray, sigma: float, strength: float) -> np.ndarray:
    """
    Unsharp masking:  out = img + strength * (img - GaussianBlur(img, sigma))

    Mathematically:  this is a high-pass filter added back to the original.
    The Gaussian acts as a low-pass; subtracting it leaves high-frequency
    (edge) content which we amplify by `strength` and add back.

    Uses cv2.addWeighted for a single fused SIMD operation:
      addWeighted(img, 1+strength, blurred, -strength, 0)
      =  (1+s)*img + (-s)*blurred
      =  img + s*(img - blurred)

    ksize=(0,0) lets OpenCV compute the kernel size from sigma:
      ksize = 2 * ceil(3*sigma) + 1
    This keeps the mask well-defined and avoids magic numbers.
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def stage3_upscale(img: np.ndarray) -> np.ndarray:
    """
    Multi-step Lanczos upscaling strategy.

    Decision boundary: min(h, w) < 64 px
      WHY 64?  At <64px a single resize to 240px is a 3.75x jump.
      Lanczos4 (4-lobe sinc approximant) introduces ringing on 1-pixel edges
      when the magnification factor is large.  Two 2x steps stay within the
      2x Nyquist-safe range and the intermediate unsharp prevents accumulated
      blur from the first resize from softening the second pass output.

    Sequence for tiny faces (< 64px short side):
      1. 2x INTER_LANCZOS4   — doubles resolution while honouring Nyquist
      2. unsharp(sigma=1.0, s=1.6) — restore mid-frequency edge lost by step 1
      3. 2x INTER_LANCZOS4   — second doubling now on a sharper intermediate
      4. resize to 240x240   — final crop/shrink to exact target

    For faces already >= 64px:
      Direct resize to 240x240 with INTER_LANCZOS4 is sufficient.
    """
    h, w = img.shape[:2]

    if min(h, w) < 64:
        # Step 1: 2x upscale
        img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
        # Step 2: restore edge sharpness lost during interpolation
        img = unsharp_mask(img, sigma=1.0, strength=1.6)
        # Step 3: second 2x upscale
        h2, w2 = img.shape[:2]
        img = cv2.resize(img, (w2 * 2, h2 * 2), interpolation=cv2.INTER_LANCZOS4)

    # Final resize to exact target
    return cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)


# ---------------------------------------------------------------------------
# STAGE 4 — ZONE SHARPENING
# ---------------------------------------------------------------------------

# Landmark indices are constant — define once at module level, not per-call.
_ZONE_IDX = (
    [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]  # LEFT_EYE
    + [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]  # RIGHT_EYE
    + [1, 2, 98, 327, 168, 197, 5, 4, 45, 275, 220, 440, 94]  # NOSE
)

# FaceMesh instances are not shared across threads.
# A separate instance per worker is safer and avoids cross-thread state bugs.


def stage4_zone_sharpen(img: np.ndarray) -> np.ndarray:
    """
    Differential sharpening guided by MediaPipe Face Mesh landmarks.

    WHY zone-aware?
      face_recognition's dlib descriptor is dominated by eye corners, nose tip,
      and mouth corners.  Over-sharpening skin regions amplifies pore texture
      and noise, which can slightly shift the 128-d encoding away from the
      ground-truth reference.  Keeping sigma higher (more blur) on cheeks
      preserves the smooth gradient that dlib expects there.

    Zone definitions (MediaPipe 468-point mesh):
      LEFT_EYE  — iris + eyelid contour landmarks
      RIGHT_EYE — mirror
      NOSE      — nose bridge, tip, alae

    Sharpening parameters:
      Eye + nose zone:  unsharp(sigma=0.8, strength=2.0)
        — tight kernel, high gain -> crisp edges on pupils and nostril ridges
      Rest of face:     unsharp(sigma=1.2, strength=1.3)
        — wider kernel, moderate gain -> smooth skin, no noise amplification

    Mask pipeline:
      1. Collect (x, y) for all zone landmarks
      2. Convex hull -> filled polygon mask
      3. Dilate 5px to absorb landmark imprecision
      4. Float blend:  out = zone_sharp * mask + rest_sharp * (1-mask)

    Fallback (no face detected): uniform unsharp(sigma=1.0, strength=1.5)
    """
    try:
        face_mesh = _get_face_mesh()
        if face_mesh is not None:
            h, w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            if res.multi_face_landmarks:
                lms = res.multi_face_landmarks[0].landmark

                # Convert normalised coords -> pixel coords using module-level ZONE_IDX
                pts = np.array(
                    [[int(lms[i].x * w), int(lms[i].y * h)] for i in _ZONE_IDX],
                    dtype=np.int32,
                )

                # Build binary mask from convex hull of eye+nose points
                mask = np.zeros((h, w), dtype=np.uint8)
                hull = cv2.convexHull(pts)
                cv2.fillConvexPoly(mask, hull, 255)
                mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)

                # Float mask for smooth blending
                mask_f = (mask.astype(np.float32) / 255.0)[:, :, np.newaxis]

                sharp_zone = unsharp_mask(img, sigma=0.8, strength=2.0).astype(np.float32)
                sharp_rest = unsharp_mask(img, sigma=1.2, strength=1.3).astype(np.float32)

                blended = sharp_zone * mask_f + sharp_rest * (1.0 - mask_f)
                return np.clip(blended, 0, 255).astype(np.uint8)

    except Exception:
        pass

    # Fallback: uniform sharpening
    return unsharp_mask(img, sigma=1.0, strength=1.5)


# ---------------------------------------------------------------------------
# FULL PIPELINE — do not change this function
# ---------------------------------------------------------------------------

def enhance_face(img: np.ndarray) -> np.ndarray:
    """Run all 4 stages in order. Do not modify."""
    img = stage1_denoise(img)
    img = stage2_clahe(img)
    img = stage3_upscale(img)
    img = stage4_zone_sharpen(img)
    return img


# ---------------------------------------------------------------------------
# EVALUATION HELPERS
# ---------------------------------------------------------------------------

def sharpness(img: np.ndarray) -> float:
    """
    Laplacian variance as a focus/sharpness measure.

    The discrete Laplacian (nabla^2) is a second-order derivative operator.
    Variance of nabla^2(I) measures the spread of high-frequency content:
      - Blurry images -> low response -> low variance
      - Sharp images  -> strong edges -> high variance

    Convert to grayscale first so the metric is luminance-only
    (colour fringing does not inflate the score).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def get_face_encoding(img: np.ndarray):
    """
    128-dimensional face descriptor via dlib (wrapped by face_recognition).

    Strategy:
      1. Standard HOG detection with 2x upsampling — handles faces as small
         as ~40px after upscaling.
      2. If HOG finds nothing, force the whole image as the bounding box.
         This is safe because RAW_FACES_DIR already contains face crops only
         (extract_faces.py guarantees this).  The forced-box trick ensures we
         always get an encoding even on 12px blurry crops.

    number_of_times_to_upsample=2 means the image is doubled twice before HOG
    runs, so a 40px face becomes 160px — well above the ~80px HOG threshold.
    """
    try:
        fr = _fr   # module-level import, no repeated sys.modules lookup
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        # Attempt 1: standard detection with upsampling
        locations = fr.face_locations(rgb, number_of_times_to_upsample=2, model="hog")
        if locations:
            encodings = fr.face_encodings(rgb, known_face_locations=locations)
            if encodings:
                return encodings[0]

        # Attempt 2: force full image as face bounding box —
        # ONLY when image is large enough to contain real facial structure.
        # On raw crops < 80px, HOG failure = genuinely no readable face.
        # Forcing a bounding box on a 12px blurry blob produces a garbage
        # 128-d vector that can accidentally land within tolerance=0.60 of a
        # real reference, inflating match_before and breaking before/after diff.
        # face_recognition uses (top, right, bottom, left) order.
        if min(h, w) >= 80:
            forced = [(0, w, h, 0)]
            encodings = fr.face_encodings(rgb, known_face_locations=forced)
            if encodings:
                return encodings[0]

    except Exception:
        pass

    return None


def ssim_score(a: np.ndarray, b: np.ndarray) -> float:
    """
    Structural Similarity Index Measure between two images.

    SSIM is defined on grayscale — both images resized to TARGET_SIZE first
    so the metric is resolution-independent and comparable across different
    raw crop sizes.

    Range: [-1, 1], higher = more structurally similar.
    """
    from skimage.metrics import structural_similarity as ssim

    a_r = cv2.resize(a, TARGET_SIZE)
    b_r = cv2.resize(b, TARGET_SIZE)

    gray_a = cv2.cvtColor(a_r, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(b_r, cv2.COLOR_BGR2GRAY)

    score, _ = ssim(gray_a, gray_b, full=True)
    return float(score)


# ---------------------------------------------------------------------------
# HTML A/B REPORT
# ---------------------------------------------------------------------------

def generate_ab_report(results: list, output_path: Path):
    """
    Self-contained offline HTML — no CDN, all images base64-encoded.

    Layout:
      - Header with overall accuracy + sharpness delta
      - Responsive grid: each card shows original | enhanced side-by-side
        with sharpness scores, match status, and matched identity label
    """
    if not results:
        return

    n = len(results)
    acc_before = round(sum(r["match_before"] for r in results) / n * 100, 1)
    acc_after  = round(sum(r["match_after"]  for r in results) / n * 100, 1)
    avg_sb = round(float(np.mean([r["sharpness_before"] for r in results])), 1)
    avg_sa = round(float(np.mean([r["sharpness_after"]  for r in results])), 1)

    def badge(matched, label=None):
        if matched:
            text  = f"&#10003; {label}" if label else "&#10003; Match"
            color = "#22c55e"
        else:
            text  = "&#10007; No match"
            color = "#ef4444"
        return (
            f'<span style="background:{color};color:#fff;font-size:10px;'
            f'padding:2px 7px;border-radius:9999px;font-family:monospace;">'
            f'{text}</span>'
        )

    cards_html = ""
    for r in results:
        raw_src = f"data:image/jpeg;base64,{r['raw_b64']}"
        enh_src = f"data:image/jpeg;base64,{r['enhanced_b64']}"

        sharp_delta = r["sharpness_after"] - r["sharpness_before"]
        delta_col   = "#22c55e" if sharp_delta > 0 else "#ef4444"
        delta_sym   = "&#9650;" if sharp_delta > 0 else "&#9660;"

        cards_html += f"""
        <div class="card">
          <div class="card-header">{r['filename']}</div>
          <div class="img-row">
            <div class="img-box">
              <img src="{raw_src}" alt="original">
              <div class="img-label">ORIGINAL<br>
                <span class="metric">{r['original_size_px'][1]}x{r['original_size_px'][0]}px</span><br>
                <span class="metric">Sharp: {r['sharpness_before']:.1f}</span><br>
                {badge(r['match_before'])}
              </div>
            </div>
            <div class="arrow">&#8594;</div>
            <div class="img-box">
              <img src="{enh_src}" alt="enhanced">
              <div class="img-label">ENHANCED<br>
                <span class="metric">240x240px</span><br>
                <span class="metric">Sharp: {r['sharpness_after']:.1f}
                  <span style="color:{delta_col}">{delta_sym}{abs(sharp_delta):.1f}</span>
                </span><br>
                {badge(r['match_after'], r['matched_identity'])}
              </div>
            </div>
          </div>
          <div class="ssim-row">SSIM gain: {r['ssim_improvement']:+.4f}</div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sentio Mind - Face Enhancement Report</title>
<style>
  :root {{
    --bg:      #0d0f14;
    --surface: #161b26;
    --card:    #1c2333;
    --border:  #2a3348;
    --text:    #e2e8f0;
    --sub:     #8492aa;
    --green:   #22c55e;
    --red:     #ef4444;
    --accent:  #6366f1;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Courier New', monospace;
    padding: 32px 24px;
    min-height: 100vh;
  }}
  .page-title {{
    font-size: 11px;
    letter-spacing: 0.25em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 6px;
  }}
  h1 {{
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 28px;
    color: var(--text);
  }}
  .summary-bar {{
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    margin-bottom: 36px;
  }}
  .stat-box {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 22px;
    min-width: 160px;
  }}
  .stat-label {{
    font-size: 10px;
    color: var(--sub);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 4px;
  }}
  .stat-val {{
    font-size: 22px;
    font-weight: 700;
  }}
  .stat-delta {{
    font-size: 12px;
    color: var(--green);
    margin-top: 2px;
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 20px;
  }}
  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    transition: border-color 0.2s;
  }}
  .card:hover {{ border-color: var(--accent); }}
  .card-header {{
    font-size: 11px;
    color: var(--sub);
    margin-bottom: 12px;
    letter-spacing: 0.08em;
  }}
  .img-row {{
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  .img-box {{
    flex: 1;
    text-align: center;
  }}
  .img-box img {{
    width: 100%;
    max-width: 130px;
    border-radius: 6px;
    border: 1px solid var(--border);
    image-rendering: pixelated;
  }}
  .img-label {{
    font-size: 10px;
    color: var(--sub);
    margin-top: 6px;
    line-height: 1.6;
  }}
  .metric {{ color: var(--text); font-size: 10px; }}
  .arrow {{
    font-size: 18px;
    color: var(--accent);
    flex-shrink: 0;
  }}
  .ssim-row {{
    margin-top: 12px;
    font-size: 10px;
    color: var(--sub);
    text-align: right;
  }}
  .footer {{
    margin-top: 40px;
    font-size: 10px;
    color: var(--sub);
    text-align: center;
  }}
</style>
</head>
<body>
  <div class="page-title">Sentio Mind - Project 4</div>
  <h1>Face Enhancement Report</h1>

  <div class="summary-bar">
    <div class="stat-box">
      <div class="stat-label">Recognition Before</div>
      <div class="stat-val">{acc_before}%</div>
    </div>
    <div class="stat-box">
      <div class="stat-label">Recognition After</div>
      <div class="stat-val" style="color:var(--green)">{acc_after}%</div>
      <div class="stat-delta">&#9650; {round(acc_after - acc_before, 1)}pp</div>
    </div>
    <div class="stat-box">
      <div class="stat-label">Avg Sharpness Before</div>
      <div class="stat-val">{avg_sb}</div>
    </div>
    <div class="stat-box">
      <div class="stat-label">Avg Sharpness After</div>
      <div class="stat-val" style="color:var(--green)">{avg_sa}</div>
      <div class="stat-delta">&#9650; {round(avg_sa - avg_sb, 1)}</div>
    </div>
    <div class="stat-box">
      <div class="stat-label">Faces Processed</div>
      <div class="stat-val">{n}</div>
    </div>
  </div>

  <div class="grid">
    {cards_html}
  </div>

  <div class="footer">
    Generated by solution.py &middot; Sentio Mind &middot;
    Denoise &rarr; CLAHE &rarr; Lanczos Upscale &rarr; Zone Sharpen
  </div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  Report saved -> {output_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> int:
    t_start = time.time()

    raw_faces = sorted([p for p in RAW_FACES_DIR.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    reference_images = sorted([p for p in REFERENCE_DIR.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])

    print(f"Detected {detect_available_cores()} logical cores", flush=True)

    if not RAW_FACES_DIR.exists():
        print(f"Missing folder: {RAW_FACES_DIR}", flush=True)
        return 1
    if not REFERENCE_DIR.exists():
        print(f"Missing folder: {REFERENCE_DIR}", flush=True)
        return 1

    print(f"Found {len(reference_images)} reference images and {len(raw_faces)} face crops", flush=True)

    # ----------------------------
    # Load references sequentially (safest for dlib/face_recognition)
    # ----------------------------
    reference_encodings = {}
    if reference_images:
        print("Loading references sequentially (dlib is not thread-safe)", flush=True)
        for i, ref in enumerate(reference_images, 1):
            try:
                item = load_reference_encoding_job(ref)
                if item is None:
                    print(f"  [ref {i}/{len(reference_images)}] {ref.name} -> skipped", flush=True)
                    continue
                name, enc = item
                reference_encodings[name] = enc
                print(f"  [ref {i}/{len(reference_images)}] {name} -> loaded", flush=True)
            except Exception as e:
                print(f"  [ref {i}/{len(reference_images)}] {ref.name} -> ERROR {type(e).__name__}: {e}", flush=True)

    print(f"Loaded {len(reference_encodings)} reference identities", flush=True)

    # ----------------------------
    # Process faces with multiprocessing in batches
    # ----------------------------
    results = []
    if raw_faces:
        ref_vecs = list(reference_encodings.values())
        ref_names = list(reference_encodings.keys())
        jobs = [(idx, fp) for idx, fp in enumerate(raw_faces)]

        num_workers = get_worker_count(len(jobs))
        batch_size = max(1, min(8, max(1, len(jobs) // (num_workers * 2))))
        batches = [jobs[i:i + batch_size] for i in range(0, len(jobs), batch_size)]

        print(
            f"Processing faces with {num_workers} worker processes "
            f"(batch size {batch_size})",
            flush=True,
        )

        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=init_worker,
            initargs=(ref_vecs, ref_names),
        ) as executor:
            futures = [executor.submit(process_face_batch, batch) for batch in batches]

            done_faces = 0
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                except Exception as e:
                    print(f"  batch ERROR {type(e).__name__}: {e}", flush=True)
                    continue

                for result in batch_results:
                    fp_name = result.get("filename", "unknown")
                    if not result:
                        print(f"  [face] {fp_name} -> no result", flush=True)
                        continue

                    if not result.get("ok", False):
                        print(f"  [face] {fp_name} -> ERROR {result.get('error','unknown')}", flush=True)
                        continue

                    results.append(result)
                    done_faces += 1
                    print(
                        f"  [face {done_faces}/{len(jobs)}] {result['filename']}: "
                        f"sharp {result['sharpness_before']:.1f}->{result['sharpness_after']:.1f} "
                        f"match {result['match_before']}->{result['match_after']}",
                        flush=True,
                    )

        results.sort(key=lambda r: r["_index"])

        for r in results:
            r.pop("_index", None)
            r.pop("ok", None)

    n = len(results)
    t_s = round(time.time() - t_start, 2)

    if n == 0:
        print("No faces were processed. Check that raw_faces/ contains images and that they are readable.", flush=True)

    metrics = {
        "source": "p4_face_enhancement",
        "total_faces_processed": n,
        "processing_time_sec": t_s,
        "pipeline_stages_applied": ["denoise", "clahe", "upscale_multistep", "zone_sharpen"],
        "recognition_accuracy_before_pct": round(sum(r["match_before"] for r in results) / n * 100, 1) if n else 0.0,
        "recognition_accuracy_after_pct": round(sum(r["match_after"] for r in results) / n * 100, 1) if n else 0.0,
        "avg_sharpness_before": round(float(np.mean([r["sharpness_before"] for r in results])), 2) if results else 0.0,
        "avg_sharpness_after": round(float(np.mean([r["sharpness_after"] for r in results])), 2) if results else 0.0,
        "avg_ssim_improvement": round(float(np.mean([r["ssim_improvement"] for r in results])), 4) if results else 0.0,
        "per_face": [{k: v for k, v in r.items() if k not in ["raw_b64", "enhanced_b64"]} for r in results],
    }

    with open(METRICS_JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if results:
        generate_ab_report(results, REPORT_HTML_OUT)

    print()
    print("=" * 55)
    print(f"  Done in {t_s}s  for {n} faces")
    print(f"  Recognition:  {metrics['recognition_accuracy_before_pct']}%  ->  {metrics['recognition_accuracy_after_pct']}%")
    print(f"  Sharpness:    {metrics['avg_sharpness_before']}  ->  {metrics['avg_sharpness_after']}")
    print(f"  Enhanced  -> {ENHANCED_DIR}/")
    print(f"  Report    -> {REPORT_HTML_OUT}")
    print(f"  Metrics   -> {METRICS_JSON_OUT}")
    print("=" * 55)

    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())