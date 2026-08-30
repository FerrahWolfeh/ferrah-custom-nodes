import base64
import io as pyio
import logging
import sys
from PIL import Image, ImageOps

logger = logging.getLogger("FerrahNodes")

# Priority list of codec candidates and option profiles to probe
_CODEC_CANDIDATES = [
    ("h264_nvenc", [{"preset": "p1", "rc": "vbr", "cq": "23"}, {"preset": "p1"}]),
    ("h264_vulkan", [{}]),
    ("h264_amf", [{"usage": "lowlatency"}, {"quality": "speed"}, {}]),
    ("h264_vaapi", [{}]),
    ("h264_qsv", [{}]),
    ("libx264", [{"preset": "ultrafast", "tune": "zerolatency", "crf": "26"}, {"preset": "ultrafast"}, {}]),
    ("libopenh264", [{}]),
]

_ACTIVE_CODEC = None
_ACTIVE_OPTIONS = None
_PROBED = False
_warned_error = False


def probe_best_encoder():
    """Probes the best working H.264/MP4 encoder in PyAV."""
    global _ACTIVE_CODEC, _ACTIVE_OPTIONS, _PROBED
    if _PROBED:
        return _ACTIVE_CODEC, _ACTIVE_OPTIONS

    _PROBED = True
    try:
        import av
    except ImportError:
        logger.warning("[FerrahNodes] PyAV (av) is not installed; hardware/MP4 preview disabled.")
        return None, None

    for codec_name, option_candidates in _CODEC_CANDIDATES:
        try:
            av.Codec(codec_name, "w")
        except Exception:
            continue

        for opts in option_candidates:
            buf = pyio.BytesIO()
            try:
                container = av.open(
                    buf,
                    mode="w",
                    format="mp4",
                    options={"movflags": "frag_keyframe+empty_moov+default_base_moof"},
                )
                stream = container.add_stream(codec_name, rate=12)
                stream.width = 128
                stream.height = 128
                stream.pix_fmt = "yuv420p"
                if opts:
                    stream.options = opts

                test_img = Image.new("RGB", (128, 128), (100, 150, 200))
                for pkt in stream.encode(av.VideoFrame.from_image(test_img)):
                    container.mux(pkt)
                for pkt in stream.encode():
                    container.mux(pkt)
                container.close()

                _ACTIVE_CODEC = codec_name
                _ACTIVE_OPTIONS = opts
                return _ACTIVE_CODEC, _ACTIVE_OPTIONS
            except Exception:
                continue

    return None, None


def _encode_mp4_universal(frames, fps, max_res):
    """
    Encodes animated video preview frames into fragmented MP4 (Base64)
    using the best probed codec (NVENC, AMF, Vulkan, VAAPI, or ultrafast libx264).
    """
    global _warned_error
    if not frames:
        return None, 0, 0

    codec, opts = probe_best_encoder()
    if not codec:
        return None, 0, 0

    try:
        import av
    except Exception:
        return None, 0, 0

    pil_frames = []
    for f in frames:
        pf = f if f.mode == "RGB" else f.convert("RGB")
        if max_res and max_res > 0 and (pf.width > max_res or pf.height > max_res):
            pf = ImageOps.contain(pf, (max_res, max_res), Image.LANCZOS)
        pil_frames.append(pf)

    w0, h0 = pil_frames[0].width, pil_frames[0].height
    out_w, out_h = w0 & ~1, h0 & ~1
    if (out_w, out_h) != (w0, h0):
        pil_frames = [pf.resize((out_w, out_h), Image.LANCZOS) for pf in pil_frames]

    if out_w < 16 or out_h < 16:
        return None, 0, 0

    buf = pyio.BytesIO()
    try:
        container = av.open(
            buf,
            mode="w",
            format="mp4",
            options={"movflags": "frag_keyframe+empty_moov+default_base_moof"},
        )
        stream = container.add_stream(codec, rate=int(max(1, fps)))
        stream.width = out_w
        stream.height = out_h
        stream.pix_fmt = "yuv420p"
        if opts:
            stream.options = opts

        for pf in pil_frames:
            for pkt in stream.encode(av.VideoFrame.from_image(pf)):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)
        container.close()

        return base64.b64encode(buf.getvalue()).decode("ascii"), out_w, out_h
    except Exception as e:
        if not _warned_error:
            _warned_error = True
            logger.warning(f"[FerrahNodes] MP4 encode with '{codec}' failed, falling back to WebP: {e}")
        return None, 0, 0


def patch_kjnodes_preview_override():
    """
    Discovers KJNodes's preview_override_node in sys.modules and patches it
    to use the universal MP4 encoder instead of strictly relying on h264_nvenc.
    """
    codec, _ = probe_best_encoder()
    if not codec:
        return False

    patched_modules = set()
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if hasattr(mod, "_encode_mp4_nvenc") and hasattr(mod, "ModelPreviewOverrideKJ"):
            if mod in patched_modules:
                continue

            # Update availability flags in kjnodes module
            setattr(mod, "_HAS_NVENC", True)
            csp_blocks = getattr(mod, "_csp_blocks_video", lambda: False)()
            setattr(mod, "_NVENC_AVAILABLE", not csp_blocks)
            setattr(mod, "_encode_mp4_nvenc", _encode_mp4_universal)
            patched_modules.add(mod)
            print(f"[FerrahNodes] Patched KJNodes ModelPreviewOverride with '{codec}' encoder ({mod_name})")

    return len(patched_modules) > 0
