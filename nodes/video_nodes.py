import av
from comfy_api.latest import io


def get_available_codecs() -> list[str]:
    """Dynamically probe available video encoders on the current system."""
    preferred_order = [
        "auto",
        "libsvtav1",
        "av1_vulkan",
        "av1_nvenc",
        "av1_amf",
        "av1_qsv",
        "av1",
        "libx264",
        "h264_vulkan",
        "h264_nvenc",
        "h264_amf",
        "h264_qsv",
        "h264",
        "libx265",
        "hevc_vulkan",
        "hevc_nvenc",
        "hevc_amf",
        "hevc_qsv",
        "hevc",
        "libvpx-vp9",
        "vp9",
    ]
    detected = set()
    for c in preferred_order:
        if c == "auto":
            continue
        try:
            codec = av.Codec(c, "w")
            if codec.type == "video":
                detected.add(c)
        except Exception:
            pass

    result = ["auto"]
    for c in preferred_order:
        if c in detected:
            result.append(c)

    return result


class VideoEncoderOptions(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        available_codecs = get_available_codecs()
        return io.Schema(
            node_id="VideoEncoderOptions",
            display_name="Video Encoder Options",
            category="FCN/video",
            description="Autodetects available system video codecs and configures rate control (CRF/QP), preset, and pixel format.",
            inputs=[
                io.Combo.Input("codec", options=available_codecs, default="auto", tooltip="The video codec / encoder to use."),
                io.Float.Input("crf", default=30.0, min=0.0, max=63.0, step=1.0, tooltip="Constant Rate Factor / Quantization Parameter. Higher CRF = smaller size. (Recommended: H.264 ~23, AV1 ~30-34, HEVC ~28)."),
                io.Combo.Input("preset", options=["auto", "ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow", "4", "5", "6", "7", "8"], default="auto", tooltip="Encoder preset / speed trade-off."),
                io.Combo.Input("pix_fmt", options=["auto", "yuv420p", "yuv420p10le", "yuva420p", "rgb24", "nv12"], default="auto", tooltip="Pixel format. yuv420p is most compatible."),
                io.Int.Input("bitrate_kbps", default=0, min=0, max=100000, step=100, tooltip="Optional target bitrate in kbps (0 for automatic CRF/QP mode)."),
            ],
            outputs=[
                io.Custom("VIDEO_ENCODER").Output("encoder_options", display_name="VIDEO_ENCODER"),
            ],
        )

    @classmethod
    def execute(cls, codec: str, crf: float, preset: str, pix_fmt: str, bitrate_kbps: int) -> io.NodeOutput:
        options = {
            "codec": codec,
            "crf": float(crf),
            "preset": preset,
            "pix_fmt": pix_fmt,
            "bitrate_kbps": int(bitrate_kbps),
        }
        return io.NodeOutput(options)


NODE_CLASS_MAPPINGS = {
    "VideoEncoderOptions": VideoEncoderOptions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoEncoderOptions": "Video Encoder Options",
}
