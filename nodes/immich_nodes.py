import numpy as np
import io
from PIL import Image
import json
from datetime import datetime
import os
import tempfile
import wave
import subprocess
import torch
import folder_paths
from ..core.immich_api import immich_api
from ..core.utils import is_true, get_metadata_exif, extract_generation_data, format_generation_data, format_software_tag

# Check for AVIF support
try:
    import pillow_avif
    avif_supported = True
except ImportError:
    avif_supported = False

# Check for JXL support
try:
    import pillow_jxl
    jxl_supported = True
except ImportError:
    jxl_supported = False

class ImmichUpload:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
                "images": ("IMAGE",),
                "format": (["AVIF", "JXL"],),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "quality": ("INT", {"default": 80, "min": 1, "max": 100}),
                "save_locally": ("BOOLEAN", {"default": False}),
                "add_to_album": ("BOOLEAN", {"default": False}),
                "album_id": (["(none/loading)"], {"default": "(none/loading)"}),
                "embed_metadata": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "extra_albums": ("IMMICH_ALBUMS",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "status",)
    FUNCTION = "upload"
    OUTPUT_NODE = True
    CATEGORY = "FCN/immich"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(s, album_id, **kwargs):
        # Always return True to allow dynamically populated album IDs from the JS frontend
        return True

    def upload(self, enabled, images, format, filename_prefix, quality, save_locally, add_to_album, album_id, embed_metadata, prompt=None, extra_pnginfo=None, unique_id=None, extra_albums=None):
        enabled_bool = is_true(enabled)
        add_to_album_bool = is_true(add_to_album)
        save_locally_bool = is_true(save_locally)
        embed_metadata_bool = is_true(embed_metadata)

        if not enabled_bool:
            return (images, json.dumps({"status": "skipped"}),)

        if format == "AVIF" and not avif_supported:
            raise ImportError("Format selected is AVIF, but 'pillow-avif-plugin' is not installed.")
        if format == "JXL" and not jxl_supported:
            raise ImportError("Format selected is JXL, but 'pillow-jxl-plugin' is not installed.")

        # Gather and validate all album IDs (from the dropdown and chained albums input)
        valid_album_ids = []

        # 1. From the widget dropdown
        if album_id and album_id not in ("(none/loading)", "(none)"):
            if "(" in album_id and album_id.endswith(")"):
                extracted_id = album_id.split("(")[-1].rstrip(")")
                if len(extracted_id) > 10:
                    valid_album_ids.append(extracted_id)
            elif len(album_id) > 10:
                valid_album_ids.append(album_id)

        # 2. From the chained albums input
        if extra_albums is not None:
            if isinstance(extra_albums, str):
                if extra_albums not in valid_album_ids:
                    valid_album_ids.append(extra_albums)
            elif isinstance(extra_albums, list):
                for a in extra_albums:
                    if isinstance(a, str) and a not in valid_album_ids:
                        valid_album_ids.append(a)

        results = {"uploaded": [], "local": [], "errors": []}

        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            now = datetime.now().astimezone()

            kwargs = {"quality": quality}
            kwargs["exif"] = get_metadata_exif(img, prompt, extra_pnginfo, unique_id, embed_metadata_bool, now)

            if format == "AVIF":
                save_format = "AVIF"
                ext = "avif"
                mime_type = "image/avif"
                if quality == 100: kwargs["lossless"] = True
            elif format == "JXL":
                save_format = "JXL"
                ext = "jxl"
                mime_type = "image/jxl"
                if quality == 100: kwargs["lossless"] = True
            else:
                raise ValueError(f"Unsupported format: {format}")

            buffer = io.BytesIO()
            img.save(buffer, save_format, **kwargs)
            buffer.seek(0)

            file_basename = f"{filename_prefix}_{now.strftime('%Y%m%d_%H%M%S%f')}.{ext}"
            
            # Save locally if requested
            if save_locally_bool:
                try:
                    output_dir = folder_paths.get_output_directory()
                    full_output_path = os.path.join(output_dir, file_basename)
                    img.save(full_output_path, save_format, **kwargs)
                    results["local"].append(full_output_path)
                except Exception as e:
                    results["errors"].append(f"Local save error: {str(e)}")

            # Upload to Immich
            upload_result = immich_api.upload_asset(file_basename, buffer, mime_type, now)
            
            if "error" in upload_result:
                results["errors"].append(f"Upload error: {upload_result['error']}")
                # Fallback save if upload failed and not already saved
                if not save_locally_bool:
                    try:
                        output_dir = folder_paths.get_output_directory()
                        full_output_path = os.path.join(output_dir, file_basename)
                        img.save(full_output_path, save_format, **kwargs)
                        results["local"].append(full_output_path)
                    except Exception as e:
                        results["errors"].append(f"Fallback save error: {str(e)}")
            else:
                asset_id = upload_result.get('id')
                results["uploaded"].append(asset_id)
                
                if add_to_album_bool and asset_id:
                    if valid_album_ids:
                        for val_album_id in valid_album_ids:
                            album_result = immich_api.add_to_album(val_album_id, asset_id)
                            if "error" in album_result:
                                results["errors"].append(f"Album add error ({val_album_id}): {album_result['error']}")
                    else:
                        results["errors"].append("Album add error: No valid album selected.")

        return (images, json.dumps(results),)


class ImmichVideoUpload:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
                "video": ("VIDEO",),
                "video_codec": (["h264", "hevc (h265)", "av1", "vp9"], {"default": "h264"}),
                "hw_accel": (["auto / cpu", "vaapi (AMD/Intel Linux)", "vulkan (Cross-platform GPU)", "nvenc (NVIDIA)", "amf (AMD AMF)"], {"default": "auto / cpu"}),
                "audio_codec": (["aac", "opus", "mp3", "flac", "none"], {"default": "aac"}),
                "format": (["mp4", "webm", "mkv", "mov"], {"default": "mp4"}),
                "crf": ("INT", {"default": 20, "min": 0, "max": 51}),
                "preset": (["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"], {"default": "medium"}),
                "filename_prefix": ("STRING", {"default": "ComfyUI_Video"}),
                "save_locally": ("BOOLEAN", {"default": False}),
                "add_to_album": ("BOOLEAN", {"default": False}),
                "album_id": (["(none/loading)"], {"default": "(none/loading)"}),
                "embed_metadata": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "extra_albums": ("IMMICH_ALBUMS",),
                "encoder_options": ("VIDEO_ENCODER",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("VIDEO", "STRING",)
    RETURN_NAMES = ("video", "status",)
    FUNCTION = "upload"
    OUTPUT_NODE = True
    CATEGORY = "FCN/immich"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(s, album_id, **kwargs):
        return True

    def _build_ffmpeg_cmd(self, video_codec, hw_accel, audio_codec, format_ext, crf, preset, w, h, fps, temp_wav_path, embed_metadata_bool, prompt, extra_pnginfo, unique_id, file_basename, now):
        vcodec_map_cpu = {
            "h264": "libx264",
            "hevc (h265)": "libx265",
            "av1": "libsvtav1",
            "vp9": "libvpx-vp9",
        }
        acodec_map = {
            "aac": "aac",
            "opus": "libopus",
            "mp3": "libmp3lame",
            "flac": "flac",
        }

        cmd = ["ffmpeg", "-y"]

        # Hardware initialization if selected
        if hw_accel == "vaapi (AMD/Intel Linux)":
            cmd.extend(["-init_hw_device", "vaapi=va:/dev/dri/renderD128", "-filter_hw_device", "va"])
        elif hw_accel == "vulkan (Cross-platform GPU)":
            cmd.extend(["-init_hw_device", "vulkan=vk", "-filter_hw_device", "vk"])

        cmd.extend([
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{w}x{h}",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "pipe:0",
        ])

        if temp_wav_path and os.path.exists(temp_wav_path):
            cmd.extend(["-i", temp_wav_path])

        if hw_accel == "vaapi (AMD/Intel Linux)":
            cmd.extend(["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=nv12,hwupload"])
            hw_codecs = {"h264": "h264_vaapi", "hevc (h265)": "hevc_vaapi", "av1": "av1_vaapi", "vp9": "vp9_vaapi"}
            cmd.extend(["-c:v", hw_codecs.get(video_codec, "h264_vaapi"), "-qp", str(crf)])
        elif hw_accel == "vulkan (Cross-platform GPU)":
            cmd.extend(["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=nv12,hwupload"])
            hw_codecs = {"h264": "h264_vulkan", "hevc (h265)": "hevc_vulkan", "av1": "av1_vulkan"}
            cmd.extend(["-c:v", hw_codecs.get(video_codec, "h264_vulkan"), "-qp", str(crf)])
        elif hw_accel == "nvenc (NVIDIA)":
            cmd.extend(["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"])
            hw_codecs = {"h264": "h264_nvenc", "hevc (h265)": "hevc_nvenc", "av1": "av1_nvenc"}
            cmd.extend(["-c:v", hw_codecs.get(video_codec, "h264_nvenc"), "-pix_fmt", "yuv420p", "-cq", str(crf)])
        elif hw_accel == "amf (AMD AMF)":
            cmd.extend(["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"])
            hw_codecs = {"h264": "h264_amf", "hevc (h265)": "hevc_amf", "av1": "av1_amf"}
            cmd.extend(["-c:v", hw_codecs.get(video_codec, "h264_amf"), "-pix_fmt", "yuv420p", "-qp_num", str(crf)])
        else:
            # Standard CPU or specific codec string
            cmd.extend(["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"])
            selected_vcodec = vcodec_map_cpu.get(video_codec, video_codec)
            cmd.extend(["-c:v", selected_vcodec])
            if selected_vcodec in ("libx264", "libx265"):
                cmd.extend(["-pix_fmt", "yuv420p"])
            cmd.extend(["-crf", str(crf)])
            if preset and preset != "auto":
                cmd.extend(["-preset", preset])

        if format_ext in ("mp4", "mov"):
            cmd.extend(["-movflags", "+use_metadata_tags+faststart"])
        else:
            cmd.extend(["-movflags", "+use_metadata_tags"])

        if temp_wav_path and os.path.exists(temp_wav_path) and audio_codec in acodec_map:
            cmd.extend(["-c:a", acodec_map[audio_codec]])
        else:
            cmd.extend(["-an"])

        # Timezone-aware metadata tags for Immich & FFmpeg container
        cmd.extend(["-metadata", f"creation_time={now.isoformat()}"])
        cmd.extend(["-metadata", f"title={file_basename}"])

        if embed_metadata_bool:
            if prompt is not None:
                prompt_str = json.dumps(prompt) if isinstance(prompt, (dict, list)) else str(prompt)
                cmd.extend(["-metadata", f"prompt={prompt_str}"])
            if extra_pnginfo is not None and isinstance(extra_pnginfo, dict):
                if "workflow" in extra_pnginfo:
                    wf_val = extra_pnginfo["workflow"]
                    wf_str = json.dumps(wf_val) if isinstance(wf_val, (dict, list)) else str(wf_val)
                    cmd.extend(["-metadata", f"workflow={wf_str}"])
                for k, v in extra_pnginfo.items():
                    if k != "workflow":
                        v_str = json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                        cmd.extend(["-metadata", f"{k}={v_str}"])
            try:
                gen_data = extract_generation_data(prompt, unique_id)
                comment_text = format_generation_data(gen_data) if gen_data else "ComfyUI Generated Video"
                software_text = format_software_tag(gen_data) if gen_data else "ComfyUI"
                cmd.extend(["-metadata", f"comment={comment_text}"])
                cmd.extend(["-metadata", f"description={comment_text}"])
                cmd.extend(["-metadata", f"software={software_text}"])
            except Exception as e:
                print(f"FerrahNodes: Error building metadata for video: {e}")

        return cmd

    def upload(self, enabled, video, video_codec, hw_accel, audio_codec, format, crf, preset, filename_prefix, save_locally, add_to_album, album_id, embed_metadata, prompt=None, extra_pnginfo=None, unique_id=None, extra_albums=None, encoder_options=None):
        if encoder_options is not None and isinstance(encoder_options, dict):
            if encoder_options.get("codec") and encoder_options["codec"] != "auto":
                video_codec = encoder_options["codec"]
            if encoder_options.get("crf") is not None:
                crf = int(encoder_options["crf"])
            if encoder_options.get("preset") and encoder_options["preset"] != "auto":
                preset = encoder_options["preset"]
        enabled_bool = is_true(enabled)
        add_to_album_bool = is_true(add_to_album)
        save_locally_bool = is_true(save_locally)
        embed_metadata_bool = is_true(embed_metadata)

        if not enabled_bool:
            return (video, json.dumps({"status": "skipped"}),)

        if video is None:
            raise ValueError("ImmichVideoUpload: 'video' input must be provided.")

        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except Exception:
            raise RuntimeError("ImmichVideoUpload: 'ffmpeg' executable was not found on system PATH.")

        # Extract components from standard ComfyUI VIDEO object
        images = None
        video_audio = None
        fps = 24.0

        if hasattr(video, "get_components"):
            try:
                comp = video.get_components()
                if comp is not None:
                    if hasattr(comp, "images") and comp.images is not None:
                        images = comp.images
                    if hasattr(comp, "audio") and comp.audio is not None:
                        video_audio = comp.audio
                    if hasattr(comp, "frame_rate") and comp.frame_rate is not None:
                        fps = float(comp.frame_rate)
            except Exception as e:
                print(f"FerrahNodes: Error extracting video components: {e}")

        if images is None:
            raise ValueError("ImmichVideoUpload: Unable to extract image frames from provided 'video' object.")

        valid_album_ids = []
        if album_id and album_id not in ("(none/loading)", "(none)"):
            if "(" in album_id and album_id.endswith(")"):
                extracted_id = album_id.split("(")[-1].rstrip(")")
                if len(extracted_id) > 10:
                    valid_album_ids.append(extracted_id)
            elif len(album_id) > 10:
                valid_album_ids.append(album_id)

        if extra_albums is not None:
            if isinstance(extra_albums, str):
                if extra_albums not in valid_album_ids:
                    valid_album_ids.append(extra_albums)
            elif isinstance(extra_albums, list):
                for a in extra_albums:
                    if isinstance(a, str) and a not in valid_album_ids:
                        valid_album_ids.append(a)

        results = {"uploaded": [], "local": [], "errors": []}

        if len(images.shape) == 3:
            images = images.unsqueeze(0)

        num_frames, h, w, _ = images.shape
        now = datetime.now().astimezone()
        ext = format.lower()
        file_basename = f"{filename_prefix}_{now.strftime('%Y%m%d_%H%M%S%f')}.{ext}"

        mime_types = {
            "mp4": "video/mp4",
            "webm": "video/webm",
            "mkv": "video/x-matroska",
            "mov": "video/quicktime",
        }
        mime_type = mime_types.get(ext, "video/mp4")

        output_dir = folder_paths.get_output_directory() if save_locally_bool else tempfile.gettempdir()
        full_output_path = os.path.join(output_dir, file_basename)

        # Audio preparation
        temp_wav_path = None
        if video_audio is not None and audio_codec != "none":
            try:
                waveform = video_audio.get("waveform")
                sample_rate = video_audio.get("sample_rate", 44100)
                if waveform is not None:
                    if len(waveform.shape) == 3:
                        waveform = waveform[0]
                    
                    if len(waveform.shape) == 2:
                        if waveform.shape[0] > waveform.shape[1]:
                            waveform = waveform.T
                        channels = waveform.shape[0]
                    else:
                        channels = 1
                        waveform = waveform.unsqueeze(0)

                    audio_int16 = (torch.clamp(waveform, -1.0, 1.0) * 32767.0).to(dtype=torch.int16, device="cpu").numpy()
                    audio_interleaved = audio_int16.T.flatten()

                    temp_wav_fd, temp_wav_path = tempfile.mkstemp(suffix=".wav")
                    os.close(temp_wav_fd)

                    with wave.open(temp_wav_path, "wb") as wf:
                        wf.setnchannels(channels)
                        wf.setsampwidth(2)
                        wf.setframerate(int(sample_rate))
                        wf.writeframes(audio_interleaved.tobytes())
            except Exception as e:
                results["errors"].append(f"Audio processing warning: {str(e)}")
                temp_wav_path = None

        def run_encoding(cmd_list):
            cmd_run = list(cmd_list)
            cmd_run.append(full_output_path)
            proc = subprocess.Popen(cmd_run, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for frame_idx in range(num_frames):
                frame_bytes = (torch.clamp(images[frame_idx], 0.0, 1.0) * 255.0).to(dtype=torch.uint8, device="cpu").numpy().tobytes()
                proc.stdin.write(frame_bytes)
            proc.stdin.close()
            _, stderr = proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"FFmpeg failed with code {proc.returncode}: {stderr.decode('utf-8', errors='replace')}")

        # Attempt encoding (with hardware acceleration fallback to CPU if needed)
        try:
            cmd = self._build_ffmpeg_cmd(video_codec, hw_accel, audio_codec, ext, crf, preset, w, h, fps, temp_wav_path, embed_metadata_bool, prompt, extra_pnginfo, unique_id, file_basename, now)
            try:
                run_encoding(cmd)
            except Exception as hw_err:
                if hw_accel != "auto / cpu":
                    print(f"FerrahNodes: Hardware acceleration '{hw_accel}' failed: {hw_err}. Falling back to software CPU encoding.")
                    cmd_cpu = self._build_ffmpeg_cmd(video_codec, "auto / cpu", audio_codec, ext, crf, preset, w, h, fps, temp_wav_path, embed_metadata_bool, prompt, extra_pnginfo, unique_id, file_basename, now)
                    run_encoding(cmd_cpu)
                else:
                    raise hw_err
        except Exception as e:
            if temp_wav_path and os.path.exists(temp_wav_path):
                try: os.remove(temp_wav_path)
                except Exception: pass
            raise RuntimeError(f"ImmichVideoUpload: Video encoding error: {str(e)}")

        if temp_wav_path and os.path.exists(temp_wav_path):
            try: os.remove(temp_wav_path)
            except Exception: pass

        if save_locally_bool:
            results["local"].append(full_output_path)

        # Upload to Immich
        try:
            with open(full_output_path, "rb") as video_file:
                upload_result = immich_api.upload_asset(file_basename, video_file, mime_type, now)
        except Exception as e:
            upload_result = {"error": str(e)}

        if "error" in upload_result:
            results["errors"].append(f"Upload error: {upload_result['error']}")
        else:
            asset_id = upload_result.get("id")
            results["uploaded"].append(asset_id)

            if add_to_album_bool and asset_id:
                if valid_album_ids:
                    for val_album_id in valid_album_ids:
                        album_result = immich_api.add_to_album(val_album_id, asset_id)
                        if "error" in album_result:
                            results["errors"].append(f"Album add error ({val_album_id}): {album_result['error']}")
                else:
                    results["errors"].append("Album add error: No valid album selected.")

        if not save_locally_bool and os.path.exists(full_output_path):
            try: os.remove(full_output_path)
            except Exception: pass

        return (video, json.dumps(results),)


class ImmichAlbum:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "album_id": (["(none/loading)"], {"default": "(none/loading)"}),
            },
            "optional": {
                "albums": ("IMMICH_ALBUMS",),
            }
        }

    RETURN_TYPES = ("IMMICH_ALBUMS",)
    RETURN_NAMES = ("albums",)
    FUNCTION = "select_album"
    CATEGORY = "FCN/immich"

    @classmethod
    def VALIDATE_INPUTS(s, album_id, **kwargs):
        # Always return True to allow dynamically populated album IDs from the JS frontend
        return True

    def select_album(self, album_id, albums=None):
        valid_album_ids = []

        # 1. Add any previously accumulated album IDs in the pipe
        if albums is not None:
            if isinstance(albums, str):
                valid_album_ids.append(albums)
            elif isinstance(albums, list):
                valid_album_ids.extend(albums)

        # 2. Add current album_id from dropdown if it is valid and not already in the list
        if album_id and album_id not in ("(none/loading)", "(none)"):
            extracted_id = None
            if "(" in album_id and album_id.endswith(")"):
                extracted_id = album_id.split("(")[-1].rstrip(")")
            elif len(album_id) > 10:
                extracted_id = album_id

            if extracted_id and extracted_id not in valid_album_ids:
                valid_album_ids.append(extracted_id)

        return (valid_album_ids,)


NODE_CLASS_MAPPINGS = {
    "immich_upload": ImmichUpload,
    "immich_video_upload": ImmichVideoUpload,
    "immich_album": ImmichAlbum
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "immich_upload": "Immich Upload (AVIF/JXL)",
    "immich_video_upload": "Immich Video Upload",
    "immich_album": "Immich Album Select / Chain"
}

