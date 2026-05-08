import numpy as np
import io
import copy
from PIL import Image
import json
from datetime import datetime
import os
import folder_paths

def is_true(val):
    """Sanitize boolean inputs that might come as strings or ints from ComfyUI."""
    if isinstance(val, bool): return val
    if isinstance(val, str): return val.strip().lower() in ("true", "1", "yes", "on")
    if isinstance(val, int): return val != 0
    return False

def resolve_link(link_val):
    if isinstance(link_val, list) and len(link_val) == 2:
        return str(link_val[0])
    return None

def recurse_text_conditioning(prompt, node_id, visited=None):
    if visited is None: visited = set()
    if node_id in visited: return []
    visited.add(node_id)
    node = prompt.get(str(node_id))
    if not node: return []
    class_type = node.get("class_type", "")
    inputs = node.get("inputs", {})
    texts = []
    if "TextEncode" in class_type or "PCLazyTextEncode" in class_type:
        t = inputs.get("text")
        if isinstance(t, str) and t.strip(): texts.append(t.strip())
    if "ConditioningConcat" in class_type:
        for key in ("conditioning_to", "conditioning_from"):
            linked = resolve_link(inputs.get(key))
            if linked: texts.extend(recurse_text_conditioning(prompt, linked, visited))
    return texts

def find_generating_node(prompt, start_node_id):
    visited = set()
    queue = [str(start_node_id)]
    while queue:
        curr_id = queue.pop(0)
        if curr_id in visited: continue
        visited.add(curr_id)
        node = prompt.get(str(curr_id))
        if not node: continue
        class_type = node.get("class_type", "")
        if "KSampler" in class_type or "SamplerCustom" in class_type: return curr_id, node
        inputs = node.get("inputs", {})
        for k, v in inputs.items():
            if k in ("images", "samples", "image", "src_image", "pixels"):
                linked = resolve_link(v)
                if linked: queue.append(linked)
    return None, None

def extract_generation_data(prompt, unique_id):
    if not prompt or not unique_id: return {}
    _, sampler_node = find_generating_node(prompt, unique_id)
    if not sampler_node: return {}
    inputs = sampler_node.get("inputs", {})
    class_type = sampler_node.get("class_type", "")
    data = {}

    s = inputs.get("seed")
    data["seed"] = s if s is not None else inputs.get("noise_seed")
    data["steps"] = inputs.get("steps")
    data["cfg"] = inputs.get("cfg")
    data["denoise"] = inputs.get("denoise")
    data["sampler_name"] = inputs.get("sampler_name")
    data["scheduler"] = inputs.get("scheduler")

    if "SamplerCustom" in class_type:
        sampler_link = resolve_link(inputs.get("sampler"))
        if sampler_link:
            s_node = prompt.get(sampler_link)
            if s_node: data["sampler_name"] = s_node.get("inputs", {}).get("sampler_name")
        sigmas_link = resolve_link(inputs.get("sigmas"))
        if sigmas_link:
            sig_node = prompt.get(sigmas_link)
            if sig_node:
                sig_in = sig_node.get("inputs", {})
                data["scheduler"] = sig_in.get("scheduler")
                if data.get("steps") is None: data["steps"] = sig_in.get("steps")
                if data.get("denoise") is None: data["denoise"] = sig_in.get("denoise")

    model_link = resolve_link(inputs.get("model"))
    if model_link:
        q, v = [model_link], set()
        while q:
            mid = q.pop(0)
            if mid in v: continue
            v.add(mid)
            mnode = prompt.get(mid)
            if not mnode: continue
            mtype = mnode.get("class_type", "")
            minputs = mnode.get("inputs", {})
            if "CheckpointLoader" in mtype:
                data["model"] = minputs.get("ckpt_name")
                break
            elif "UNETLoader" in mtype:
                data["model"] = minputs.get("unet_name")
                break
            else:
                link = resolve_link(minputs.get("model"))
                if link: q.append(link)

    pos_link = resolve_link(inputs.get("positive"))
    if pos_link: data["positive_prompt"] = ", ".join(recurse_text_conditioning(prompt, pos_link))
    neg_link = resolve_link(inputs.get("negative"))
    if neg_link: data["negative_prompt"] = ", ".join(recurse_text_conditioning(prompt, neg_link))
    return data

def format_generation_data(data):
    parts = []
    if data.get("model"):               parts.append(f"Model: {data['model']}")
    if data.get("steps") is not None:   parts.append(f"Steps: {data['steps']}")
    if data.get("sampler_name"):        parts.append(f"Sampler: {data['sampler_name']}")
    if data.get("scheduler"):           parts.append(f"Scheduler: {data['scheduler']}")
    if data.get("cfg") is not None:     parts.append(f"CFG: {data['cfg']}")
    if data.get("seed") is not None:    parts.append(f"Seed: {data['seed']}")
    if data.get("denoise") is not None: parts.append(f"Denoise: {data['denoise']}")
    if data.get("positive_prompt"):     parts.append(f"Positive: {data['positive_prompt']}")
    if data.get("negative_prompt"):     parts.append(f"Negative: {data['negative_prompt']}")
    return "\n".join(parts)

def format_software_tag(data):
    parts = []
    if data.get("model"):           parts.append(f"Model: {data['model']}")
    if data.get("sampler_name"):    parts.append(f"Sampler: {data['sampler_name']}")
    if data.get("steps") is not None: parts.append(f"Steps: {data['steps']}")
    if data.get("cfg") is not None: parts.append(f"CFG: {data['cfg']}")
    if data.get("seed") is not None: parts.append(f"Seed: {data['seed']}")
    return " | ".join(parts) if parts else "ComfyUI"

def get_metadata_exif(img, prompt, extra_pnginfo=None, unique_id=None, embed_metadata=True, now=None):
    exif = img.getexif()
    exif_ifd = exif.get_ifd(0x8769)

    if now:
        now_str = now.strftime("%Y:%m:%d %H:%M:%S")
        offset_str = now.strftime("%z")
        if len(offset_str) == 5:
            offset_str = f"{offset_str[:3]}:{offset_str[3:]}"
        elif not offset_str:
            offset_str = "+00:00"

        exif[0x0132] = now_str
        exif_ifd[0x9003] = now_str
        exif_ifd[0x9004] = now_str
        exif_ifd[0x9010] = offset_str
        exif_ifd[0x9011] = offset_str
        exif_ifd[0x9012] = offset_str

    if embed_metadata:
        metadata = {}
        if prompt is not None:
            safe_prompt = copy.deepcopy(prompt)
            # Remove any potentially sensitive data from the prompt copy
            # (though we are moving them to config.json, it's good practice)
            if unique_id and unique_id in safe_prompt:
                inputs = safe_prompt[unique_id].get("inputs", {})
                for key in ["immich_api_key", "immich_url", "api_key", "url"]:
                    if key in inputs:
                        inputs[key] = "**HIDDEN**"
            metadata["prompt"] = safe_prompt

        if extra_pnginfo is not None:
            metadata.update(extra_pnginfo)

        prompt_json = metadata.get("prompt")
        if prompt_json:
            exif[0x010f] = "Prompt: " + json.dumps(prompt_json)

        workflow_json = metadata.get("workflow")
        if workflow_json:
            exif[0x010e] = "Workflow: " + json.dumps(workflow_json)

    try:
        gen_data = extract_generation_data(prompt, unique_id)
        if gen_data:
            user_comment_str = format_generation_data(gen_data)
            exif_ifd[0x9286] = b"UNICODE\x00" + user_comment_str.encode("utf-16-le", errors="replace")
            exif[0x0131] = format_software_tag(gen_data)
    except Exception as e:
        print(f"FerrahNodes: Could not extract generation metadata: {e}")

    return exif.tobytes()
