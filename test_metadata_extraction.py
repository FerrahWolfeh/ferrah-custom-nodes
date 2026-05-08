import json
import os
import traceback


# ---------------------------------------------------------------------------
# Core extraction logic (mirrors what will go into convert_to_avif.py)
# ---------------------------------------------------------------------------

def get_node_by_id(prompt, node_id):
    return prompt.get(str(node_id))


def resolve_node_link(link_val):
    """Return the source node id string if link_val is a [node_id, slot] pair."""
    if isinstance(link_val, list) and len(link_val) == 2:
        return str(link_val[0])
    return None


def recurse_text_conditioning(prompt, node_id, visited=None):
    """Recursively collect text strings from conditioning nodes."""
    if visited is None:
        visited = set()
    if node_id in visited:
        return []
    visited.add(node_id)

    node = get_node_by_id(prompt, node_id)
    if not node:
        return []

    class_type = node.get("class_type", "")
    inputs = node.get("inputs", {})
    texts = []

    # Any node that directly encodes text
    if "TextEncode" in class_type or "PCLazyTextEncode" in class_type:
        t = inputs.get("text")
        if isinstance(t, str) and t.strip():
            texts.append(t.strip())

    # ConditioningConcat chains multiple conditioning nodes
    if "ConditioningConcat" in class_type:
        for key in ("conditioning_to", "conditioning_from"):
            linked = resolve_node_link(inputs.get(key))
            if linked:
                texts.extend(recurse_text_conditioning(prompt, linked, visited))

    return texts


def find_generating_node(prompt, start_node_id):
    """BFS back from start_node_id following image-like edges to find a sampler."""
    visited = set()
    queue = [str(start_node_id)]

    while queue:
        curr_id = queue.pop(0)
        if curr_id in visited:
            continue
        visited.add(curr_id)

        node = get_node_by_id(prompt, curr_id)
        if not node:
            continue

        class_type = node.get("class_type", "")

        # Match both KSampler variants and SamplerCustom
        if "KSampler" in class_type or "SamplerCustom" in class_type:
            return curr_id, node

        inputs = node.get("inputs", {})
        for k, v in inputs.items():
            if k in ("images", "samples", "image", "src_image", "pixels"):
                linked = resolve_node_link(v)
                if linked:
                    queue.append(linked)

    return None, None


def extract_generation_data(prompt, unique_id):
    """
    Traverse the prompt graph starting from the ConvertToAvif node
    and extract human-readable generation parameters.

    Returns a dict with keys: seed, steps, cfg, denoise,
    sampler_name, scheduler, model, positive_prompt, negative_prompt.
    """
    data = {}

    sampler_id, sampler_node = find_generating_node(prompt, unique_id)
    if not sampler_node:
        return {}

    inputs = sampler_node.get("inputs", {})
    class_type = sampler_node.get("class_type", "")

    # --- Basic scalar params ---
    data["seed"] = inputs.get("seed") or inputs.get("noise_seed")
    data["steps"] = inputs.get("steps")
    data["cfg"] = inputs.get("cfg")
    data["denoise"] = inputs.get("denoise")
    data["sampler_name"] = inputs.get("sampler_name")
    data["scheduler"] = inputs.get("scheduler")

    # --- SamplerCustom: sampler and sigmas come from separate nodes ---
    if "SamplerCustom" in class_type:
        sampler_link = resolve_node_link(inputs.get("sampler"))
        if sampler_link:
            s_node = get_node_by_id(prompt, sampler_link)
            if s_node:
                data["sampler_name"] = s_node.get("inputs", {}).get("sampler_name")

        sigmas_link = resolve_node_link(inputs.get("sigmas"))
        if sigmas_link:
            sig_node = get_node_by_id(prompt, sigmas_link)
            if sig_node:
                sig_inputs = sig_node.get("inputs", {})
                data["scheduler"] = sig_inputs.get("scheduler")
                if data.get("steps") is None:
                    data["steps"] = sig_inputs.get("steps")
                if data.get("denoise") is None:
                    data["denoise"] = sig_inputs.get("denoise")

    # --- Model name: trace back through Lora loaders / sampling wrappers ---
    model_link = resolve_node_link(inputs.get("model"))
    if model_link:
        q = [model_link]
        v = set()
        while q:
            mid = q.pop(0)
            if mid in v:
                continue
            v.add(mid)
            mnode = get_node_by_id(prompt, mid)
            if not mnode:
                continue
            mtype = mnode.get("class_type", "")
            minputs = mnode.get("inputs", {})

            if "CheckpointLoader" in mtype:
                data["model"] = minputs.get("ckpt_name")
                break
            elif "UNETLoader" in mtype:
                data["model"] = minputs.get("unet_name")
                break
            else:
                # Pass-through nodes like Lora loaders or ModelSampling wrappers
                link = resolve_node_link(minputs.get("model"))
                if link:
                    q.append(link)

    # --- Prompts ---
    pos_link = resolve_node_link(inputs.get("positive"))
    if pos_link:
        texts = recurse_text_conditioning(prompt, pos_link)
        data["positive_prompt"] = ", ".join(texts)

    neg_link = resolve_node_link(inputs.get("negative"))
    if neg_link:
        texts = recurse_text_conditioning(prompt, neg_link)
        data["negative_prompt"] = ", ".join(texts)

    return data


def format_generation_data(data):
    """Format the extracted dict into a short human-readable string for UserComment."""
    parts = []
    if data.get("model"):
        parts.append(f"Model: {data['model']}")
    if data.get("steps") is not None:
        parts.append(f"Steps: {data['steps']}")
    if data.get("sampler_name"):
        parts.append(f"Sampler: {data['sampler_name']}")
    if data.get("scheduler"):
        parts.append(f"Scheduler: {data['scheduler']}")
    if data.get("cfg") is not None:
        parts.append(f"CFG: {data['cfg']}")
    if data.get("seed") is not None:
        parts.append(f"Seed: {data['seed']}")
    if data.get("denoise") is not None:
        parts.append(f"Denoise: {data['denoise']}")
    if data.get("positive_prompt"):
        parts.append(f"Positive: {data['positive_prompt']}")
    if data.get("negative_prompt"):
        parts.append(f"Negative: {data['negative_prompt']}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


BASE = "/mnt/SSD/AI/ComfyUI/custom_nodes/ferrah-custom-nodes/"
PASS = 0
FAIL = 0


def run_test(name, path, unique_id, expected):
    global PASS, FAIL
    print(f"\n{'='*50}")
    print(f"TEST: {name}")
    try:
        prompt = load_json(os.path.join(BASE, path))
        result = extract_generation_data(prompt, unique_id)
        print(json.dumps(result, indent=2))
        print("--- Formatted ---")
        print(format_generation_data(result))
        print("-----------------")
        for key, val in expected.items():
            actual = result.get(key)
            assert actual == val, f"  FAIL [{key}]: expected {val!r}, got {actual!r}"
            print(f"  PASS [{key}]: {actual!r}")
        PASS += 1
    except AssertionError as e:
        print(f"ASSERTION ERROR: {e}")
        FAIL += 1
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}")
        traceback.print_exc()
        FAIL += 1


run_test(
    "SDXL RouWei Immich Upload",
    "SDXL RouWei Immich Upload.json",
    unique_id="48",
    expected={
        # HiRes fix sampler (node 17) is the one feeding into the image chain
        "seed": 854079388497582,
        "steps": 30,
        "sampler_name": "euler_ancestral",
        "scheduler": "simple",
        "cfg": 4,
        "model": "waiSHUFFLENOOB_vPred04.safetensors",
    },
)

run_test(
    "Anima Simple Custom Workflow (via PreviewImage node)",
    "Anima simple custom workflow.json",
    unique_id="65",
    expected={
        "seed": 661354905635813,
        "steps": 30,
        "sampler_name": "sa_solver_pece",
        "scheduler": "simple",
        "cfg": 4,
        "model": "anima-preview.safetensors",
    },
)

run_test(
    "Anima Immich Upload Standard Flow",
    "Anima Immich upload - standard flow.json",
    unique_id="48",
    expected={
        "seed": 309794055581996,
        "steps": 30,  # node 16 (SamplerCustom) -> sigmas -> node 12 (30 steps)
        "sampler_name": "dpmpp_3m_sde",
        "scheduler": "simple",
        "model": "anima-preview.safetensors",
    },
)

print(f"\n{'='*50}")
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL > 0:
    exit(1)
