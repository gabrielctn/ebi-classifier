"""Gradio inference UI for EBI classification.

Simple interface for biologists to upload TIFF triplets, run predictions,
and download a .pop file with the sample IDs classified as EBI.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import timm
import gradio as gr
from safetensors.torch import load_file as load_safetensors

from dataset import (
    REQUIRED_CHANNELS_DEFAULT,
    RGB_CHANNEL_ORDER,
    TripletSample,
    _parse_channel_id,
    _parse_sample_id,
    _is_tiff,
    _read_ome_tiff_2d,
    _to_float01,
    letterbox_resize_pad,
)


# Increase file descriptor limit for large uploads
try:
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(10000, hard), hard))
except Exception:
    pass  # Windows or permission denied


# ---------------------------------------------------------------------------
# Channel mapping for biological types
# ---------------------------------------------------------------------------
# The model was trained with: R=Ch7 (erythroblasts), G=Ch11 (macrophages), B=Ch3 (neutrophils)
# This is the "training" order that the model expects: (erythroblasts, macrophages, neutrophils)
TRAINING_RGB_ORDER = (7, 11, 3)  # (R, G, B) = (erythroblasts, macrophages, neutrophils)

# Available channels in ImageStream data (1-12)
AVAILABLE_CHANNELS = [str(i) for i in range(1, 13)]

# Default channel assignments for the training data
DEFAULT_ERYTHROBLASTS_CHANNEL = "7"
DEFAULT_MACROPHAGES_CHANNEL = "11"
DEFAULT_NEUTROPHILS_CHANNEL = "3"


def build_user_channel_order(
    erythroblasts_ch: int,
    macrophages_ch: int,
    neutrophils_ch: int,
) -> Tuple[int, int, int]:
    """Build the RGB channel order from biological type assignments.
    
    Returns (R, G, B) channel numbers in the order needed to match training:
    R = erythroblasts, G = macrophages, B = neutrophils
    """
    # Training order is (erythroblasts=R, macrophages=G, neutrophils=B)
    return (erythroblasts_ch, macrophages_ch, neutrophils_ch)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[nn.Module, dict]:
    """Load a trained model from checkpoint."""
    if checkpoint_path.suffix == ".safetensors":
        config = json.loads(checkpoint_path.with_suffix(".json").read_text(encoding="utf-8"))
        state = load_safetensors(str(checkpoint_path), device="cpu")
    else:
        # Legacy training checkpoints must come from a trusted source.
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config = ckpt.get("config", {})
        state = ckpt["model_state"]

    model_name = config.get("model", "convnextv2_tiny.fcmae_ft_in22k_in1k")
    drop_path_rate = config.get("drop_path_rate", 0.2)

    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=2,
        drop_path_rate=drop_path_rate,
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model, config


def get_final_model_checkpoint() -> Optional[Path]:
    """Find the most recent final.pt checkpoint in runs/.
    
    After CV training, the final model (trained on all data) is saved as final.pt.
    This is the production-ready model that should be used for inference.
    """
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return None

    # Distribution packages use a non-pickle model format at a fixed path.
    release_model = runs_dir / "release/final.safetensors"
    if release_model.is_file():
        return release_model
    
    # Find all final.pt files (from final_model directories)
    final_pts = list(runs_dir.glob("**/final.pt"))
    if final_pts:
        # Sort by mtime descending (newest first)
        final_pts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return final_pts[0]
    
    # Fallback: look for best.pt if no final.pt exists (legacy runs)
    best_pts = list(runs_dir.glob("**/best.pt"))
    if best_pts:
        best_pts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return best_pts[0]
    
    # Last resort: any .pt file
    all_pts = list(runs_dir.glob("**/*.pt"))
    if all_pts:
        return Path(all_pts[0])
    
    return None


# ---------------------------------------------------------------------------
# Inference logic
# ---------------------------------------------------------------------------

def group_uploaded_files(
    file_paths: List[str],
    required_channels: Optional[set] = None,
) -> Dict[str, Dict[int, Path]]:
    """Group uploaded files by sample ID and channel.
    
    Args:
        file_paths: List of uploaded file paths.
        required_channels: Set of channel numbers to include. If None, uses default.
    """
    if required_channels is None:
        required_channels = set(REQUIRED_CHANNELS_DEFAULT)
    
    groups: Dict[str, Dict[int, Path]] = {}

    for fp in file_paths:
        p = Path(fp)
        if not _is_tiff(p):
            continue

        ch = _parse_channel_id(p.name)
        if ch is None:
            continue
        if ch not in required_channels:
            continue

        sid = _parse_sample_id(p.name)
        groups.setdefault(sid, {})[ch] = p

    return groups


def prepare_image(
    channels: Dict[int, Path],
    rgb_order: Tuple[int, ...] = RGB_CHANNEL_ORDER,
    image_size: int = 224,
    intensity_norm: str = "percentile",
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> torch.Tensor:
    """Load and preprocess a triplet into a model-ready tensor."""
    rgb_tensors = []
    for ch in rgb_order:
        arr = _read_ome_tiff_2d(channels[ch])
        t = _to_float01(
            arr,
            intensity_norm=intensity_norm,
            percentile_low=percentile_low,
            percentile_high=percentile_high,
        )
        rgb_tensors.append(t)

    img = torch.stack(rgb_tensors, dim=0)  # [3, H, W]
    img = letterbox_resize_pad(img, size=image_size, fill=0.0)
    return img


@torch.no_grad()
def run_inference(
    model: nn.Module,
    samples: List[Tuple[str, Dict[int, Path]]],
    config: dict,
    device: torch.device,
    threshold: float = 0.67,
    user_channel_order: Tuple[int, int, int] = TRAINING_RGB_ORDER,
) -> List[Tuple[str, float, bool, torch.Tensor]]:
    """Run inference on samples.
    
    Returns list of (sample_id, probability, is_ebi, image_tensor).
    
    Args:
        user_channel_order: Channel numbers in order (erythroblasts, macrophages, neutrophils).
                            Maps to (R, G, B) in the training order.
    """
    image_size = config.get("image_size", 224)
    intensity_norm = config.get("intensity_norm", "percentile")
    percentile_low = config.get("percentile_low", 1.0)
    percentile_high = config.get("percentile_high", 99.0)

    # user_channel_order provides channels in (erythroblasts, macrophages, neutrophils) order
    # which directly maps to (R, G, B) for the model

    results = []
    for sample_id, channels in samples:
        # Load image with user's channel assignments mapped to RGB
        img = prepare_image(
            channels,
            rgb_order=user_channel_order,
            image_size=image_size,
            intensity_norm=intensity_norm,
            percentile_low=percentile_low,
            percentile_high=percentile_high,
        )
        
        img_batch = img.unsqueeze(0).to(device)  # [1, 3, H, W]

        logits = model(img_batch)
        probs = torch.softmax(logits, dim=1)
        prob_ebi = float(probs[0, 1].cpu().item())
        is_ebi = prob_ebi >= threshold

        results.append((sample_id, prob_ebi, is_ebi, img.cpu()))

    return results


def tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    """Convert a [3, H, W] tensor in [0,1] to PIL Image."""
    arr = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def generate_pop_file(ebi_ids: List[str]) -> str:
    """Generate .pop file content with EBI sample IDs."""
    lines = [f"{sid}," for sid in ebi_ids]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

# Global model cache
_model_cache: Dict[str, Tuple[nn.Module, dict, torch.device]] = {}


def get_or_load_model() -> Tuple[nn.Module, dict, torch.device]:
    """Get cached model or load from the final model checkpoint."""
    ckpt = get_final_model_checkpoint()
    if ckpt is None:
        raise ValueError(
            "No model found. Please train a model first using: python train.py --cv-folds 5"
        )
    checkpoint_path = str(ckpt)

    if checkpoint_path in _model_cache:
        return _model_cache[checkpoint_path]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")

    model, config = load_model(Path(checkpoint_path), device)
    _model_cache[checkpoint_path] = (model, config, device)
    return model, config, device


def refresh_ui(data_state: Dict[str, List]) -> Tuple[Optional[str], str, List, List, Dict[str, List]]:
    """Regenerate all UI outputs from the current state."""
    ebi_list = data_state["ebi"]
    not_ebi_list = data_state["not_ebi"]

    # 1. Generate .pop file
    ebi_ids = [sid for sid, _, _ in ebi_list]
    pop_content = generate_pop_file(ebi_ids)
    
    # Save to unique temp directory with a fixed filename for clarity
    tmp_dir = tempfile.mkdtemp()
    pop_path = Path(tmp_dir) / "EBI_population_ids.pop"
    pop_path.write_text(pop_content)
    pop_file_path = str(pop_path)

    # 2. Summary
    summary = f"✅ Processed **{len(ebi_list) + len(not_ebi_list)}** samples: **{len(ebi_list)} EBI**, {len(not_ebi_list)} NOT_EBI"

    # 3. Galleries
    # Sort lists for display:
    # EBI: descending probability (best first)
    ebi_list.sort(key=lambda x: -x[1])
    # NOT_EBI: ascending probability (best "not ebi" first, i.e. lowest ebi probability)
    not_ebi_list.sort(key=lambda x: x[1])

    ebi_gallery = [
        (tensor_to_pil(img), f"{sid} (p={prob:.3f})")
        for sid, prob, img in ebi_list
    ]
    not_ebi_gallery = [
        (tensor_to_pil(img), f"{sid} (p={prob:.3f})")
        for sid, prob, img in not_ebi_list
    ]

    return pop_file_path, summary, ebi_gallery, not_ebi_gallery, data_state


def handle_selection(
    evt: gr.SelectData,
    section: str,
    data_state: Dict[str, List],
    selection_state: Optional[List[Dict]],
):
    """Handle click on gallery item (toggles selection in list)."""
    if not data_state:
        return "Prediction not ready.", []
    
    # Initialize selection list if needed
    if selection_state is None:
        selection_state = []
    
    # Identify which item was clicked
    lst = data_state.get(section, [])
    if evt.index >= len(lst):
        return "Error selecting item.", selection_state
    
    sid = lst[evt.index][0]
    new_selection = {"section": section, "index": evt.index, "sample_id": sid}
    
    # Toggle: if already selected, remove it; otherwise add it
    existing_idx = None
    for i, sel in enumerate(selection_state):
        if sel["sample_id"] == sid and sel["section"] == section:
            existing_idx = i
            break
    
    if existing_idx is not None:
        selection_state.pop(existing_idx)
    else:
        selection_state.append(new_selection)
    
    # Build info message
    if not selection_state:
        msg = "ℹ️ No items selected. Click images to select them."
    else:
        ebi_selected = [s["sample_id"] for s in selection_state if s["section"] == "ebi"]
        not_ebi_selected = [s["sample_id"] for s in selection_state if s["section"] == "not_ebi"]
        parts = []
        if ebi_selected:
            parts.append(f"**EBI:** {len(ebi_selected)} selected")
        if not_ebi_selected:
            parts.append(f"**NOT_EBI:** {len(not_ebi_selected)} selected")
        msg = " | ".join(parts)
    
    return msg, selection_state


def move_selected_to_not_ebi(data_state: Dict[str, List], selection_state: Optional[List[Dict]]):
    """Move all selected EBI items to NOT_EBI."""
    if not data_state or not selection_state:
        return (*refresh_ui(data_state if data_state else {"ebi": [], "not_ebi": []})[:-1], data_state, "No selection", [])

    # Filter selections to only EBI items
    ebi_selections = [s for s in selection_state if s["section"] == "ebi"]
    
    if not ebi_selections:
        return (*refresh_ui(data_state)[:-1], data_state, "⚠️ No EBI items selected. Select items from the EBI gallery first.", selection_state)
    
    # Collect sample IDs to move
    ids_to_move = {s["sample_id"] for s in ebi_selections}
    
    # Move items by ID (safer than index after sorting)
    moved_ids = []
    remaining_ebi = []
    for item in data_state["ebi"]:
        if item[0] in ids_to_move:
            data_state["not_ebi"].append(item)
            moved_ids.append(item[0])
        else:
            remaining_ebi.append(item)
    data_state["ebi"] = remaining_ebi
    
    if moved_ids:
        msg = f"✅ Moved **{len(moved_ids)}** item(s) to NOT_EBI."
    else:
        msg = "⚠️ Could not find selected items in EBI list."

    # Refresh UI and clear selection
    *ui_outs, _ = refresh_ui(data_state)
    return (*ui_outs, data_state, msg, [])


def move_selected_to_ebi(data_state: Dict[str, List], selection_state: Optional[List[Dict]]):
    """Move all selected NOT_EBI items to EBI."""
    if not data_state or not selection_state:
        return (*refresh_ui(data_state if data_state else {"ebi": [], "not_ebi": []})[:-1], data_state, "No selection", [])

    # Filter selections to only NOT_EBI items
    not_ebi_selections = [s for s in selection_state if s["section"] == "not_ebi"]
    
    if not not_ebi_selections:
        return (*refresh_ui(data_state)[:-1], data_state, "⚠️ No NOT_EBI items selected. Select items from the NOT_EBI gallery first.", selection_state)
    
    # Collect sample IDs to move
    ids_to_move = {s["sample_id"] for s in not_ebi_selections}
    
    # Move items by ID (safer than index after sorting)
    moved_ids = []
    remaining_not_ebi = []
    for item in data_state["not_ebi"]:
        if item[0] in ids_to_move:
            data_state["ebi"].append(item)
            moved_ids.append(item[0])
        else:
            remaining_not_ebi.append(item)
    data_state["not_ebi"] = remaining_not_ebi
    
    if moved_ids:
        msg = f"✅ Moved **{len(moved_ids)}** item(s) to EBI."
    else:
        msg = "⚠️ Could not find selected items in NOT_EBI list."

    *ui_outs, _ = refresh_ui(data_state)
    return (*ui_outs, data_state, msg, [])


def predict(
    files: List[str],
    erythroblasts_ch: str,
    macrophages_ch: str,
    neutrophils_ch: str,
) -> Tuple[Optional[str], str, List[Tuple[Image.Image, str]], List[Tuple[Image.Image, str]], Dict[str, List]]:
    """Main prediction function for Gradio.
    
    Returns (pop_file_path, summary_text, ebi_gallery, not_ebi_gallery, data_state).
    """
    if not files:
        return None, "⚠️ No files uploaded.", [], [], {}

    # Build channel order from biological type assignments
    user_channel_order = build_user_channel_order(
        int(erythroblasts_ch),
        int(macrophages_ch),
        int(neutrophils_ch),
    )

    # Group files by sample ID, filtering for the user's selected channels
    required_channels = {int(erythroblasts_ch), int(macrophages_ch), int(neutrophils_ch)}
    groups = group_uploaded_files(files, required_channels)
    if not groups:
        return None, "⚠️ No valid TIFF triplets found.", [], [], {}

    # Filter to complete triplets only
    required = required_channels
    complete_samples = [
        (sid, ch_map)
        for sid, ch_map in groups.items()
        if set(ch_map.keys()) == required
    ]

    if not complete_samples:
        incomplete = list(groups.keys())[:5]
        return None, f"⚠️ No complete triplets found. Incomplete samples: {incomplete}...", [], [], {}

    # Load model (always uses the final model)
    try:
        model, config, device = get_or_load_model()
    except ValueError as e:
        return None, f"❌ {e}", [], [], {}

    # Get threshold from config
    threshold = config.get("val_metrics", {}).get("threshold", 0.67)
    if not isinstance(threshold, float):
        threshold = 0.67

    # Run inference with user's channel mapping
    results = run_inference(
        model, complete_samples, config, device, threshold,
        user_channel_order=user_channel_order,
    )

    # Separate EBI and NOT_EBI to create initial state
    ebi_results = [(sid, prob, img) for sid, prob, is_ebi, img in results if is_ebi]
    not_ebi_results = [(sid, prob, img) for sid, prob, is_ebi, img in results if not is_ebi]

    data_state = {
        "ebi": ebi_results,
        "not_ebi": not_ebi_results
    }
    
    return refresh_ui(data_state)


def create_app() -> gr.Blocks:
    """Create the Gradio app."""
    
    # Check if a model is available
    model_path = get_final_model_checkpoint()
    model_status = f"✅ Model found: `{model_path.name}`" if model_path else "⚠️ No model found. Train one first."

    with gr.Blocks(
        title="EBI Classifier",
        analytics_enabled=False,
    ) as app:
        gr.Markdown(
            f"""
            # 🔬 EBI Classifier
            
            Upload your ImageStream TIFF triplets to classify Erythroblastic Islands (EBI).
            
            **Expected file format:** `{{sample_id}}_ChX.ome.tif` where X is the channel number (1-12)
            
            {model_status}
            """
        )

        # Helper state to store results for interaction
        data_state = gr.State({})
        # selection_state is a list of {"section": "ebi"/"not_ebi", "index": int, "sample_id": str}
        selection_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 🔬 Channel Mapping")
                gr.Markdown("*Specify which ImageStream channel contains each cell type in your files:*")
                with gr.Row():
                    erythroblasts_ch = gr.Dropdown(
                        label="Erythroblasts",
                        choices=AVAILABLE_CHANNELS,
                        value=DEFAULT_ERYTHROBLASTS_CHANNEL,
                        info="Channel for erythroblasts",
                    )
                    macrophages_ch = gr.Dropdown(
                        label="Macrophages",
                        choices=AVAILABLE_CHANNELS,
                        value=DEFAULT_MACROPHAGES_CHANNEL,
                        info="Channel for macrophages",
                    )
                    neutrophils_ch = gr.Dropdown(
                        label="Neutrophils",
                        choices=AVAILABLE_CHANNELS,
                        value=DEFAULT_NEUTROPHILS_CHANNEL,
                        info="Channel for neutrophils",
                    )
                file_input = gr.File(
                    label="Upload TIFF files",
                    height=200,
                    file_count="multiple",
                    type="filepath",
                )
                predict_btn = gr.Button("🚀 Run Classification", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_file = gr.File(label="Download EBI IDs (.pop)", interactive=False)
                summary_output = gr.Markdown(label="Summary")

        with gr.Accordion("📊 View Prediction Details", open=False):
            with gr.Row():
                selection_info = gr.Markdown("ℹ️ **Usage:** Click images to select/deselect them (multi-select supported), then use the buttons to move them.")
                btn_clear_selection = gr.Button("🗑️ Clear Selection", variant="secondary", size="sm")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### ✅ EBI (Positive)")
                    ebi_gallery = gr.Gallery(
                        label="EBI Predictions",
                        columns=4,
                        rows=3,
                        height="auto",
                        object_fit="contain",
                        show_label=False,
                        allow_preview=True, 
                    )
                    btn_move_to_not_ebi = gr.Button("⬇️ Move Selected to NOT_EBI", variant="secondary")
                
                with gr.Column():
                    gr.Markdown("### ❌ NOT_EBI (Negative)")
                    not_ebi_gallery = gr.Gallery(
                        label="NOT_EBI Predictions",
                        columns=4,
                        rows=3,
                        height="auto",
                        object_fit="contain",
                        show_label=False,
                        allow_preview=True,
                    )
                    btn_move_to_ebi = gr.Button("⬆️ Move Selected to EBI", variant="secondary")

        # Wire up the prediction
        predict_btn.click(
            fn=predict,
            inputs=[file_input, erythroblasts_ch, macrophages_ch, neutrophils_ch],
            outputs=[output_file, summary_output, ebi_gallery, not_ebi_gallery, data_state],
        )

        # Selection Handlers
        def select_ebi_wrapper(evt: gr.SelectData, data_st, sel_st):
            return handle_selection(evt, "ebi", data_st, sel_st)
            
        def select_not_ebi_wrapper(evt: gr.SelectData, data_st, sel_st):
            return handle_selection(evt, "not_ebi", data_st, sel_st)
        
        def clear_selection():
            return "ℹ️ Selection cleared. Click images to select them.", []

        ebi_gallery.select(
            fn=select_ebi_wrapper,
            inputs=[data_state, selection_state],
            outputs=[selection_info, selection_state],
        )

        not_ebi_gallery.select(
            fn=select_not_ebi_wrapper,
            inputs=[data_state, selection_state],
            outputs=[selection_info, selection_state],
        )
        
        btn_clear_selection.click(
            fn=clear_selection,
            inputs=[],
            outputs=[selection_info, selection_state],
        )

        # Move Handlers
        btn_move_to_not_ebi.click(
            fn=move_selected_to_not_ebi,
            inputs=[data_state, selection_state],
            outputs=[output_file, summary_output, ebi_gallery, not_ebi_gallery, data_state, selection_info, selection_state],
        )

        btn_move_to_ebi.click(
            fn=move_selected_to_ebi,
            inputs=[data_state, selection_state],
            outputs=[output_file, summary_output, ebi_gallery, not_ebi_gallery, data_state, selection_info, selection_state],
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(theme=gr.themes.Default(), inbrowser=True, server_name="127.0.0.1", share=False)
