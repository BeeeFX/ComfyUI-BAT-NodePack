"""
bat_preview — the node-face thumbnail the BAT loader shows after a run.

Writes one small PNG into ComfyUI's temp directory and returns the
``{filename, subfolder, type}`` descriptor the frontend's image widget expects.
Deliberately capped at 1024px: this is drawn a couple of hundred pixels wide on
a node, and shipping a full-resolution PNG of a 4K plate to the browser on every
execution is 10 MB+ for no visible difference.
"""

import os
import uuid
import logging
import tempfile
import numpy as np
import torch
from typing import List, Dict, Optional, Union, Tuple
from PIL import Image

try:
    import folder_paths
except ImportError:
    folder_paths = None

logger = logging.getLogger(__name__)


class PreviewGenerator:
    """Standardized preview generator for ComfyUI nodes"""
    
    def __init__(self, max_preview_size: int = 1024, enable_full_size: bool = False):
        self.max_preview_size = max_preview_size
        self.enable_full_size = enable_full_size
        self.temp_dir = folder_paths.get_temp_directory() if folder_paths else tempfile.gettempdir()
    
    def generate_preview_for_comfyui(self, image_tensor: torch.Tensor, 
                                   source_path: str = "", 
                                   is_sequence: bool = False,
                                   frame_index: int = 0,
                                   full_size: bool = False) -> Optional[List[Dict]]:
        """Generate preview image for ComfyUI"""
        try:
            if not self.temp_dir:
                return None
            
            # Use requested frame or first frame
            if is_sequence and image_tensor.shape[0] > 1:
                preview_tensor = image_tensor[min(frame_index, image_tensor.shape[0] - 1)]
            else:
                preview_tensor = image_tensor[0]
            
            # Convert tensor to PIL Image
            if preview_tensor.dim() == 3:  # [H, W, C]
                image_array = preview_tensor.cpu().numpy()
            else:
                return None
            
            image_array = np.clip(image_array, 0, 1).astype(np.float32)
            image_array = (image_array * 255).astype(np.uint8)
            
            if image_array.shape[2] == 1:
                pil_image = Image.fromarray(image_array.squeeze(2), mode='L')
            elif image_array.shape[2] == 3:
                pil_image = Image.fromarray(image_array, mode='RGB')
            elif image_array.shape[2] == 4:
                pil_image = Image.fromarray(image_array, mode='RGBA')
            else:
                pil_image = Image.fromarray(image_array[:, :, :3], mode='RGB')
            
            # Resize
            if not full_size and not self.enable_full_size:
                if pil_image.width > self.max_preview_size or pil_image.height > self.max_preview_size:
                    ratio = min(self.max_preview_size / pil_image.width, self.max_preview_size / pil_image.height)
                    pil_image = pil_image.resize((int(pil_image.width * ratio), int(pil_image.height * ratio)), Image.Resampling.LANCZOS)
            
            # Generate filename
            unique_id = uuid.uuid4().hex[:8]
            preview_filename = f"bat_preview_{unique_id}.png"
            preview_path = os.path.join(self.temp_dir, preview_filename)
            
            # This is a node-face thumbnail, so spend as little as possible on
            # it: optimize=True runs extra compression trials for a file that is
            # written once, read once and thrown away, and on a grainy 4K frame
            # that alone cost seconds per execution.
            pil_image.save(preview_path, format='PNG', compress_level=1)
            
            return [{"filename": preview_filename, "subfolder": "", "type": "temp"}]
            
        except Exception as e:
            logger.warning(f"Failed to generate preview: {e}")
            return None

# Shared instance. The 1024 cap is honoured by default; a caller that genuinely
# wants the original passes full_size=True per call.
preview_generator = PreviewGenerator(max_preview_size=1024, enable_full_size=False)

def generate_preview_for_comfyui(image_tensor: torch.Tensor, 
                               source_path: str = "", 
                               is_sequence: bool = False,
                               frame_index: int = 0,
                               full_size: bool = False) -> Optional[List[Dict]]:
    """Convenience function for generating ComfyUI previews"""
    return preview_generator.generate_preview_for_comfyui(image_tensor, source_path, is_sequence, frame_index, full_size)