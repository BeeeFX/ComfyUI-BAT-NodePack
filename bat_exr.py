"""
bat_exr — reading multi-layer OpenEXR for the BAT loader.

What an EXR actually contains, and why this file is the size it is: a render
from Nuke or an offline renderer is not one RGBA image. It is a flat list of
named channels (``diffuse.R``, ``N.X``, ``depth.Z``, ``crypto_object00.r``…)
that this module groups back into usable layers, deciding per group whether it
is RGB(A), a vector (XYZ), a single-channel mask, or a cryptomatte — plus two
rectangles (the data and display windows) that a render with overscan does not
keep equal. See conform_to_display_window() and _spec_windows().

Needs OpenImageIO. Without it OIIO_AVAILABLE is False and the loader falls back
to its 8-bit paths rather than failing to import.
"""

import os
import logging
import numpy as np
import torch
import json
from typing import Tuple, Dict, List, Optional, Union, Any

try:
    import OpenImageIO as oiio
    OIIO_AVAILABLE = True
except ImportError:
    OIIO_AVAILABLE = False

from .bat_preview import generate_preview_for_comfyui

logger = logging.getLogger(__name__)


def _own_tensor(array: np.ndarray) -> torch.Tensor:
    """`array` as a float32 tensor that owns its own memory.

    Slicing a channel out of a loaded EXR — `all_data[:, :, i]`, or
    `subimage[:, :, :3]` — gives a *strided view* over the whole decoded frame.
    torch.from_numpy keeps that view's base alive, so a single 8.8 MB alpha
    channel pinned the entire 35 MB RGBA frame, and on a 60-channel
    cryptomatte EXR one such view pinned ~150 MB. Across a 100-frame sequence
    that was the difference between holding the channels we return and holding
    every channel in the file.

    So: copy the slice out if it isn't already contiguous. The copy is the
    *cheaper* option — it's the size of the channel, not of the frame — and it
    lets the decoded frame be freed as soon as the loop moves on. Measured on
    40 frames of 2K RGBA: peak 4878 MB -> 3944 MB, and slightly faster.

    A contiguous input (the `np.stack` results, which are already fresh
    copies) passes through untouched, so this is never a second copy.
    """
    if not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array)
    return torch.from_numpy(array).float()


def _spec_windows(spec):
    """`spec`'s data and display windows as ``(x, y, width, height)`` pairs.

    EXR carries two rectangles in absolute pixel coordinates: the *display
    window* is the format (what Nuke calls the shot's resolution) and the
    *data window* is the bounding box the pixels were actually written for.
    OIIO exposes the data window as ``spec.x/y/width/height`` — which is what
    ``read_image`` hands back — and the display window as ``spec.full_*``.

    They are equal for a normal render. A Nuke render with overscan has a data
    window LARGER than the display window; anything that shrank its bbox
    (a Crop, a rotoshape, an element that moved partly off frame) has one
    smaller, and either can move and resize from frame to frame.
    """
    return ((spec.x, spec.y, spec.width, spec.height),
            (spec.full_x, spec.full_y, spec.full_width, spec.full_height))


def conform_to_display_window(array, data_window, display_window):
    """`array` (H, W, C over the data window) re-fitted to the display window.

    Overscan outside the format is cropped away and any part of the format the
    data window doesn't cover is filled with black — i.e. exactly the frame
    Nuke shows you at 1:1 with the viewer's bbox overlay off. Since every
    frame of a render shares one display window, this is what makes a sequence
    whose bbox breathes from frame to frame batchable at all.

    Black, rather than edge-replicate, because that is what the EXR spec says
    lives outside the data window and what OIIO/Nuke read there.

    Returns `array` itself when the two windows already match, so a normal
    render pays nothing.
    """
    dx, dy, dw, dh = data_window
    fx, fy, fw, fh = display_window
    if fw <= 0 or fh <= 0:
        # No usable display window (a file written without one). Nothing to
        # conform to — leave the pixels alone rather than guess.
        return array
    if (dx, dy, dw, dh) == (fx, fy, fw, fh):
        return array

    y0, y1 = max(dy, fy), min(dy + dh, fy + fh)
    x0, x1 = max(dx, fx), min(dx + dw, fx + fw)
    if y1 <= y0 or x1 <= x0:
        # The bbox is entirely outside the format. Legal (an element that has
        # travelled off screen) and the frame really is empty.
        return np.zeros((fh, fw, array.shape[2]), dtype=array.dtype)

    cropped = array[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    if (x0, y0, x1 - x0, y1 - y0) == (fx, fy, fw, fh):
        # Pure overscan: the data window covers the whole format, so the crop
        # IS the answer — returned as a view, deliberately.
        #
        # Materialising it here would allocate a second full frame beside the
        # one just decoded (331 MB on the 60-channel 1408x981 comp EXR in this
        # pack), and buy nothing: every consumer of this array pulls channels
        # out one at a time and copies them (np.stack, or _own_tensor, which
        # calls ascontiguousarray on any non-contiguous input). Those copies
        # are made from the view, so they are already the cropped size, and
        # the decoded frame behind it is released when the caller drops the
        # array — exactly as it is on the no-overscan path. The channels of an
        # EXR are interleaved, so every one of those copies was strided
        # already; cropping adds no new penalty.
        return cropped
    out = np.zeros((fh, fw, array.shape[2]), dtype=array.dtype)
    out[y0 - fy:y1 - fy, x0 - fx:x1 - fx] = cropped
    return out


class ExrProcessor:
    """Shared EXR processing functionality for loader nodes"""
    
    @staticmethod
    def check_oiio_availability():
        """Check if OpenImageIO is available"""
        if not OIIO_AVAILABLE:
            raise ImportError("OpenImageIO is required for EXR loading but not available")
    
    @staticmethod
    def scan_exr_metadata(image_path: str) -> Dict[str, Any]:
        """
        Scan the EXR file to extract metadata about available subimages without loading pixel data.
        Returns a dictionary of subimage information including names, channels, dimensions, etc.
        """
        ExrProcessor.check_oiio_availability()

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"EXR file not found: {image_path}")
            
        input_file = None
        try:
            input_file = oiio.ImageInput.open(image_path)
            if not input_file:
                raise IOError(f"Could not open {image_path}")
                
            metadata = {}
            subimages = []
            
            current_subimage = 0
            more_subimages = True
            
            while more_subimages:
                spec = input_file.spec()
                
                width = spec.width
                height = spec.height
                channels = spec.nchannels
                channel_names = [spec.channel_name(i) for i in range(channels)]
                
                subimage_name = "default"
                if "name" in spec.extra_attribs:
                    subimage_name = spec.getattribute("name")
                
                # width/height above are the DATA window's — the size
                # read_image returns. Carry both windows so a caller can tell
                # overscan from format without reopening the file.
                data_window, display_window = _spec_windows(spec)
                subimage_info = {
                    "index": current_subimage,
                    "name": subimage_name,
                    "width": width,
                    "height": height,
                    "channels": channels,
                    "channel_names": channel_names,
                    "data_window": list(data_window),
                    "display_window": list(display_window)
                }
                
                extra_attribs = {}
                for i in range(len(spec.extra_attribs)):
                    name = spec.extra_attribs[i].name
                    value = spec.extra_attribs[i].value
                    extra_attribs[name] = value
                
                subimage_info["extra_attributes"] = extra_attribs
                subimages.append(subimage_info)
                
                more_subimages = input_file.seek_subimage(current_subimage + 1, 0)
                current_subimage += 1
            
            metadata["subimages"] = subimages
            metadata["is_multipart"] = len(subimages) > 1
            metadata["subimage_count"] = len(subimages)
            metadata["file_path"] = image_path
            
            return metadata
            
        except Exception as e:
            logger.error(f"[Bat_Loader] error scanning EXR metadata from {image_path}: {e}")
            raise
            
        finally:
            if input_file:
                input_file.close()

    @staticmethod
    def read_windows(image_path: str):
        """``(data_window, display_window)`` of subimage 0, header only.

        Opens the file and reads no pixels, which is what lets the sequence
        loader decide what to do about overscan before it starts decoding —
        see bat_loader._plan_overscan.
        """
        ExrProcessor.check_oiio_availability()
        input_file = oiio.ImageInput.open(image_path)
        if not input_file:
            raise IOError(f"Could not open {image_path}")
        try:
            return _spec_windows(input_file.spec())
        finally:
            input_file.close()

    @staticmethod
    def load_all_data(image_path: str, conform: bool = False) -> Dict[int, np.ndarray]:
        """
        Load all pixel data from all subimages in the EXR file.
        Returns a dictionary mapping subimage index to numpy array of shape (height, width, channels).

        `conform=True` re-fits every subimage to its display window, so the
        arrays come back at the format size with the overscan gone — see
        conform_to_display_window.
        """
        input_file = None
        try:
            input_file = oiio.ImageInput.open(image_path)
            if not input_file:
                raise IOError(f"Could not open {image_path}")
            
            all_subimage_data = {}
            
            current_subimage = 0
            more_subimages = True
            
            while more_subimages:
                spec = input_file.spec()
                width = spec.width
                height = spec.height
                channels = spec.nchannels
                
                pixels = input_file.read_image()
                if pixels is None:
                    logger.warning(f"[Bat_Loader] failed to read subimage {current_subimage} "
                                   f"from {image_path}")
                else:
                    # asarray, NOT array: OIIO's Python read_image() already
                    # returns float32 (verified on OIIO 3.1.9, for both half and
                    # float EXRs), so np.array's default copy=True was cloning
                    # the entire decoded frame for nothing — 150 MB per frame on
                    # a 60-channel 2K cryptomatte EXR, times every decode thread
                    # in flight. asarray is a free re-label when the dtype
                    # already matches and still converts if a build ever hands
                    # back something else.
                    array = np.asarray(
                        pixels, dtype=np.float32).reshape(height, width, channels)
                    if conform:
                        array = conform_to_display_window(array, *_spec_windows(spec))
                    all_subimage_data[current_subimage] = array
                
                more_subimages = input_file.seek_subimage(current_subimage + 1, 0)
                current_subimage += 1
            
            return all_subimage_data
            
        finally:
            if input_file:
                input_file.close()

    @staticmethod
    def load_rgba_only(image_path: str, metadata: Dict[str, Any],
                       conform: bool = False):
        """Subimage 0's RGB(A) channels only, as ``(array, channel_names)``.

        The fast path for a plain plate load — see `want_layers` in
        process_exr_data. A comp EXR carries far more than the beauty pass: the
        cryptomatte file in this pack has 60 channels, so reading the whole
        frame to hand back 4 of them decoded (and held, per decode thread) ~15x
        the pixels the graph asked for.

        Only taken when R/G/B(/A) sit in one contiguous run, which is the
        normal layout — OIIO presents channels in file order and renderers put
        the beauty pass first. Anything else falls back to the full read, so a
        file with interleaved channels is slower but never wrong.

        Returns None when the fast path doesn't apply, and the caller reads the
        whole file as before.
        """
        subimages = metadata.get("subimages") or []
        if not subimages:
            return None
        names = subimages[0].get("channel_names") or []
        wanted = [n for n in ("R", "G", "B", "A") if n in names]
        if len(wanted) < 3:
            return None
        indices = [names.index(n) for n in wanted]
        lo, hi = min(indices), max(indices)
        if hi - lo + 1 != len(indices):
            return None  # not contiguous — not worth the index bookkeeping

        input_file = oiio.ImageInput.open(image_path)
        if not input_file:
            raise IOError(f"Could not open {image_path}")
        try:
            spec = input_file.spec()
            # read_image(subimage, miplevel, chbegin, chend, format); chend is
            # exclusive.
            pixels = input_file.read_image(0, 0, lo, hi + 1, oiio.FLOAT)
            if pixels is None:
                return None
            array = np.asarray(pixels, dtype=np.float32).reshape(
                spec.height, spec.width, hi + 1 - lo)
            if conform:
                array = conform_to_display_window(array, *_spec_windows(spec))
        finally:
            input_file.close()
        return array, names[lo:hi + 1]

    @staticmethod
    def is_cryptomatte_layer(group_name: str) -> bool:
        """Determine if a layer is a cryptomatte layer based on its name"""
        group_name_lower = group_name.lower()
        return (
            "cryptomatte" in group_name_lower or 
            group_name_lower.startswith("crypto") or
            any(crypto_key for crypto_key in ("cryptoasset", "cryptomaterial", "cryptoobject", "cryptoprimvar") 
                if crypto_key in group_name_lower) or
            any(part.lower().startswith("crypto") for part in group_name.split('.'))
        )

    @staticmethod
    def process_default_channels(all_data, channel_names, height, width, normalize):
        """Process default RGB and Alpha channels"""
        rgb_tensor = None
        if 'R' in channel_names and 'G' in channel_names and 'B' in channel_names:
            r_idx = channel_names.index('R')
            g_idx = channel_names.index('G')
            b_idx = channel_names.index('B')
            
            rgb_array = np.stack([
                all_data[:, :, r_idx],
                all_data[:, :, g_idx],
                all_data[:, :, b_idx]
            ], axis=2)
            
            rgb_tensor = _own_tensor(rgb_array)
            rgb_tensor = rgb_tensor.unsqueeze(0)  # [1, H, W, 3]
            
            if normalize:
                rgb_range = rgb_tensor.max() - rgb_tensor.min()
                if rgb_range > 0:
                    rgb_tensor = (rgb_tensor - rgb_tensor.min()) / rgb_range
        else:
            if all_data.shape[2] >= 3:
                rgb_array = all_data[:, :, :3]
            else:
                rgb_array = np.stack([all_data[:, :, 0]] * 3, axis=2)
            
            rgb_tensor = _own_tensor(rgb_array)
            rgb_tensor = rgb_tensor.unsqueeze(0)  # [1, H, W, 3]
            
            if normalize:
                rgb_range = rgb_tensor.max() - rgb_tensor.min()
                if rgb_range > 0:
                    rgb_tensor = (rgb_tensor - rgb_tensor.min()) / rgb_range
        
        alpha_tensor = None
        if 'A' in channel_names:
            a_idx = channel_names.index('A')
            alpha_array = all_data[:, :, a_idx]
            
            alpha_tensor = _own_tensor(alpha_array)
            alpha_tensor = alpha_tensor.unsqueeze(0)  # [1, H, W]
            
            if normalize:
                alpha_tensor = alpha_tensor.clamp(0, 1)
        else:
            alpha_tensor = torch.ones((1, height, width))
            
        return rgb_tensor, alpha_tensor

    @staticmethod
    def process_rgb_type_layer(group_name, r_suffix, g_suffix, b_suffix, a_suffix, 
                               channel_names, all_data, normalize, is_cryptomatte, 
                               layers_dict, cryptomatte_dict):
        """Process RGB/RGBA type layers with various naming conventions"""
        try:
            r_channel = f"{group_name}.{r_suffix}"
            g_channel = f"{group_name}.{g_suffix}"
            b_channel = f"{group_name}.{b_suffix}"
            a_channel = f"{group_name}.{a_suffix}"
            
            try:
                r_idx = channel_names.index(r_channel)
                g_idx = channel_names.index(g_channel)
                b_idx = channel_names.index(b_channel)
            except ValueError:
                logger.warning(f"[Bat_Loader] could not find RGB channels for {group_name}")
                return
            
            has_alpha = False
            a_idx = -1
            try:
                a_idx = channel_names.index(a_channel)
                has_alpha = True
            except ValueError:
                pass
            
            if is_cryptomatte:
                # For Cryptomatte, we often want all 4 channels intact (ID, Coverage pairs)
                # and we definitely do NOT want to normalize IDs.
                rgb_array = np.stack([
                    all_data[:, :, r_idx],
                    all_data[:, :, g_idx],
                    all_data[:, :, b_idx]
                ], axis=2)
                
                rgb_tensor_layer = _own_tensor(rgb_array)
                rgb_tensor_layer = rgb_tensor_layer.unsqueeze(0)  # [1, H, W, 3]

                if has_alpha:
                    alpha_array = all_data[:, :, a_idx]
                    alpha_tensor_layer = _own_tensor(alpha_array)
                    alpha_tensor_layer = alpha_tensor_layer.unsqueeze(0).unsqueeze(-1) # [1, H, W, 1]
                    # Combine into [1, H, W, 4] for cryptomatte processing
                    cryptomatte_dict[group_name] = torch.cat([rgb_tensor_layer, alpha_tensor_layer], dim=-1)
                else:
                    cryptomatte_dict[group_name] = rgb_tensor_layer
                return # Exit early for cryptomatte

            rgb_array = np.stack([
                all_data[:, :, r_idx],
                all_data[:, :, g_idx],
                all_data[:, :, b_idx]
            ], axis=2)
            
            rgb_tensor_layer = _own_tensor(rgb_array)
            rgb_tensor_layer = rgb_tensor_layer.unsqueeze(0)  # [1, H, W, 3]
            
            if normalize:
                rgb_range = rgb_tensor_layer.max() - rgb_tensor_layer.min()
                if rgb_range > 0:
                    rgb_tensor_layer = (rgb_tensor_layer - rgb_tensor_layer.min()) / rgb_range
            
            layers_dict[group_name] = rgb_tensor_layer
                
            if has_alpha:
                alpha_array = all_data[:, :, a_idx]
                
                alpha_tensor_layer = _own_tensor(alpha_array)
                alpha_tensor_layer = alpha_tensor_layer.unsqueeze(0)  # [1, H, W]
                
                if normalize:
                    alpha_tensor_layer = alpha_tensor_layer.clamp(0, 1)
                
                alpha_layer_name = f"{group_name}_alpha"
                layers_dict[alpha_layer_name] = alpha_tensor_layer
        except ValueError as e:
            logger.warning(f"[Bat_Loader] error processing RGB layer {group_name}: {e}")

    @staticmethod
    def process_xyz_type_layer(group_name, x_suffix, y_suffix, z_suffix, 
                               channel_names, all_data, normalize, layers_dict):
        """Process XYZ type vector layers with various naming conventions"""
        try:
            x_channel = f"{group_name}.{x_suffix}"
            y_channel = f"{group_name}.{y_suffix}"
            z_channel = f"{group_name}.{z_suffix}"
            
            try:
                x_idx = channel_names.index(x_channel)
                y_idx = channel_names.index(y_channel)
                z_idx = channel_names.index(z_channel)
            except ValueError:
                logger.warning(f"[Bat_Loader] could not find XYZ channels for {group_name}")
                return
            
            xyz_array = np.stack([
                all_data[:, :, x_idx],
                all_data[:, :, y_idx],
                all_data[:, :, z_idx]
            ], axis=2)
            
            xyz_tensor = _own_tensor(xyz_array)
            xyz_tensor = xyz_tensor.unsqueeze(0)  # [1, H, W, 3]
            
            if normalize:
                max_abs = xyz_tensor.abs().max()
                if max_abs > 0:
                    xyz_tensor = xyz_tensor / max_abs
            
            layers_dict[group_name] = xyz_tensor
        except ValueError as e:
            logger.warning(f"[Bat_Loader] error processing XYZ layer {group_name}: {e}")

    @staticmethod
    def process_single_channel(group_name, suffixes, group_indices, channel_names, all_data, normalize, layers_dict):
        """Process single channel data like depth maps or Z channels"""
        idx = -1
        if 'Z' in suffixes:
            z_channel = f"{group_name}.Z"
            z_channel_lower = f"{group_name}.z"
            try:
                idx = channel_names.index(z_channel)
            except ValueError:
                try:
                    idx = channel_names.index(z_channel_lower)
                except ValueError:
                    idx = group_indices[0]
        else:
            idx = group_indices[0]
        
        if idx >= 0:
            channel_array = all_data[:, :, idx]
            is_mask_type = any(keyword in group_name.lower() for keyword in ['depth', 'mask', 'matte', 'alpha', 'id', 'z'])
            
            if group_name == 'Z':
                is_mask_type = True
                logger.debug("[Bat_Loader] processing Z channel as a mask")
            
            if is_mask_type:
                mask_tensor = _own_tensor(channel_array).unsqueeze(0)  # [1, H, W]
                if normalize:
                    mask_range = mask_tensor.max() - mask_tensor.min()
                    if mask_range > 0:
                        mask_tensor = (mask_tensor - mask_tensor.min()) / mask_range
                layers_dict[group_name] = mask_tensor
            else:
                rgb_array = np.stack([channel_array] * 3, axis=2)
                channel_tensor = _own_tensor(rgb_array)
                channel_tensor = channel_tensor.unsqueeze(0)  # [1, H, W, 3]
                if normalize:
                    channel_range = channel_tensor.max() - channel_tensor.min()
                    if channel_range > 0:
                        channel_tensor = (channel_tensor - channel_tensor.min()) / channel_range
                layers_dict[group_name] = channel_tensor

    @staticmethod
    def process_multi_channel(group_name, group_indices, all_data, normalize, is_cryptomatte, layers_dict, cryptomatte_dict):
        """Process multi-channel data that doesn't fit standard patterns"""
        channels_to_use = min(3, len(group_indices))
        array_channels = []
        for i in range(channels_to_use):
            array_channels.append(all_data[:, :, group_indices[i]])
        while len(array_channels) < 3:
            array_channels.append(array_channels[-1])
        multi_array = np.stack(array_channels, axis=2)
        multi_tensor = _own_tensor(multi_array).unsqueeze(0)  # [1, H, W, 3]
        if normalize:
            multi_range = multi_tensor.max() - multi_tensor.min()
            if multi_range > 0:
                multi_tensor = (multi_tensor - multi_tensor.min()) / multi_range
        if is_cryptomatte:
            cryptomatte_dict[group_name] = multi_tensor
        else:
            layers_dict[group_name] = multi_tensor

    @staticmethod
    def process_layer_groups(channel_groups, cryptomatte_dict, metadata):
        """Process groups of related layers (like cryptomatte layer groups)"""
        for group_name, suffixes in channel_groups.items():
            if not group_name.endswith('_layer_group'):
                continue
            base_name = group_name[:-12]
            if not suffixes:
                continue
            is_crypto_layer_group = ExrProcessor.is_cryptomatte_layer(base_name)
            in_crypto_dict = any(group_part in cryptomatte_dict for group_part in suffixes)
            if 'layer_groups' not in metadata:
                metadata['layer_groups'] = {}
            metadata['layer_groups'][base_name] = suffixes
            if is_crypto_layer_group or in_crypto_dict:
                cryptomatte_dict[group_name] = [cryptomatte_dict.get(part, None) for part in suffixes]

    @staticmethod
    def store_layer_type_metadata(layers_dict, metadata):
        """Store information about layer types in metadata"""
        layer_types = {}
        for layer_name, tensor in layers_dict.items():
            if len(tensor.shape) >= 4 and tensor.shape[3] == 3:
                layer_types[layer_name] = "IMAGE"
            else:
                layer_types[layer_name] = "MASK"
        metadata["layer_types"] = layer_types

    @staticmethod
    def create_processed_layer_names(layers_dict, cryptomatte_dict):
        """Create a sorted list of processed layer names"""
        processed_layer_names = list(layers_dict.keys())
        for crypto_name in cryptomatte_dict.keys():
            processed_layer_names.append(f"crypto:{crypto_name}")
        processed_layer_names.sort()
        return processed_layer_names

    @staticmethod
    def get_channel_groups(channel_names: List[str]) -> Dict[str, List[str]]:
        """Group channel names by their prefix"""
        groups = {}
        layer_group_prefixes = set()
        for channel in channel_names:
            if '.' in channel:
                parts = channel.split('.')
                prefix = '.'.join(parts[:-1]) if len(parts) > 2 else parts[0]
                suffix = parts[-1]
                base_prefix = prefix
                for i in range(10):
                    if prefix.endswith(f"{i:02d}"):
                        base_prefix = prefix[:-2]
                        layer_group_prefixes.add(base_prefix)
                        break
                if prefix not in groups:
                    groups[prefix] = []
                groups[prefix].append(suffix)
            else:
                if channel not in groups:
                    groups[channel] = []
                groups[channel].append(None)
        
        if all(c in channel_names for c in 'RGB'):
            groups['RGB'] = ['R', 'G', 'B']
        if all(c in channel_names for c in 'XYZ'):
            groups['XYZ'] = ['X', 'Y', 'Z']
        
        depth_channels = [c for c in channel_names if c in ('Z', 'zDepth', 'zDepth1') or (('depth' in c.lower() or 'z' in c.lower()) and not '.' in c)]
        if depth_channels:
            groups['Depth'] = depth_channels

        crypto_prefixes = {p[:-2] for p in groups.keys() if ('crypto' in p.lower() or p.startswith('Crypto')) and any(p.endswith(f"{i:02d}") for i in range(10))}
        for crypto_base in crypto_prefixes:
            groups[f"{crypto_base}_layer_group"] = [f"{crypto_base}{i:02d}" for i in range(10) if f"{crypto_base}{i:02d}" in groups]
        for group_base in layer_group_prefixes:
            if group_base not in crypto_prefixes:
                groups[f"{group_base}_layer_group"] = [f"{group_base}{i:02d}" for i in range(10) if f"{group_base}{i:02d}" in groups]
        return groups

    @staticmethod
    def process_exr_data(image_path: str, normalize: bool, node_id: str = None,
                         layer_data: Dict = None, want_preview: bool = True,
                         want_layers: bool = True, conform: bool = False) -> List:
        """Main EXR processing function

        `want_preview=False` skips the node-face thumbnail. The sequence loader
        passes it: this function was writing a full PNG per frame (three
        full-frame temporaries plus a LANCZOS resize each, on every one of the
        decode threads) and the caller discarded all of them, because
        BatLoader.load makes its own thumbnail from the middle of the
        assembled batch.

        `want_layers=False` builds ONLY the image and mask, and reads only the
        beauty channels off disk. Every layer in the file used to be decoded
        and materialised whether or not the graph had anything plugged into the
        layer outputs — on a 60-channel cryptomatte EXR that is 28 layer
        tensors (~390 MB per 2K frame) built to be thrown away. The caller
        decides: BatLoader.load passes False only when nothing in the
        prompt reads any of the layer-derived outputs (see
        _consumes_slots), so the outputs that go empty here are exactly
        the ones nobody was reading.

        `conform=True` re-fits every channel to the EXR's display window: the
        overscan a Nuke render carries outside the format is cropped off and a
        bbox smaller than the format is padded back out with black. Every
        output — image, mask, layers, cryptomatte — comes from the same decoded
        array, so doing it at read time covers all of them at once.
        """
        if image_path is None:
            raise ValueError("image_path cannot be None.")

        try:
            metadata = layer_data if layer_data else ExrProcessor.scan_exr_metadata(image_path)

            all_subimage_data = None
            if not want_layers:
                # Beauty channels only, when they're contiguous. Falls through
                # to the full read otherwise.
                fast = ExrProcessor.load_rgba_only(image_path, metadata,
                                                   conform=conform)
                if fast is not None:
                    array, names = fast
                    all_subimage_data = {0: array}
                    metadata = dict(metadata)
                    metadata["subimages"] = [
                        {**metadata["subimages"][0], "name": "default",
                         "channels": array.shape[2], "channel_names": names}
                    ]
            if all_subimage_data is None:
                all_subimage_data = ExrProcessor.load_all_data(image_path,
                                                               conform=conform)

            if conform:
                # The arrays are the display window now, so the sizes reported
                # on the `metadata` output have to follow — a stale data-window
                # size there would describe a frame nobody was handed.
                conformed = []
                for idx, info in enumerate(metadata["subimages"]):
                    part = all_subimage_data.get(idx)
                    conformed.append(info if part is None else
                                     {**info, "height": part.shape[0],
                                      "width": part.shape[1],
                                      "conformed_to_display_window": True})
                metadata = {**metadata, "subimages": conformed}

            layers_dict = {}
            cryptomatte_dict = {}
            all_channel_names = []

            for subimage_idx, subimage_info in enumerate(metadata["subimages"]):
                if subimage_idx not in all_subimage_data:
                    continue
                
                subimage_data = all_subimage_data[subimage_idx]
                subimage_name = subimage_info["name"]
                channel_names = subimage_info["channel_names"]
                all_channel_names.extend(channel_names)
                height, width, channels = subimage_data.shape
                
                if subimage_idx == 0:
                    channel_groups = ExrProcessor.get_channel_groups(channel_names)
                    metadata["channel_groups"] = channel_groups
                    rgb_tensor, alpha_tensor = ExrProcessor.process_default_channels(subimage_data, channel_names, height, width, normalize)
                
                if subimage_name != "default" and want_layers:
                    if channels >= 3:
                        rgb_array = subimage_data[:, :, :3]
                        rgb_tensor_layer = _own_tensor(rgb_array).unsqueeze(0)
                        if normalize:
                            rgb_range = rgb_tensor_layer.max() - rgb_tensor_layer.min()
                            if rgb_range > 0:
                                rgb_tensor_layer = (rgb_tensor_layer - rgb_tensor_layer.min()) / rgb_range
                        layers_dict[subimage_name] = rgb_tensor_layer
                        if channels >= 4:
                            alpha_array = subimage_data[:, :, 3]
                            alpha_tensor_layer = _own_tensor(alpha_array).unsqueeze(0)
                            if normalize:
                                alpha_tensor_layer = alpha_tensor_layer.clamp(0, 1)
                            layers_dict[f"{subimage_name}_alpha"] = alpha_tensor_layer
                    elif channels == 1:
                        channel_array = subimage_data[:, :, 0]
                        is_mask_type = any(keyword in subimage_name.lower() for keyword in ['depth', 'mask', 'matte', 'alpha', 'id', 'z'])
                        if is_mask_type or subimage_name == 'depth':
                            mask_tensor = _own_tensor(channel_array).unsqueeze(0)
                            if normalize:
                                mask_range = mask_tensor.max() - mask_tensor.min()
                                if mask_range > 0:
                                    mask_tensor = (mask_tensor - mask_tensor.min()) / mask_range
                            layers_dict[subimage_name] = mask_tensor
                        else:
                            rgb_array = np.stack([channel_array] * 3, axis=2)
                            channel_tensor = _own_tensor(rgb_array).unsqueeze(0)
                            if normalize:
                                channel_range = channel_tensor.max() - channel_tensor.min()
                                if channel_range > 0:
                                    channel_tensor = (channel_tensor - channel_tensor.min()) / channel_range
                            layers_dict[subimage_name] = channel_tensor

                if subimage_idx == 0 and want_layers:
                    for group_name, suffixes in channel_groups.items():
                        if group_name in ('R', 'G', 'B', 'A', 'RGB', 'XYZ') or group_name.endswith('_layer_group'):
                            continue
                        is_cryptomatte = ExrProcessor.is_cryptomatte_layer(group_name)
                        group_indices = [i for i, c in enumerate(channel_names) if c == group_name or c.startswith(f"{group_name}.")]
                        if not group_indices: continue
                        if all(s in suffixes for s in ['R', 'G', 'B']):
                            ExrProcessor.process_rgb_type_layer(group_name, 'R', 'G', 'B', 'A', channel_names, subimage_data, normalize, is_cryptomatte, layers_dict, cryptomatte_dict)
                        elif all(s in suffixes for s in ['r', 'g', 'b']):
                            ExrProcessor.process_rgb_type_layer(group_name, 'r', 'g', 'b', 'a', channel_names, subimage_data, normalize, is_cryptomatte, layers_dict, cryptomatte_dict)
                        elif all(s in suffixes for s in ['X', 'Y', 'Z']):
                            ExrProcessor.process_xyz_type_layer(group_name, 'X', 'Y', 'Z', channel_names, subimage_data, normalize, layers_dict)
                        elif len(group_indices) == 1 or 'Z' in suffixes:
                            ExrProcessor.process_single_channel(group_name, suffixes, group_indices, channel_names, subimage_data, normalize, layers_dict)
                        else:
                            ExrProcessor.process_multi_channel(group_name, group_indices, subimage_data, normalize, is_cryptomatte, layers_dict, cryptomatte_dict)

            if want_layers:
                ExrProcessor.process_layer_groups(channel_groups, cryptomatte_dict, metadata)
                ExrProcessor.store_layer_type_metadata(layers_dict, metadata)
            metadata_json = json.dumps(metadata)
            processed_layer_names = ExrProcessor.create_processed_layer_names(layers_dict, cryptomatte_dict)
            preview_result = (generate_preview_for_comfyui(
                rgb_tensor, image_path, is_sequence=False, frame_index=0)
                if want_preview else None)
            
            # Combine everything for the CRYPTOMATTE output so non-crypto layers can be selected too
            all_for_crypto = {**layers_dict, **cryptomatte_dict}
            result = [rgb_tensor, alpha_tensor, all_for_crypto, layers_dict, processed_layer_names, all_channel_names, metadata_json]
            return {"ui": {"images": preview_result}, "result": result} if preview_result else result
        except Exception as e:
            logger.error(f"[Bat_Loader] error loading EXR file {image_path}: {e}")
            raise