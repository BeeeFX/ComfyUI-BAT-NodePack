"""
Weights for 🦇 Lines: where they live, where they come from, and the download.

Every file is pinned twice — to an immutable URL (a Hugging Face revision or a
GitHub commit, never a branch) and to its sha256 — and the download is only
moved into place once the hash matches. A mirror that changes its bytes, a
proxy that serves an error page, or a half-finished transfer can therefore
never end up in the models folder looking like a checkpoint.

The files live in their own ``lines`` model folder (``models/lines/`` plus any
``lines:`` entry in extra_model_paths.yaml), one subfolder per upstream project,
so a licence row in the folder's LICENSES.md names exactly one project:

    lines/teed/7_model.pth
    lines/informative_drawings/sk_model.pth, sk_model2.pth
    lines/anime2sketch/netG.pth
    lines/manga_line/erika.pth
    lines/mlsd/mlsd_large_512_fp32.pth
    lines/hed/ControlNetHED.pth

Licences (weights, which is what counts for commercial use): TEED MIT,
informative-drawings MIT, Anime2Sketch MIT, MangaLineExtraction MIT, M-LSD
Apache-2.0, ControlNet HED Apache-2.0 (granted in lllyasviel's source header).
TEED deliberately comes from the author's own repository: the copy
comfyui_controlnet_aux downloads (bdsqlsz/qinglong_controlnet-lllite) is
published under CC-BY-NC-SA-4.0 and is a different file.

In production the files are placed by hand by RnD and ``auto_download`` never
has to fire; on a personal install it fetches what a filter needs on first use.
"""

import hashlib
import logging
import os
import threading
import time
import urllib.request

logger = logging.getLogger("[Bat_Lines]")

FOLDER = "lines"

_ANNOTATORS = ("https://huggingface.co/lllyasviel/Annotators/resolve/"
               "982e7edaec38759d914a963c48c4726685de7d96/")

# key -> where the file goes, where it comes from, and what it must hash to.
MODELS = {
    "teed": {
        "path": "teed/7_model.pth",
        "url": ("https://raw.githubusercontent.com/xavysp/TEED/"
                "40fa4b1391dc6424f88989d0ca75d5b592c8681d/checkpoints/BIPED/7/7_model.pth"),
        "sha256": "d0109e7f40e7d9f1f495d34947eb08167e8fbb0a13b4e6ab3121261fb8d5a416",
        "size": 249_139,
        "licence": "MIT",
        "project": "https://github.com/xavysp/TEED",
    },
    "lineart": {
        "path": "informative_drawings/sk_model.pth",
        "url": _ANNOTATORS + "sk_model.pth",
        "sha256": "c686ced2a666b4850b4bb6ccf0748031c3eda9f822de73a34b8979970d90f0c6",
        "size": 17_173_511,
        "licence": "MIT",
        "project": "https://github.com/carolineec/informative-drawings",
    },
    "lineart_coarse": {
        "path": "informative_drawings/sk_model2.pth",
        "url": _ANNOTATORS + "sk_model2.pth",
        "sha256": "30a534781061f34e83bb9406b4335da4ff2616c95d22a585c1245aa8363e74e0",
        "size": 17_173_511,
        "licence": "MIT",
        "project": "https://github.com/carolineec/informative-drawings",
    },
    "anime": {
        "path": "anime2sketch/netG.pth",
        "url": _ANNOTATORS + "netG.pth",
        "sha256": "ccabdcc3f5cf3c07cf65d58776acb21df7dfda825cdc70c9766a93fd62bfc488",
        "size": 217_631_959,
        "licence": "MIT",
        "project": "https://github.com/Mukosame/Anime2Sketch",
    },
    "manga": {
        "path": "manga_line/erika.pth",
        "url": _ANNOTATORS + "erika.pth",
        "sha256": "badbd6baf013cefbd98993307b02cc14a26c770d067416e4fdecc8720b88feeb",
        "size": 172_789_563,
        "licence": "MIT",
        "project": "https://github.com/ljsabc/MangaLineExtraction_PyTorch",
    },
    "mlsd": {
        "path": "mlsd/mlsd_large_512_fp32.pth",
        "url": _ANNOTATORS + "mlsd_large_512_fp32.pth",
        "sha256": "5696f168eb2c30d4374bbfd45436f7415bb4d88da29bea97eea0101520fba082",
        "size": 6_341_481,
        "licence": "Apache-2.0",
        "project": "https://github.com/lhwcv/mlsd_pytorch",
    },
    "hed": {
        "path": "hed/ControlNetHED.pth",
        "url": _ANNOTATORS + "ControlNetHED.pth",
        "sha256": "5ca93762ffd68a29fee1af9d495bf6aab80ae86f08905fb35472a083a4c7a8fa",
        "size": 29_444_406,
        "licence": "Apache-2.0",
        "project": "https://huggingface.co/lllyasviel/Annotators",
    },
}

try:
    import folder_paths
except ImportError:          # tests, or the module imported outside ComfyUI
    folder_paths = None


def _register_folder():
    """Make ``lines`` a known model folder type, keeping any existing entries."""
    if folder_paths is None:
        return
    try:
        target = os.path.join(folder_paths.models_dir, FOLDER)
        if FOLDER in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path(FOLDER, target)
        else:
            folder_paths.folder_names_and_paths[FOLDER] = (
                [target], set(folder_paths.supported_pt_extensions))
    except Exception as e:  # never let this take the pack down at import
        logger.warning("could not register the models/%s folder: %s", FOLDER, e)


_register_folder()


def search_dirs():
    """Every configured ``lines`` folder, in ComfyUI's priority order."""
    dirs = []
    if folder_paths is not None:
        try:
            dirs = list(folder_paths.get_folder_paths(FOLDER))
        except KeyError:
            dirs = []
        default = os.path.join(folder_paths.models_dir, FOLDER)
        if default not in dirs:
            dirs.append(default)
    return dirs


def find(key, dirs=None):
    """Path of an already-present weight file, or None."""
    rel = MODELS[key]["path"]
    for d in (search_dirs() if dirs is None else dirs):
        p = os.path.join(d, rel)
        if os.path.isfile(p):
            return p
    return None


def _target_dir(dirs):
    """First ``lines`` folder we can write into, creating it if needed."""
    for d in dirs:
        try:
            os.makedirs(d, exist_ok=True)
            if os.access(d, os.W_OK):
                return d
        except OSError:
            continue
    return None


def manual_instructions(key, dirs=None):
    m = MODELS[key]
    dirs = search_dirs() if dirs is None else dirs
    where = dirs[0] if dirs else "models/lines"
    return (
        f"Download it by hand instead:\n"
        f"  {m['url']}\n"
        f"and save it as\n"
        f"  {os.path.join(where, m['path'])}\n"
        f"(sha256 {m['sha256']}, {m['size']:,} bytes; licence {m['licence']}, "
        f"{m['project']})."
    )


_DOWNLOAD_LOCK = threading.Lock()


def ensure(key, auto_download=True, dirs=None, _urlopen=None):
    """Path of the weights for `key`, downloading them first if allowed.

    Serialised on a module lock so the node and the live preview can't race each
    other into the same file. `_urlopen` is for the tests.
    """
    dirs = search_dirs() if dirs is None else list(dirs)
    found = find(key, dirs)
    if found:
        return found
    if not auto_download:
        raise RuntimeError(
            f"🦇 Lines: the weights for this filter are not installed and "
            f"auto_download is off.\n" + manual_instructions(key, dirs))

    with _DOWNLOAD_LOCK:
        found = find(key, dirs)          # another thread may have finished
        if found:
            return found
        target = _target_dir(dirs)
        if target is None:
            raise RuntimeError(
                "🦇 Lines: no writable models/lines folder to download into.\n"
                + manual_instructions(key, dirs))
        return _download(key, target, _urlopen or urllib.request.urlopen)


def _download(key, target, urlopen):
    m = MODELS[key]
    root = os.path.realpath(target)
    dest = os.path.realpath(os.path.join(root, m["path"]))
    # The relative paths are constants above, but a check is cheaper than an
    # argument: nothing this module writes may land outside the lines folder.
    if os.path.commonpath([root, dest]) != root:
        raise RuntimeError(f"🦇 Lines: refusing to write outside {root}: {dest}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = f"{dest}.part-{os.getpid()}-{threading.get_ident()}"

    logger.info("downloading %s (%.1f MB) from %s", m["path"],
                m["size"] / 1e6, m["url"])
    print(f"[🦇 Lines] downloading {m['path']} ({m['size'] / 1e6:.1f} MB) "
          f"into {root} — one-off")
    h = hashlib.sha256()
    got = 0
    t0 = time.time()
    try:
        req = urllib.request.Request(m["url"], headers={"User-Agent": "ComfyUI-BAT-NodePack"})
        with urlopen(req, timeout=60) as r, open(part, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                h.update(chunk)
                got += len(chunk)
        digest = h.hexdigest()
        if digest != m["sha256"]:
            raise RuntimeError(
                f"🦇 Lines: {m['path']} downloaded with the wrong checksum "
                f"(got {digest}, {got:,} bytes; expected {m['sha256']}, "
                f"{m['size']:,} bytes). Nothing was installed.")
        os.replace(part, dest)
    except Exception as e:
        try:
            os.remove(part)
        except OSError:
            pass
        if isinstance(e, RuntimeError):
            raise RuntimeError(f"{e}\n" + manual_instructions(key, [target])) from e
        raise RuntimeError(
            f"🦇 Lines: downloading {m['path']} failed: {e}\n"
            + manual_instructions(key, [target])) from e
    print(f"[🦇 Lines] {m['path']} ok ({got / 1e6:.1f} MB in {time.time() - t0:.0f}s)")
    return dest
