# bat_lines_nets — third-party attribution

The network definitions in this directory are **vendored third-party code**, so
🦇 Lines runs without `comfyui_controlnet_aux` installed. Each file starts with
a header naming its origin, its licence and what was changed (only imports and
the detector wrappers removed; the layers are untouched, so the published
weights load with `strict=True`). `__init__.py` is BAT code: it builds the
networks and does each one's pre- and post-processing.

| File | Origin | Code licence | Weights (`models/lines/…`) and their licence |
|---|---|---|---|
| `ted.py`, `_smish.py` | [xavysp/TEED](https://github.com/xavysp/TEED) @ `40fa4b13` | MIT, © 2022 Xavier Soria Poma | `teed/7_model.pth` from the same commit — MIT. **Not** the `bdsqlsz/qinglong_controlnet-lllite` copy that controlnet_aux uses (CC-BY-NC-SA-4.0, a different file). |
| `informative_drawings.py` | [carolineec/informative-drawings](https://github.com/carolineec/informative-drawings) | MIT, © 2022 Caroline Chan | `informative_drawings/sk_model.pth`, `sk_model2.pth` — MIT (mirror: HF `lllyasviel/Annotators`) |
| `anime2sketch.py` | [Mukosame/Anime2Sketch](https://github.com/Mukosame/Anime2Sketch) (U-Net from pix2pix) | MIT, © 2021 Xiaoyu Xiang; pix2pix BSD-2-Clause | `anime2sketch/netG.pth` — MIT (mirror: HF `lllyasviel/Annotators`) |
| `manga_line.py` | [ljsabc/MangaLineExtraction_PyTorch](https://github.com/ljsabc/MangaLineExtraction_PyTorch) | MIT, © 2021 Miaomiao Li | `manga_line/erika.pth` — MIT (GitHub release `v1`; same bytes on HF `lllyasviel/Annotators`) |
| `mlsd_large.py`, `mlsd_utils.py` | [lhwcv/mlsd_pytorch](https://github.com/lhwcv/mlsd_pytorch), port of [navervision/mlsd](https://github.com/navervision/mlsd) | Apache-2.0, © NAVER Corp. | `mlsd/mlsd_large_512_fp32.pth` — Apache-2.0 |
| `hed.py` | lllyasviel's ControlNet HED re-implementation | Apache-2.0, granted in the source header | `hed/ControlNetHED.pth` — Apache-2.0 per that header (the HF card says "other") |

All except TEED and `_smish.py` were taken from
[comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux)
1.1.5 (Apache-2.0), which carries them under the same upstream licences. Licence
texts are in `licenses/`. Note: controlnet_aux's copy of the Anime2Sketch
licence credits Caroline Chan by mistake; `licenses/Anime2Sketch-MIT.txt` is
the upstream file.

Download URLs and sha256 checksums for every weight file are in
`../bat_lines_models.py`.
