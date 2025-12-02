# Project Report

## Overview
- Goal: Multi-agent pipeline that generates short English illustrated novels across diverse genres, producing images and a compiled PDF per novel.
- Environment: Conda env "chemist" on Windows (PowerShell). Key deps in `comic_agent/requirements.txt`.
- Outputs: Five finished novels under `output_comics/`, each with `illustration_*.png` and a final PDF.

## Models & Algorithms
- Model: `ogkalu/Comic-Diffusion` (SD 1.5 family) via `diffusers.StableDiffusionPipeline`. Custom CSA UNet disabled to avoid shape mismatch.
- Agent pipeline:
  - `ScriptWriterAgent`: creates narrative segments with image markers (English, concise, coherent).
  - `RefinerAgent`: uses `CharacterMemory`, `SceneMemory`, and `PromptNormalizer` to compress each image prompt into ~10–12 English keywords; emphasizes identity tokens and consistency.
  - `ImageGeneratorAgent`: default quality mode is `high` (more steps / stronger CFG), generates images sequentially; consolidated negative prompt reduces artifacts (blur, hands, watermark/text, logos).
  - `ComposerAgent`: assembles text + images into PDF using `fpdf2`; registers Unicode TTF fonts to fix Helvetica/Unicode and bold style issues.
- Key strategies:
  - Enforce English-only text and prompts; short, high-signal keywords.
  - Strengthened negative prompts and identity token weighting for character consistency.
  - Sequential generation for stability; per-novel folder outputs.

## Batch Results Summary
- `output_comics/cozy_cafe`:
  - PDF: `cozy_cafe_novel.pdf`
  - Images: `illustration_1.png`, `illustration_2.png`, `illustration_3.png`, `illustration_4.png`, `illustration_5.png`
- `output_comics/cyberpunk_city`:
  - PDF: `cyberpunk_city_novel.pdf`
  - Images: `illustration_1.png`, `illustration_2.png`, `illustration_3.png`, `illustration_4.png`, `illustration_5.png`
- `output_comics/detective_blackwood`:
  - PDF: `detective_blackwood_novel.pdf`
  - Images: `illustration_1.png`, `illustration_2.png`, `illustration_3.png`, `illustration_4.png`, `illustration_5.png`, `illustration_6.png`
- `output_comics/epic_fantasy`:
  - PDF: `epic_fantasy_novel.pdf`
  - Images: `illustration_1.png`, `illustration_2.png`, `illustration_3.png`, `illustration_4.png`, `illustration_5.png`, `illustration_6.png`
- `output_comics/space_odyssey`:
  - PDF: `space_odyssey_novel.pdf`
  - Images: `illustration_1.png`, `illustration_2.png`, `illustration_3.png`, `illustration_4.png`, `illustration_5.png`, `illustration_6.png`

## Featured Images
<div align="center">
  <img src="output_comics/2025-12-02%20222706.png" alt="Sample 2025-12-02 222706" width="320" />
  <img src="output_comics/2025-12-02%20225543.png" alt="Sample 2025-12-02 225543" width="320" />
  <img src="output_comics/2025-12-02%20225648.png" alt="Sample 2025-12-02 225648" width="320" />
</div>

## Quality & Stability
- Default quality mode: `high`. Single-run and batch runs complete reliably.
- Non-blocking warnings: Hugging Face cache lacks `.safetensors` for UNet/VAE; `diffusers` falls back to unsafe serialization. Set `allow_pickle=False` to force error if desired. Current outputs are unaffected.
- PDF encoding: Unicode TTF font registration ensures proper text rendering and styles.

## Paths & Usage
- Single-run example: `comic_agent/run_holmes_comic.py`.
- Batch script: `comic_agent/generate_batch_novels.py` (produces all five novels).
- PowerShell commands:
```powershell
python .\comic_agent\generate_batch_novels.py
```

## Next Steps (Suggestions)
- Further tune negative prompts and identity token weights for finer consistency.
- Add per-segment fallback/skip logic on image generation for robustness.
- For higher fidelity, extend `quality_modes` in `project_config.json` or switch to a stronger base model.