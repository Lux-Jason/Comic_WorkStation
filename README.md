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

# Comic WorkStation — One-Page README

## Summary
- Multi-agent pipeline that generates short English illustrated novels (images + PDF per novel).
- Environment: Conda "chemist" on Windows (PowerShell). Deps in `comic_agent/requirements.txt`.
- Outputs: five folders under `output_comics/` with final PDFs and images.

## Model & Pipeline (very brief)
- Base model: `ogkalu/Comic-Diffusion` via `diffusers.StableDiffusionPipeline` (CSA disabled).
- Agents: ScriptWriter → Refiner (short English prompts + identity consistency) → ImageGenerator (quality=high) → Composer (PDF with Unicode fonts).

## Outputs (PDF names)
- `output_comics/cozy_cafe/cozy_cafe_novel.pdf`
- `output_comics/cyberpunk_city/cyberpunk_city_novel.pdf`
- `output_comics/detective_blackwood/detective_blackwood_novel.pdf`
- `output_comics/epic_fantasy/epic_fantasy_novel.pdf`
- `output_comics/space_odyssey/space_odyssey_novel.pdf`

## Featured Images
<div align="center">
  <img src="output_comics/2025-12-02%20222706.png" alt="Sample 2025-12-02 222706" width="220" />
  <img src="output_comics/2025-12-02%20225543.png" alt="Sample 2025-12-02 225543" width="220" />
  <img src="output_comics/2025-12-02%20225648.png" alt="Sample 2025-12-02 225648" width="220" />
</div>

## Run
```powershell
python .\comic_agent\generate_batch_novels.py
```

More details: `PROJECT_REPORT_EN.md` (EN) / `PROJECT_REPORT.md` (CN).