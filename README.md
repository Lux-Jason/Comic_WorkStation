# Comic WorkStation — One Page

## Summary
- Multi-agent pipeline to generate short English illustrated novels (images + PDF per novel).
- Env: Conda "chemist" on Windows (PowerShell). Deps in `comic_agent/requirements.txt`.
- Model: `ogkalu/Comic-Diffusion` via `diffusers.StableDiffusionPipeline` (CSA disabled). Quality mode default = high; Unicode-safe PDF.

## Outputs (PDFs)
- `output_comics/cozy_cafe/cozy_cafe_novel.pdf`
- `output_comics/cyberpunk_city/cyberpunk_city_novel.pdf`
- `output_comics/detective_blackwood/detective_blackwood_novel.pdf`
- `output_comics/epic_fantasy/epic_fantasy_novel.pdf`
- `output_comics/space_odyssey/space_odyssey_novel.pdf`

## Featured Images
<div align="center">
  <img src="output_comics/2025-12-02%20222706.png" alt="Sample 2025-12-02 222706" width="120" />
  <img src="output_comics/2025-12-02%20225543.png" alt="Sample 2025-12-02 225543" width="120" />
  <img src="output_comics/2025-12-02%20225648.png" alt="Sample 2025-12-02 225648" width="120" />
</div>

## Run
```powershell
python .\comic_agent\generate_batch_novels.py
```

More details: `PROJECT_REPORT_EN.md` (EN) · `PROJECT_REPORT.md` (CN)

