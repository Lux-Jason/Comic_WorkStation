# Comic WorkStation — Agents, Methods, Results

## DeepSeek Notes
- Purpose: LLM for story generation and prompt refinement (`ScriptWriterAgent`, `RefinerAgent`).
- Model: `deepseek-chat` by default (configure in `comic_agent/project_config.json` under `agents.script_writer.model` and `agents.refiner.model`).
- Auth: `DEEPSEEK_API_KEY` is read in `comic_agent/config.py`; `DEEPSEEK_BASE_URL` defaults to `https://api.deepseek.com`.
- Client: OpenAI-compatible client (see `comic_agent/agents.py`, `BaseAgent` initialization).
- Quick check: `python comic_agent/demo_run.py` validates the key and connectivity.
- Security: Keep secrets in local env or an untracked `.env`; avoid committing keys (ignore `comic_agent/.env` in `.gitignore`).
- Tuning: Adjust `model_params.temperature` and `max_tokens` in `project_config.json`.

## Agents
- ScriptWriterAgent: drafts story segments with explicit image markers to guide illustrations.
- RefinerAgent: Compresses each prompt to ~10–12 high-signal keywords; applies CharacterMemory + SceneMemory for identity and setting consistency; normalizes style tokens.
- ImageGeneratorAgent: uses `diffusers.StableDiffusionPipeline` with `ogkalu/Comic-Diffusion`; default quality=high; sequential generation; consolidated negative prompts to suppress blur, extra fingers, text/watermarks, logos; supports seed control and identity-weighting.
## Implementation Highlights
- Short English prompts with identity tokens for character consistency across panels.
- Unified negative prompt policy to reduce common diffusion artifacts.
- Sequential image generation for stability and reproducibility; quality modes tuned with higher steps/CFG by default.
- PDF composition with registered TTF fonts to ensure cross-platform rendering.

## Results (delivered artifacts)
- Folders with PDF + images under `output_comics/`
- Sample images:
<div align="center">
  <img src="output_comics/2025-12-02%20222706.png" alt="Sample 2025-12-02 222706" width="120" />
  <img src="output_comics/2025-12-02%20225543.png" alt="Sample 2025-12-02 225543" width="120" />
  <img src="output_comics/2025-12-02%20225648.png" alt="Sample 2025-12-02 225648" width="120" />
</div>

