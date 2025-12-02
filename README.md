# Comic WorkStation — Technical Overview & Model Analysis

## Quick Summary
- Purpose: Multi-agent comic creation pipeline that produces English light‑novel text, concise image prompts, coherent illustrations, and a final PDF per story.
- Focus: Short high‑signal prompts, cross‑panel character/scene consistency, and strong negative prompt policy to suppress artifacts.
- Outputs: Per‑story folder under `output_comics/` with `illustration_*.png` and a compiled PDF.

## Architecture
- Flow: `ScriptWriterAgent → RefinerAgent → ImageGeneratorAgent → ComposerAgent`.
- Contracts:
  - Text segments with markers: `[ILLUSTRATION: ...]` (Scene Illustration | Character Portrait).
  - Refined image segments: `final_prompt`, `panel_id`, optional `dialogue`.
  - Images saved as `illustration_{panel_id}.png`, then assembled into PDF.

## Agents
- ScriptWriterAgent: Writes English‑only prose and inserts at least five explicit illustration markers per chapter.
- RefinerAgent: Compresses each marker to 10–12 English keywords; enforces identity/setting tokens; normalizes style tokens.
- ImageGeneratorAgent: Uses `diffusers.StableDiffusionPipeline` with `ogkalu/Comic-Diffusion`; default quality=high; sequential generation; consolidated negative prompts; optional seed control and identity weighting.
- ComposerAgent: Builds Unicode‑safe PDFs via `fpdf2` with registered TTF fonts; normalizes curly quotes and dashes; robust cross‑platform rendering.

## DeepSeek LLM
- Model: `deepseek-chat` (configurable in `comic_agent/project_config.json` at `agents.script_writer` and `agents.refiner`).
- Client & Auth: OpenAI‑compatible client in `comic_agent/agents.py` (`BaseAgent`); credentials read in `comic_agent/config.py` (`DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL` default `https://api.deepseek.com`).
- Prompting Strategy:
  - ScriptWriterAgent: Enforces English‑only, structured markers, explicit Scene/Portrait tags.
  - RefinerAgent: Extracts concise, high‑signal tokens; preserves identity and scene continuity cues.
- Parameters: Tune `temperature` (≈0.5–0.7) and `max_tokens` (≈512–3000) in `project_config.json`.
- Safety & Checks: Keep API keys in local env or untracked `.env` (ignore `comic_agent/.env`); run `python comic_agent/demo_run.py` to validate connectivity.

## Diffusion Model & Pipeline
- Visual Model: `ogkalu/Comic-Diffusion` (Stable Diffusion 1.5 family; anime/comic prior).
- Pipeline: `diffusers.StableDiffusionPipeline` (CSA off to avoid UNet shape mismatch; StoryDiffusion CSA module exists but is disabled by default).
- Scheduler: DPMSolverMultistep (configured by name in `project_config.json`).
- Defaults: 512×768 portrait; `guidance_scale=7.0`; `num_inference_steps=28` (standard).

### Quality Modes
| Mode     | Steps | Guidance | Notes                         |
|----------|-------|----------|-------------------------------|
| fast     | 20    | 6.0      | For quick previews            |
| standard | 28    | 7.0      | Balanced quality & latency    |
| high     | 42    | 6.5      | Default; best clarity/consistency |

### Negative Prompt Policy
- Core: `lowres, bad hands, malformed, extra/missing/fused fingers, distorted, wrong anatomy, duplicate, watermark, signature, text artifact, blurry, jpeg artifacts, grainy, noisy, oversaturated, neon bleed`.
- Style: `worst quality, low quality, normal quality, out of frame, cropped, skewed perspective, unnatural lighting`.
- Applied per panel to suppress text/watermarks, hand anomalies, blur/noise.

### Generation Strategy
- Sequential per‑panel rendering to stabilize style and reduce later‑frame drift.
- Slight step boost on the first panel to establish style tokens.
- Seed control supported; identity tokens in early positions are emphasized via parentheses weighting.

## Consistency Mechanisms
- CharacterMemory: Extracts stable identity tokens (e.g., hair/attire/props) and reuses across prompts.
- SceneMemory: Caches setting cues (time, weather, palette, camera) for continuity.
- PromptNormalizer: De‑duplicates and limits to 10–12 high‑signal keywords; avoids long phrases.
- Identity Weighting: Parentheses amplify the first few identity tokens in `final_prompt`.

## Configuration Map
| Component/Config     | Path                                      | Key Fields / Notes |
|----------------------|-------------------------------------------|--------------------|
| LLM credentials      | `comic_agent/config.py`                    | `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL` |
| Project config       | `comic_agent/project_config.json`          | Agents, models, modes, negatives |
| Agents implementation| `comic_agent/agents.py`                    | Quality/seed setters, negatives, identity weighting, sequential gen |
| CSA/Story modules    | `comic_agent/story_diffusion.py`, `mccd_pipeline.py` | Present but disabled by default |
| Output directory     | `comic_agent/config.py` (`OUTPUT_DIR`)     | Default: `output_comics` |

## Model Analysis & Trade‑offs
- DeepSeek (`deepseek-chat`):
  - Strengths: concise keyword extraction; controllable verbosity; reliable when guided by strict templates.
  - Risks: may produce verbose prose if constraints weaken; mitigated by templates + normalization.
- Comic‑Diffusion (SD 1.5):
  - Strengths: anime/comic prior yields appealing line art and color at short prompts.
  - Risks: classic SD1.5 issues (hands/anatomy, unintended text/watermarks) — mitigated via consolidated negatives and sequential generation. CSA currently disabled due to UNet shape mismatch.
- Performance: `high` mode (42 steps, guidance 6.5) improves clarity/consistency at higher latency; recommended for final renders.

## Results
- Artifacts are under `output_comics/` (per‑story subfolders with PNGs + PDF).
- Sample images:
<div align="center">
  <img src="output_comics/2025-12-02%20225648.png" alt="Sample 2025-12-02 225648" width="250" />
  <img src="output_comics/2025-12-02%20222706.png" alt="Sample 2025-12-02 222706" width="250" />
  <img src="output_comics/2025-12-02%20225543.png" alt="Sample 2025-12-02 225543" width="250" />
</div>

