# Multi-Agent Comic Creation System Design

## 1. Overview
This system uses a multi-agent architecture to generate high-quality comics using DeepSeek (LLM) and Animagine XL 3.1 (Image Generation). The workflow is designed to solve common issues like inconsistent characters, prompt truncation, and VAE artifacts ("colorful blocks").

## 2. Agent Roles

### 🎭 Director (Orchestrator)
- **Responsibility**: Manages the overall workflow and passes data between agents.
- **Implementation**: Implicitly handled in the main execution loop.

### 👤 Character Designer Agent
- **Responsibility**: Converts raw character descriptions into consistent visual tags.
- **System Prompt**: "You are a Lead Character Designer... Create consistent 'Character Sheets' using Danbooru tags."
- **Output**: JSON Dictionary `{"CharacterName": "1girl, blue_hair, robe..."}`.
- **Why**: Ensures that every time "Wizard" appears, they look the same (same tags).

### 📝 Screenwriter Agent
- **Responsibility**: Writes the narrative script broken down into panels.
- **System Prompt**: "You are a Comic Screenwriter... Create a 4-panel comic script..."
- **Output**: JSON List of Panel Objects (Description, Dialogue, Characters Present).

### 🎬 Storyboard Agent (Prompt Engineer)
- **Responsibility**: Converts narrative descriptions into optimized SDXL prompts.
- **System Prompt**: "You are an expert AI Prompt Engineer for SDXL... START with quality tags... INSERT character tags... KEEP IT CONCISE."
- **Strategy**:
    1.  **Prefix**: `masterpiece, best quality, very aesthetic, absurdres`
    2.  **Character**: Injects tags from Character Designer.
    3.  **Action/Scene**: Extracts keywords from script description.
    4.  **Limit**: Targets < 75 tokens to avoid CLIP truncation.

### 🎨 Illustration Agent (Art Director)
- **Responsibility**: Generates the actual images using the Diffusion pipeline.
- **Technical Fixes**:
    -   **VAE Upcast**: Forces the VAE to run in `float32` while keeping the UNet in `float16`. This fixes the "meaningless colorful blocks" / NaN issue common with SDXL on some GPUs.
    -   **Parameters**: Uses `guidance_scale=7.0` and `steps=28` (optimal for Animagine).

## 3. Technical Solutions

### 🌈 "Colorful Blocks" Fix (VAE)
The issue is caused by numerical instability (overflow) in the VAE when running in half-precision (`fp16`).
**Fix Applied**:
```python
if ComicConfig.UPCAST_VAE and device == "cuda":
    self.pipe.vae.to(dtype=torch.float32)
```

### ✂️ Long Prompt Handling
SDXL has a soft limit of 77 tokens for the CLIP text encoder. Long narrative prompts get truncated, leading to ignored instructions.
**Strategy**:
-   The **Storyboard Agent** is explicitly instructed to use *comma-separated keywords* instead of full sentences.
-   Prioritizes: Quality Tags > Character Tags > Action > Background.

## 4. Workflow Data Flow
1.  **User Input**: Theme & Characters.
2.  **Char Designer**: `Raw Text` -> `{"Hero": "tags..."}`
3.  **Screenwriter**: `Theme` + `Char Names` -> `[Panel 1 Desc, Panel 2 Desc...]`
4.  **Storyboarder**: `Panel Desc` + `{"Hero": "tags..."}` -> `Final SDXL Prompt`
5.  **Illustrator**: `Final SDXL Prompt` -> `Image File`

## 5. Usage
Run the GUI:
```bash
python comic_agent/gui.py
```
Or run the CLI:
```bash
python comic_agent/comic_system.py
```
