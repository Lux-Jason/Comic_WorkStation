import json
import os
import re
from typing import List, Dict, Any, Optional
from openai import OpenAI
from colorama import Fore, Style
from PIL import Image, ImageDraw, ImageFont
from fpdf import FPDF
import textwrap

try:
    from config import ComicConfig
except ImportError:
    from .config import ComicConfig

try:
    from story_diffusion import apply_story_diffusion
except ImportError:
    from .story_diffusion import apply_story_diffusion

# MCCDSceneParser import moved to LayoutPlannerAgent to avoid circular import


def _extract_keywords(text: str, limit: int = 4) -> List[str]:
    """Lightweight keyword splitter used for prompt compression."""
    if not text:
        return []
    cleaned = text.replace("/", " ")
    chunks = re.split(r"[\n;,]", cleaned)
    keywords: List[str] = []
    for chunk in chunks:
        tokens = [tok for tok in chunk.strip().split() if tok]
        if not tokens:
            continue
        keywords.append(" ".join(tokens[:2]))
        if len(keywords) >= limit:
            break
    return keywords


class CharacterMemory:
    """Tracks author-provided personas so prompts stay consistent."""

    def __init__(self, design_text: str):
        self.design_text = design_text or ""
        self.profiles = self._parse_design(self.design_text)

    def _parse_design(self, raw: str) -> Dict[str, Dict[str, List[str]]]:
        entries = [seg.strip() for seg in re.split(r"[;\n]", raw) if seg.strip()]
        profiles: Dict[str, Dict[str, List[str]]] = {}
        for entry in entries:
            if ":" in entry:
                name, desc = entry.split(":", 1)
            else:
                bits = entry.split()
                if not bits:
                    continue
                name, desc = bits[0], " ".join(bits[1:])
            name = name.strip()
            if not name:
                continue
            profiles[name] = {
                "keywords": _extract_keywords(desc, limit=6) or [name],
                "aliases": self._build_aliases(name)
            }
        return profiles

    def _build_aliases(self, name: str) -> List[str]:
        alias = [name.lower()]
        parts = name.lower().split()
        if parts:
            alias.append(parts[0])
        return list(dict.fromkeys(alias))

    def detect_mentions(self, text: str, object_names: List[str]) -> List[str]:
        haystack = (text or "").lower()
        if object_names:
            haystack += " " + " ".join(object_names).lower()
        matches = []
        for name, meta in self.profiles.items():
            if any(alias in haystack for alias in meta.get("aliases", [])):
                matches.append(name)
        return matches

    def tokens_for(self, names: List[str], max_per_char: int = 4) -> List[str]:
        tokens: List[str] = []
        for name in names:
            profile = self.profiles.get(name)
            if not profile:
                continue
            tokens.append(name)
            tokens.extend(profile.get("keywords", [])[:max_per_char])
        return tokens

    def all_tokens(self) -> List[str]:
        tokens: List[str] = []
        for name in self.profiles:
            tokens.extend(self.tokens_for([name]))
        return tokens

    def serialize(self) -> str:
        parts = []
        for name, profile in self.profiles.items():
            parts.append(f"{name}: {', '.join(profile.get('keywords', []))}")
        return " | ".join(parts)


class SceneMemory:
    """Caches background context to keep locations coherent."""

    def __init__(self):
        self.primary_background = ""

    def remember_background(self, candidate: str) -> str:
        candidate = (candidate or "").strip()
        if candidate:
            self.primary_background = candidate
        return self.primary_background


class PromptNormalizer:
    """Ensures prompts stay under the requested word budget."""

    def __init__(self, max_words: int = 12):
        self.max_words = max_words

    def normalize(self, phrases: List[str], fallback: List[str] = None) -> str:
        words: List[str] = []
        seen = set()
        for phrase in phrases:
            if not phrase:
                continue
            for word in re.split(r"[\s,]+", phrase.lower()):
                clean = re.sub(r"[^a-z0-9-]", "", word)
                if not clean or clean in seen:
                    continue
                seen.add(clean)
                words.append(clean)
                if len(words) >= self.max_words:
                    break
            if len(words) >= self.max_words:
                break
        if fallback and len(words) < max(6, self.max_words // 2):
            for word in fallback:
                clean = re.sub(r"[^a-z0-9-]", "", word.lower())
                if not clean or clean in seen:
                    continue
                seen.add(clean)
                words.append(clean)
                if len(words) >= self.max_words:
                    break
        if not words:
            return "focus, anime"
        return ", ".join(words[:self.max_words])


class BaseAgent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.client = OpenAI(
            api_key=ComicConfig.DEEPSEEK_API_KEY,
            base_url=ComicConfig.DEEPSEEK_BASE_URL
        )
        self.config = ComicConfig.PROJECT_CONFIG["agents"]

    def log(self, message: str, color=Fore.WHITE):
        print(f"{color}[{self.name}]: {message}{Style.RESET_ALL}")

    def chat(self, prompt: str, system_prompt: str = "", model_config: Dict = None) -> str:
        if model_config is None:
            model_config = {"model": "deepseek-chat", "temperature": 0.7}
            
        try:
            response = self.client.chat.completions.create(
                model=model_config.get("model", "deepseek-chat"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=model_config.get("temperature", 0.7),
                max_tokens=model_config.get("max_tokens", 2048)
            )
            return response.choices[0].message.content
        except Exception as e:
            self.log(f"Error calling LLM: {e}", Fore.RED)
            return ""

class ScriptWriterAgent(BaseAgent):
    def __init__(self):
        super().__init__("ScriptWriter", "Novel Generation")
        self.agent_config = self.config["script_writer"]

    def write_novel(self, theme: str, characters: str, scene_desc: str) -> List[Dict]:
        self.log(f"Writing novel chapter for theme: {theme}", Fore.CYAN)
        
        prompt = self.agent_config["prompt_template"].format(
            theme=theme,
            characters=characters,
            scene_desc=scene_desc
        ) + "\nIMPORTANT: Write the story and descriptions in English only."
        
        response = self.chat(
            prompt, 
            system_prompt="You are a professional light novel author. Write the story in English.",
            model_config=self.agent_config
        )
        
        # Parse the response into segments
        segments = []
        lines = response.split('\n')
        current_text = []
        
        for line in lines:
            line = line.strip()
            if not line: continue
            
            if "[ILLUSTRATION:" in line and "]" in line:
                # Save accumulated text
                if current_text:
                    segments.append({
                        "type": "text",
                        "content": "\n".join(current_text)
                    })
                    current_text = []
                
                # Extract illustration description
                start = line.find("[ILLUSTRATION:") + len("[ILLUSTRATION:")
                end = line.find("]")
                desc = line[start:end].strip()
                
                segments.append({
                    "type": "image",
                    "description": desc,
                    "panel_id": len([s for s in segments if s["type"] == "image"]) + 1
                })
            else:
                current_text.append(line)
                
        # Append remaining text
        if current_text:
            segments.append({
                "type": "text",
                "content": "\n".join(current_text)
            })
            
        return segments

    # Backward-compatible alias used by other modules/tests
    def generate_script(self, theme: str, characters: str, scene_desc: str) -> List[Dict]:
        return self.write_novel(theme, characters, scene_desc)

class LayoutPlannerAgent(BaseAgent):
    def __init__(self):
        super().__init__("LayoutPlanner", "Layout & Composition")
        self.agent_config = self.config["layout_planner"]
        try:
            from mccd_agents import MCCDSceneParser
        except ImportError:
            from .mccd_agents import MCCDSceneParser
        self.mccd_parser = MCCDSceneParser()

    def plan_layout(self, panels: List[Dict]) -> List[Dict]:
        self.log("Planning layout using MCCD Multi-Agent Parsing...", Fore.MAGENTA)
        
        for panel in panels:
            if panel['type'] != 'image':
                continue
                
            desc = panel.get('description', '')
            self.log(f"Parsing Panel {panel['panel_id']}...", Fore.CYAN)
            
            # Use MCCD Parser to get detailed scene breakdown
            parsed_scene = self.mccd_parser.parse_scene(desc)
            
            # Store the parsed data in the panel object
            panel['mccd_data'] = parsed_scene
            
            # Construct a rich prompt from the parsed data
            # "Background: ..., Objects: [Obj1 (desc), Obj2 (desc)]"
            objects_desc = ", ".join([f"{k}: {v}" for k, v in parsed_scene['objects'].items()])
            background = parsed_scene['background']
            
            # Update description to be more detailed for the Refiner
            panel['rich_description'] = f"Background: {background}. Objects: {objects_desc}."
            
            self.log(f"Panel {panel['panel_id']} Layout: {len(parsed_scene['layout'])} objects positioned.", Fore.GREEN)
            
        return panels

class RefinerAgent(BaseAgent):
    def __init__(self):
        super().__init__("Refiner", "Quality Control")
        self.agent_config = self.config["refiner"]
        self.scene_memory = SceneMemory()
        self.prompt_normalizer = PromptNormalizer()
        self.character_memory: Optional[CharacterMemory] = None

    def refine_prompts(self, segments: List[Dict], character_design: str) -> List[Dict]:
        self.log("Refining prompts for consistency...", Fore.MAGENTA)
        self.character_memory = CharacterMemory(character_design)
        
        for segment in segments:
            if segment['type'] != 'image':
                continue
                
            desc = segment.get('rich_description', segment.get('description', ''))
            metadata = segment.get('mccd_data', {})
            structured_prompt = self._build_structured_prompt(desc, metadata)

            refined_prompt = self.chat(
                structured_prompt,
                system_prompt="You compress scenes into <=12 English keywords.",
                model_config=self.agent_config
            ).strip().strip('"')

            keyword_candidates = self._collect_keyword_candidates(segment, refined_prompt)
            fallback_tokens = self.character_memory.all_tokens() if self.character_memory else []
            minimal_prompt = self.prompt_normalizer.normalize(keyword_candidates, fallback=fallback_tokens)

            segment['final_prompt'] = minimal_prompt
            self.log(f"Image {segment['panel_id']} Final Prompt: {minimal_prompt}", Fore.MAGENTA)
            
        return segments

    def _build_structured_prompt(self, description: str, metadata: Dict[str, Any]) -> str:
        character_block = self.character_memory.serialize() if self.character_memory else ""
        object_block = json.dumps(metadata.get('objects', {}), ensure_ascii=False) if metadata else "{}"
        background = metadata.get('background') if metadata else ""
        remembered_background = self.scene_memory.remember_background(background)
        layout_block = json.dumps(metadata.get('layout', {}), ensure_ascii=False) if metadata else "{}"

        return self.agent_config["prompt_template"].format(
            prompt=f"{description}\nObjects: {object_block}\nBackground: {remembered_background}\nLayout: {layout_block}",
            character_design=character_block
        )

    def _collect_keyword_candidates(self, segment: Dict[str, Any], llm_output: str) -> List[str]:
        metadata = segment.get('mccd_data', {}) or {}
        objects = metadata.get('objects', {}) or {}
        background = metadata.get('background') or ""
        remembered_background = self.scene_memory.remember_background(background)

        object_keywords: List[str] = []
        for name, desc in objects.items():
            object_keywords.extend(_extract_keywords(f"{name} {desc}", limit=2))

        description = segment.get('description', '')
        character_tokens: List[str] = []
        if self.character_memory:
            mentions = self.character_memory.detect_mentions(description, list(objects.keys()))
            character_tokens = self.character_memory.tokens_for(mentions)

        background_keywords = _extract_keywords(remembered_background, limit=3)

        return [llm_output] + character_tokens + object_keywords + background_keywords

class ImageGeneratorAgent(BaseAgent):
    def __init__(self):
        super().__init__("ImageGenerator", "Illustrator")
        self.agent_config = self.config["image_generator"]
        self.pipe = None
        # Default quality mode set to 'high' per user request
        self.quality_mode = "high"
        self.seed = None
        self._load_model()

    def _load_model(self):
        try:
            import torch
            from diffusers import StableDiffusionPipeline as SDPipeline
            
            model_id = self.agent_config["model"]
            self.log(f"Loading Model: {model_id} with MCCD Pipeline...", Fore.GREEN)
            use_cuda = torch.cuda.is_available()
            dtype = torch.bfloat16 if use_cuda else torch.float32
            device_map = "cuda" if use_cuda else None
            
            # Use standard Stable Diffusion pipeline for stability
            self.pipe = SDPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype
            )
            if use_cuda:
                self.pipe = self.pipe.to("cuda")
            else:
                self.pipe = self.pipe.to("cpu")
            if not use_cuda:
                self.pipe.to("cpu")
            
            # Set scheduler if specified
            if "scheduler" in self.agent_config["model_params"] and hasattr(self.pipe, "scheduler"):
                from diffusers import DPMSolverMultistepScheduler, PNDMScheduler, EulerDiscreteScheduler
                scheduler_name = self.agent_config["model_params"]["scheduler"]
                if scheduler_name == "DPMSolverMultistepScheduler":
                    self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
                elif scheduler_name == "PNDMScheduler":
                    self.pipe.scheduler = PNDMScheduler.from_config(self.pipe.scheduler.config)
            
            # Temporarily disable StoryDiffusion CSA to avoid UNet shape mismatch.
            # Once CSA is stabilized, re-enable for cross-panel consistency.
            # apply_story_diffusion(self.pipe)
                
            # Optional performance / quality tweaks
            try:
                if use_cuda:
                    self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
            try:
                self.pipe.enable_vae_tiling()
            except Exception:
                pass
            # Disable safety checker for speed if present (assumes safe content in pipeline)
            if hasattr(self.pipe, "safety_checker"):
                try:
                    self.pipe.safety_checker = None
                except Exception:
                    pass
            self.log("Model loaded (standard pipeline, CSA disabled).", Fore.GREEN)
        except Exception as e:
            self.log(f"Model load failed: {e}", Fore.RED)

    def set_quality_mode(self, mode: str):
        if mode in self.agent_config.get("quality_modes", {}):
            self.quality_mode = mode
            self.log(f"Quality mode set to {mode}", Fore.CYAN)
        else:
            self.log(f"Unknown quality mode '{mode}', keeping '{self.quality_mode}'", Fore.YELLOW)

    def set_seed(self, seed: int | None):
        self.seed = seed
        self.log(f"Seed set to {seed}", Fore.CYAN)

    def _compose_negative_prompt(self) -> str:
        core = self.agent_config.get("negative_prompt_core", "")
        style = self.agent_config.get("negative_prompt_style", "")
        # Merge and deduplicate tokens
        tokens = []
        for block in [core, style]:
            for t in block.split(','):
                tt = t.strip()
                if tt and tt not in tokens:
                    tokens.append(tt)
        return ', '.join(tokens)

    def _assemble_prompt(self, segment: Dict) -> str:
        base = segment.get('final_prompt') or segment.get('description', '')
        # Emphasize core character tokens with parentheses for weighting
        if 'final_prompt' in segment:
            parts = [p.strip() for p in base.split(',') if p.strip()]
            emphasized = []
            for i, p in enumerate(parts):
                if i < 4: # emphasize first few identity tokens
                    emphasized.append(f"({p})")
                else:
                    emphasized.append(p)
            base = ', '.join(emphasized)
        style_suffix = "anime style" if 'anime style' in self.agent_config.get('prompt_template','anime style') else ""
        prompt = base if not style_suffix else f"{base}, {style_suffix}".strip(', ')
        return prompt

    def generate_images(self, segments: List[Dict]) -> List[Dict]:
        if not self.pipe:
            self.log("Pipeline not loaded, skipping generation.", Fore.RED)
            return segments
        
        # Select quality mode overrides
        qm = self.agent_config.get("quality_modes", {}).get(self.quality_mode, {})
        params = self.agent_config["model_params"].copy()
        params.update(qm)
        width = params.get("width", 512)
        height = params.get("height", 768)
        steps = params.get("num_inference_steps", 28)
        guidance = params.get("guidance_scale", 7.0)
        
        template = self.agent_config["prompt_template"]
        
        os.makedirs(ComicConfig.OUTPUT_DIR, exist_ok=True)
        
        # 1. Collect all prompts for batch generation
        image_segments = [s for s in segments if s['type'] == 'image']
        if not image_segments:
            return segments
            
        prompts = []
        neg_prompts = []
        neg_prompt_text = self._compose_negative_prompt()

        for segment in image_segments:
            prompt = self._assemble_prompt(segment)
            prompts.append(prompt)
            neg_prompts.append(neg_prompt_text)
            
        self.log(f"Generating {len(prompts)} images sequentially (mode={self.quality_mode}, steps={steps}, guidance={guidance})...", Fore.GREEN)

        # Sequential generation to reduce later frame distortion
        for idx, segment in enumerate(image_segments):
            prompt = prompts[idx]
            neg_prompt = neg_prompts[idx]
            # Slightly boost steps for first image to establish style tokens
            dynamic_steps = steps + 5 if idx == 0 else steps
            try:
                generator = None
                if self.seed is not None:
                    import torch
                    generator = torch.Generator(device=self.pipe.device).manual_seed(self.seed + idx)
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=neg_prompt,
                    width=width,
                    height=height,
                    guidance_scale=guidance,
                    num_inference_steps=dynamic_steps,
                    generator=generator
                )
                image = result.images[0]
                # Optional light sharpen using PIL (enhance edges subtly)
                try:
                    from PIL import ImageFilter
                    image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=3))
                except Exception:
                    pass
                filename = f"illustration_{segment['panel_id']}.png"
                path = os.path.join(ComicConfig.OUTPUT_DIR, filename)
                image.save(path)
                segment['image_path'] = path
                self.log(f"Saved Image {segment['panel_id']} (steps={dynamic_steps})", Fore.GREEN)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.log(f"Generation failed for panel {segment['panel_id']}: {e}", Fore.RED)
                
        return segments

class ComposerAgent(BaseAgent):
    def __init__(self):
        super().__init__("Composer", "Final Assembly")
        self.agent_config = self.config["composer"]

    def create_comic_strip(self, panels: List[Dict], output_name="comic_strip.png"):
        self.log("Assembling comic strip...", Fore.YELLOW)
        
        layout_cfg = self.agent_config["layout"]
        panels_per_row = layout_cfg.get("panels_per_row", 2)
        spacing = layout_cfg.get("panel_spacing", 10)
        bg_color = layout_cfg.get("background_color", "white")
        
        # Filter panels with images
        valid_panels = [p for p in panels if 'image_path' in p and os.path.exists(p['image_path'])]
        if not valid_panels:
            self.log("No images to assemble.", Fore.RED)
            return

        # Load images to get dimensions
        images = [Image.open(p['image_path']) for p in valid_panels]
        if not images: return
        
        # Assume all images have same size (or resize to first one)
        w, h = images[0].size
        
        # Calculate canvas size
        num_panels = len(images)
        num_rows = (num_panels + panels_per_row - 1) // panels_per_row
        
        canvas_w = (w * panels_per_row) + (spacing * (panels_per_row + 1))
        canvas_h = (h * num_rows) + (spacing * (num_rows + 1))
        
        # Create canvas
        canvas = Image.new('RGB', (canvas_w, canvas_h), bg_color)
        
        # Paste images
        for idx, img in enumerate(images):
            row = idx // panels_per_row
            col = idx % panels_per_row
            
            x = spacing + col * (w + spacing)
            y = spacing + row * (h + spacing)
            
            canvas.paste(img, (x, y))
            
            # Add Dialogue (Simple overlay for now)
            # In a real system, we'd do speech bubbles. Here we just print text at bottom.
            draw = ImageDraw.Draw(canvas)
            panel_data = valid_panels[idx]
            dialogues = panel_data.get('dialogue', [])
            
            if dialogues:
                text_y = y + h - 100 # Start text area 100px from bottom
                # Draw semi-transparent box
                draw.rectangle([(x, text_y), (x+w, y+h)], fill=(255, 255, 255, 200))
                
                # Draw text
                full_text = ""
                for d in dialogues:
                    if isinstance(d, dict):
                        full_text += f"{d.get('speaker', '')}: {d.get('text', '')}\n"
                    else:
                        full_text += str(d) + "\n"
                
                # Try to load font, fallback to default
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except:
                    font = ImageFont.load_default()
                
                draw.text((x+10, text_y+10), full_text, fill="black", font=font)

        output_path = os.path.join(ComicConfig.OUTPUT_DIR, output_name)
        canvas.save(output_path)
        self.log(f"Comic strip saved to {output_path}", Fore.GREEN)

    def create_pdf_novel(self, segments: List[Dict], title="Comic Novel", output_name="comic_novel.pdf"):
        self.log("Assembling PDF novel...", Fore.YELLOW)

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=20)

        # Register a Unicode TrueType font to avoid encoding errors
        # Try common Windows fonts first; fall back to DejaVu/Noto if present in repo.
        def register_unicode_font() -> str:
            candidates = [
                # Windows Latin fonts
                r"C:\\Windows\\Fonts\\arial.ttf",
                r"C:\\Windows\\Fonts\\calibri.ttf",
                r"C:\\Windows\\Fonts\\segoeui.ttf",
                # Windows CJK fonts (if present; fpdf2 only supports TTF, not TTC)
                r"C:\\Windows\\Fonts\\simhei.ttf",
                # Repo-provided fonts (optional): place under comic_agent/fonts/
                os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans.ttf"),
                os.path.join(os.path.dirname(__file__), "fonts", "NotoSans-Regular.ttf"),
            ]

            for path in candidates:
                try:
                    if path and os.path.exists(path):
                        pdf.add_font("Uni", "", path, uni=True)
                        return "Uni"
                except Exception:
                    continue
            # As a last resort, fall back to core font (may not support unicode fully)
            return "Arial"

        font_family = register_unicode_font()

        # Title Page
        pdf.add_page()
        # For custom Unicode TTF (family="Uni"), bold style isn't auto-synthesized.
        # Use regular style for title if using the Unicode font; use bold only for core fonts.
        if font_family == "Uni":
            pdf.set_font(font_family, '', 24)
        else:
            pdf.set_font(font_family, 'B', 24)
        try:
            pdf.cell(0, 60, title, 0, 1, 'C')
        except Exception:
            # Fallback: strip to ASCII if needed
            pdf.cell(0, 60, title.encode('ascii', 'ignore').decode('ascii'), 0, 1, 'C')

        pdf.set_font(font_family, '', 12)
        pdf.cell(0, 10, "Generated by AI Comic Studio", 0, 1, 'C')

        pdf.add_page()

        for segment in segments:
            if segment['type'] == 'text':
                pdf.set_font(font_family, '', 12)
                text = segment['content']
                # Normalize common curly quotes and dashes
                text = (
                    text.replace('’', "'")
                        .replace('‘', "'")
                        .replace('“', '"')
                        .replace('”', '"')
                        .replace('\u2014', '-')
                        .replace('\u2013', '-')
                )

                try:
                    pdf.multi_cell(0, 8, text)
                except Exception as e:
                    self.log(f"Encoding error for text segment: {e}. Skipping text.", Fore.RED)
                    # Fallback to ASCII-only rendering
                    try:
                        ascii_text = text.encode('ascii', 'ignore').decode('ascii')
                        pdf.multi_cell(0, 8, ascii_text)
                    except Exception:
                        pass
                pdf.ln(5)

            elif segment['type'] == 'image':
                if 'image_path' in segment and os.path.exists(segment['image_path']):
                    img_w = 120  # mm
                    x_offset = (210 - img_w) / 2
                    try:
                        # Page break if near bottom
                        if pdf.get_y() > 200:
                            pdf.add_page()

                        pdf.ln(5)
                        pdf.image(segment['image_path'], x=x_offset, w=img_w)
                        pdf.ln(10)
                    except Exception as e:
                        self.log(f"Failed to add image to PDF: {e}", Fore.RED)

        output_path = os.path.join(ComicConfig.OUTPUT_DIR, output_name)
        try:
            pdf.output(output_path)
            self.log(f"PDF novel saved to {output_path}", Fore.GREEN)
        except Exception as e:
            self.log(f"Failed to save PDF: {e}", Fore.RED)
