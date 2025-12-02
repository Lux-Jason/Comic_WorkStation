import os
from colorama import Fore, Style
from typing import List, Dict

from config import ComicConfig
from agents import ScriptWriterAgent, RefinerAgent, ImageGeneratorAgent, ComposerAgent

# Batch definitions: different genres & lengths
NOVELS = [
    {
        "folder": "detective_blackwood",
        "title": "The Mystery of Lord Blackwood",
        "theme": "Victorian London detective mystery, atmospheric clues, deductive tension",
        "length_style": "short ~6 scenes",
        "characters": "Detective Arthur Pendelton: analytical, pipe; Dr. William Vance: supportive physician; Lady Blackwood: secretive aristocrat",
        "scene_desc": "Initial discovery in a lamplit study"
    },
    {
        "folder": "space_odyssey",
        "title": "Echoes Beyond Proxima",
        "theme": "Space opera exploration near Proxima Centauri, alien artifact, cosmic awe",
        "length_style": "medium ~8 scenes",
        "characters": "Commander Lyra Chen: resolute; Xenobiologist Malik Rao: curious; AI Navigator ORBIT: calm guidance; Artifact Sentinel: enigmatic",
        "scene_desc": "Approach to drifting alien monolith"
    },
    {
        "folder": "epic_fantasy",
        "title": "Shards of the Emerald Throne",
        "theme": "Epic fantasy quest, ancient dragons, court intrigue, mystic relic shards",
        "length_style": "long ~10 scenes",
        "characters": "Arin Stormward: young mage; Ser Caldre: knight guardian; Lady Myriel: court seer; Vaelrix: ancient emerald dragon",
        "scene_desc": "Storm-swept ridge overlooking shattered citadel"
    },
    {
        "folder": "cyberpunk_city",
        "title": "Neon Shadows of District Zero",
        "theme": "Cyberpunk noir, megacorp infiltration, neural hacking, rain-soaked alleys",
        "length_style": "short ~6 scenes",
        "characters": "Rin Kaito: netrunner; Shade-7: covert android; Exec Helena Arq: ruthless; Street Broker Vox: info dealer",
        "scene_desc": "Rainy alley with flickering holo signage"
    },
    {
        "folder": "cozy_cafe",
        "title": "Whispers at the Midnight Cafe",
        "theme": "Cozy slice-of-life night cafe, quiet revelations, warm ambience",
        "length_style": "short ~5 scenes",
        "characters": "Barista Lila: gentle listener; Novelist Ezra: introspective; Traveler Mina: wistful; Old Clock: silent presence",
        "scene_desc": "Soft glow interior, antique clock ticking"
    },
]


def ensure_directory(path: str):
    os.makedirs(path, exist_ok=True)


def generate_single_novel(meta: Dict):
    # Use a stable base to avoid accidental nested paths if OUTPUT_DIR mutates
    base_output_dir = getattr(ComicConfig, "OUTPUT_DIR", "output_comics")
    folder_path = os.path.join(base_output_dir, meta["folder"])
    ensure_directory(folder_path)

    # Override global output directory for this run
    ComicConfig.OUTPUT_DIR = folder_path

    print(f"\n{Fore.CYAN}=== Generating Novel: {meta['title']} ({meta['folder']}) ==={Style.RESET_ALL}")

    # Instantiate agents
    script_writer = ScriptWriterAgent()
    refiner = RefinerAgent()
    image_gen = ImageGeneratorAgent()
    composer = ComposerAgent()

    # Ensure high quality mode explicitly (default already set to high after patch)
    image_gen.set_quality_mode("high")

    # Build theme directive adding length request
    theme_directive = f"{meta['theme']} | Write a {meta['length_style']} narrative."

    # Write novel (segments include text + illustration markers)
    segments = script_writer.write_novel(theme_directive, meta['characters'], meta['scene_desc'])

    # Extract illustration segments for refinement & image generation
    illustration_segments: List[Dict] = []
    final_segments: List[Dict] = []

    for seg in segments:
        if seg['type'] == 'image':
            illustration_segments.append(seg)
        final_segments.append(seg)

    # Prepare segments in the format expected by RefinerAgent
    segments_for_refiner = []
    for idx, s in enumerate(illustration_segments, start=1):
        segments_for_refiner.append({
            'type': 'image',
            'panel_id': idx,
            'description': s.get('description', ''),
            'rich_description': s.get('description', ''),
            'mccd_data': s.get('mccd_data', {})
        })

    refined_segments = refiner.refine_prompts(segments_for_refiner, character_design=meta['characters'])

    # Generate images using refined segments (the generator saves and returns paths)
    generated_segments = image_gen.generate_images(refined_segments)

    # Attach generated image paths and prompts back into original illustration segments
    for i, gen_seg in enumerate(generated_segments, start=1):
        if i-1 < len(illustration_segments):
            orig = illustration_segments[i-1]
            orig['image_path'] = gen_seg.get('image_path')
            orig['refined_prompt'] = gen_seg.get('final_prompt', gen_seg.get('description', ''))

    # Compose PDF novel
    pdf_name = f"{meta['folder']}_novel.pdf"
    composer.create_pdf_novel(final_segments, title=meta['title'], output_name=pdf_name)

    # Restore original output dir
    ComicConfig.OUTPUT_DIR = base_output_dir

    print(f"{Fore.GREEN}Completed: {meta['title']} -> {folder_path}{Style.RESET_ALL}")


def main():
    print(f"{Fore.MAGENTA}=== Batch Novel Generation (High Quality) ==={Style.RESET_ALL}")
    for novel in NOVELS:
        try:
            generate_single_novel(novel)
        except Exception as e:
            print(f"{Fore.RED}Failed '{novel['title']}': {e}{Style.RESET_ALL}")

    print(f"\n{Fore.MAGENTA}Batch generation finished. Check individual folders under: {getattr(ComicConfig, 'OUTPUT_DIR', 'output_comics')}{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
