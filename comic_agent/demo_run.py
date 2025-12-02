import sys
import os
import time
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Ensure we can import from the same directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from agents import (
    DirectorAgent,
    CharacterDesignerAgent,
    ScreenwriterAgent,
    StoryboardAgent,
    IllustratorAgent
)
from config import ComicConfig

def run_demo():
    print(f"{Fore.CYAN}=== Comic System Quality Assurance Demo ==={Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Goal: Verify fix for CLIP token limit & Image Quality{Style.RESET_ALL}\n")

    # 1. Initialize Agents
    if not ComicConfig.DEEPSEEK_API_KEY:
        print(f"{Fore.RED}Error: DEEPSEEK_API_KEY not found. Check your .env path in config.py{Style.RESET_ALL}")
        return

    try:
        director = DirectorAgent()
        char_designer = CharacterDesignerAgent()
        screenwriter = ScreenwriterAgent()
        storyboarder = StoryboardAgent()
        illustrator = IllustratorAgent()
    except Exception as e:
        print(f"{Fore.RED}Agent Initialization Failed: {e}{Style.RESET_ALL}")
        return

    # 2. Predefined Complex Theme (to test limits)
    theme = "A steampunk alchemist discovering a glowing spirit in his cluttered laboratory"
    print(f"Theme: {theme}")

    # 3. Director
    print(f"\n{Fore.CYAN}[1/5] Director: World Building...{Style.RESET_ALL}")
    world_bible = director.analyze_request(theme)
    print(f"Title: {world_bible.get('title')}")
    print(f"Style: {world_bible.get('style_prompt')}")

    # 4. Character Design (Should be concise now)
    print(f"\n{Fore.CYAN}[2/5] Character Designer: Designing (Concise Mode)...{Style.RESET_ALL}")
    char_designs = char_designer.design_characters(world_bible.get('character_briefs', []))
    for name, tags in char_designs.items():
        print(f"  - {name}: {tags[:50]}... (Len: {len(tags)})")

    # 5. Scripting
    print(f"\n{Fore.CYAN}[3/5] Screenwriter: Drafting Script...{Style.RESET_ALL}")
    script = screenwriter.write_script(
        world_bible.get('title'), 
        world_bible.get('setting_description'), 
        char_designs
    )
    print(f"Generated {len(script)} panels.")

    # 6. Storyboarding (Should optimize prompts)
    print(f"\n{Fore.CYAN}[4/5] Storyboarder: Optimizing Prompts (Target < 75 tokens)...{Style.RESET_ALL}")
    final_panels = storyboarder.generate_prompts(
        script, 
        char_designs, 
        world_bible.get('style_prompt')
    )
    
    # Check prompt lengths
    for p in final_panels:
        prompt_len = len(p['sd_prompt'].split())
        print(f"  - Panel {p['panel_id']} Prompt Words: {prompt_len}")
        print(f"    Prompt: {p['sd_prompt'][:100]}...")

    # 7. Illustration
    print(f"\n{Fore.CYAN}[5/5] Illustrator: Rendering...{Style.RESET_ALL}")
    
    # Create demo specific output folder
    demo_output_dir = os.path.join(ComicConfig.OUTPUT_DIR, "demo_run")
    os.makedirs(demo_output_dir, exist_ok=True)
    
    if not final_panels:
        print(f"{Fore.RED}No panels to render! Check previous steps.{Style.RESET_ALL}")
        return

    for panel in final_panels:
        start_time = time.time()
        # Temporarily override output dir for demo
        original_output_dir = ComicConfig.OUTPUT_DIR
        ComicConfig.OUTPUT_DIR = demo_output_dir
        
        path = illustrator.draw_panel(panel)
        
        # Restore
        ComicConfig.OUTPUT_DIR = original_output_dir
        
        duration = time.time() - start_time
        if path:
            abs_path = os.path.abspath(path)
            print(f"  - Panel {panel['panel_id']} rendered in {duration:.1f}s")
            print(f"    Saved to: {Fore.GREEN}{abs_path}{Style.RESET_ALL}")
        else:
            print(f"  - Panel {panel['panel_id']} FAILED.")

    print(f"\n{Fore.GREEN}=== Demo Complete ==={Style.RESET_ALL}")

if __name__ == "__main__":
    run_demo()