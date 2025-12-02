import sys
import os
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Ensure we can import from the same directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from config import ComicConfig
from agents import (
    ScriptWriterAgent,
    LayoutPlannerAgent,
    RefinerAgent,
    ImageGeneratorAgent,
    ComposerAgent
)

def main():
    print(f"{Fore.CYAN}=== AI Comic Studio (Multi-Agent System) ==={Style.RESET_ALL}")
    print(f"{Fore.CYAN}Mode: Advanced Pipeline (Script -> Layout -> Refine -> Render -> Compose){Style.RESET_ALL}")
    
    # 1. Init Agents
    script_writer = ScriptWriterAgent()
    layout_planner = LayoutPlannerAgent()
    refiner = RefinerAgent()
    image_generator = ImageGeneratorAgent()
    composer = ComposerAgent()
    
    # 2. User Input
    theme = input("Enter Comic Theme (e.g., 'Cyberpunk detective in rain'): ")
    if not theme: theme = "Cyberpunk detective in rain"
    
    characters = input("Enter Characters (e.g., 'Detective: silver hair, trench coat'): ")
    if not characters: characters = "Detective: silver hair, trench coat"
    
    scene_desc = input("Enter Scene Description (e.g., 'Neon-lit rainy alleyway'): ")
    if not scene_desc: scene_desc = "Neon-lit rainy alleyway"
    
    # 3. Script Generation
    print(f"\n{Fore.CYAN}--- Step 1: Script Generation ---{Style.RESET_ALL}")
    script = script_writer.generate_script(theme, characters, scene_desc)
    print(f"Generated {len(script)} panels.")
    
    # 4. Layout Planning
    print(f"\n{Fore.CYAN}--- Step 2: Layout Planning ---{Style.RESET_ALL}")
    panels_with_layout = layout_planner.plan_layout(script)
    
    # 5. Refinement (Prompt Engineering & Consistency)
    print(f"\n{Fore.CYAN}--- Step 3: Refinement & Consistency Check ---{Style.RESET_ALL}")
    refined_panels = refiner.refine_prompts(panels_with_layout, characters)
    
    # 6. Image Generation
    print(f"\n{Fore.CYAN}--- Step 4: Image Generation ---{Style.RESET_ALL}")
    final_panels = image_generator.generate_images(refined_panels)
    
    # 7. Composition
    print(f"\n{Fore.CYAN}--- Step 5: Final Composition ---{Style.RESET_ALL}")
    composer.create_comic_strip(final_panels)
        
    print(f"\n{Fore.GREEN}Comic Creation Complete! Output in {ComicConfig.OUTPUT_DIR}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()