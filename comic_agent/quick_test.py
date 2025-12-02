import sys
import os
from colorama import Fore, Style, init

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
    init(autoreset=True)
    print(f"{Fore.CYAN}=== AI Comic Studio: Quick Test Mode ==={Style.RESET_ALL}")
    
    # Pre-defined Test Scenario (Cyberpunk Theme)
    theme = "Cyberpunk detective investigating a crime scene"
    characters = "Detective Kael: gritty man, stubble, trench coat, cybernetic eye. Android Unit 7: sleek white robot, glowing blue visor."
    scene_desc = "A dark, rain-soaked alleyway with neon signs reflecting on puddles. Police tape in the background."
    
    print(f"{Fore.YELLOW}Using Test Inputs:{Style.RESET_ALL}")
    print(f"Theme: {theme}")
    print(f"Characters: {characters}")
    print(f"Scene: {scene_desc}")
    
    # 1. Init Agents
    print(f"\n{Fore.CYAN}Initializing Agents...{Style.RESET_ALL}")
    script_writer = ScriptWriterAgent()
    layout_planner = LayoutPlannerAgent()
    refiner = RefinerAgent()
    image_generator = ImageGeneratorAgent()
    composer = ComposerAgent()
    
    # 2. Pipeline Execution
    print(f"\n{Fore.CYAN}--- Step 1: Script Generation ---{Style.RESET_ALL}")
    script = script_writer.generate_script(theme, characters, scene_desc)
    print(f"Generated {len(script)} panels.")
    
    print(f"\n{Fore.CYAN}--- Step 2: Layout Planning ---{Style.RESET_ALL}")
    panels_with_layout = layout_planner.plan_layout(script)
    
    print(f"\n{Fore.CYAN}--- Step 3: Refinement ---{Style.RESET_ALL}")
    refined_panels = refiner.refine_prompts(panels_with_layout, characters)
    
    print(f"\n{Fore.CYAN}--- Step 4: Image Generation ---{Style.RESET_ALL}")
    final_panels = image_generator.generate_images(refined_panels)
    
    print(f"\n{Fore.CYAN}--- Step 5: Composition ---{Style.RESET_ALL}")
    composer.create_comic_strip(final_panels, output_name="quick_test_result.png")
        
    print(f"\n{Fore.GREEN}Test Complete! Check output_comics/quick_test_result.png{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
