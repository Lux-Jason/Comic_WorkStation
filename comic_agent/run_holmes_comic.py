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
    print(f"{Fore.CYAN}=== AI Comic Studio: Holmes Special Edition ==={Style.RESET_ALL}")
    
    # 1. Init Agents
    script_writer = ScriptWriterAgent()
    layout_planner = LayoutPlannerAgent()
    refiner = RefinerAgent()
    image_generator = ImageGeneratorAgent()
    composer = ComposerAgent()
    
    # 2. Hardcoded Inputs for Quick Test
    theme = "Victorian London Detective Mystery, Sherlock Holmes style"
    
    characters = """
    1. Detective Arthur Pendelton: Sharp features, deerstalker hat, smoking pipe, long trench coat, intelligent eyes.
    2. Dr. William Vance: Round glasses, mustache, tweed suit, carrying a notebook, worried expression.
    3. Inspector Lestrade-type: Police uniform, stern face, mutton chops, holding a lantern.
    4. The Victim - Lord Blackwood: Aristocratic, elderly, grey hair, fine suit, lying on floor.
    5. The Suspect - Lady Blackwood: Elegant red dress, holding a fan, secretive look, dark hair.
    """
    
    scene_desc = """
    A dark and foggy Victorian London. 
    Scene 1: Baker Street Study - messy, books, chemistry set.
    Scene 2: Carriage Ride - foggy streets, gaslamps.
    Scene 3: Crime Scene - The Library - dark, body on floor, books scattered.
    Scene 4: Interrogation Room - dim light, wooden table.
    Scene 5: The Chase - Rooftops at night, moonlit.
    Scene 6: The Arrest - Street corner, gaslight, police present.
    """
    
    print(f"{Fore.YELLOW}Theme: {theme}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Characters: {characters[:50]}...{Style.RESET_ALL}")
    
    # 3. Script Generation (Novel Mode)
    print(f"\n{Fore.CYAN}--- Step 1: Novel Writing ---{Style.RESET_ALL}")
    # We override the prompt slightly to ensure we get 6 scenes
    segments = script_writer.write_novel(theme, characters, scene_desc)
    print(f"Generated {len(segments)} segments (Text + Images).")
    
    # 4. Layout Planning (Skipped for Novel)
    # print(f"\n{Fore.CYAN}--- Step 2: Layout Planning ---{Style.RESET_ALL}")
    # panels_with_layout = layout_planner.plan_layout(script)
    
    # 5. Refinement
    print(f"\n{Fore.CYAN}--- Step 2: Refinement & Consistency Check ---{Style.RESET_ALL}")
    refined_segments = refiner.refine_prompts(segments, characters)
    
    # 6. Image Generation
    print(f"\n{Fore.CYAN}--- Step 3: Image Generation ---{Style.RESET_ALL}")
    final_segments = image_generator.generate_images(refined_segments)
    
    # 7. Composition (PDF)
    print(f"\n{Fore.CYAN}--- Step 4: Final Composition (PDF Novel) ---{Style.RESET_ALL}")
    composer.create_pdf_novel(final_segments, title="The Mystery of Lord Blackwood", output_name="holmes_novel.pdf")
        
    print(f"\n{Fore.GREEN}Comic Novel Creation Complete! Output in {ComicConfig.OUTPUT_DIR}/holmes_novel.pdf{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
