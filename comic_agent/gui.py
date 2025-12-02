import gradio as gr
import os
import sys
import json

# Add the current directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from agents import (
    DirectorAgent,
    CharacterDesignerAgent, 
    ScreenwriterAgent, 
    StoryboardAgent, 
    IllustratorAgent
)
from config import ComicConfig

# Initialize Agents
print("Initializing Agents...")
try:
    director = DirectorAgent()
    char_designer = CharacterDesignerAgent()
    screenwriter = ScreenwriterAgent()
    storyboarder = StoryboardAgent()
    illustrator = IllustratorAgent()
except Exception as e:
    print(f"Error initializing agents: {e}")
    director = None
    char_designer = None
    screenwriter = None
    storyboarder = None
    illustrator = None

def generate_comic(theme, progress=gr.Progress()):
    if not director or not illustrator:
        return "Error: Agents not initialized properly. Check console logs.", [], ""

    # 1. Director: World Building
    progress(0.1, desc="Director: Establishing World Bible...")
    try:
        world_bible = director.analyze_request(theme)
        title = world_bible.get('title', 'Untitled')
        style = world_bible.get('style_prompt', 'Anime')
        setting = world_bible.get('setting_description', 'Generic')
        briefs = world_bible.get('character_briefs', [])
        
        log_text = f"Title: {title}\nStyle: {style}\nSetting: {setting}\nBriefs: {briefs}\n\n"
    except Exception as e:
        return f"Error in Director Phase: {e}", [], ""

    # 2. Character Design: Locking Visuals
    progress(0.2, desc="Character Designer: Locking Visual Tags...")
    try:
        char_designs = char_designer.design_characters(briefs)
        log_text += f"--- Character Designs (Immutable Tags) ---\n{json.dumps(char_designs, indent=2)}\n\n"
    except Exception as e:
        return f"Error in Character Design: {e}\n{log_text}", [], ""

    # 3. Scripting: Narrative
    progress(0.4, desc="Screenwriter: Drafting Script...")
    try:
        script = screenwriter.write_script(title, setting, char_designs)
        log_text += f"--- Script ---\n{json.dumps(script, indent=2)}\n\n"
    except Exception as e:
        return f"Error in Scripting: {e}\n{log_text}", [], ""
    
    # 4. Storyboarding: Strict Prompt Assembly
    progress(0.6, desc="Storyboarder: Assembling Prompts...")
    try:
        final_panels = storyboarder.generate_prompts(script, char_designs, style)
    except Exception as e:
        return f"Error in Storyboarding: {e}\n{log_text}", [], ""

    # 5. Illustration: Rendering
    progress(0.8, desc="Illustrator: Rendering Panels...")
    image_paths = []
    try:
        for i, panel in enumerate(final_panels):
            progress(0.8 + (0.2 * (i / len(final_panels))), desc=f"Rendering Panel {i+1}...")
            path = illustrator.draw_panel(panel)
            if path:
                caption = f"Panel {panel['panel_id']}: {panel.get('action_description', '')}"
                image_paths.append((path, caption))
            
    except Exception as e:
        return f"Error generating images: {e}\n{log_text}", [], ""

    return log_text, image_paths, json.dumps(world_bible, indent=2)

# Custom CSS
custom_css = """
.container { max-width: 1200px; margin: auto; }
"""

with gr.Blocks(title="Multi-Agent Comic Studio (Strict Mode)", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# 🎨 Multi-Agent Comic Studio (Strict Consistency Mode)")
    gr.Markdown("Powered by DeepSeek & Animagine XL. Agents enforce strict character and style consistency.")
    
    with gr.Row():
        with gr.Column(scale=1):
            theme_input = gr.Textbox(label="Comic Theme", placeholder="e.g. A cyberpunk detective investigating a neon-lit alleyway", lines=3)
            generate_btn = gr.Button("Generate Comic", variant="primary")
            
            with gr.Accordion("Debug Info (World Bible)", open=False):
                debug_output = gr.JSON(label="Project Info")
                
        with gr.Column(scale=2):
            gallery = gr.Gallery(label="Generated Comic", columns=2, height="auto")
            log_output = gr.Textbox(label="Process Log", lines=10)

    generate_btn.click(
        fn=generate_comic,
        inputs=[theme_input],
        outputs=[log_output, gallery, debug_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)