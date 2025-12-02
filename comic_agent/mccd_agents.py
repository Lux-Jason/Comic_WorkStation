import json
from typing import Dict, List, Any
from colorama import Fore, Style

try:
    from agents import BaseAgent
except ImportError:
    from .agents import BaseAgent

class MCCDSceneParser(BaseAgent):
    """
    Implements the Multi-agent Collaboration-based Scene Parsing (MCP) module 
    from MCCD (arXiv:2505.02648).
    """
    def __init__(self):
        super().__init__("MCCD_Parser", "Scene Parsing")
        # We simulate the 6 sub-agents within this class to avoid overhead of 6 separate API calls if possible,
        # or we can make distinct calls. For fidelity to the paper, we should use distinct prompts/personas.

    def parse_scene(self, scene_description: str) -> Dict[str, Any]:
        self.log(f"Parsing scene using MCCD Multi-Agent System...", Fore.CYAN)
        
        # 1. Object Extraction Agent
        objects = self._run_agent("Object Extraction", 
            "Extract all physical objects and their visual characteristics from the text. Return JSON list.", 
            scene_description)
        
        # 2. Background Extraction Agent
        background = self._run_agent("Background Extraction", 
            "Extract the object-independent background description. Return string.", 
            scene_description)
        
        # 3. Action & Spatial Relations (Combined for efficiency, or separate as per paper)
        relations = self._run_agent("Spatial/Action Relations", 
            "Describe the spatial positions and action relationships between objects.", 
            scene_description)
        
        # 4. Layout Agent (The most critical one for HCD)
        # We need Bounding Boxes [x, y, w, h] normalized 0-1.
        layout_prompt = f"""
        Based on the objects: {objects}
        And relations: {relations}
        Generate a JSON dictionary where keys are object names and values are bounding boxes [x, y, w, h] (0.0-1.0).
        Example: {{"cat": [0.1, 0.5, 0.2, 0.2]}}
        """
        layout = self._run_agent("Layout Agent", "You are a Layout Composition Expert.", layout_prompt)
        
        # 5. Aesthetics Enhancement Agent
        # Refine the object descriptions
        enhanced_objects = self._run_agent("Aesthetics Enhancement", 
            "Enhance the object descriptions for high-quality artistic generation. Return JSON dict {name: description}.", 
            str(objects))

        return {
            "background": background,
            "layout": self._parse_json(layout),
            "objects": self._parse_json(enhanced_objects)
        }

    def _run_agent(self, agent_name: str, system_prompt: str, user_prompt: str) -> str:
        self.log(f"[{agent_name}] working...", Fore.MAGENTA)
        response = self.chat(user_prompt, system_prompt=system_prompt)
        return response

    def _parse_json(self, text: str):
        try:
            # Simple cleanup to find JSON structure
            start = text.find('{')
            end = text.rfind('}') + 1
            if start == -1 or end == 0:
                start = text.find('[')
                end = text.rfind(']') + 1
            
            if start != -1 and end != 0:
                return json.loads(text[start:end])
            return text
        except:
            return text
