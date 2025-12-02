import torch
from diffusers import DiffusionPipeline
import os

def verify_model():
    print("Starting model verification...")
    
    # Ensure output directory exists
    output_dir = "output_comics/verification"
    os.makedirs(output_dir, exist_ok=True)
    
    model_id = "ogkalu/Comic-Diffusion"
    print(f"Loading model: {model_id}")
    
    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_id, 
            dtype=torch.bfloat16, 
            device_map="cuda"
        )
        print("Model loaded successfully.")
        
        prompts = [
            "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
            "A cyberpunk detective standing in the rain, neon lights, anime style",
            "A magical forest with glowing mushrooms, fantasy art style"
        ]
        
        for i, prompt in enumerate(prompts):
            print(f"Generating image for prompt: {prompt}")
            image = pipe(prompt).images[0]
            save_path = os.path.join(output_dir, f"test_image_{i+1}.png")
            image.save(save_path)
            print(f"Saved to {save_path}")
            
        print("Verification complete. Please check the images in output_comics/verification/")
        
    except Exception as e:
        print(f"Verification failed: {e}")

if __name__ == "__main__":
    verify_model()
