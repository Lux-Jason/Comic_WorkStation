import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor2_0

class ConsistentSelfAttentionProcessor(AttnProcessor2_0):
    """
    Implements Consistent Self-Attention (CSA) from StoryDiffusion (arXiv:2405.01434).
    This allows a batch of images to attend to each other during generation, 
    ensuring character and scene consistency.
    """
    def __init__(self):
        super().__init__()
        print("DEBUG: ConsistentSelfAttentionProcessor initialized")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale=1.0,
    ):
        # --- Consistent Self-Attention Logic ---
        
        batch_size, sequence_length, _ = hidden_states.shape
        print(f"DEBUG: StoryDiffusion __call__ executed. Batch={batch_size}, Seq={sequence_length}")
        # If encoder_hidden_states is not None, it's Cross-Attention (Text-to-Image).
        # We only want to modify Self-Attention (Image-to-Image consistency).
        if encoder_hidden_states is not None:
            return super().__call__(
                attn, 
                hidden_states, 
                encoder_hidden_states, 
                attention_mask, 
                temb, 
                scale
            )

        # --- Consistent Self-Attention Logic ---
        
        batch_size, sequence_length, _ = hidden_states.shape
        # print(f"DEBUG: StoryDiffusion Input: batch={batch_size}, seq={sequence_length}")
        
        # If batch_size is 1, we can't do consistency across images.
        if batch_size <= 1:
            return super().__call__(
                attn, 
                hidden_states, 
                encoder_hidden_states, 
                attention_mask, 
                temb, 
                scale
            )

        # 1. Project Q, K, V
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # Reshape to (Batch, Heads, Seq, Dim)
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        print(f"DEBUG: Query shape: {query.shape}")
        print(f"DEBUG: Key shape: {key.shape}")

        # 2. Construct Shared Keys and Values
        # We handle Classifier Free Guidance (CFG) by assuming the batch is [Uncond, Cond].
        # We only want to share attention within the "Cond" part (and maybe Uncond part separately).
        # But mixing them is bad.
        
        # Heuristic: If batch_size is even, we assume it's [Uncond, Cond] split.
        # This is a simplification but works for standard pipelines.
        is_cfg = batch_size % 2 == 0 and batch_size > 1
        
        if is_cfg:
            half = batch_size // 2
            # Split into Uncond and Cond
            key_uncond, key_cond = key.chunk(2, dim=0)
            value_uncond, value_cond = value.chunk(2, dim=0)
            
            # Process Uncond (Share within Uncond group)
            k_u_shared = key_uncond.transpose(0, 1).reshape(1, attn.heads, -1, head_dim)
            v_u_shared = value_uncond.transpose(0, 1).reshape(1, attn.heads, -1, head_dim)
            k_u_shared = k_u_shared.repeat(half, 1, 1, 1)
            v_u_shared = v_u_shared.repeat(half, 1, 1, 1)
            
            # Process Cond (Share within Cond group)
            k_c_shared = key_cond.transpose(0, 1).reshape(1, attn.heads, -1, head_dim)
            v_c_shared = value_cond.transpose(0, 1).reshape(1, attn.heads, -1, head_dim)
            k_c_shared = k_c_shared.repeat(half, 1, 1, 1)
            v_c_shared = v_c_shared.repeat(half, 1, 1, 1)
            
            # Concatenate back
            try:
                key_shared = torch.cat([k_u_shared, k_c_shared], dim=0)
                value_shared = torch.cat([v_u_shared, v_c_shared], dim=0)
            except RuntimeError as e:
                print(f"Error in StoryDiffusion cat: {e}")
                print(f"k_u_shared: {k_u_shared.shape}")
                print(f"k_c_shared: {k_c_shared.shape}")
                raise e
            
        else:
            # No CFG, share across all
            key_shared = key.transpose(0, 1).reshape(1, attn.heads, -1, head_dim)
            value_shared = value.transpose(0, 1).reshape(1, attn.heads, -1, head_dim)
            key_shared = key_shared.repeat(batch_size, 1, 1, 1)
            value_shared = value_shared.repeat(batch_size, 1, 1, 1)
            
        # 3. Perform Attention
        # Q: (Batch, Heads, Seq, Dim)
        # K: (Batch, Heads, Batch/2 * Seq, Dim) if CFG
        # V: (Batch, Heads, Batch/2 * Seq, Dim) if CFG
        
        # IMPORTANT: If attention_mask is provided, it usually matches the original sequence length.
        # Since we expanded the Key/Value sequence length, the mask will mismatch.
        # For Self-Attention in Image Generation, masking is usually not needed (except for padding which we don't have here).
        # So we force attn_mask to None to avoid shape mismatch errors.
        
        try:
            hidden_states = F.scaled_dot_product_attention(
                query, key_shared, value_shared, attn_mask=None, dropout_p=0.0, is_causal=False
            )
        except RuntimeError as e:
            print(f"Error in StoryDiffusion attention: {e}")
            print(f"query: {query.shape}")
            print(f"key_shared: {key_shared.shape}")
            print(f"value_shared: {value_shared.shape}")
            raise e

        # 4. Reshape back
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        print(f"DEBUG: Output shape before projection: {hidden_states.shape}")

        # 5. Output Projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states        # 4. Reshape back
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # 5. Output Projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

def apply_story_diffusion(pipe):
    """
    Applies the Consistent Self-Attention processor to the pipeline's UNet.
    """
    print("Applying StoryDiffusion (Consistent Self-Attention) to UNet...")
    # We only want to apply this to Self-Attention layers.
    # In SD/SDXL UNet, these are usually in the transformer blocks.
    
    new_attn_processors = {}
    
    for name, processor in pipe.unet.attn_processors.items():
        # 'attn1' is usually Self-Attention
        # 'attn2' is usually Cross-Attention
        if name.endswith("attn1.processor"):
            new_attn_processors[name] = ConsistentSelfAttentionProcessor()
        else:
            new_attn_processors[name] = processor # Keep existing (likely AttnProcessor2_0)
            
    pipe.unet.set_attn_processor(new_attn_processors)
    print("StoryDiffusion applied successfully.")

def remove_story_diffusion(pipe):
    """
    Restores standard attention processors.
    """
    from diffusers.models.attention_processor import AttnProcessor2_0
    print("Removing StoryDiffusion hooks...")
    pipe.unet.set_attn_processor(AttnProcessor2_0())
