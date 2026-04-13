import torch
import os
import sys
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor

sys.path.append(str(Path(__file__).parent.parent))

from internnav.model.basemodel.internvla_n1.internvla_n1_system1 import InternVLAN1System1
from internnav.model.basemodel.internvla_n1.internvla_n1_system2 import InternVLAN1System2

def verify_system_split(checkpoint_dir):
    s1_path = os.path.join(checkpoint_dir, "system1", "model.safetensors")
    s2_dir = os.path.join(checkpoint_dir, "system2")

    extracted_latents = None

    print("=== 1. System 2 Loading and Inference Test ===")
    try:
        model2 = InternVLAN1System2.from_pretrained_system2(
            s2_dir, 
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(s2_dir)
        print("✅ System 2 loaded successfully")
        
        image = Image.new('RGB', (224, 224), color='red')
        prompt = "just stop right now and do nothing."
        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model2.device)
        
        traj_tokens = torch.full((1, 4), 151667, dtype=torch.long, device=model2.device)
        inputs['input_ids'] = torch.cat([inputs['input_ids'], traj_tokens], dim=1)
        
        with torch.no_grad():
            model2(**inputs)
            extracted_latents = model2.get_last_latents()
            
            generated_ids = model2.generate(**inputs, max_new_tokens=20, do_sample=False)
            response = processor.tokenizer.decode(generated_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
        print(f"✅ System 2 inference successful!")
        print(f"💬 VLM Response: [ {response} ]")
        print(f"🎯 Extracted Latent Shape: {extracted_latents.shape}")
        
    except Exception as e:
        print(f"❌ System 2 test failed: {e}")

    print("\n=== 2. System 1 Loading and Integration Test ===")
    try:
        model1 = InternVLAN1System1.from_pretrained_system1(
            s1_path, 
            device="cuda:1" if torch.cuda.is_available() else "cpu", 
            dtype=torch.bfloat16
        )
        print("✅ System 1 loaded successfully")

        assert extracted_latents is not None, "Failed to extract latents from System 2. System 1 test cannot proceed."
        input_latent = extracted_latents.to(model1.device)

        dummy_images = torch.randn(1, 2, 224, 224, 3, device=model1.device, dtype=torch.bfloat16)
        traj = model1.generate_traj(input_latent, dummy_images, num_inference_steps=10)
        
        print("\n📈 Generated Trajectory Samples (Top 5 waypoints [x, y, yaw]):")
        sample_traj = traj[0, :5, :].detach().cpu().float().numpy() 
        for i, pt in enumerate(sample_traj):
            print(f"  Step {i+1}: x={pt[0]:.4f}, y={pt[1]:.4f}, yaw={pt[2]:.4f}")

    except Exception as e:
        print(f"❌ System 1 test failed: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str)
    args = parser.parse_args()
    verify_system_split(args.checkpoint_dir)