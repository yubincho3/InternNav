import argparse
import json
import os
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

from PIL import Image

import torch
from transformers import AutoProcessor

sys.path.append(str(Path(__file__).parent.parent))
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
from internnav.model.basemodel.internvla_n1.internvla_n1_system1 import InternVLAN1System1
from internnav.model.basemodel.internvla_n1.internvla_n1_system2 import InternVLAN1System2

SYSTEM1_PREFIXES = (
    'model.traj_dit.',
    'model.action_encoder.',
    'model.action_decoder.',
    'model.pos_encoding.',
    'model.rgb_model.',
    'model.memory_encoder.',
    'model.rgb_resampler.',
)

def split_model(args):
    print('\n===== Starting model split process =====')

    try:
        # 1. Load model
        print(f'Loading model: {args.model_dir}')
        model = InternVLAN1ForCausalLM.from_pretrained(
            args.model_dir,
            torch_dtype=torch.bfloat16,
            device_map={'': args.device}
        )
        model.eval()

        # 2. Divide state_dict
        print('Splitting model state_dict into System 1 and System 2 components...')
        system1_sd, system2_sd = OrderedDict(), OrderedDict()
        for key, value in model.state_dict().items():
            if any(key.startswith(p) for p in SYSTEM1_PREFIXES):
                system1_sd[key] = value
            else:
                system2_sd[key] = value

        # 3. Save
        from safetensors.torch import save_file

        s1_dir = os.path.join(args.output_dir, 'system1')
        s2_dir = os.path.join(args.output_dir, 'system2')
        os.makedirs(s1_dir, exist_ok=True)
        os.makedirs(s2_dir, exist_ok=True)

        print(f'Saving System 1 model to: {s1_dir}')
        save_file({k: v.cpu() for k, v in system1_sd.items()}, os.path.join(s1_dir, 'model.safetensors'))
        print(f'Saving System 2 model to: {s2_dir}')
        save_file({k: v.cpu() for k, v in system2_sd.items()}, os.path.join(s2_dir, 'model.safetensors'))

        # Copy metadata files (config.json, etc.) from original model to system2
        for file in os.listdir(args.model_dir):
            if file.endswith('.json') or 'tokenizer' in file or 'merges.txt' in file:
                src_path = os.path.join(args.model_dir, file)
                dst_path = os.path.join(s2_dir, file)
                print(f'Copying {file} to {dst_path}')
                if file == 'config.json':
                    with open(src_path, 'r') as f:
                        config_data = json.load(f)
                    config_data['model_type'] = 'qwen2_5_vl'
                    config_data['architectures'] = ['Qwen2_5_VLForConditionalGeneration']
                    with open(dst_path, 'w') as f:
                        json.dump(config_data, f, indent=2)
                else:
                    shutil.copy(src_path, dst_path)

        # Copy config/tokenizer/processor files for System2
        print('Copying tokenizer and config files to System 2 directory...')
        S2_FILES = [
            'generation_config.json',
            'preprocessor_config.json',
            'tokenizer_config.json',
            'vocab.json',
            'merges.txt',
            'added_tokens.json',
            'special_tokens_map.json',
            'chat_template.json',
        ]
        for fname in S2_FILES:
            src = os.path.join(args.model_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(s2_dir, fname))

        print(f'✅ Split complete: {args.output_dir}')
    except Exception as e:
        print(f'❌ Error during model split: {e}')
        raise

def test_splited(checkpoint_dir):
    print('\n===== Starting verification of split models =====')

    s1_path = os.path.join(checkpoint_dir, 'system1', 'model.safetensors')
    s2_dir = os.path.join(checkpoint_dir, 'system2')

    extracted_latents = None

    print('=== 1. System 2 Loading and Inference Test ===')
    try:
        model2 = InternVLAN1System2.from_pretrained_system2(
            s2_dir, 
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
        processor = AutoProcessor.from_pretrained(s2_dir)
        print('✅ System 2 loaded successfully')
        
        image = Image.new('RGB', (224, 224), color='red')
        prompt = 'stop right now!!!'
        messages = [
            {'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': prompt}]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors='pt').to(model2.device)

        traj_tokens = torch.full((1, 4), 151667, dtype=torch.long, device=model2.device)
        latent_input_ids = torch.cat([inputs['input_ids'], traj_tokens], dim=1)

        with torch.inference_mode():
            # Latent extraction: input_ids WITH traj tokens
            latent_inputs = dict(inputs)
            latent_inputs['input_ids'] = latent_input_ids
            model2(**latent_inputs)
            extracted_latents = model2.get_last_latents()

            # Text generation: input_ids WITHOUT traj tokens (original)
            generated_ids = model2.generate(**inputs, max_new_tokens=20, do_sample=False)
            response = processor.tokenizer.decode(generated_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        print(f'✅ System 2 inference successful!')
        print(f'VLM Response: [ {repr(response)} ]')
        print(f'Extracted Latent Shape: {extracted_latents.shape}')

    except Exception as e:
        print(f'❌ System 2 test failed: {e}')

    print('\n=== 2. System 1 Loading and Integration Test ===')
    try:
        model1 = InternVLAN1System1.from_pretrained_system1(
            s1_path, 
            device='cuda' if torch.cuda.is_available() else 'cpu', 
            dtype=torch.bfloat16
        )
        print('✅ System 1 loaded successfully')

        assert extracted_latents is not None, 'Failed to extract latents from System 2. System 1 test cannot proceed.'
        input_latent = extracted_latents.to(model1.device)

        dummy_images = torch.randn(1, 2, 224, 224, 3, device=model1.device, dtype=torch.bfloat16)
        traj = model1.generate_traj(input_latent, dummy_images, num_inference_steps=10)
        
        print('\nGenerated Trajectory Samples (Top 5 waypoints [x, y, yaw]):')
        sample_traj = traj[0, :5, :].detach().cpu().float().numpy() 
        for i, pt in enumerate(sample_traj):
            print(f'  Step {i+1}: x={pt[0]:.4f}, y={pt[1]:.4f}, yaw={pt[2]:.4f}')

    except Exception as e:
        print(f'❌ System 1 test failed: {e}')
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    try:
        split_model(args)
        test_splited(args.output_dir)
    except Exception as e:
        print(f'❌ Error in main execution: {e}')
