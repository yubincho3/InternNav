import argparse
import os
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM

SYSTEM1_PREFIXES = (
    'model.traj_dit.',
    'model.action_encoder.',
    'model.action_decoder.',
    'model.cond_projector.',
    'model.pos_encoding.',
    'model.rgb_model.',
    'model.memory_encoder.',
    'model.rgb_resampler.',
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    # 1. Load model
    print(f'Loading model: {args.model_path}')
    model = InternVLAN1ForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={'': 'cpu'}
    )
    model.eval()

    # 2. Divide state_dict
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

    save_file({k: v.cpu() for k, v in system1_sd.items()}, os.path.join(s1_dir, "model.safetensors"))
    save_file({k: v.cpu() for k, v in system2_sd.items()}, os.path.join(s2_dir, "model.safetensors"))

    # Copy config/tokenizer file for System2
    for fname in os.listdir(args.model_path):
        if fname.endswith('.safetensors') or fname.endswith('.bin'):
            continue
        src = os.path.join(args.model_path, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(s2_dir, fname))

    print(f'Complete: {args.output_dir}')

if __name__ == '__main__':
    main()
