import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import tensorrt as trt

sys.path.append(str(Path(__file__).parent.parent))
from internnav.model.basemodel.internvla_n1.internvla_n1_system1 import InternVLAN1System1

class TRTSystem1Runner:
    """Full-model TRT runner.

    The BF16 engine was built from an FP32 ONNX model, so this runner
    casts inputs → fp32 before TRT, and casts the fp32 output
    back to the input dtype.
    """

    def __init__(self, engine_path, logger_level=trt.Logger.INFO):
        self.logger = trt.Logger(logger_level)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream_pt = torch.cuda.Stream()
        self.stream = self.stream_pt.cuda_stream

    def generate_traj(
        self,
        traj_latents: torch.Tensor,
        images_dp: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        num_inference_steps: int = 10,
        num_sample_trajs: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        # Cast input → fp32 for the FP32-ONNX TRT engine
        latents = traj_latents.float().contiguous()
        images = images_dp.float().contiguous()
        noise = noise.float().contiguous() if noise is not None else None

        # Output is FP32, shape (batch, 32, 3)
        output = torch.empty(
            (latents.shape[0], 32, 3),
            dtype=torch.float32,
            device=traj_latents.device,
        )

        self.context.set_input_shape('traj_latents', tuple(latents.shape))
        self.context.set_input_shape('images_dp', tuple(images.shape))
        if noise is not None:
            self.context.set_input_shape('noise', tuple(noise.shape))
        self.context.set_tensor_address('trajectory', output.data_ptr())

        self.context.set_tensor_address('traj_latents', latents.data_ptr())
        self.context.set_tensor_address('images_dp', images.data_ptr())
        if noise is not None:
            self.context.set_tensor_address('noise', noise.data_ptr())
        self.context.execute_async_v3(self.stream)
        self.stream_pt.synchronize()

        # Cast back to caller's dtype
        return output.to(traj_latents.dtype)

def check_correctness(pt_out, trt_out, atol=1e-2, rtol=1e-2, mse_threshold=1e-3, cos_threshold=0.999):
    pt_flat = pt_out.detach().view(-1).float()
    trt_flat = trt_out.detach().view(-1).float()

    cos_sim = F.cosine_similarity(pt_flat, trt_flat, dim=0).item()
    cos_pass = cos_sim >= cos_threshold

    diff = (pt_flat - trt_flat).abs()
    mse = torch.mean(diff**2).item()
    mse_pass = mse < mse_threshold
    max_diff = torch.max(diff).item()

    is_close = torch.allclose(pt_out.float(), trt_out.float(), atol=atol, rtol=rtol)

    print(f'--- Correctness Test Result ---')
    print(f'{"✅" if cos_pass else "❌"} 1. Cosine Similarity: {cos_sim:.6f} (Target: >{cos_threshold})')
    print(f'{"✅" if mse_pass else "⚠️"} 2. MSE: {mse:.2e}')
    print(f'{"✅" if is_close else "❌"} 3. Max Diff (w/ rtol): {max_diff:.2e}')

    if not is_close or not cos_pass:
        diff_2d = (pt_out.detach().float() - trt_out.detach().float()).abs().squeeze(0)  # (32, 3)
        print(f'    Per-axis Max Diff │ vx: {diff_2d[:, 0].max():.4f}  vy: {diff_2d[:, 1].max():.4f}  vyaw: {diff_2d[:, 2].max():.4f}')
        worst_step = diff_2d.max(dim=1).values.argmax().item()
        print(f'    Worst step: [{worst_step}]  diff={diff_2d[worst_step]}  pt={pt_out[0, worst_step].float()}  trt={trt_out[0, worst_step].float()}')

    if is_close and cos_pass:
        print('✅ Correctness: PASS')
        return True
    else:
        print('❌ Correctness: FAIL')
        return False


def benchmark_system1(pt_model_path, engine_path):
    device = 'cuda:1'
    torch.cuda.set_device(device)
    dtype = torch.bfloat16

    # Load models
    pt_model = InternVLAN1System1.from_pretrained_system1(
        pt_model_path, device=device, dtype=dtype
    )
    pt_model.eval()
    trt_model = TRTSystem1Runner(engine_path)

    # ------------------------------------------------------------------
    # Correctness Test (100 runs, both models with num_inference_steps=10)
    # ------------------------------------------------------------------
    print(f'\n{"="*60}')
    print(f' Correctness Test (num_inference_steps=10, baked in TRT)')
    print(f'{"="*60}')

    pass_count = 0
    total_count = 100

    failures = []
    for i in range(total_count):
        latents_in = torch.randn(1, 4, 768).to(device, dtype)
        images_in = torch.randn(1, 2, 224, 224, 3).to(device, dtype)
        noise_in = torch.randn(1, 32, 3).to(device, dtype)

        with torch.inference_mode():
            pt_out = pt_model.generate_traj(
                latents_in, images_in, noise=noise_in,
                num_inference_steps=10, num_sample_trajs=1,
            )
            trt_out = trt_model.generate_traj(
                latents_in, images_in, noise=noise_in,
            )

        if not check_correctness(pt_out, trt_out, atol=1e-2, rtol=1e-2):
            diff = (pt_out.float() - trt_out.float()).abs()
            failures.append(diff.max().item())
        else:
            pass_count += 1

    print(f'\n{"="*60}')
    print(f' Correctness Summary: {pass_count}/{total_count} PASSED')
    print(f'Fail max diffs: {sorted(failures)}')
    print(f'{"="*60}')

    # ------------------------------------------------------------------
    # Latency Benchmark
    # ------------------------------------------------------------------
    print(f'{"="*60}')
    print(f' Latency Benchmark (num_inference_steps=10)')
    print(f'{"="*60}')

    latents_in = torch.randn(1, 4, 768).to(device, dtype)
    images_in = torch.randn(1, 2, 224, 224, 3).to(device, dtype)
    noise_in = torch.randn(1, 32, 3).to(device, dtype)
    num_runs = 50

    # --- PyTorch warmup + benchmark ---
    with torch.inference_mode():
        for _ in range(5):
            _ = pt_model.generate_traj(
                latents_in, images_in, noise=noise_in,
                num_inference_steps=10, num_sample_trajs=1,
            )

    torch.cuda.synchronize()
    start_time = time.time()
    with torch.inference_mode():
        for _ in range(num_runs):
            _ = pt_model.generate_traj(
                latents_in, images_in, noise=noise_in,
                num_inference_steps=10, num_sample_trajs=1,
            )
    torch.cuda.synchronize()
    pt_latency = (time.time() - start_time) / num_runs * 1000

    # --- TRT warmup + benchmark ---
    for _ in range(5):
        _ = trt_model.generate_traj(latents_in, images_in, noise=noise_in)

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        _ = trt_model.generate_traj(latents_in, images_in, noise=noise_in)
    torch.cuda.synchronize()
    trt_latency = (time.time() - start_time) / num_runs * 1000

    print('-' * 50)
    print(f'Safetensors Latency: {pt_latency:.2f} ms ({1000/pt_latency:.2f} Hz)')
    print(f'TensorRT Latency:    {trt_latency:.2f} ms ({1000/trt_latency:.2f} Hz)')
    print(f'Speedup: {pt_latency/trt_latency:.2f}x')
    print('-' * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--engine_path', type=str, required=True)
    args = parser.parse_args()
    benchmark_system1(args.model_path, args.engine_path)
