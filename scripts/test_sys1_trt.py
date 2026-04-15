import torch
import os
import sys
import time
import argparse
import tensorrt as trt
from pathlib import Path
from typing import cast, Optional

sys.path.append(str(Path(__file__).parent.parent))
from internnav.model.basemodel.internvla_n1.internvla_n1_system1 import InternVLAN1System1

class TRTSystem1Runner:
    def __init__(self, engine_path, logger_level=trt.Logger.INFO):
        self.logger = trt.Logger(logger_level)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.current_stream().cuda_stream

    def generate_traj(
        self,
        traj_latents: torch.Tensor, 
        images_dp: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        num_inference_steps: int = 10,
        num_sample_trajs: int = 1,
        **kwargs
    ) -> torch.Tensor:
        self.context.set_input_shape("traj_latents", traj_latents.shape)
        self.context.set_input_shape("images_dp", images_dp.shape)
        if noise is not None:
            self.context.set_input_shape("noise", noise.shape)

        output_shape = (traj_latents.shape[0], 32, 3)
        output_tensor = torch.empty(output_shape, dtype=torch.bfloat16, device='cuda')

        self.context.set_tensor_address("traj_latents", traj_latents.data_ptr())
        self.context.set_tensor_address("images_dp", images_dp.data_ptr())
        if noise is not None:
            self.context.set_tensor_address("noise", noise.data_ptr())
        self.context.set_tensor_address("trajectory", output_tensor.data_ptr())

        self.context.execute_async_v3(self.stream)
        torch.cuda.synchronize()
        return output_tensor

def benchmark_system1(pt_model_path, engine_path):
    device = "cuda"
    dtype = torch.bfloat16
    
    latents_in = torch.randn(1, 8, 768).to(device, dtype)
    images_in = torch.randn(1, 2, 224, 224, 3).to(device, dtype)
    noise_in = torch.randn(1, 32, 3).to(device, dtype)

    pt_model = InternVLAN1System1.from_pretrained_system1(pt_model_path, device=device, dtype=dtype)
    pt_model.eval()
    
    trt_model = cast(InternVLAN1System1, TRTSystem1Runner(engine_path))

    with torch.no_grad():
        pt_out = pt_model.generate_traj(latents_in, images_in, noise=noise_in, num_inference_steps=10, num_sample_trajs=1)
        trt_out = trt_model.generate_traj(latents_in, images_in, noise=noise_in)

    diff = (pt_out - trt_out).abs()
    mse = torch.mean(diff**2).item()
    print(f"✅ Correctness Test - MSE: {mse:.10f}")

    num_runs = 50
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = pt_model.generate_traj(latents_in, images_in, noise=noise_in, num_inference_steps=10, num_sample_trajs=1)
    torch.cuda.synchronize()
    pt_latency = (time.time() - start_time) / num_runs * 1000

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        _ = trt_model.generate_traj(latents_in, images_in, noise=noise_in)
    torch.cuda.synchronize()
    trt_latency = (time.time() - start_time) / num_runs * 1000

    print("-" * 50)
    print(f"Safetensors Latency: {pt_latency:.2f} ms ({1000/pt_latency:.2f} Hz)")
    print(f"TensorRT Latency:    {trt_latency:.2f} ms ({1000/trt_latency:.2f} Hz)")
    print(f"Speedup: {pt_latency/trt_latency:.2f}x")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--engine_path", type=str, default="checkpoints/splited/system1/system1.engine")
    args = parser.parse_args()
    benchmark_system1(args.model_path, args.engine_path)
