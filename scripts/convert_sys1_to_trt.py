import argparse
import os
import sys
from pathlib import Path

import tensorrt as trt
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))
from internnav.model.basemodel.internvla_n1.internvla_n1_system1 import InternVLAN1System1

# This patches ALL Transformer components, including MultiheadAttention,
# to decompose them into basic matrix operations. This is necessary to
# bypass 'aten::_native_multi_head_attention' and other fused kernels.
def patch_transformer_for_onnx():
    # 0. Manual MultiheadAttention implementation with full projection support
    def manual_mha_decomposed(mha, q, k, v, attn_mask=None, key_padding_mask=None, is_causal=False):
        batch_first = getattr(mha, "batch_first", False)
        embed_dim = mha.embed_dim
        num_heads = mha.num_heads
        head_dim = embed_dim // num_heads
        device = q.device
        dtype = q.dtype
        
        # Linear Projections
        if mha._qkv_same_embed_dim:
            if mha.in_proj_weight is not None:
                q_w, k_w, v_w = mha.in_proj_weight.chunk(3)
                q_b, k_b, v_b = mha.in_proj_bias.chunk(3) if mha.in_proj_bias is not None else (None, None, None)
                q = F.linear(q, q_w, q_b)
                k = F.linear(k, k_w, k_b)
                v = F.linear(v, v_w, v_b)
            else:
                q = F.linear(q, mha.q_proj_weight, mha.bias_q if hasattr(mha, 'bias_q') else None)
                k = F.linear(k, mha.k_proj_weight, mha.bias_k if hasattr(mha, 'bias_k') else None)
                v = F.linear(v, mha.v_proj_weight, mha.bias_v if hasattr(mha, 'bias_v') else None)
        else:
            q = F.linear(q, mha.q_proj_weight, mha.bias_q if hasattr(mha, 'bias_q') else None)
            k = F.linear(k, mha.k_proj_weight, mha.bias_k if hasattr(mha, 'bias_k') else None)
            v = F.linear(v, mha.v_proj_weight, mha.bias_v if hasattr(mha, 'bias_v') else None)

        # Reshape for heads
        if batch_first:
            B, Lq, _ = q.shape
            _, Lk, _ = k.shape
            q = q.view(B, Lq, num_heads, head_dim).transpose(1, 2)
            k = k.view(B, Lk, num_heads, head_dim).transpose(1, 2)
            v = v.view(B, Lk, num_heads, head_dim).transpose(1, 2)
        else:
            Lq, B, _ = q.shape
            Lk, _, _ = k.shape
            q = q.view(Lq, B, num_heads, head_dim).permute(1, 2, 0, 3)
            k = k.view(Lk, B, num_heads, head_dim).permute(1, 2, 0, 3)
            v = v.view(Lk, B, num_heads, head_dim).permute(1, 2, 0, 3)

        # Scaled Dot Product (Explicitly avoid float64 promotion)
        scaling = torch.tensor(head_dim, dtype=dtype, device=device).pow(-0.5)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scaling
        
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_weights.masked_fill_(attn_mask.unsqueeze(1), float("-inf"))
            else:
                attn_weights = attn_weights + attn_mask.unsqueeze(1)
                
        if key_padding_mask is not None:
            pm = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(pm, float("-inf"))
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        if batch_first:
            attn_output = attn_output.transpose(1, 2).contiguous().view(B, Lq, embed_dim)
        else:
            attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(Lq, B, embed_dim)
            
        # Final projection
        attn_output = mha.out_proj(attn_output)
        return attn_output, None

    # Patch Encoder/Decoder functions
    def new_encoder_forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src
        if getattr(self, "norm_first", False):
            nx = self.norm1(x)
            attn_output = manual_mha_decomposed(self.self_attn, nx, nx, nx, attn_mask=src_mask, 
                                                key_padding_mask=src_key_padding_mask, is_causal=is_causal)[0]
            x = x + self.dropout1(attn_output)
            nx = self.norm2(x)
            ff_output = self.linear2(self.dropout(self.activation(self.linear1(nx))))
            x = x + self.dropout2(ff_output)
        else:
            attn_output = manual_mha_decomposed(self.self_attn, x, x, x, attn_mask=src_mask, 
                                                key_padding_mask=src_key_padding_mask, is_causal=is_causal)[0]
            x = self.norm1(x + self.dropout1(attn_output))
            ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
            x = self.norm2(x + self.dropout2(ff_output))
        return x

    def new_decoder_forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                            tgt_key_padding_mask=None, memory_key_padding_mask=None,
                            tgt_is_causal=False, memory_is_causal=False):
        x = tgt
        if getattr(self, "norm_first", False):
            nx = self.norm1(x)
            attn_output = manual_mha_decomposed(self.self_attn, nx, nx, nx, attn_mask=tgt_mask, 
                                                key_padding_mask=tgt_key_padding_mask, is_causal=tgt_is_causal)[0]
            x = x + self.dropout1(attn_output)
            nx = self.norm2(x)
            attn_output = manual_mha_decomposed(self.multihead_attn, nx, memory, memory, attn_mask=memory_mask, 
                                                key_padding_mask=memory_key_padding_mask, is_causal=memory_is_causal)[0]
            x = x + self.dropout2(attn_output)
            nx = self.norm3(x)
            ff_output = self.linear2(self.dropout(self.activation(self.linear1(nx))))
            x = x + self.dropout3(ff_output)
        else:
            attn_output = manual_mha_decomposed(self.self_attn, x, x, x, attn_mask=tgt_mask, 
                                                key_padding_mask=tgt_key_padding_mask, is_causal=tgt_is_causal)[0]
            x = self.norm1(x + self.dropout1(attn_output))
            attn_output = manual_mha_decomposed(self.multihead_attn, x, memory, memory, attn_mask=memory_mask, 
                                                key_padding_mask=memory_key_padding_mask, is_causal=memory_is_causal)[0]
            x = self.norm2(x + self.dropout2(attn_output))
            ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
            x = self.norm3(x + self.dropout3(ff_output))
        return x

    def new_encoder_container_forward(self, src, mask=None, src_key_padding_mask=None, is_causal=None):
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, is_causal=is_causal)
        if self.norm is not None:
            output = self.norm(output)
        return output

    # Direct patching
    nn.TransformerEncoderLayer.forward = new_encoder_forward
    nn.TransformerDecoderLayer.forward = new_decoder_forward
    nn.TransformerEncoder.forward = new_encoder_container_forward
    
    # Force decomposition by disabling fast path
    if hasattr(nn.TransformerEncoder, '_can_use_fast_path'):
        nn.TransformerEncoder._can_use_fast_path = lambda *args, **kwargs: False
    if hasattr(nn.TransformerDecoder, '_can_use_fast_path'):
        nn.TransformerDecoder._can_use_fast_path = lambda *args, **kwargs: False
        
    print("Transformer & MHA patches applied (Manual decomposition).")

def export_onnx(model_path, onnx_path, device, dtype):
    patch_transformer_for_onnx()
    model = InternVLAN1System1.from_pretrained_system1(model_path, device=device, dtype=dtype)
    model.eval()

    dummy_latents = torch.randn(1, 8, 768).to(device, dtype)
    dummy_images = torch.randn(1, 2, 224, 224, 3).to(device, dtype)
    dummy_noise = torch.randn(1, 32, 3).to(device, dtype)

    from contextlib import nullcontext
    ctx = nullcontext()
    if hasattr(torch.backends.cuda, "sdp_kernel"):
        ctx = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
    elif hasattr(torch.nn, "attention") and hasattr(torch.nn.attention, "sdp_kernel"):
        ctx = torch.nn.attention.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)

    with ctx:
        torch.onnx.export(
            model, (dummy_latents, dummy_images, dummy_noise), onnx_path,
            input_names=['traj_latents', 'images_dp', 'noise'],
            output_names=['trajectory'],
            dynamic_axes={
                'traj_latents': {0: 'batch_size'},
                'images_dp': {0: 'batch_size', 1: 'seq_len'},
                'noise': {0: 'batch_size'},
                'trajectory': {0: 'batch_size'}
            },
            opset_version=14,
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL
        )
    print(f"ONNX Export Complete: {onnx_path}")
    
    del model
    torch.cuda.empty_cache()

def build_engine(onnx_path, engine_path):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False

    config = builder.create_builder_config()
    config.builder_optimization_level = 5
    config.avg_timing_iterations = 8

    try:
        config.set_flag(trt.BuilderFlag.BF16)
    except:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    profile.set_shape("traj_latents", (1, 8, 768), (1, 8, 768), (1, 8, 768))
    profile.set_shape("images_dp", (1, 1, 224, 224, 3), (1, 2, 224, 224, 3), (1, 4, 224, 224, 3))
    profile.set_shape("noise", (1, 32, 3), (1, 32, 3), (1, 32, 3))
    config.add_optimization_profile(profile)

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024)
    
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        return False

    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--engine_path", type=str, required=True)
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.engine_path), exist_ok=True)
    
    # Use a temporary ONNX path
    onnx_path = args.engine_path + ".temp.onnx"
    
    try:
        print(f"Starting System 1 TRT Conversion...")
        export_onnx(args.model_path, onnx_path, "cuda", torch.bfloat16)
        
        if build_engine(onnx_path, args.engine_path):
            print(f"TRT Engine Build Successful: {args.engine_path}")
        else:
            print(f"Error: TRT Engine Build Failed.")
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
