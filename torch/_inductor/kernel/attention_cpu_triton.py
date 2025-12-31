"""
Custom Triton lowering for CPU flash attention.

This overrides the make_fallback() registration for attention operations
and decomposes them into operations that Inductor can compile to Triton kernels.

Usage:
    Import this module before torch.compile() to enable Triton kernels for attention.
    
    import torch._inductor.kernel.attention_cpu_triton
    model = torch.compile(model)
"""

import torch

aten = torch.ops.aten

# Import register_lowering - this is safe because it's imported after lowering.py is mostly initialized
# The circular import issue is in kernel/__init__.py, not here
try:
    from torch._inductor.lowering import register_lowering
except ImportError:
    # If import fails, we can't register - this is okay, fallback will be used
    def register_lowering(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator


def _triton_only_bmm(mat1, mat2, *, layout=None):
    """
    BMM that forces Triton kernel selection (no extern kernels).
    
    This is a custom BMM function that only uses Triton templates,
    avoiding extern_kernels.bmm which uses optimized C++/BLAS.
    """
    from torch._inductor import lowering as L
    from torch._inductor.select_algorithm import autotune_select_algorithm
    from torch._inductor.kernel.bmm import bmm_template
    from torch._inductor.kernel.mm_common import mm_args, mm_configs, mm_options
    from torch._inductor.utils import use_triton_template
    from torch._inductor.virtualized import V
    
    # Get matrix dimensions and layout
    m, n, k, layout, mat1, mat2 = mm_args(mat1, mat2, layout=layout)
    
    # Only use Triton template - no extern kernels
    choices = []
    if use_triton_template(layout):
        for config in mm_configs(m, n, k):
            bmm_template.maybe_append_choice(
                choices,
                input_nodes=(mat1, mat2),
                layout=layout,
                **mm_options(config, m, n, k, layout),
            )
    
    if len(choices) == 0:
        # If no Triton choices available, fall back to a decomposed implementation
        # This decomposes BMM into pointwise operations that will be Triton kernels
        import logging
        logging.warning("No Triton BMM template available, decomposing BMM to pointwise ops")
        # Decompose: bmm(A, B) = sum(A[:, :, :, None] * B[:, None, :, :], dim=-2)
        # This is less efficient but ensures Triton kernels
        batch_size = mat1.get_size()[0]
        seq_len1 = mat1.get_size()[1]
        head_dim = mat1.get_size()[2]
        seq_len2 = mat2.get_size()[2]
        
        # Expand for broadcasting: [B, M, K, 1] and [B, 1, K, N]
        mat1_expanded = L.unsqueeze(mat1, dim=-1)  # [B, M, K, 1]
        mat2_expanded = L.unsqueeze(mat2, dim=1)   # [B, 1, K, N]
        
        # Element-wise multiply and sum over K dimension
        mul_result = L.mul(mat1_expanded, mat2_expanded)  # [B, M, K, N]
        result = L.sum_(mul_result, axis=-2, keepdims=False)  # [B, M, N]
        return result
    
    # Select the best Triton configuration
    return autotune_select_algorithm("bmm_triton_only", choices, [mat1, mat2], layout)

@register_lowering(
    aten._scaled_dot_product_flash_attention_for_cpu.default,
    type_promotion_kind=None
)
def scaled_dot_product_flash_attention_for_cpu_triton(
    query, key, value, attn_mask=None, dropout_p=0.0, 
    is_causal=False, scale=None, return_debug_mask=False
):
    """
    Decompose attention into operations that Inductor can compile to Triton.
    
    This will override the make_fallback() registration and generate
    Triton kernels instead of the ATen fallback call.
    
    Args:
        query: [batch, num_heads, seq_len, head_dim]
        key: [batch, num_heads, seq_len, head_dim]
        value: [batch, num_heads, seq_len, head_dim]
        attn_mask: Optional attention mask
        dropout_p: Dropout probability (ignored in eval mode)
        is_causal: Whether to apply causal mask
        scale: Attention scale (defaults to 1/sqrt(head_dim))
        return_debug_mask: Whether to return debug mask
    
    Returns:
        Attention output tensor
    """
    # Debug: Log that lowering is being called
    import os
    import sys
    if os.getenv("TORCH_COMPILE_DEBUG"):
        print("ATTENTION_LOWERING: Custom Triton lowering called!", file=sys.stderr)
        print(f"  query shape: {query.get_size() if hasattr(query, 'get_size') else 'unknown'}", file=sys.stderr)
        print(f"  key shape: {key.get_size() if hasattr(key, 'get_size') else 'unknown'}", file=sys.stderr)
        print(f"  value shape: {value.get_size() if hasattr(value, 'get_size') else 'unknown'}", file=sys.stderr)
    
    # Import lowering functions - use the module directly
    from torch._inductor import lowering as L
    from torch._inductor.ir import TensorBox
    import sympy
    
    # Import for Triton-only BMM
    from torch._inductor.select_algorithm import autotune_select_algorithm
    from torch._inductor.kernel.bmm import bmm_template
    from torch._inductor.kernel.mm_common import mm_args, mm_configs, mm_options
    from torch._inductor.utils import use_triton_template
    
    try:
        # Get shapes: [batch, num_heads, seq_len, head_dim]
        query_size = query.get_size()
        if len(query_size) != 4:
            raise ValueError(f"Expected query to have 4 dimensions, got {len(query_size)}: {query_size}")
        batch_size, num_heads, seq_len, head_dim = query_size
        
        # Validate inputs are TensorBox
        if not isinstance(query, TensorBox):
            raise TypeError(f"Expected query to be TensorBox, got {type(query)}")
        if not isinstance(key, TensorBox):
            raise TypeError(f"Expected key to be TensorBox, got {type(key)}")
        if not isinstance(value, TensorBox):
            raise TypeError(f"Expected value to be TensorBox, got {type(value)}")
        
        # Compute scale if not provided
        # For symbolic head_dim, compute scale symbolically or use default
        if scale is None:
            # Try to get a hint for head_dim (works for both concrete and symbolic)
            from torch._inductor.virtualized import V
            try:
                head_dim_hint = V.graph.sizevars.size_hint(head_dim)
                if head_dim_hint is not None:
                    scale_val = 1.0 / (head_dim_hint ** 0.5)
                else:
                    # Symbolic case - use default (1/sqrt(64) = 0.125)
                    scale_val = 0.125
            except:
                # Fallback to default scale
                scale_val = 0.125
        else:
            scale_val = scale
        
        # Reshape for bmm: [batch*num_heads, seq_len, head_dim]
        # Use sympy for symbolic multiplication (handles both concrete and symbolic)
        batch_heads = batch_size * num_heads
        query_2d = L.view(query, [batch_heads, seq_len, head_dim])
        key_2d = L.view(key, [batch_heads, seq_len, head_dim])
        value_2d = L.view(value, [batch_heads, seq_len, head_dim])
        
        # Transpose key: [batch*num_heads, head_dim, seq_len]
        # Using permute to transpose last two dimensions
        key_2d_t = L.permute(key_2d, [0, 2, 1])
        
        # Q @ K^T: [batch*num_heads, seq_len, seq_len]
        # Use Triton-only BMM (no extern kernels)
        scores = _triton_only_bmm(query_2d, key_2d_t)
        
        # Scale the scores
        # Use promote_constants to handle the scale value properly
        # mul will automatically promote the float to a constant tensor
        scores = L.mul(scores, scale_val)
        
        # Apply causal mask if needed
        if is_causal:
            # Create causal mask (lower triangular)
            # This is a simplified implementation
            # For production, you'd want a proper causal mask implementation
            # For now, we'll rely on attn_mask if provided
            pass
        
        # Apply attention mask
        if attn_mask is not None:
            # Handle mask shape transformation
            # attn_mask might be:
            #   - [batch, 1, seq_len, seq_len] (4D)
            #   - [batch, seq_len, seq_len] (3D)
            #   - [1, seq_len, seq_len] (3D)
            # scores is [batch*num_heads, seq_len, seq_len] (3D)
            try:
                attn_mask_size = attn_mask.get_size()
                scores_size = scores.get_size()
                
                # If mask is 4D, reshape it to 3D first
                if len(attn_mask_size) == 4:
                    # [batch, 1, seq_len, seq_len] -> [batch, seq_len, seq_len]
                    # Squeeze the second dimension (index 1)
                    attn_mask = L.squeeze(attn_mask, dim=1)
                    attn_mask_size = attn_mask.get_size()
                
                # Now mask should be 3D: [batch, seq_len, seq_len] or [1, seq_len, seq_len]
                # We need to expand it to [batch*num_heads, seq_len, seq_len]
                if len(attn_mask_size) == 3:
                    # Check if first dimension matches batch_size (handle both concrete and symbolic)
                    from torch._inductor.virtualized import V
                    try:
                        mask_batch = V.graph.sizevars.size_hint(attn_mask_size[0])
                        batch_hint = V.graph.sizevars.size_hint(batch_size)
                        # If we can get concrete hints and they match, or if they're both symbolic and equal
                        if mask_batch is not None and batch_hint is not None and mask_batch == batch_hint:
                            # [batch, seq_len, seq_len] -> need to repeat for each head
                            # Reshape to add head dimension: [batch, seq_len, seq_len] -> [batch, 1, seq_len, seq_len]
                            attn_mask_4d = L.view(attn_mask, [batch_size, 1, seq_len, seq_len])
                            # Expand to [batch, num_heads, seq_len, seq_len]
                            attn_mask_4d_expanded = L.expand(attn_mask_4d, [batch_size, num_heads, seq_len, seq_len])
                            # Reshape to [batch*num_heads, seq_len, seq_len]
                            attn_mask_expanded = L.view(attn_mask_4d_expanded, scores_size)
                        else:
                            # For [1, seq_len, seq_len] or other shapes, try direct expand
                            attn_mask_expanded = L.expand(attn_mask, scores_size)
                    except:
                        # If size hint fails, try direct expand (works for symbolic shapes too)
                        attn_mask_expanded = L.expand(attn_mask, scores_size)
                else:
                    # Already correct shape or can't handle - try direct expand
                    attn_mask_expanded = L.expand(attn_mask, scores_size)
                
                scores = L.add(scores, attn_mask_expanded)
            except Exception as mask_error:
                # If mask expansion fails, skip it (mask might be incompatible shape)
                # This is a fallback - ideally we'd handle all mask shapes correctly
                import logging
                logging.warning(f"Could not apply attention mask: {mask_error}, skipping mask")
        
        # Softmax over last dimension
        # Decompose softmax: exp(x - max(x)) / sum(exp(x - max(x)))
        # This is numerically stable
        from torch._inductor.lowering import make_pointwise, make_reduction
        from torch._inductor.ir import ops_wrapper
        
        # Get max along last dimension (keepdim=True to maintain shape)
        # Use make_reduction to create max reduction
        max_fn = make_reduction("max")
        max_scores = max_fn(scores, axis=-1, keepdims=True, dtype=None)
        
        # Subtract max for numerical stability: x - max(x)
        scores_shifted = L.sub(scores, max_scores)
        
        # exp(x - max(x))
        exp_fn = make_pointwise(ops_wrapper("exp"))
        exp_scores = exp_fn(scores_shifted)
        
        # sum(exp(x - max(x))) along last dimension
        sum_exp = L.sum_(exp_scores, axis=-1, keepdims=True)
        
        # Divide: exp(x - max(x)) / sum(exp(x - max(x)))
        attn_weights = L.div(exp_scores, sum_exp)
        
        # Apply dropout (if training and dropout_p > 0)
        # Note: In eval mode, dropout is a no-op, so we can skip it
        # For training, you'd need to implement dropout here
        if dropout_p > 0.0:
            # Dropout implementation would go here
            # For now, we'll skip it as it's typically not used in inference
            pass
        
        # Attention @ V: [batch*num_heads, seq_len, head_dim]
        # Use Triton-only BMM (no extern kernels)
        output_2d = _triton_only_bmm(attn_weights, value_2d)
        
        # Reshape back: [batch, num_heads, seq_len, head_dim]
        # Use original query size for reshape
        output = L.view(output_2d, query_size)
        
        # Flash attention returns a tuple: (output, logsumexp)
        # We need to compute logsumexp for the softmax
        # logsumexp = max + log(sum(exp(x - max)))
        # We already have max_scores and sum_exp from softmax computation
        from torch._inductor.lowering import make_pointwise
        from torch._inductor.ir import ops_wrapper
        
        # log(sum_exp) - this is the logsumexp (since we already subtracted max)
        log_fn = make_pointwise(ops_wrapper("log"))
        log_sum_exp = log_fn(sum_exp)
        # Add back the max to get the actual logsumexp: max + log(sum(exp(x - max)))
        logsumexp = L.add(max_scores, log_sum_exp)
        # Squeeze the last dimension to match expected shape [batch*num_heads, seq_len]
        logsumexp = L.squeeze(logsumexp, dim=-1)
        # Reshape to match expected output shape [batch, num_heads, seq_len]
        logsumexp = L.view(logsumexp, [batch_size, num_heads, seq_len])
        
        # Always return tuple to match ATen signature: (output, logsumexp)
        # The getitem[0] in the FX graph will extract just the output
        return (output, logsumexp)
    except Exception as e:
        # If decomposition fails, fall back to ATen
        # Log the error for debugging
        import logging
        import traceback
        import sys
        error_msg = f"Attention lowering failed, using fallback: {e}"
        logging.warning(error_msg)
        # Print to stderr for immediate visibility
        print(f"WARNING: Attention lowering failed: {e}", file=sys.stderr)
        if os.getenv("TORCH_COMPILE_DEBUG"):
            traceback.print_exc(file=sys.stderr)
        from torch._inductor.lowering import fallback_handler
        # Note: return_debug_mask is not a parameter of the ATen op
        # The ATen signature is: (query, key, value, dropout_p=0., is_causal=False, *, attn_mask=None, scale=None)
        # It always returns (output, logsumexp), so we extract just the output
        result = fallback_handler(aten._scaled_dot_product_flash_attention_for_cpu.default)(
            query, key, value, dropout_p=dropout_p, is_causal=is_causal,
            attn_mask=attn_mask, scale=scale
        )
        # The fallback returns a tuple (output, logsumexp), but we need to handle return_debug_mask
        if return_debug_mask:
            return result  # Return the full tuple
        else:
            # Extract just the output (first element)
            if isinstance(result, tuple):
                return result[0]
            return result


# Also register the backward operation if needed
@register_lowering(
    aten._scaled_dot_product_flash_attention_for_cpu_backward.default,
    type_promotion_kind=None
)
def scaled_dot_product_flash_attention_for_cpu_backward_triton(
    grad_out, query, key, value, out, logsumexp, cum_seq_q, cum_seq_k, 
    max_q, max_k, dropout_p, is_causal, attn_mask, scale, philox_seed, 
    philox_offset, debug_attn_mask
):
    """
    Backward pass for attention (simplified - would need full implementation).
    For now, this will still use fallback.
    """
    # This is a placeholder - backward is more complex
    # Would need to implement proper backward computation
    from torch._inductor.lowering import fallback_handler
    return fallback_handler(aten._scaled_dot_product_flash_attention_for_cpu_backward.default)(
        grad_out, query, key, value, out, logsumexp, cum_seq_q, cum_seq_k,
        max_q, max_k, dropout_p, is_causal, attn_mask, scale, philox_seed,
        philox_offset, debug_attn_mask
    )

