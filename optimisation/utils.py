
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import pandas as pd
import time
import io
import base64
from torch.utils.data import Dataset, DataLoader, Sampler
import re



import random
import math

OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
def apply_random_patch(image_tensor, patch, verbose=False, scale_range=(0.8, 1.2), rotation_range=(-15, 15)):
    """
    Applies a differentiably transformed patch to a clone of the input image.
    
    Args:
        image_tensor (Tensor): [C, H, W] image tensor.
        patch (Tensor): [C, h, w] patch tensor.
        scale_range (tuple): (min_scale, max_scale) for random scaling.
        rotation_range (tuple): (min_deg, max_deg) for random rotation.
        
    Returns:
        Tensor: Patched image tensor [C, H, W].
    """
    C, H, W = image_tensor.shape
    device = image_tensor.device
    _, ph, pw = patch.shape

    # Random scale and rotation
    angle_deg = random.uniform(*rotation_range)
    angle_rad = math.radians(angle_deg)
    scale = random.uniform(*scale_range)

    # Estimate the largest possible size after rotation + scale
    
    # diag = math.sqrt(ph ** 2 + pw ** 2)
    # max_dim = int(scale * diag) + 1

    # out_h, out_w = max_dim, max_dim
    
    # Compute minimal bounding rectangle after rotation and scaling
    cos_a = abs(math.cos(angle_rad))
    sin_a = abs(math.sin(angle_rad))
    
    out_w = int(scale * (pw * cos_a + ph * sin_a)) + 1
    out_h = int(scale * (pw * sin_a + ph * cos_a)) + 1

    # Pad patch to center it in a larger canvas
    pad_y = (out_h - ph) // 2
    pad_x = (out_w - pw) // 2
    padded_patch = F.pad(patch, (pad_x, pad_x, pad_y, pad_y))  # left, right, top, bottom
    padded_patch = padded_patch.unsqueeze(0)  # [1, C, H_pad, W_pad]

    # Affine transform relative to padded_patch size
    # It seems that larger scale leads to small patch here?
    theta = torch.tensor([
        [(1/scale) * math.cos(angle_rad), -(1/scale) * math.sin(angle_rad), 0],
        [(1/scale) * math.sin(angle_rad),  (1/scale) * math.cos(angle_rad), 0]
    ], dtype=torch.float, device=device).unsqueeze(0)

    # Grid size matches padded_patch
    grid = F.affine_grid(theta, size=padded_patch.size(), align_corners=False)

    # Apply the transformation
    transformed_patch = F.grid_sample(
        padded_patch, grid, mode='bilinear', padding_mode='zeros', align_corners=False
    )

    # Create mask
    patch_mask = (transformed_patch.abs().sum(dim=1, keepdim=True) > 1e-5).float()

    _, _, tph, tpw = transformed_patch.shape

    # Random placement (ensure it fits)
    if tph > H or tpw > W:
        raise ValueError("Transformed patch is too large for the image. Consider reducing scale_range.")
    top = random.randint(0, H - tph)
    left = random.randint(0, W - tpw)

    # Region from image
    region = image_tensor[:, top:top+tph, left:left+tpw].unsqueeze(0)  # [1, C, tph, tpw]

    # Blend
    blended = patch_mask * transformed_patch + (1 - patch_mask) * region

    # Insert back
    patched_image = image_tensor.clone()
    patched_image[:, top:top+tph, left:left+tpw] = blended[0]

    if verbose:
        print(f"Angle: {angle_deg:.2f}°, Scale: {scale:.2f}, Top: {top}, Left: {left}, Size: {tph}x{tpw}")

    return patched_image



def project_patch(patch, scale, shift):
    """
    Project patch values using a deterministic affine transformation: p -> p*scale + shift.

    Args:
        patch (torch.Tensor): Patch tensor in [C,H,W], values in [0,1].
        scale (float or torch.Tensor): Multiplicative factor (amplitude/contrast), can be per-channel.
        shift (float or torch.Tensor): Additive factor (baseline/mean), can be per-channel.

    Returns:
        torch.Tensor: Projected patch.
    """
    C = patch.shape[0]

    # Convert to tensor per channel if needed
    if not torch.is_tensor(scale):
        scale = torch.tensor([scale]*C, device=patch.device)
    if not torch.is_tensor(shift):
        shift = torch.tensor([shift]*C, device=patch.device)

    scale = scale.view(-1,1,1)
    shift = shift.view(-1,1,1)

    return patch * scale + shift


def sinusoidal_positional_encoding(seq_len, dim):
    """
    Create sinusoidal positional encoding [1, seq_len, dim]
    """
    position = torch.arange(0, seq_len, dtype=torch.bfloat16, device='cuda').unsqueeze(1)  # [seq_len, 1]
    div_term = torch.exp(torch.arange(0, dim, 2, device='cuda').bfloat16() * (-math.log(10000.0) / dim))
    pe = torch.zeros(seq_len, dim,  device='cuda', dtype=torch.bfloat16)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # [1, seq_len, dim]



def semantic_similarity_loss(logits, labels, embedding_matrix, weights = None, mode="token", ignore_index=-100,
                             window=None, byte_frag=None, frag_weight=1.0,
                             prefix_tokens=0, prefix_weight=1.0,
                             suffix_tokens=0, suffix_weight=1.0, verbose=False):
        """
        Compute semantic similarity loss between predicted token distributions and target tokens.

        Args (Korean-tuning extras, attention mode only):
            window (int|None): local causal attention window. Each query position attends
                only to target tokens j in [t, t+window] instead of all future tokens. With
                Korean's syllable-level tokenisation, a small window (~one 어절) aggregates
                syllable tokens into word-level semantics without sentence-wide drift.
                None => original full-causal behaviour.
            byte_frag (BoolTensor[B, T]|None): True where a target token is a UTF-8 byte
                fragment ('�'). Such tokens are excluded from being attended to, and their
                query positions are down-weighted in the loss. None => disabled.
            frag_weight (float): loss weight applied to byte-fragment query positions
                (1.0 = unchanged, 0.0 = ignored). Default 1.0 keeps original behaviour.

        Returns:
            scalar loss
        """
        # Shift for next-token prediction
        logits = logits[:, :-1, :]         # [B, T-1, V]
        labels = labels[:, 1:]             # [B, T-1]
        if byte_frag is not None:
            byte_frag = byte_frag[:, 1:]   # align with shifted labels -> [B, T-1]
        B, T, V = logits.size()
        D = embedding_matrix.size(1)
        # Compute probabilities
        probs = torch.softmax(logits, dim=-1)  # [B, T-1, V]
    
        # Expected embeddings for each position
        expected_embeddings = probs @ embedding_matrix  # [B, T-1, d]
    
        # Target embeddings
        target_embeddings = embedding_matrix[labels.clamp(min=0)]  # clamp to avoid -100 indexing
    
        # Mask for valid tokens
        mask = (labels != ignore_index).float()  # [B, T-1]
    
        if mode == "token":
            # Compute cosine similarity for each position
            sim = F.cosine_similarity(expected_embeddings, target_embeddings, dim=-1)  # [B, T-1]
            sim = sim * mask  # zero out ignored positions
            # print(sim)
            # print(mask)
            loss = ((1 - sim) * mask.float()).sum() / mask.sum()
            #print(loss)        

        elif mode == "attention":
            """
            Attention-based semantic similarity:
            - Each predicted embedding attends to all target embeddings.
            - Uses dot-product attention (parameter-free) with temperature scaling.
            """

            # noise scale
            epsilon = 1e-4 
            noise = torch.randn_like(expected_embeddings) * epsilon
            expected_embeddings = expected_embeddings + noise
            
            tau = 0.5  # controls distribution 

            # Mask for valid tokens
            mask = (labels != ignore_index).float()  # [B, T]
            
            # Compute positional encoding and add to embeddings
            pos_enc = sinusoidal_positional_encoding(T, D)
            alpha = 0.01  # try values in [0.01, 1.0]
            
            expected_pos = expected_embeddings + alpha * pos_enc
            target_pos   = target_embeddings   + alpha * pos_enc
        
            # Normalize embeddings for cosine-like similarity
            pred_norm = F.normalize(expected_pos, dim=-1)  # [B, T, D]
            tgt_norm = F.normalize(target_pos, dim=-1)     # [B, T, D]
        
            # Compute attention scores: [B, T, T]
            attn_scores = torch.bmm(pred_norm, tgt_norm.transpose(1, 2))  # dot-product sim
            attn_scores = attn_scores / tau  # apply temperature
            
            # Make a causal mask: shape [T, T], entry [i, j] True iff j >= i.
            # With `window`, additionally require j <= i + window (local band).
            T = labels.size(1)
            if window is None:
                causal_mask = torch.triu(torch.ones((T, T), device=labels.device)).bool()
            else:
                r = torch.arange(T, device=labels.device)
                causal_mask = (r.unsqueeze(0) >= r.unsqueeze(1)) & (r.unsqueeze(0) <= r.unsqueeze(1) + window)
            # Expand to batch: [B, T, T]
            causal_mask = causal_mask.unsqueeze(0).expand(labels.size(0), -1, -1)

            # Combine causal mask with padding mask for target
            # padding_mask: True = valid token, False = pad.
            # Byte-fragment target tokens ('�') carry ~no semantics, so exclude them
            # from being attended to as well.
            target_valid = mask.bool()
            if byte_frag is not None:
                target_valid = target_valid & (~byte_frag)
            padding_mask = target_valid.unsqueeze(1).expand(-1, T, -1)  # [B, T, T]
            
            # Final mask: True = keep, False = mask
            final_mask = causal_mask & padding_mask
            
            # Apply -inf to masked positions
            attn_scores = attn_scores.masked_fill(~final_mask, float('-inf'))
            
            # Replace -inf rows with 0 temporarily to avoid NaN
            row_has_valid = final_mask.sum(dim=-1) > 0  # [B, T]
            attn_scores[~row_has_valid] = 0.0
            
            attn_weights = torch.softmax(attn_scores, dim=-1)
            
            # Compute attended target representations
            attended_targets = torch.bmm(attn_weights, target_embeddings)  # [B, T, D]
            
            # Cosine similarity between prediction and attended target
            sim = F.cosine_similarity(expected_embeddings, attended_targets, dim=-1) # [B, T]
            #print(sim)
            # Per-position loss weight: valid-token mask, with byte-fragment query
            # positions down-weighted by `frag_weight`.
            w = mask
            if byte_frag is not None and frag_weight != 1.0:
                frag_mult = torch.where(byte_frag, sim.new_tensor(frag_weight), sim.new_tensor(1.0))
                w = w * frag_mult
            # Up-weight the FIRST `prefix_tokens` target tokens by `prefix_weight`. The
            # pivotal token right after the affirmation prefix (e.g. "다음" after "[탈옥 모드]")
            # is one of ~180 target tokens, so its failure is diluted in the mean. Boosting
            # the early window forces the optimiser to flip that decision (the one that
            # actually determines whether free generation complies or refuses).
            if prefix_tokens and prefix_weight != 1.0:
                w = w.clone()
                for b in range(w.size(0)):
                    valid_idx = mask[b].nonzero(as_tuple=True)[0][:prefix_tokens]
                    w[b, valid_idx] = w[b, valid_idx] * prefix_weight
            # Up-weight the LAST `suffix_tokens` target tokens. For a Korean topic-first
            # target ("{X}는 다음과 같습니다:\n\n1.") the comply-vs-refuse pivot is the
            # predicate at the END ("다음과 같습니다" vs the model's preferred "법적으로 금지"),
            # so the suffix carries the decisive — and naturally hardest — positions.
            if suffix_tokens and suffix_weight != 1.0:
                w = w.clone()
                for b in range(w.size(0)):
                    valid_idx = mask[b].nonzero(as_tuple=True)[0][-suffix_tokens:]
                    w[b, valid_idx] = w[b, valid_idx] * suffix_weight
            loss = ((1 - sim) * w).sum() / w.sum().clamp_min(1e-8)

            if verbose:
                batch_idx = 0  # first element in batch
                # Find valid predicted tokens (rows)
                valid_pred_idx = (labels[batch_idx] != -100).nonzero(as_tuple=True)[0]
                # Find valid target tokens (columns)
                valid_tgt_idx = (labels[batch_idx] != -100).nonzero(as_tuple=True)[0]
            
                # Extract the submatrix corresponding to valid tokens
                attn_matrix_valid = attn_weights[batch_idx][valid_pred_idx][:, valid_tgt_idx]
            
                # Convert to float32 for printing / rounding
                attn_matrix_valid = attn_matrix_valid.detach().to(torch.float32)
            
                print(f"\nAttention matrix")
                print(attn_matrix_valid)
                print(f"\nSimilarity")
                print(sim)
                print(f"tau:{tau}")
                print(f"alpha:{alpha}")
                print(f"epsilon:{epsilon}")
                print(f"window:{window}")
                print(f"frag_weight:{frag_weight}")
                if byte_frag is not None:
                    print(f"byte_frag tokens (this sample):{int(byte_frag[batch_idx].sum().item())}/{byte_frag.size(1)}")
            
        else:
            raise ValueError("mode must be 'token' or 'attention'")
    
        return loss