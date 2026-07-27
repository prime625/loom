"""
Loom VAE — Tiny imagination engine for texture patches.
Trains with PyTorch, infers with pure NumPy (no PyTorch needed at runtime).
"""
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# ---------------------------------------------------------------------------
# Pure NumPy VAE (inference only)
# ---------------------------------------------------------------------------

class TinyVAENumpy:
    """
    Tiny CNN VAE: 64x64x3 -> 256-dim latent -> 64x64x3
    ~2.2M params, ~9MB float32, runs on any CPU.
    """
    
    def __init__(self, latent_dim: int = 256):
        self.latent_dim = latent_dim
        self.weights = {}
    
    def load_weights(self, weight_dict: dict):
        """Load exported PyTorch weights as NumPy arrays."""
        self.weights = weight_dict
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def _tanh(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def _conv2d(self, x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int = 2) -> np.ndarray:
        """
        x: (C, H, W)
        w: (out_C, in_C, K, K)
        b: (out_C,)
        """
        in_C, H, W = x.shape
        out_C, _, K, _ = w.shape
        out_H = (H - K) // stride + 1
        out_W = (W - K) // stride + 1
        out = np.zeros((out_C, out_H, out_W), dtype=np.float32)
        
        for oc in range(out_C):
            for ic in range(in_C):
                for oh in range(out_H):
                    for ow in range(out_W):
                        h0 = oh * stride
                        w0 = ow * stride
                        patch = x[ic, h0:h0+K, w0:w0+K]
                        out[oc, oh, ow] += np.sum(patch * w[oc, ic])
            out[oc] += b[oc]
        return out
    
    def _conv_transpose2d(self, x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int = 2) -> np.ndarray:
        """
        x: (C, H, W)
        w: (in_C, out_C, K, K)
        b: (out_C,)
        """
        in_C, H, W = x.shape
        _, out_C, K, _ = w.shape
        out_H = (H - 1) * stride + K
        out_W = (W - 1) * stride + K
        out = np.zeros((out_C, out_H, out_W), dtype=np.float32)
        
        for ic in range(in_C):
            for oh in range(H):
                for ow in range(W):
                    h0 = oh * stride
                    w0 = ow * stride
                    out[:, h0:h0+K, w0:w0+K] += x[ic, oh, ow] * w[ic]
        
        for oc in range(out_C):
            out[oc] += b[oc]
        return out
    
    def _linear(self, x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        return x @ w.T + b
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        x: (3, 64, 64) in [-1, 1]
        Returns: (latent_dim,) latent vector
        """
        w = self.weights
        
        # Conv1: 3x64x64 -> 16x32x32
        h = self._conv2d(x, w['enc_conv1_w'], w['enc_conv1_b'], stride=2)
        h = self._relu(h)
        
        # Conv2: 16x32x32 -> 32x16x16
        h = self._conv2d(h, w['enc_conv2_w'], w['enc_conv2_b'], stride=2)
        h = self._relu(h)
        
        # Conv3: 32x16x16 -> 64x8x8
        h = self._conv2d(h, w['enc_conv3_w'], w['enc_conv3_b'], stride=2)
        h = self._relu(h)
        
        # Flatten -> Linear
        h_flat = h.flatten()
        z = self._linear(h_flat, w['enc_fc_w'], w['enc_fc_b'])
        return z
    
    def decode(self, z: np.ndarray) -> np.ndarray:
        """
        z: (latent_dim,)
        Returns: (3, 64, 64) in [-1, 1]
        """
        w = self.weights
        
        # Linear -> 64x8x8
        h = self._linear(z, w['dec_fc_w'], w['dec_fc_b'])
        h = self._relu(h)
        h = h.reshape(64, 8, 8)
        
        # Deconv1: 64x8x8 -> 32x16x16
        h = self._conv_transpose2d(h, w['dec_deconv1_w'], w['dec_deconv1_b'], stride=2)
        h = self._relu(h)
        
        # Deconv2: 32x16x16 -> 16x32x32
        h = self._conv_transpose2d(h, w['dec_deconv2_w'], w['dec_deconv2_b'], stride=2)
        h = self._relu(h)
        
        # Deconv3: 16x32x32 -> 3x64x64
        h = self._conv_transpose2d(h, w['dec_deconv3_w'], w['dec_deconv3_b'], stride=2)
        h = self._tanh(h)
        
        return h


# ---------------------------------------------------------------------------
# PyTorch Training (only used during corpus building)
# ---------------------------------------------------------------------------

def train_vae(patches: List[np.ndarray], latent_dim: int = 256, epochs: int = 50, lr: float = 1e-3) -> dict:
    """
    Train Tiny VAE on texture patches.
    patches: list of (64, 64, 3) uint8 arrays
    Returns: weight dict for NumPy inference
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        print("PyTorch not installed. Cannot train VAE.")
        print("Install: pip install torch --index-url https://download.pytorch.org/whl/cpu")
        return None
    
    device = torch.device("cpu")
    
    # Normalize patches to [-1, 1]
    data = np.stack([p.astype(np.float32) / 127.5 - 1.0 for p in patches])
    data = torch.from_numpy(data).permute(0, 3, 1, 2).to(device)  # (N, 3, 64, 64)
    dataset = torch.utils.data.TensorDataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    class TinyVAE(nn.Module):
        def __init__(self, latent_dim=256):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 16, 4, 2, 1),   # 32x32
                nn.ReLU(),
                nn.Conv2d(16, 32, 4, 2, 1),  # 16x16
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2, 1),  # 8x8
                nn.ReLU(),
            )
            self.fc_enc = nn.Linear(64 * 8 * 8, latent_dim)
            self.fc_dec = nn.Linear(latent_dim, 64 * 8 * 8)
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 16x16
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, 4, 2, 1),  # 32x32
                nn.ReLU(),
                nn.ConvTranspose2d(16, 3, 4, 2, 1),   # 64x64
                nn.Tanh(),
            )
        
        def forward(self, x):
            h = self.encoder(x)
            h = h.view(h.size(0), -1)
            z = self.fc_enc(h)
            h = self.fc_dec(z)
            h = h.view(h.size(0), 64, 8, 8)
            return self.decoder(h), z
    
    model = TinyVAE(latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"Training VAE on {len(patches)} patches, {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0.0
        for (batch,) in loader:
            recon, z = model(batch)
            loss = F.mse_loss(recon, batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
    
    # Export weights to NumPy
    state = model.state_dict()
    weight_dict = {
        'enc_conv1_w': state['encoder.0.weight'].cpu().numpy(),      # (16, 3, 4, 4)
        'enc_conv1_b': state['encoder.0.bias'].cpu().numpy(),      # (16,)
        'enc_conv2_w': state['encoder.2.weight'].cpu().numpy(),    # (32, 16, 4, 4)
        'enc_conv2_b': state['encoder.2.bias'].cpu().numpy(),      # (32,)
        'enc_conv3_w': state['encoder.4.weight'].cpu().numpy(),    # (64, 32, 4, 4)
        'enc_conv3_b': state['encoder.4.bias'].cpu().numpy(),      # (64,)
        'enc_fc_w': state['fc_enc.weight'].cpu().numpy(),          # (256, 4096)
        'enc_fc_b': state['fc_enc.bias'].cpu().numpy(),            # (256,)
        'dec_fc_w': state['fc_dec.weight'].cpu().numpy(),          # (4096, 256)
        'dec_fc_b': state['fc_dec.bias'].cpu().numpy(),             # (4096,)
        'dec_deconv1_w': state['decoder.0.weight'].cpu().numpy(),  # (64, 32, 4, 4)
        'dec_deconv1_b': state['decoder.0.bias'].cpu().numpy(),     # (32,)
        'dec_deconv2_w': state['decoder.2.weight'].cpu().numpy(),  # (32, 16, 4, 4)
        'dec_deconv2_b': state['decoder.2.bias'].cpu().numpy(),     # (16,)
        'dec_deconv3_w': state['decoder.4.weight'].cpu().numpy(),  # (16, 3, 4, 4)
        'dec_deconv3_b': state['decoder.4.bias'].cpu().numpy(),     # (3,)
        'latent_dim': latent_dim,
    }
    
    # Transpose conv weights for NumPy format
    weight_dict['dec_deconv1_w'] = weight_dict['dec_deconv1_w'].transpose(1, 0, 2, 3)
    weight_dict['dec_deconv2_w'] = weight_dict['dec_deconv2_w'].transpose(1, 0, 2, 3)
    weight_dict['dec_deconv3_w'] = weight_dict['dec_deconv3_w'].transpose(1, 0, 2, 3)
    
    print("VAE training complete.")
    return weight_dict


def compute_prototypes(latents: np.ndarray, categories: List[str]) -> dict:
    """
    Compute mean latent vector per category.
    latents: (N, latent_dim)
    categories: (N,) list of category strings
    """
    prototypes = {}
    cat_array = np.array(categories)
    
    for cat in set(categories):
        mask = cat_array == cat
        if mask.sum() == 0:
            continue
        cat_latents = latents[mask]
        prototypes[cat] = {
            'mean': cat_latents.mean(axis=0).astype(np.float32),
            'std': cat_latents.std(axis=0).astype(np.float32) + 1e-6,
        }
    
    # Add a "neutral" prototype from all data
    prototypes['neutral'] = {
        'mean': latents.mean(axis=0).astype(np.float32),
        'std': latents.std(axis=0).astype(np.float32) + 1e-6,
    }
    
    return prototypes


def sample_imagined_latent(prototypes: dict, keywords: List[str], seed: int = 0) -> np.ndarray:
    """
    Blend prototype latents based on keywords.
    'cat' + 'ocean' -> latent between cat and ocean textures.
    """
    np.random.seed(seed)
    
    matched = []
    for kw in keywords:
        if kw in prototypes:
            matched.append(prototypes[kw])
    
    if not matched:
        matched = [prototypes.get('neutral', list(prototypes.values())[0])]
    
    # Blend means
    if len(matched) == 1:
        proto = matched[0]
        z = proto['mean'] + np.random.randn(*proto['mean'].shape) * proto['std'] * 0.3
    else:
        # Interpolate between concepts (IMAGINATION)
        weights = np.random.dirichlet(np.ones(len(matched)))
        mean = sum(w * m['mean'] for w, m in zip(weights, matched))
        std = sum(w * m['std'] for w, m in zip(weights, matched))
        z = mean + np.random.randn(*mean.shape) * std * 0.3
    
    return z.astype(np.float32)