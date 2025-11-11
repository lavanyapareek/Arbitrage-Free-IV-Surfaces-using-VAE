"""
Conditional VAE Model for Single Heston Parameters
Architecture: Conditions on 8 market/macro variables to learn p(θ|c)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add parent directory to path for Heston model import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from heston_model_ql import HestonModelQL


class ConditionalVAE_SingleHeston(nn.Module):
    """
    Conditional Variational Autoencoder for single Heston parameter sets.
    
    Input: 
        - θ: 5 parameters [kappa, theta, sigma_v, rho, v0]
        - c: 8 conditioning variables [crude_oil_30d_mean, crude_oil_7d_mean, ...]
    
    Output: 
        - Reconstructed parameters conditioned on c
        - Latent distribution parameters
    """
    
    def __init__(
        self,
        param_dim=5,
        conditioning_dim=8,
        latent_dim=4,
        hidden_dims=[128, 64, 32],
        encoder_activation='tanh',
        decoder_activation='relu',
        dropout=0.15,
        feller_penalty_weight=1.0,
        beta=0.1,
        arbitrage_penalty_weight=2.0
    ):
        super().__init__()
        
        self.param_dim = param_dim
        self.conditioning_dim = conditioning_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.feller_penalty_weight = feller_penalty_weight
        self.arbitrage_penalty_weight = arbitrage_penalty_weight
        self.dropout = dropout
        
        # Activation functions
        self.encoder_act = self._get_activation(encoder_activation)
        self.decoder_act = self._get_activation(decoder_activation)
        
        # ====================================================================
        # ENCODER: p(z|θ,c)
        # Input: concatenate θ (5D) + c (8D) = 13D
        # ====================================================================
        encoder_layers = []
        encoder_input_dim = param_dim + conditioning_dim  # 5 + 8 = 13
        prev_dim = encoder_input_dim
        
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(self.encoder_act)
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # ====================================================================
        # DECODER: p(θ|z,c)
        # Input: concatenate z (4D) + c (8D) = 12D
        # ====================================================================
        decoder_layers = []
        decoder_input_dim = latent_dim + conditioning_dim  # 4 + 8 = 12
        
        # From latent+conditioning to first hidden layer
        decoder_layers.append(nn.Linear(decoder_input_dim, hidden_dims[-1]))
        decoder_layers.append(nn.BatchNorm1d(hidden_dims[-1]))
        decoder_layers.append(self.decoder_act)
        if dropout > 0:
            decoder_layers.append(nn.Dropout(dropout))
        
        # Reverse hidden layers
        for i in range(len(hidden_dims) - 1, 0, -1):
            decoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i-1]))
            decoder_layers.append(nn.BatchNorm1d(hidden_dims[i-1]))
            decoder_layers.append(self.decoder_act)
            if dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))
        
        # Final layer to output (parameters)
        decoder_layers.append(nn.Linear(hidden_dims[0], param_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def _get_activation(self, name):
        """Get activation function by name"""
        if name == 'relu':
            return nn.ReLU()
        elif name == 'leaky_relu':
            return nn.LeakyReLU(0.2)
        elif name == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {name}")
    
    def encode(self, x, c):
        """
        Encode input parameters and conditioning to latent distribution parameters
        
        Args:
            x: Parameters (batch_size, param_dim)
            c: Conditioning variables (batch_size, conditioning_dim)
        
        Returns:
            mu, logvar: Latent distribution parameters
        """
        # Concatenate parameters and conditioning
        input_combined = torch.cat([x, c], dim=1)
        
        # Pass through encoder
        h = self.encoder(input_combined)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, c):
        """
        Decode latent vector and conditioning to parameters
        
        Args:
            z: Latent vector (batch_size, latent_dim)
            c: Conditioning variables (batch_size, conditioning_dim)
        
        Returns:
            Reconstructed parameters (batch_size, param_dim)
        """
        # Concatenate latent and conditioning
        input_combined = torch.cat([z, c], dim=1)
        
        # Pass through decoder
        return self.decoder(input_combined)
    
    def forward(self, x, c):
        """
        Full forward pass
        
        Args:
            x: Parameters (batch_size, param_dim)
            c: Conditioning variables (batch_size, conditioning_dim)
        
        Returns:
            recon: Reconstructed parameters
            mu, logvar: Latent distribution parameters
        """
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar
    
    def compute_feller_penalty(self, params_normalized, norm_mean, norm_std):
        """
        Compute Feller condition penalty: 2*kappa*theta > sigma_v^2
        
        Args:
            params_normalized: Normalized parameters [kappa, theta, sigma_v, rho, v0]
            norm_mean: Normalization mean
            norm_std: Normalization std
        
        Returns:
            Feller penalty (scalar)
        """
        # Denormalize parameters
        params = params_normalized * norm_std + norm_mean
        
        # Apply inverse transforms
        kappa = torch.exp(params[:, 0])      # log -> exp
        theta = torch.exp(params[:, 1])      # log -> exp
        sigma_v = torch.exp(params[:, 2])    # log -> exp
        
        # Feller condition: 2*kappa*theta > sigma_v^2
        feller_lhs = 2 * kappa * theta
        feller_rhs = sigma_v ** 2
        
        # Penalty for violation (ReLU ensures only violations are penalized)
        violation = F.relu(feller_rhs - feller_lhs)
        
        return violation.mean()
    
    def compute_arbitrage_penalty(self, params_normalized, norm_mean, norm_std, 
                                   spot=7000.0, r=0.067, q=0.0):
        """
        Compute arbitrage penalty for actual violations.
        Penalizes:
        - Static arbitrage: call prices not decreasing in strike
        - Butterfly arbitrage: call prices not convex in strike
        
        Args:
            params_normalized: Normalized parameters [kappa, theta, sigma_v, rho, v0]
            norm_mean: Normalization mean
            norm_std: Normalization std
            spot: Spot price
            r: Risk-free rate
            q: Dividend yield
        
        Returns:
            Arbitrage penalty (scalar)
        """
        # Denormalize parameters
        params = params_normalized * norm_std + norm_mean
        
        # Apply inverse transforms to get actual Heston parameters
        kappa = torch.exp(params[:, 0])
        theta = torch.exp(params[:, 1])
        sigma_v = torch.exp(params[:, 2])
        rho = torch.tanh(params[:, 3])
        v0 = torch.exp(params[:, 4])
        
        # Convert to numpy for Heston model (QuantLib requires numpy/python types)
        kappa_np = kappa.detach().cpu().numpy()
        theta_np = theta.detach().cpu().numpy()
        sigma_v_np = sigma_v.detach().cpu().numpy()
        rho_np = rho.detach().cpu().numpy()
        v0_np = v0.detach().cpu().numpy()
        
        # Sample strikes and maturities
        logm_samples = np.array([-0.1, -0.05, 0.0, 0.05, 0.1])  # 5 strikes
        tau_samples = np.array([0.5, 1.0])  # 2 maturities
        
        penalties = []
        
        # For each parameter set in batch
        for i in range(params.shape[0]):
            try:
                # Create Heston model with current parameters
                model = HestonModelQL(
                    kappa=float(kappa_np[i]),
                    theta=float(theta_np[i]),
                    sigma_v=float(sigma_v_np[i]),
                    rho=float(rho_np[i]),
                    v0=float(v0_np[i]),
                    r=r,
                    q=q
                )
                
                # For each maturity, check static and butterfly arbitrage
                for tau in tau_samples:
                    # Get price ratios for all strikes
                    try:
                        price_ratios = np.array([model.price_ratio(logm, tau) for logm in logm_samples])
                    except:
                        continue
                    
                    if np.any(np.isnan(price_ratios)) or np.any(price_ratios < 0):
                        continue
                    
                    # Check static arbitrage: prices should decrease in strike
                    for j in range(len(price_ratios) - 1):
                        if price_ratios[j+1] > price_ratios[j]:  # Violation
                            violation = price_ratios[j+1] - price_ratios[j]
                            penalties.append(violation * 100.0)  # Heavy penalty
                    
                    # Check butterfly arbitrage: prices should be convex in strike
                    for j in range(len(price_ratios) - 2):
                        butterfly = price_ratios[j] - 2*price_ratios[j+1] + price_ratios[j+2]
                        if butterfly < 0:  # Violation
                            penalties.append(-butterfly * 100.0)  # Heavy penalty
                
            except Exception as e:
                # If model creation or pricing fails, skip this parameter set
                continue
        
        # Convert penalties to tensor
        if len(penalties) > 0:
            penalty_tensor = torch.tensor(np.mean(penalties), 
                                         dtype=params.dtype, 
                                         device=params.device)
            return penalty_tensor
        else:
            return torch.tensor(0.0, dtype=params.dtype, device=params.device)
    
    def loss_function(self, recon, x, mu, logvar, norm_mean, norm_std):
        """
        Compute total VAE loss with conditioning
        
        Args:
            recon: Reconstructed parameters (conditioned on c)
            x: True parameters
            mu, logvar: Latent distribution parameters
            norm_mean, norm_std: Normalization parameters
        
        Returns:
            total_loss, recon_loss, kl_loss, feller_loss, arbitrage_loss
        """
        batch_size = x.size(0)
        
        # Reconstruction loss (MSE)
        # This enforces that decode(encode(θ,c), c) ≈ θ
        # Forces model to learn conditional distribution p(θ|c)
        recon_loss = F.mse_loss(recon, x, reduction='sum') / batch_size
        
        # KL divergence: KL(q(z|θ,c) || p(z))
        # Note: Currently uses standard normal prior p(z) = N(0,I)
        # Could be upgraded to conditional prior p(z|c) later
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        
        # Feller penalty (physical constraint)
        feller_loss = self.compute_feller_penalty(recon, norm_mean, norm_std)
        
        # Arbitrage penalty (market realism)
        if self.arbitrage_penalty_weight > 0:
            arbitrage_loss = self.compute_arbitrage_penalty(recon, norm_mean, norm_std)
        else:
            arbitrage_loss = torch.tensor(0.0, device=x.device)
        
        # Total loss
        # Balance between:
        # - Reconstruction (1.0): Truth to conditioning
        # - KL (0.1): Mild regularization
        # - Feller (1.0): Physical realism
        # - Arbitrage (2.0): Market realism
        total_loss = (recon_loss + 
                     self.beta * kl_loss + 
                     self.feller_penalty_weight * feller_loss +
                     self.arbitrage_penalty_weight * arbitrage_loss)
        
        return total_loss, recon_loss, kl_loss, feller_loss, arbitrage_loss
    
    def sample(self, num_samples, conditioning, device='cpu'):
        """
        Generate random samples conditioned on given market variables
        
        Args:
            num_samples: Number of samples to generate
            conditioning: Conditioning variables (num_samples, conditioning_dim)
                         or (1, conditioning_dim) to broadcast
            device: Device to generate on
        
        Returns:
            Generated samples in normalized space (num_samples, param_dim)
        """
        with torch.no_grad():
            # Sample from standard normal
            z = torch.randn(num_samples, self.latent_dim).to(device)
            
            # Broadcast conditioning if needed
            if conditioning.shape[0] == 1 and num_samples > 1:
                conditioning = conditioning.repeat(num_samples, 1)
            
            conditioning = conditioning.to(device)
            
            # Decode with conditioning
            samples = self.decode(z, conditioning)
        
        return samples
    
    def generate_for_conditions(self, conditions_dict, num_samples_per_condition, 
                                cond_mean, cond_std, device='cpu'):
        """
        Generate samples for multiple conditioning scenarios
        
        Args:
            conditions_dict: Dict of {name: conditioning_values}
            num_samples_per_condition: Samples per scenario
            cond_mean, cond_std: Normalization params for conditioning
            device: Device
        
        Returns:
            Dict of {name: generated_samples}
        """
        results = {}
        
        for name, cond_values in conditions_dict.items():
            # Normalize conditioning
            cond_normalized = (cond_values - cond_mean) / cond_std
            cond_tensor = torch.tensor(cond_normalized, dtype=torch.float32).unsqueeze(0)
            
            # Generate samples
            samples = self.sample(num_samples_per_condition, cond_tensor, device)
            results[name] = samples
        
        return results
