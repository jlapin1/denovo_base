import itertools
import math
import os
import typing
from dataclasses import dataclass

#import hydra.utils
#import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
#import torchmetrics
#import transformers
from torch import Tensor

#import dataloader
#import models
import models.mdlm.noise_schedule as noise_schedule
import models.mdlm.ema as ema
import models.mdlm.utils as utils
from tqdm.auto import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LOG2 = math.log(2)


def _sample_categorical(categorical_probs, **kwargs):
  gumbel_norm = (
    1e-10
    - (torch.rand_like(categorical_probs) + 1e-10).log())
  return (categorical_probs / gumbel_norm).argmax(dim=-1)

def _sample_top_categorical(categorical_probs, top=2):
    sorted_probs, sorted_inds = categorical_probs.sort(-1)
    sorted_probs_ = sorted_probs[..., -top:]
    gumbel_norm = (
        1e-10 - (torch.rand_like(sorted_probs_) + 1e-10).log()
    )
    winner = (sorted_probs_ / gumbel_norm).argmax(dim=-1)
    output = sorted_inds[...,-top:].gather(-1, winner[:,:,None]).squeeze()
    return output

def _unsqueeze(x, reference):
  return x.view(
    * x.shape,
    * ((1,) * (len(reference.shape) - len(x.shape))))


@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  token_mask: torch.FloatTensor

def CosineSampler(num_steps):
    t_array = torch.arange(num_steps, -1, -1).type(torch.float32)
    return torch.cos( (np.pi / 2) * (1 - t_array / len(t_array)) ).clamp(1e-5)

def SigmoidSampler(num_steps, zero_value=0.00247):
    t_array = torch.arange(num_steps, -1, -1).type(torch.float32)
    offset = np.log((1-zero_value) / zero_value)
    multiplier = (offset - np.log(zero_value / (1-zero_value))) / num_steps
    multiplier2 = 1 / (1 - 2*zero_value)
    output = (multiplier2*(1 / (1+torch.exp(offset - multiplier*t_array)))-zero_value).clamp(1e-5)
    return output

def CubeSampler(num_steps):
    t_array = torch.arange(num_steps, -1, -1).type(torch.float32)
    return (1.9073486328125e-06*(t_array - num_steps/2)**3 + 0.5).clamp(1e-5)

class Diffusion:
  def __init__(
      self,
      config,
      dictionary,
      backbone,
      tokenizer=None,
  ):
    self.config = config

    self.tokenizer = tokenizer
    self.vocab_size = np.max(list(dictionary.values())) + 1
    self.sampler = self.config['sampling']['predictor']
    self.antithetic_sampling = self.config['training']['antithetic_sampling']
    self.importance_sampling = self.config['training']['importance_sampling']
    self.change_of_variables = self.config['training']['change_of_variables']
    self.use_guidance = self.config['guidance']['p_uncond'] > 0

    self.mask_index = dictionary['<MASK>']
    self.eos_token_id = dictionary['<EOS>']
    self.NT = dictionary['X']
    self.parameterization = self.config['parameterization']
    
    self.backbone = backbone
    self.device = device
    self.dtype = torch.float32

    self.T = self.config['T']
    self.subs_masking = self.config['subs_masking']
    self.SL = self.config['model']['length']
    self.steps = self.config['sampling']['steps']

    self.softplus = torch.nn.Softplus()
    self.eval_model_tokenizer = None

    self.noise = noise_schedule.get_noise(self.config, dtype=torch.float32)
    """if self.config['training']['ema'] > 0:
      self.ema = ema.ExponentialMovingAverage(
        itertools.chain(self.backbone.parameters(),
                        self.noise.parameters()),
        decay=self.config['training']['ema'])
    else:
      self.ema = None"""
    
    #self.lr = self.config.optim.lr
    self.sampling_eps = self.config['training']['sampling_eps']
    self.time_conditioning = self.config['time_conditioning']
    self.neg_infinity = -1000000.0
    self.fast_forward_epochs = None
    self.fast_forward_batches = None
    self._validate_configuration()

  def _validate_configuration(self):
    assert not (self.change_of_variables
                and self.importance_sampling)
    #if self.diffusion != 'absorbing_state':
    #  assert self.parameterization not in {'ar', 'subs'}
    if self.T > 0:
      assert self.parameterization in {'d3pm', 'subs'}
    if self.subs_masking:
      assert self.parameterization == 'd3pm'

  def _subs_parameterization(self, logits, xt):
    # log prob at the mask index = - infinity
    logits[:, :, self.mask_index] += self.neg_infinity
    
    # Normalize the logits such that x.exp() is
    # a probability distribution over vocab_size.
    logits = logits - torch.logsumexp(logits, dim=-1,
                                      keepdim=True)

    # Apply updates directly in the logits matrix.
    # For the logits of the unmasked tokens, set all values
    # to -infinity except for the indices corresponding to
    # the unmasked tokens.
    unmasked_indices = (xt != self.mask_index)
    logits[unmasked_indices] = self.neg_infinity
    logits[unmasked_indices, xt[unmasked_indices]] = 0
    return logits

  def _d3pm_parameterization(self, logits):
    if self.subs_masking:
      logits[:, :, self.mask_index] += self.neg_infinity
    logits = logits - torch.logsumexp(logits, dim=-1,
                                      keepdim=True)
    return logits

  def _sedd_parameterization(self, logits, xt, sigma):
    esigm1_log = torch.where(
      sigma < 0.5,
      torch.expm1(sigma),
      sigma.exp() - 1).log().to(logits.dtype)
    # logits shape
    # (batch_size, diffusion_model_input_length, vocab_size)
    logits = logits - esigm1_log[:, None, None] - np.log(
      logits.shape[-1] - 1)
    # The below scatter operation sets the log score
    # for the input word to 0.
    logits = torch.scatter(logits, -1, xt[..., None],
                           torch.zeros_like(logits[..., :1]))
    return logits

  def _process_sigma(self, sigma):
    if sigma is None:
      assert self.parameterization == 'ar'
      return sigma
    if sigma.ndim > 1:
      sigma = sigma.squeeze(-1)
    if not self.time_conditioning:
      sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma

  def add_unconditional_to_batch(self, x, model_kwargs):
      # Double up sequence
      x = torch.cat([x, x], dim=0)
      
      # Create conditional/unconditional mass vector
      #  then put it back in kwargs
      mass = model_kwargs.pop('mass')
      if mass.shape[0] != x.shape[0]:  
        mass = torch.cat([mass, 0*mass], dim=0)
      model_kwargs['mass'] = mass
      
      charge = model_kwargs.pop("charge")
      if charge.shape[0] != x.shape[0]:
        charge = torch.cat([charge, 0*charge], dim=0)
      model_kwargs['charge'] = charge

      #kv_features = model_kwargs.pop("kv_features")
      #if kv_features.shape[0] != x.shape[0]:
      #    kv_features = torch.cat([kv_features, 0*kv_features], dim=0)
      #model_kwargs['kv_features'] = kv_features

      # Double up kwargs
      for key, value in model_kwargs.items():
          if value.shape[0] != x.shape[0]:
            model_kwargs[key] = torch.cat([value, value], axis=0)
     
      return x, model_kwargs

  def forward(self, x, sigma, model_kwargs={}, training=False):
    """Returns log score."""
    sigma = self._process_sigma(sigma)
    model_kwargs['timesteps'] = sigma
    if self.use_guidance and not training:
        x_, model_kwargs = self.add_unconditional_to_batch(x, model_kwargs)
    else:
        x_ = x
    #with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float32):
    logits = self.backbone(x_, **model_kwargs)['out']
    if 'self_conditions' in model_kwargs:
        model_kwargs['self_conditions'] = logits
    if self.use_guidance and not training:
        cond, uncond = logits.chunk(2)
        logits = self.config['guidance']['w']*(cond-uncond) + uncond

    if self.parameterization == 'subs':
      out = self._subs_parameterization(logits=logits.clone(),
                                         xt=x)
      return out, logits

    elif self.parameterization == 'sedd':
      return self._sedd_parameterization(logits=logits,
                                         xt=x,
                                         sigma=sigma)
    elif self.parameterization == 'd3pm':
      return self._d3pm_parameterization(logits=logits), logits
    
    return logits, None

  def _d3pm_loss(self, model_output, xt, x0, t):
    dt = 1 / self.T

    if torch.is_tensor(t):
      t = t[:, None]
      assert t.ndim == 2
      t = t.clamp(0., 1. - 1e-4)
    alpha_t = 1 - t + torch.zeros_like(xt)
    alpha_s = 1 - (t - dt) + torch.zeros_like(xt)

    log_x_theta_at_x0 = torch.gather(
      model_output, -1, x0[:, :, None]).squeeze(-1)
    log_x_theta_at_m = model_output[:, :, self.mask_index]
    x_theta_at_m = log_x_theta_at_m.exp()
    
    term_1_coef = dt / t
    term_1_log_nr = torch.log(alpha_t * x_theta_at_m / t + 1)
    term_1_log_dr = log_x_theta_at_x0
    
    term_2_coef = 1 - dt / t
    term_2_log_nr = term_1_log_nr
    term_2_log_dr = torch.log(alpha_s * x_theta_at_m / (t - dt) + 1)

    L_vb_masked = (
      term_1_coef * (term_1_log_nr - term_1_log_dr)
      + term_2_coef * (term_2_log_nr - term_2_log_dr))

    L_vb = L_vb_masked * (xt == self.mask_index)

    return self.T * L_vb

  def _compute_loss(self, batch, prefix):
    if 'attention_mask' in batch:
      attention_mask = batch['attention_mask']
    else:
      attention_mask = None
    losses = self._loss(batch['input_ids'], attention_mask)
    loss = losses.loss

    if prefix == 'train':
      self.train_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.train_metrics
    elif prefix == 'val':
      self.valid_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.valid_metrics
    elif prefix == 'test':
      self.test_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.test_metrics
    else:
      raise ValueError(f'Invalid prefix: {prefix}')

    self.log_dict(metrics,
                  on_step=False,
                  on_epoch=True,
                  sync_dist=True)
    return loss

  def on_train_epoch_start(self):
    self.backbone.train()
    self.noise.train()

  def training_step(self, batch, batch_idx):
    loss = self._compute_loss(batch, prefix='train')
    self.log(name='trainer/loss',
             value=loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    return loss
  
  def q_xt(self, x, move_chance):
    """Computes the noisy sample xt.

    Args:
      x: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input. 
      move_chance: float torch.Tensor with shape (batch_size, 1).
    """
    move_indices = torch.rand(
      * x.shape, device=x.device) < move_chance
    xt = torch.where(move_indices, self.mask_index, x)
    return xt

  def _sample_prior(self, *batch_dims):
    return self.mask_index * torch.ones(
      * batch_dims, dtype=torch.int64)

  def _ddpm_caching_update(
    self,
    x,
    t,
    dt,
    p_x0=None,
    top=1,
    model_kwargs={},
    guide_model=None,
    gamma=1.,
  ):
    # Timesteps
    assert self.config['noise']['type'] == 'loglinear'
    sigma_t, _ = self.noise(t)
    if t.ndim > 1:
      t = t.squeeze(-1)
    assert t.ndim == 1
    # Move chance
    move_chance_t = t[:, None, None]
    move_chance_s = (t - dt)[:, None, None]
    assert move_chance_t.ndim == 3, move_chance_t.shape
    # Model
    if p_x0 is None:
      logp_x0, logits = self.forward(x, sigma_t, model_kwargs)
      p_x0 = logp_x0.exp()
      if guide_model is not None:
          # Compute unguided posterior
          diffusion_log_probs = logp_x0 + torch.log(1. - move_chance_s / move_chance_t) # + torch.log(move_chance_t)
          diffusion_log_probs[..., self.mask_index] = torch.log(move_chance_s / move_chance_t)[:, :, 0] # + torch.log(move_chance_t)
          diffusion_log_probs.detach()
          # Compute guided posterior
          if True: # approx
              with torch.enable_grad():
                guide_out = guide_model(x, t=sigma_t)
                xt_one_hot = guide_out['one_hot']
                guide_loss = guide_model.get_backprop_prop(guide_out['out'], model_kwargs['mass'], model_kwargs['charge'])
                #classifiger_log_prob_xt.sum().backward()
                guide_loss.backward(torch.ones_like(guide_loss))
                guide_log_prob_xt = xt_one_hot.grad
              # 2*all_grads - current_token_grads
              guide_log_prob_ratio = (guide_log_prob_xt - (xt_one_hot * guide_log_prob_xt).sum(dim=-1, keepdim=True)).detach().requires_grad_(False)
              guide_log_prob = (guide_log_prob_ratio + guide_log_prob_xt).detach().requires_grad_(False)
        
              # Apply guidance
              with torch.no_grad():
                guided_log_probs = (gamma * guide_log_prob) + diffusion_log_probs
                copy_flag = x != self.mask_index
                guided_log_probs[copy_flag] = self.neg_infinity
                guided_log_probs[copy_flag, x[copy_flag]] = 0.0     
              q_xs = guided_log_probs.softmax(dim=-1)
      else:
          #assert move_chance_t.ndim == p_x0.ndim
          # Sampling
          q_xs = p_x0 * (move_chance_t - move_chance_s) * (p_x0 > self.config['sampling']['min_prob']).float()
          q_xs[p_x0 > self.config['sampling']['max_prob']] = 1e10
          q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    sampler = _sample_top_categorical #if self.config['sampling']['top'] else _sample_categorical
    _x = sampler(q_xs, np.maximum(2, top))
    
    copy_flag = (x != self.mask_index).to(x.dtype)
    return p_x0, copy_flag * x + (1 - copy_flag) * _x, logits

  def _ddpm_update(self, x, t, dt, model_kwargs={}):
    sigma_t, _ = self.noise(t)
    sigma_s, _ = self.noise(t - dt)
    if sigma_t.ndim > 1:
      sigma_t = sigma_t.squeeze(-1)
    if sigma_s.ndim > 1:
      sigma_s = sigma_s.squeeze(-1)
    assert sigma_t.ndim == 1, sigma_t.shape
    assert sigma_s.ndim == 1, sigma_s.shape
    move_chance_t = 1 - torch.exp(-sigma_t)
    move_chance_s = 1 - torch.exp(-sigma_s)
    move_chance_t = move_chance_t[:, None, None]
    move_chance_s = move_chance_s[:, None, None]
    log_p_x0 = self.forward(x, sigma_t, model_kwargs)
    assert move_chance_t.ndim == log_p_x0.ndim
    # Technically, this isn't q_xs since there's a division
    # term that is missing. This division term doesn't affect
    # the samples.
    q_xs = log_p_x0.exp() * (move_chance_t
                             - move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    _x = _sample_categorical(q_xs)

    copy_flag = (x != self.mask_index).to(x.dtype)
    return copy_flag * x + (1 - copy_flag) * _x

  def _ar_sampler(self, bsz):
    # precompute token buffer
    num_pred_tokens = self.config.model.length - 1
    x = torch.zeros(
      (bsz, num_pred_tokens + 1),
      dtype=torch.long,
      device=self.device)
    x[:, 0] = self.tokenizer.bos_token_id
    # precompute noise
    noise = (torch.distributions.Gumbel(0, 1)
             .sample((bsz, num_pred_tokens, self.vocab_size))
             .to(self.device))
    for i in range(num_pred_tokens):
      next_logits = self.forward(x[:, :i + 1], None)[:, -1]
      y = (next_logits + noise[:, i]).argmax(-1)
      x[:, i + 1] = y
    return x

  def fill_null(self, x):
      A,B = torch.where(x==self.eos_token_id)
      msk = torch.arange(x.shape[1], device=x.device)[None].tile([x.shape[0], 1])
      C,D = torch.where(msk[A] > B[:,None])
      x[A.gather(0, C), D] = self.NT
      return x

  @torch.no_grad()
  def _sample(
      self,
      x=None,
      num_steps=None, 
      eps=1e-5,
      top=None,
      model_kwargs={},
      guide_model=None,
      gamma=1.,
      save_x=False, 
      save_p=False, 
      progress=False
  ):
      """Generate samples from the model."""
      batch_size_per_gpu = len(model_kwargs['charge'])
      
      if self.parameterization == 'ar':
          return self._ar_sampler(batch_size_per_gpu)
      
      if num_steps is None:
          num_steps = self.steps
      
      # Initialize variables
      if x == None:
          x = self._sample_prior(batch_size_per_gpu, self.SL).to(self.device)
      if self.config['sampling']['sampler'] == 'cosine':
          # Delays unmasking
          timesteps = CosineSampler(num_steps)
      elif self.config['sampling']['sampler'] == 'sigmoid':
          # Increases unmasking in the middle of trajectory
          timesteps = SigmoidSampler(num_steps, self.config['sampling']['sigmoid_zero_value'])
      elif self.config['sampling']['sampler'] == 'cube':
          # Increases unmasking at beginning and end
          timesteps = CubeSampler(num_steps)
      else:
          # Unmasking rate is constant throughout trajectory
          timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
      
      #
      Logits = torch.zeros(batch_size_per_gpu, x.shape[1], self.vocab_size, device=self.device)
      p_x0_cache = None
      if self.config['model']['self_condition']:
          model_kwargs['self_conditions'] = torch.zeros(
              2*batch_size_per_gpu if self.use_guidance else batch_size_per_gpu, x.shape[1], self.vocab_size, 
              device=self.device,
          )
      if save_x:
          xsave = torch.zeros(num_steps+2, x.shape[0], x.shape[1], dtype=torch.int32, device=self.device)
          xsave[0] = x
      if save_p: 
          psave = torch.zeros(num_steps+1, x.shape[0], x.shape[1], self.vocab_size, device=self.device)
      
      # Sampling loops
      pbar = tqdm(range(num_steps)) if progress else range(num_steps)
      for i in pbar:
          t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
          dt = timesteps[i] - timesteps[i+1]
          #model_kwargs['timesteps'] = t # Set in self.forward()
          top = self.config['sampling']['top'] if top is None else top

          if self.sampler == 'ddpm':
              x = self._ddpm_update(x, t, dt, model_kwargs, top=top)
          elif self.sampler == 'ddpm_cache':
              p_x0_cache, x_next, logits = self._ddpm_caching_update(
                  x,
                  t,
                  dt,
                  p_x0=p_x0_cache,
                  top=top,
                  model_kwargs=model_kwargs,
                  guide_model=guide_model,
                  gamma=gamma,
              )
          
          Logits[x_next!=x] = logits[x_next!=x]
          if save_p: 
              psave[i] = p_x0_cache
          if True:#(not torch.allclose(x_next, x) or self.time_conditioning):
              # Disable caching
              p_x0_cache = None
              x = x_next
          #else:
          #    x = self._analytic_update(x, t, dt)
          if save_x: 
              xsave[i+1] = x
      
      # Final decisions
      if self.config['sampling']['noise_removal']:
          t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
          if self.sampler == 'analytic':
              x = self._denoiser_update(x, t, model_kwargs)
          else:
              sigma_t = self.noise(t)[0]
              final_, logits = self.forward(x, sigma_t, model_kwargs, training=False)
              final = final_.argmax(dim=-1)
              Logits[final!=x] = logits[final!=x]
              if save_x: xsave[-1] = final
              if save_p: psave[-1] = final_.exp()
      
      # Output
      output = {'prediction': final, 'logits': Logits}
      if save_x:
          output['x_save'] = xsave.transpose(0,1)
      if save_p:
          output['p_save'] = psave.transpose(0,1)
      
      return output

  #def _sample(
  #  self,
  #  x=None,
  #  eps=1e-5,
  #  num_steps=0,
  #  model_kwargs={},
  #  save_x=False,
  #  save_p=False,
  #  progress=False,
  #  **kwargs
  #):
  #  batch_size_per_gpu = len(model_kwargs['charge'])
  #  
  #  # Initialize variables
  #  if self.backbone.block_size is not None:
  #      x_ = self._sample_prior(batch_size_per_gpu, self.backbone.block_size).to(self.device)
  #      if x is not None:
  #          # Extending the sequence
  #          x = torch.cat([x, x_], dim=1)
  #      else:
  #          # Starting off
  #          x = x_
  #  else:
  #      x = self._sample_prior(batch_size_per_gpu, self.SL).to(self.device)
  #  #steps_btw = num_steps
  #  num_steps = self.SL #(1+num_steps)*self.SL
  #  timesteps = torch.linspace(1, eps, num_steps, device=self.device)

  #  p_x0_cache = None
  #  if self.config['model']['self_condition']:
  #    model_kwargs['self_conditions'] = torch.zeros(
  #      batch_size_per_gpu, self.SL, self.vocab_size, device=self.device
  #    )
  #  if save_x:
  #    xsave = torch.zeros(num_steps+1, x.shape[0], x.shape[1], dtype=torch.int32, device=self.device)
  #    xsave[0] = x
  #  if save_p: 
  #    psave = torch.zeros(num_steps, x.shape[0], x.shape[1], self.vocab_size, device=self.device)
  #    
  #  pbar = tqdm(range(num_steps)) if progress else range(num_steps)
  #  for i in pbar:
  #    # Timestep
  #    t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
  #    if t.ndim > 1:
  #      t = t.squeeze(-1)
  #    sigma_t, _ = self.noise(t)
  #      
  #    # Model forward pass
  #    if p_x0_cache is None:
  #      logp_x0, logits = self.forward(x, sigma_t, model_kwargs)
  #      p_x0 = logp_x0.exp()
  #      p_x0_ = p_x0[...,:-1]
  #      
  #      # Set self-conditions
  #      if self.config['model']['self_condition']:
  #        model_kwargs['self_conditions'] = logits
  #      
  #      if True:#i % (steps_btw+1) == 0:
  #          # Automatic switchovers
  #          auto_unmask = p_x0 > self.config['sampling']['max_prob']
  #          where = torch.where(auto_unmask)
  #          x[where[0], where[1]] = where[2]
  #          
  #          maxlog, maxcat = p_x0.max(-1)
  #          maxlog[x!=self.mask_index] = 0
  #          maxlog2, maxind = maxlog.max(-1)
  #          bd = torch.arange(len(maxind), device=self.device)
  #          x[bd, maxind] = maxcat[bd, maxind]
  #      if (x!=self.mask_index).all():
  #          break
  #      
  #      if save_p: 
  #          psave[i] = p_x0
  #      #if (not torch.allclose(x_next, x) or self.time_conditioning):
  #      #    # Disable caching
  #      #    p_x0_cache = None
  #      #    x = x_next
  #      #else:
  #      #    x = self._analytic_update(x, t, dt)
  #      if save_x: 
  #          xsave[i+1] = x
  #    
  #  assert (x!=self.mask_index).all()

  #  # Output
  #  output = {'prediction': x, 'logits': logits}
  #  if save_x:
  #    output['x_save'] = xsave.transpose(0,1)
  #  if save_p:
  #    output['p_save'] = psave.transpose(0,1)
  #    
  #  return output

  def restore_model_and_sample(self, num_steps, eps=1e-5):
    """Generate samples from the model."""
    # Lightning auto-casting is not working in this method for some reason
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    samples = self._sample(num_steps=num_steps, eps=eps)
    if self.ema:
      self.ema.restore(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.train()
    self.noise.train()
    return samples

  def get_score(self, x, sigma, model_kwargs={}):
    model_output = self.forward(x, sigma, model_kwargs)
    if self.parameterization == 'subs':
      # score(x, t) = p_t(y) / p_t(x)
      # => log score(x, t) = log p_t(y) - log p_t(x)
      
      # case 1: x = masked
      #   (i) y = unmasked
      #     log score(x, t) = log p_\theta(x)|_y + log k
      #     where k = exp(- sigma) / (1 - exp(- sigma))
      #   (ii) y = masked
      #     log score(x, t) = 0

      # case 2: x = unmasked
      #   (i) y != masked, y != x
      #     log score(x_i, t) = - inf
      #   (ii) y = x 
      #     log score(x_i, t) = 0
      #   (iii) y = masked token
      #     log score(x_i, t) = - log k
      #     where k = exp(- sigma) / (1 - exp(- sigma))
      
      log_k = - torch.log(torch.expm1(sigma)).squeeze(-1)
      assert log_k.ndim == 1
      
      masked_score = model_output + log_k[:, None, None]
      masked_score[:, :, self.mask_index] = 0

      unmasked_score = self.neg_infinity * torch.ones_like(
        model_output)
      unmasked_score = torch.scatter(
        unmasked_score,
        -1,
        x[..., None],
        torch.zeros_like(unmasked_score[..., :1]))
      unmasked_score[:, :, self.mask_index] = - (
        log_k[:, None] * torch.ones_like(x))
      
      masked_indices = (x == self.mask_index).to(
        model_output.dtype)[:, :, None]
      model_output = (
        masked_score * masked_indices
        + unmasked_score * (1 - masked_indices))
    return model_output.exp()

  def _staggered_score(self, score, dsigma):
    score = score.clone()
    extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
    score *= dsigma.exp()[:, None]
    score[..., self.mask_index] += extra_const
    return score

  def _analytic_update(self, x, t, step_size, model_kwargs):
    curr_sigma, _ = self.noise(t)
    next_sigma, _ = self.noise(t - step_size)
    dsigma = curr_sigma - next_sigma
    score = self.get_score(x, curr_sigma, model_kwargs)
    stag_score = self._staggered_score(score, dsigma)
    probs = stag_score * self._transp_transition(x, dsigma)
    return _sample_categorical(probs)

  def _denoiser_update(self, x, t, model_kwargs):
    sigma, _ = self.noise(t)
    score = self.get_score(x, sigma, model_kwargs)
    stag_score = self._staggered_score(score, sigma)
    probs = stag_score * self._transp_transition(x, sigma)
    probs[..., self.mask_index] = 0
    samples = _sample_categorical(probs)
    return samples

  def _transp_transition(self, i, sigma):
    sigma = _unsqueeze(sigma, reference=i[..., None])
    edge = torch.exp(-sigma) * F.one_hot(
      i, num_classes=self.vocab_size)
    edge += torch.where(i == self.mask_index,
                        1 - torch.exp(-sigma).squeeze(-1),
                        0)[..., None]
    return edge

  def _sample_t(self, n, device):
    _eps_t = torch.rand(n, device=device)
    if self.antithetic_sampling:
      offset = torch.arange(n, device=device) / n
      _eps_t = (_eps_t / n + offset) % 1
    t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
    if self.importance_sampling:
      return self.noise.importance_sampling_transformation(t)
    return t

  def _maybe_sub_sample(self, x0, attention_mask):
    seqlen = x0.shape[1]
    if seqlen > self.config.model.length:
      assert seqlen == 2 * self.config.model.length
      # cropping is needed for text8-crop dataset
      # try the same starting point for now
      start = np.random.choice(self.config.model.length)
      end = start + self.config.model.length
      input_tokens = x0[:, start: end]
      output_tokens = x0[:, start + 1: end + 1]
      new_attention_mask = attention_mask[:, start: end]

      # Helps with validation PPL, since the val
      # examples will all start and end with BOS/EOS
      input_tokens[:, 0] = self.tokenizer.bos_token_id
      output_tokens[:, -1] = self.tokenizer.eos_token_id
    elif self.parameterization == 'ar':
      input_tokens = x0[:, :-1]
      output_tokens = x0[:, 1:]
      new_attention_mask = attention_mask[:, 1:]
    else:
      input_tokens = x0
      output_tokens = None
      new_attention_mask = attention_mask
    return input_tokens, output_tokens, new_attention_mask

  def _reconstruction_loss(self, x0, model_kwargs={}):
    t0 = torch.zeros(x0.shape[0], dtype=self.dtype,
                     device=self.device)
    assert self.config['noise']['type'] == 'loglinear'
    # The above assert is for d3pm parameterization
    unet_conditioning = self.noise(t0)[0][:, None]
    model_output_t0 = self.forward(x0, unet_conditioning, model_kwargs=model_kwargs)[0]
    return - torch.gather(input=model_output_t0,
                          dim=-1,
                          index=x0[:, :, None]).squeeze(-1)

  def _forward_pass_diffusion(self, x0, model_kwargs, block_training=False):
    bs, sl = x0.shape
    t = self._sample_t(bs, x0.device)#[torch.randperm(x0.shape[0])]
    if self.T > 0:
      t = (t * self.T).to(torch.int)
      t = t / self.T
      # t \in {1/T, 2/T, ..., 1}
      t += (1 / self.T)

    if self.change_of_variables:
      model_kwargs['timesteps'] = t[:, None]
      f_T = torch.log1p(- torch.exp(- self.noise.sigma_max))
      f_0 = torch.log1p(- torch.exp(- self.noise.sigma_min))
      move_chance = torch.exp(f_0 + t * (f_T - f_0))
      move_chance = move_chance[:, None]
    else:
      sigma, dsigma = self.noise(t)
      model_kwargs['timesteps'] = sigma if self.time_conditioning else torch.zeros_like(sigma)
      move_chance = 1 - torch.exp(-sigma[:, None])
    
    xt = self.q_xt(x0, move_chance)
    masked_token_mask = xt==self.mask_index
    if block_training:
        xt = torch.cat([xt, x0], dim=1)

    if self.config['model']['self_condition']:
        model_kwargs['self_conditions'] = torch.zeros(xt.shape[0], xt.shape[1], self.vocab_size, device=device)
        if np.random.uniform() > 0.5:
            with torch.no_grad():
                model_output = self.backbone(xt, **model_kwargs)['out']
            model_kwargs['self_conditions'] = model_output.detach()
    
    model_output = (
        self.backbone(xt, **model_kwargs)['out']
        if self.config['custom_loss'] else
        self.forward(xt, sigma[:, None], model_kwargs=model_kwargs, training=True)[0]
    )

    if block_training:
        model_output = model_output[:, :sl]
        masked_token_mask = masked_token_mask[:, :sl]
    utils.print_nans(model_output, 'model_output')
    
    if self.config['custom_loss']:
        return model_output, dsigma / torch.expm1(sigma), masked_token_mask, t
    
    if self.parameterization == 'sedd':
      return dsigma[:, None] * self._score_entropy(
        model_output, sigma[:, None], xt, x0)
    
    if self.T > 0:
      diffusion_loss = self._d3pm_loss(
        model_output=model_output, xt=xt, x0=x0, t=t)
      if self.parameterization == 'd3pm':
        if self.config['model']['self_condition']:
            model_kwargs['self_conditions'] = torch.zeros(xt.shape[0], xt.shape[1], self.vocab_size, device=device)
        reconstruction_loss = self._reconstruction_loss(x0, model_kwargs=model_kwargs)
      elif self.parameterization == 'subs':
        reconstruction_loss = torch.zeros(())
      return reconstruction_loss + diffusion_loss, {'reconstruction_loss': reconstruction_loss.mean().item(), 'diffusion_loss': diffusion_loss.mean().item()}
    
    # SUBS parameterization, continuous time.
    log_p_theta = torch.gather(
      input=model_output,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    
    if self.change_of_variables or self.importance_sampling:
      return log_p_theta * torch.log1p(
        - torch.exp(- self.noise.sigma_min))
    
    return - log_p_theta * (
      dsigma / torch.expm1(sigma))[:, None], {}

  def _loss(self, x0, attention_mask):
    (input_tokens, output_tokens,
     attention_mask) = self._maybe_sub_sample(
       x0, attention_mask)

    if self.parameterization == 'ar':
      logprobs = self.backbone(input_tokens, None)
      loss = - logprobs.gather(
        -1, output_tokens[:, :, None])[:, :, 0]
    else:
      loss = self._forward_pass_diffusion(input_tokens)
    
    nlls = loss * attention_mask
    count = attention_mask.sum()

    batch_nll = nlls.sum()
    token_nll = batch_nll / count

    return Loss(loss=token_nll,
                nlls=nlls,
                token_mask=attention_mask)

  def _score_entropy(self, log_score, sigma, xt, x0):
    """Computes the SEDD loss.

    Args:
      log_score: float torch.Tensor with shape (batch_size,
          diffusion_model_input_length, vocab_size),
          log score, output of the denoising network.
      xt: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      x0: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      sigma: float torch.Tensor with shape (batch_size, 1).

    Returns:
      loss with shape (batch_size, diffusion_model_input_length)
    """
    masked_indices = xt == self.mask_index

    expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
    q_ratio = 1 / expsig_minus_1[masked_indices]

    words_that_were_masked = x0[masked_indices]

    neg_term = q_ratio * torch.gather(
      log_score[masked_indices],
      -1,
      words_that_were_masked[..., None]).squeeze(-1)
    score = log_score[masked_indices].exp()
    if self.mask_index == self.vocab_size - 1:
      pos_term = score[:, :-1].sum(dim=-1)
    else:
      pos_term = score[:, : self.mask_index].sum(
        dim=-1) + score[:, self.mask_index + 1:].sum(dim=-1)
    const = q_ratio * (q_ratio.log() - 1)

    entropy = torch.zeros(* xt.shape, device=xt.device)
    entropy[masked_indices] += pos_term - neg_term + const
    return entropy

  @torch.no_grad
  def sample_subs_guidance(
    self, n_samples, stride_length, num_strides, dt=0.001):
    ones = torch.ones(n_samples, dtype=self.dtype,
                      device=self.device)

    num_steps = int(1 / dt)
    sampling_steps = 0
    intermediate_tokens = []
    target = None
    for _ in range(num_strides + 1):
      p_x0_cache = None
      x = self._sample_prior(
        n_samples,
        self.config.model.length).to(self.device)
      if target is not None:
        x[:, : -stride_length] = target
      for i in range(num_steps + 1):
        p_x0_cache, x_next = self._ddpm_caching_update(
          x=x, t=(1 - i * dt) * ones, dt=dt, p_x0=p_x0_cache)
        if (not torch.allclose(x_next, x)
            or self.time_conditioning):
          p_x0_cache = None
          sampling_steps += 1
        x = x_next
      x = self.forward(x, 0 * ones).argmax(dim=-1)
      intermediate_tokens.append(
        x[:, :stride_length].cpu().numpy())
      target = x[:, stride_length:]
    
    intermediate_tokens.append(target.cpu().numpy())
    intermediate_text_samples = []
    sequence_lengths = ((
      np.concatenate(intermediate_tokens, axis=1)[:, 1:]
      == self.tokenizer.eos_token_id).cumsum(-1) == 0).sum(-1)
    for i in range(2, len(intermediate_tokens) + 1):
      intermediate_text_samples.append(
        self.tokenizer.batch_decode(
          np.concatenate(intermediate_tokens[:i], axis=1)))
    return (sampling_steps, intermediate_text_samples,
            sequence_lengths)

