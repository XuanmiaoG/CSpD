from typing import Optional
import math
import torch
from tqdm import tqdm
from models.mar import MAR

class DistributionProcessor:
    def __init__(self, x1_mean, x1_std, x1_log_var, x2_mean, x2_std, x2_log_var, dim, temperature=42.):
        """x has shape BC. x1 for q(x), x2 for p(x)."""
        self.x1_mean = x1_mean
        self.x1_std = x1_std
        self.x1_log_var = x1_log_var
        self.x2_mean = x2_mean
        self.x2_std = x2_std
        self.x2_log_var = x2_log_var
        self.temperature = temperature
        self.dim = dim
        self.fixed_term = torch.exp((x2_log_var - x1_log_var).sum(dim=-1))
        
        # 初始化接受计数器
        self.accepted_tokens = 0
        self.total_tokens = 0

    def x1_pdf_unormalize(self, x):
        return torch.exp(-0.5 * (((x - self.x1_mean) / self.x1_std / self.temperature) ** 2).sum(-1))

    def x2_pdf_unormalize(self, x):
        return torch.exp(-0.5 * (((x - self.x2_mean) / self.x2_std / self.temperature) ** 2).sum(-1))

    def x1_pdf(self, x):
        return torch.exp(-0.5 * (((x - self.x1_mean) / self.x1_std) ** 2).sum(-1)) / (
                (2 * math.pi) ** (self.dim / 2) * self.x1_std.prod(-1))

    def x2_pdf(self, x):
        return torch.exp(-0.5 * (((x - self.x2_mean) / self.x2_std) ** 2).sum(-1)) / (
                (2 * math.pi) ** (self.dim / 2) * self.x2_std.prod(-1))

    def judge_accept(self, x):
        q_x = self.x1_pdf_unormalize(x)
        p_x = self.x2_pdf_unormalize(x)
        return p_x * self.fixed_term * self.x1_std.prod(-1) / (q_x * self.x2_std.prod(-1))

    def subtracted_distribution_unormalize(self, x):
        ret = self.x2_pdf(x) - self.x1_pdf(x) / self.fixed_term
        proposal = self.x2_pdf(x)
        return ret.clamp(min=0) / proposal

    def sample_distribution(self):
        mask = torch.zeros((self.x2_mean.shape[0],), device=self.x2_mean.device).bool()
        sample = torch.zeros_like(self.x2_mean)
        for i in range(100):  # try at most 100 times
            proposal = torch.randn_like(self.x2_mean) * self.x2_std + self.x2_mean
            alpha = self.subtracted_distribution_unormalize(proposal)
            eps = torch.rand_like(alpha)

            accept_mask = eps < alpha
            sample[accept_mask] = proposal[accept_mask]
            mask = torch.logical_or(accept_mask, mask)

            if torch.all(mask):
                break
        return sample

    def speculative_sample(self, x):
        """执行推测采样，根据接受概率生成样本，并返回接受掩码。
        
        返回：
        - updated_x: 更新后的样本。
        - accept_mask: 布尔张量，指示每个样本是否被接受。
        """
        prob = self.judge_accept(x)
        guess = torch.rand_like(prob)
        accept_mask = prob > guess
        
        # 更新计数器
        self.accepted_tokens += accept_mask.sum().item()
        self.total_tokens += accept_mask.numel()
        
        if torch.all(accept_mask):
            return x, accept_mask

        return torch.where(accept_mask.unsqueeze(-1), x, self.sample_distribution()), accept_mask


class SpecDecodeRunner:
    """
    A demonstration of partial acceptance with local refinement 
    plus acceptance mask printing + overall acceptance rate tracking.
    """

    def __init__(
        self,
        draft_model: MAR,
        target_model: MAR,
        inner_temperature: float = 42.,
        ratio: float = 0.15
    ):
        self.draft_model = draft_model
        self.target_model = target_model

        self.seq_len = draft_model.seq_len
        self.token_embed_dim = draft_model.token_embed_dim

        self.sample_orders = draft_model.sample_orders
        self.unpatchify = draft_model.unpatchify
        self.temperature = inner_temperature
        self.ratio = ratio

        # acceptance stats
        self.verify_steps = 0
        self.accepted_total = 0
        self.seen_total = 0
        
        # 新增：记录每次“验证批次”的接受率
        self.batch_accept_rates = []

    def run(
        self,
        bsz: int,
        num_iter=64,
        num_draft=8,
        cfg_schedule="linear",
        labels: Optional[torch.LongTensor] = None,
        temperature=1.0,
        cfg=1.0,
        progress=False
    ):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        mask_draft = torch.ones(bsz, self.seq_len, device=device)
        mask_target = torch.ones(bsz, self.seq_len, device=device)

        draft = torch.zeros(bsz, self.seq_len, self.token_embed_dim, device=device)
        # store distribution stats
        draft_mean_map    = torch.zeros_like(draft)
        draft_var_map     = torch.zeros_like(draft)
        draft_log_var_map = torch.zeros_like(draft)

        noise = torch.randn(bsz, self.seq_len, self.draft_model.diffloss.in_channels, device=device)
        ns = self.draft_model.diffloss.gen_diffusion.num_timesteps
        inter_noise = torch.randn(bsz, self.seq_len, ns, self.draft_model.diffloss.in_channels, device=device)

        orders = self.sample_orders(bsz)

        # get class embeddings
        draft_class_emb  = self.draft_model.class_embed_cfg(bsz, labels, cfg)
        target_class_emb = self.target_model.class_embed_cfg(bsz, labels, cfg)

        # expand for CFG
        noise, inter_noise = self.draft_model.expand_batch(noise, inter_noise, cfg=cfg)

        iteration_range = range(num_iter)
        milestone = int(num_iter * self.ratio)

        if progress:
            iteration_range = tqdm(iteration_range)

        for step in iteration_range:
            # If step <= milestone, we do a "target-based" draft
            if step <= milestone:
                draft, mask_draft, draft_aux = self.target_model.sample_tokens_step(
                    draft,
                    mask_draft,
                    noise,
                    inter_noise,
                    orders,
                    bsz,
                    target_class_emb,
                    step, num_iter,
                    cfg_schedule, temperature,
                    cfg
                )
                mask_target = mask_draft.clone()

            else:
                # step > milestone => use the draft model
                draft, mask_draft, draft_aux = self.draft_model.sample_tokens_step(
                    draft,
                    mask_draft,
                    noise,
                    inter_noise,
                    orders,
                    bsz,
                    draft_class_emb,
                    step, num_iter,
                    cfg_schedule, temperature,
                    cfg
                )

                # gather distribution from draft
                draft_dist = self.draft_model.get_distribution()
                draft_dist.mean, draft_dist.variance, draft_dist.log_var, draft_dist.sample = \
                    self.draft_model.recover_batch(
                        draft_dist.mean,
                        draft_dist.variance,
                        draft_dist.log_var,
                        draft_dist.sample,
                        cfg=cfg
                    )
                idx_pred = draft_aux['index_to_pred']
                draft_mean_map[idx_pred]    = draft_dist.mean
                draft_var_map[idx_pred]     = draft_dist.variance
                draft_log_var_map[idx_pred] = draft_dist.log_var

                # local refine chunk
                refined = self.local_refine_chunk(
                    draft_tokens = draft[idx_pred],
                    distribution = draft_dist,
                    n_refine_steps = 3, # tune
                    micro_chunk = 2    # tune
                )
                draft[idx_pred] = refined

                # final partial verify every num_draft steps or last step
                if (step % num_draft == 0 and step != 0) or (step == num_iter - 1):
                    target, mask_target, target_aux = self.target_model.sample_tokens_step_with_mask(
                        draft.clone(),
                        mask_target,
                        noise,
                        inter_noise,
                        orders,
                        bsz,
                        target_class_emb,
                        step,
                        num_iter,
                        mask_draft,
                        draft_aux['mask_len'],
                        cfg_schedule,
                        temperature,
                        cfg
                    )
                    idx_pred_t = target_aux['index_to_pred']

                    # target distribution
                    target_dist = self.target_model.get_distribution()
                    target_dist.mean, target_dist.variance, target_dist.log_var, target_dist.sample = \
                        self.target_model.recover_batch(
                            target_dist.mean,
                            target_dist.variance,
                            target_dist.log_var,
                            target_dist.sample,
                            cfg=cfg
                        )
                    # gather relevant stats
                    draft_samples = draft[idx_pred_t]
                    dmean = draft_mean_map[idx_pred_t]
                    dvar  = draft_var_map[idx_pred_t]
                    dlogv = draft_log_var_map[idx_pred_t]

                    # final acceptance
                    processor = DistributionProcessor(
                        x1_mean=dmean,
                        x1_std=dvar,
                        x1_log_var=dlogv,
                        x2_mean=target_dist.mean,
                        x2_std=target_dist.variance,
                        x2_log_var=target_dist.log_var,
                        dim=self.token_embed_dim,
                        temperature=self.temperature
                    )
                    updated, accept_mask = processor.speculative_sample(draft_samples)
                    draft[idx_pred_t] = updated

                    # Convert accept_mask => 't'/'f' string
                    mask_str = ''.join(
                        't' if a else 'f'
                        for a in accept_mask.cpu().tolist()
                    )
                    print(f"Step={step}, acceptance mask: {mask_str}")

                    # 仅记录该批 acceptance rate，不在此打印 overall
                    # （accept_mask 只是当前这 batch 的接受情况）
                    local_rate = 100.0 * accept_mask.sum().item() / accept_mask.numel()
                    self.batch_accept_rates.append(local_rate)

                    # 仍然统计全局，但不打印
                    self.verify_steps += 1
                    self.accepted_total += accept_mask.sum().item()
                    self.seen_total     += accept_mask.numel()

        # finalize
        tokens = self.unpatchify(draft)

        # 结算：在所有步骤完成后，统一打印每个验证批次的接受率 & overall
        print("\n==== Acceptance Rate Report ====")
        for i, rate in enumerate(self.batch_accept_rates, start=1):
            print(f" Batch verify {i}: {rate:.2f}% acceptance")
        overall_rate = 100.0 * self.accepted_total / max(1, self.seen_total)
        print(f"Overall acceptance across all verify steps: {overall_rate:.2f}%\n")

        return tokens


    def local_refine_chunk(
        self,
        draft_tokens: torch.Tensor,
        distribution,
        n_refine_steps=3,
        micro_chunk=2
    ) -> torch.Tensor:
        """
        A 'smarter' local correction approach:
          - We refine in micro-chunks (partial acceptance).
          - We measure whether updated_slice is closer to `distribution.mean`
            than the old slice. If so, we accept; else we may randomize acceptance.
          - This tends to push tokens closer to the local distribution's mean
            (i.e., more 'likely' region).
        """
        refined = draft_tokens.clone()        # shape [N, D]
        N = refined.shape[0]
        device = refined.device

        # We'll interpret distribution.mean, distribution.variance
        # as the draft model's local distribution stats for these tokens.
        for _ in range(n_refine_steps):
            start = 0
            while start < N:
                end = min(start + micro_chunk, N)
                slice_ref = refined[start:end]                 # shape [m, D]
                slice_mean = distribution.mean[start:end]      # shape [m, D]

                # 1) SHIFT tokens closer to mean + noise injection
                direction = slice_mean - slice_ref
                step_noise = 0.02 * torch.randn_like(slice_ref)
                updated_slice = slice_ref + 0.5 * direction + step_noise

                # 2) Compute distance to mean (old vs. new)
                dist_old = (slice_ref  - slice_mean).norm(dim=-1)   # shape [m]
                dist_new = (updated_slice - slice_mean).norm(dim=-1)# shape [m]

                # 3) local_ratio = old_distance / new_distance
                # If ratio>1 => the updated slice is strictly closer => good sign
                local_ratio = dist_old / (dist_new + 1e-20)

                # 4) partial acceptance
                guess = torch.rand_like(local_ratio)  # shape [m]
                accept_mask = local_ratio > guess     # boolean [m]

                # 5) finalize partial acceptance
                slice_out = torch.where(
                    accept_mask.unsqueeze(-1),
                    updated_slice,
                    slice_ref
                )
                refined[start:end] = slice_out
                start = end

        return refined
