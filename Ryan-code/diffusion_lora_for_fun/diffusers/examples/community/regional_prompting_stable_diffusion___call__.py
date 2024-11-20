@torch.no_grad()
def __call__(self, prompt: str, height: int=512, width: int=512,
    num_inference_steps: int=50, guidance_scale: float=7.5, negative_prompt:
    str=None, num_images_per_prompt: Optional[int]=1, eta: float=0.0,
    generator: Optional[torch.Generator]=None, latents: Optional[torch.
    Tensor]=None, output_type: Optional[str]='pil', return_dict: bool=True,
    rp_args: Dict[str, str]=None):
    active = KBRK in prompt[0] if isinstance(prompt, list) else KBRK in prompt
    if negative_prompt is None:
        negative_prompt = '' if isinstance(prompt, str) else [''] * len(prompt)
    device = self._execution_device
    regions = 0
    self.power = int(rp_args['power']) if 'power' in rp_args else 1
    prompts = prompt if isinstance(prompt, list) else [prompt]
    n_prompts = negative_prompt if isinstance(prompt, str) else [
        negative_prompt]
    self.batch = batch = num_images_per_prompt * len(prompts)
    all_prompts_cn, all_prompts_p = promptsmaker(prompts, num_images_per_prompt
        )
    all_n_prompts_cn, _ = promptsmaker(n_prompts, num_images_per_prompt)
    equal = len(all_prompts_cn) == len(all_n_prompts_cn)
    if Compel:
        compel = Compel(tokenizer=self.tokenizer, text_encoder=self.
            text_encoder)

        def getcompelembs(prps):
            embl = []
            for prp in prps:
                embl.append(compel.build_conditioning_tensor(prp))
            return torch.cat(embl)
        conds = getcompelembs(all_prompts_cn)
        unconds = getcompelembs(all_n_prompts_cn)
        embs = getcompelembs(prompts)
        n_embs = getcompelembs(n_prompts)
        prompt = negative_prompt = None
    else:
        conds = self.encode_prompt(prompts, device, 1, True)[0]
        unconds = self.encode_prompt(n_prompts, device, 1, True)[0
            ] if equal else self.encode_prompt(all_n_prompts_cn, device, 1,
            True)[0]
        embs = n_embs = None
    if not active:
        pcallback = None
        mode = None
    else:
        if any(x in rp_args['mode'].upper() for x in ['COL', 'ROW']):
            mode = 'COL' if 'COL' in rp_args['mode'].upper() else 'ROW'
            ocells, icells, regions = make_cells(rp_args['div'])
        elif 'PRO' in rp_args['mode'].upper():
            regions = len(all_prompts_p[0])
            mode = 'PROMPT'
            reset_attnmaps(self)
            self.ex = 'EX' in rp_args['mode'].upper()
            self.target_tokens = target_tokens = tokendealer(self,
                all_prompts_p)
            thresholds = [float(x) for x in rp_args['th'].split(',')]
        orig_hw = height, width
        revers = True

        def pcallback(s_self, step: int, timestep: int, latents: torch.
            Tensor, selfs=None):
            if 'PRO' in mode:
                self.step = step
                if len(self.attnmaps_sizes) > 3:
                    self.history[step] = self.attnmaps.copy()
                    for hw in self.attnmaps_sizes:
                        allmasks = []
                        basemasks = [None] * batch
                        for tt, th in zip(target_tokens, thresholds):
                            for b in range(batch):
                                key = f'{tt}-{b}'
                                _, mask, _ = makepmask(self, self.attnmaps[
                                    key], hw[0], hw[1], th, step)
                                mask = mask.unsqueeze(0).unsqueeze(-1)
                                if self.ex:
                                    allmasks[b::batch] = [(x - mask) for x in
                                        allmasks[b::batch]]
                                    allmasks[b::batch] = [torch.where(x > 0,
                                        1, 0) for x in allmasks[b::batch]]
                                allmasks.append(mask)
                                basemasks[b] = mask if basemasks[b
                                    ] is None else basemasks[b] + mask
                        basemasks = [(1 - mask) for mask in basemasks]
                        basemasks = [torch.where(x > 0, 1, 0) for x in
                            basemasks]
                        allmasks = basemasks + allmasks
                        self.attnmasks[hw] = torch.cat(allmasks)
                    self.maskready = True
            return latents

        def hook_forward(module):

            def forward(hidden_states: torch.Tensor, encoder_hidden_states:
                Optional[torch.Tensor]=None, attention_mask: Optional[torch
                .Tensor]=None, temb: Optional[torch.Tensor]=None, scale:
                float=1.0) ->torch.Tensor:
                attn = module
                xshape = hidden_states.shape
                self.hw = h, w = split_dims(xshape[1], *orig_hw)
                if revers:
                    nx, px = hidden_states.chunk(2)
                else:
                    px, nx = hidden_states.chunk(2)
                if equal:
                    hidden_states = torch.cat([px for i in range(regions)] +
                        [nx for i in range(regions)], 0)
                    encoder_hidden_states = torch.cat([conds] + [unconds])
                else:
                    hidden_states = torch.cat([px for i in range(regions)] +
                        [nx], 0)
                    encoder_hidden_states = torch.cat([conds] + [unconds])
                residual = hidden_states
                args = () if USE_PEFT_BACKEND else (scale,)
                if attn.spatial_norm is not None:
                    hidden_states = attn.spatial_norm(hidden_states, temb)
                input_ndim = hidden_states.ndim
                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel,
                        height * width).transpose(1, 2)
                batch_size, sequence_length, _ = (hidden_states.shape if 
                    encoder_hidden_states is None else
                    encoder_hidden_states.shape)
                if attention_mask is not None:
                    attention_mask = attn.prepare_attention_mask(attention_mask
                        , sequence_length, batch_size)
                    attention_mask = attention_mask.view(batch_size, attn.
                        heads, -1, attention_mask.shape[-1])
                if attn.group_norm is not None:
                    hidden_states = attn.group_norm(hidden_states.transpose
                        (1, 2)).transpose(1, 2)
                args = () if USE_PEFT_BACKEND else (scale,)
                query = attn.to_q(hidden_states, *args)
                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states
                elif attn.norm_cross:
                    encoder_hidden_states = attn.norm_encoder_hidden_states(
                        encoder_hidden_states)
                key = attn.to_k(encoder_hidden_states, *args)
                value = attn.to_v(encoder_hidden_states, *args)
                inner_dim = key.shape[-1]
                head_dim = inner_dim // attn.heads
                query = query.view(batch_size, -1, attn.heads, head_dim
                    ).transpose(1, 2)
                key = key.view(batch_size, -1, attn.heads, head_dim).transpose(
                    1, 2)
                value = value.view(batch_size, -1, attn.heads, head_dim
                    ).transpose(1, 2)
                hidden_states = scaled_dot_product_attention(self, query,
                    key, value, attn_mask=attention_mask, dropout_p=0.0,
                    is_causal=False, getattn='PRO' in mode)
                hidden_states = hidden_states.transpose(1, 2).reshape(
                    batch_size, -1, attn.heads * head_dim)
                hidden_states = hidden_states.to(query.dtype)
                hidden_states = attn.to_out[0](hidden_states, *args)
                hidden_states = attn.to_out[1](hidden_states)
                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(
                        batch_size, channel, height, width)
                if attn.residual_connection:
                    hidden_states = hidden_states + residual
                hidden_states = hidden_states / attn.rescale_output_factor
                if any(x in mode for x in ['COL', 'ROW']):
                    reshaped = hidden_states.reshape(hidden_states.size()[0
                        ], h, w, hidden_states.size()[2])
                    center = reshaped.shape[0] // 2
                    px = reshaped[0:center] if equal else reshaped[0:-batch]
                    nx = reshaped[center:] if equal else reshaped[-batch:]
                    outs = [px, nx] if equal else [px]
                    for out in outs:
                        c = 0
                        for i, ocell in enumerate(ocells):
                            for icell in icells[i]:
                                if 'ROW' in mode:
                                    out[0:batch, int(h * ocell[0]):int(h *
                                        ocell[1]), int(w * icell[0]):int(w *
                                        icell[1]), :] = out[c * batch:(c + 
                                        1) * batch, int(h * ocell[0]):int(h *
                                        ocell[1]), int(w * icell[0]):int(w *
                                        icell[1]), :]
                                else:
                                    out[0:batch, int(h * icell[0]):int(h *
                                        icell[1]), int(w * ocell[0]):int(w *
                                        ocell[1]), :] = out[c * batch:(c + 
                                        1) * batch, int(h * icell[0]):int(h *
                                        icell[1]), int(w * ocell[0]):int(w *
                                        ocell[1]), :]
                                c += 1
                    px, nx = (px[0:batch], nx[0:batch]) if equal else (px[0
                        :batch], nx)
                    hidden_states = torch.cat([nx, px], 0
                        ) if revers else torch.cat([px, nx], 0)
                    hidden_states = hidden_states.reshape(xshape)
                elif 'PRO' in mode:
                    px, nx = torch.chunk(hidden_states
                        ) if equal else hidden_states[0:-batch], hidden_states[
                        -batch:]
                    if (h, w) in self.attnmasks and self.maskready:

                        def mask(input):
                            out = torch.multiply(input, self.attnmasks[h, w])
                            for b in range(batch):
                                for r in range(1, regions):
                                    out[b] = out[b] + out[r * batch + b]
                            return out
                        px, nx = (mask(px), mask(nx)) if equal else (mask(
                            px), nx)
                    px, nx = (px[0:batch], nx[0:batch]) if equal else (px[0
                        :batch], nx)
                    hidden_states = torch.cat([nx, px], 0
                        ) if revers else torch.cat([px, nx], 0)
                return hidden_states
            return forward

        def hook_forwards(root_module: torch.nn.Module):
            for name, module in root_module.named_modules():
                if ('attn2' in name and module.__class__.__name__ ==
                    'Attention'):
                    module.forward = hook_forward(module)
        hook_forwards(self.unet)
    output = StableDiffusionPipeline(**self.components)(prompt=prompt,
        prompt_embeds=embs, negative_prompt=negative_prompt,
        negative_prompt_embeds=n_embs, height=height, width=width,
        num_inference_steps=num_inference_steps, guidance_scale=
        guidance_scale, num_images_per_prompt=num_images_per_prompt, eta=
        eta, generator=generator, latents=latents, output_type=output_type,
        return_dict=return_dict, callback_on_step_end=pcallback)
    if 'save_mask' in rp_args:
        save_mask = rp_args['save_mask']
    else:
        save_mask = False
    if mode == 'PROMPT' and save_mask:
        saveattnmaps(self, output, height, width, thresholds, 
            num_inference_steps // 2, regions)
    return output
