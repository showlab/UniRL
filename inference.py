
import json
import logging
import math
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import random
from typing import Any, List, Tuple, Union
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, ListConfig, OmegaConf
from models import Showo
from prompting_v7_predict_next import UniversalPrompting, create_attention_mask, create_attention_mask_predict_next, create_attention_mask_for_mmu
from transformers import AutoTokenizer

try:
    import apex

    is_apex_available = True
except ImportError:
    is_apex_available = False

def get_config():

    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf


def flatten_omega_conf(cfg: Any, resolve: bool = False) -> List[Tuple[str, Any]]:
    ret = []

    def handle_dict(key: Any, value: Any, resolve: bool) -> List[Tuple[str, Any]]:
        return [(f"{key}.{k1}", v1) for k1, v1 in flatten_omega_conf(value, resolve=resolve)]

    def handle_list(key: Any, value: Any, resolve: bool) -> List[Tuple[str, Any]]:
        return [(f"{key}.{idx}", v1) for idx, v1 in flatten_omega_conf(value, resolve=resolve)]

    if isinstance(cfg, DictConfig):
        for k, v in cfg.items_ex(resolve=resolve):
            if isinstance(v, DictConfig):
                ret.extend(handle_dict(k, v, resolve=resolve))
            elif isinstance(v, ListConfig):
                ret.extend(handle_list(k, v, resolve=resolve))
            else:
                ret.append((str(k), v))
    elif isinstance(cfg, ListConfig):
        for idx, v in enumerate(cfg._iter_ex(resolve=resolve)):
            if isinstance(v, DictConfig):
                ret.extend(handle_dict(idx, v, resolve=resolve))
            elif isinstance(v, ListConfig):
                ret.extend(handle_list(idx, v, resolve=resolve))
            else:
                ret.append((str(idx), v))
    else:
        assert False

    return ret


def get_vq_model_class(model_type):
    if model_type == "vqgan":
        return VQGANModel
    elif model_type == "movq":
        return MOVQ
    elif model_type == "maskgit_vqgan":
        return MaskGitVQGAN
    elif model_type == "paella_vq":
        return PaellaVQModel
    elif model_type == "magvitv2":
        return MagViTv2
    else:
        raise ValueError(f"model_type {model_type} not supported for VQGAN")


def soft_target_cross_entropy(logits, targets, soft_targets):
    # ignore the first token from logits and targets (class id token)
    logits = logits[:, 1:]
    targets = targets[:, 1:]

    logits = logits[..., : soft_targets.shape[-1]]

    log_probs = F.log_softmax(logits, dim=-1)
    padding_mask = targets.eq(-100)

    loss = torch.sum(-soft_targets * log_probs, dim=-1)
    loss.masked_fill_(padding_mask, 0.0)

    # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
    num_active_elements = padding_mask.numel() - padding_mask.long().sum()
    loss = loss.sum() / num_active_elements
    return loss


def get_loss_weight(t, mask, min_val=0.3):
    return 1 - (1 - mask) * ((1 - t) * (1 - min_val))[:, None]


def mask_or_random_replace_tokens(image_tokens, mask_id, config, mask_schedule, is_train=True):
    batch_size, seq_len = image_tokens.shape

    if not is_train and config.training.get("eval_mask_ratios", None):
        mask_prob = random.choices(config.training.eval_mask_ratios, k=batch_size)
        mask_prob = torch.tensor(mask_prob, device=image_tokens.device)
    else:
        # Sample a random timestep for each image
        timesteps = torch.rand(batch_size, device=image_tokens.device)
        # Sample a random mask probability for each image using timestep and cosine schedule
        mask_prob = mask_schedule(timesteps)
        mask_prob = mask_prob.clip(config.training.min_masking_rate)

    # creat a random mask for each image
    num_token_masked = (seq_len * mask_prob).round().clamp(min=1)

    mask_contiguous_region_prob = config.training.get("mask_contiguous_region_prob", None)

    if mask_contiguous_region_prob is None:
        mask_contiguous_region = False
    else:
        mask_contiguous_region = random.random() < mask_contiguous_region_prob

    if not mask_contiguous_region:
        batch_randperm = torch.rand(batch_size, seq_len, device=image_tokens.device).argsort(dim=-1)
        mask = batch_randperm < num_token_masked.unsqueeze(-1)
    else:
        resolution = int(seq_len ** 0.5)
        mask = torch.zeros((batch_size, resolution, resolution), device=image_tokens.device)

        # TODO - would be nice to vectorize
        for batch_idx, num_token_masked_ in enumerate(num_token_masked):
            num_token_masked_ = int(num_token_masked_.item())

            # NOTE: a bit handwavy with the bounds but gets a rectangle of ~num_token_masked_
            num_token_masked_height = random.randint(
                math.ceil(num_token_masked_ / resolution), min(resolution, num_token_masked_)
            )
            num_token_masked_height = min(num_token_masked_height, resolution)

            num_token_masked_width = math.ceil(num_token_masked_ / num_token_masked_height)
            num_token_masked_width = min(num_token_masked_width, resolution)

            start_idx_height = random.randint(0, resolution - num_token_masked_height)
            start_idx_width = random.randint(0, resolution - num_token_masked_width)

            mask[
            batch_idx,
            start_idx_height: start_idx_height + num_token_masked_height,
            start_idx_width: start_idx_width + num_token_masked_width,
            ] = 1

        mask = mask.reshape(batch_size, seq_len)
        mask = mask.to(torch.bool)

    # mask images and create input and labels
    if config.training.get("noise_type", "mask"):
        input_ids = torch.where(mask, mask_id, image_tokens)
    elif config.training.get("noise_type", "random_replace"):
        # sample random tokens from the vocabulary
        random_tokens = torch.randint_like(
            image_tokens, low=0, high=config.model.codebook_size, device=image_tokens.device
        )
        input_ids = torch.where(mask, random_tokens, image_tokens)
    else:
        raise ValueError(f"noise_type {config.training.noise_type} not supported")

    if (
            config.training.get("predict_all_tokens", False)
            or config.training.get("noise_type", "mask") == "random_replace"
    ):
        labels = image_tokens
        loss_weight = get_loss_weight(mask_prob, mask.long())
    else:
        labels = torch.where(mask, image_tokens, -100)
        loss_weight = None

    return input_ids, labels, loss_weight, mask_prob

if __name__ == '__main__':

    config = get_config()

    resume_wandb_run = config.wandb.resume
    run_id = config.wandb.get("run_id", None)
    run_id = 'dkoh9qng'
    if run_id is None:
        resume_wandb_run = False
        run_id = wandb.util.generate_id()
        config.wandb.run_id = run_id

    wandb_init_kwargs = dict(
        name=config.experiment.name,
        id=run_id,
        resume=resume_wandb_run,
        entity=config.wandb.get("entity", None),
        config_exclude_keys=[],
    )
    wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
    wandb_config.pop("experiment.resume_from_checkpoint")

    wandb.init(
        project="demo",
        name=config.experiment.name,
        notes="tweak baseline",
        tags=["baseline", "paper1"],
        config=wandb_config,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model.transformer.pretrained_model_path, padding_side="left")

    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=(
                                           "soi", "eoi", "sovi", "eovi", "t2i", "mmu", "t2v", "v2v", "lvg"),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)
    print(uni_prompting.sptids_dict)
    vq_model = get_vq_model_class(config.model.vq_model.type)().to(device)
    vq_model.load_state_dict(torch.load(config.model.vq_model.pretrained))

    # Freeze the text model and VQGAN
    vq_model.requires_grad_(False)
    vq_model.eval()
    model = Showo(**config.model.transformer).to(device)

    # directly load state dict. we resume a 175000 model to add qk_norm and overall training loss on a sequence
    # ****************************************** choose a checkpoint ******************************************
    # checkpoint-30000 is the pre-trained model
    # state_dict = torch.load('phi_journeydb_refinedweb_6x8_gpus_with_lm_img_no_attend_to_text_pad_zero_grad_sp_head_muse_setting_predict_next_qk_norm_weightfp32_mixedbf16_sa1b_cap_trial2/checkpoint-10000/pytorch_model.bin')
    # state_dict = torch.load('phi_journeydb_refinedweb_6x8_gpus_with_lm_img_no_attend_to_text_pad_zero_grad_sp_head_muse_setting_predict_next_qk_norm_weightfp32_mixedbf16_recaptionedlaion12m_cap/checkpoint-40000/pytorch_model.bin')
    # state_dict = torch.load('phi_journeydb_refinedweb_6x8_gpus_with_lm_img_no_attend_to_text_pad_zero_grad_sp_head_muse_setting_predict_next_qk_norm_weightfp32_mixedbf16_recaptionedlaion12m_only_generation/checkpoint-190000/pytorch_model.bin')
    # state_dict = torch.load('phi_6x8_gpus_pretraining_stage/checkpoint-100000/pytorch_model.bin')
    # state_dict = torch.load('phi_6x8_gpus_pretraining_stage_imagenet/checkpoint-260000/unwrapped_model/pytorch_model.bin')
    # state_dict = torch.load(
    #     'phi_6x8_gpus_pretraining_stage_imagenet_adjust_lr_0_00002/checkpoint-430000/unwrapped_model/pytorch_model.bin')
    # state_dict = torch.load('phi_6x8_gpus_pretraining_stage_imagenet_only_generation/checkpoint-260000/unwrapped_model/pytorch_model.bin')
    # state_dict = torch.load('phi_6x8_gpus_pretraining_stage_long_caption/checkpoint-400000/unwrapped_model/pytorch_model.bin')
    state_dict = torch.load('phi_6x8_gpus_pretraining_stage_long_caption_sft12m/checkpoint-10000/unwrapped_model/pytorch_model.bin')
    # state_dict = torch.load('phi_6x8_gpus_pretraining_stage_long_caption_512x512/checkpoint-150000/unwrapped_model/pytorch_model.bin')
    # state_dict = torch.load('phi_journeydb_refinedweb_6x8_gpus_with_lm_img_no_attend_to_text_pad_zero_grad_sp_head_muse_setting_predict_next_qk_norm_weightfp32_mixedbf16_sa1b_cap_new/checkpoint-180000/pytorch_model.bin')
    # state_dict = torch.load('phi_journeydb_refinedweb_6x8_gpus_with_lm_img_no_attend_to_text_pad_zero_grad_sp_head_muse_setting_predict_next_qk_norm_weightfp32_mixedbf16_sa1b_cap_journeydb_cap/checkpoint-30000/pytorch_model.bin')
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    # model.to(dtype=torch.bfloat16)
    del state_dict
    # ****************************************** choose a checkpoint ******************************************

    mask_id = model.config.mask_token_id
    output_size = model.output_size

    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params
    total_batch_size_without_accum = config.training.batch_size * 1
    total_batch_size = (
            config.training.batch_size * 1 * config.training.gradient_accumulation_steps
    )

    config.dataset.params.validation_prompts_file = 'validation_prompts/partiprompts_subset.txt'
    with open(config.dataset.params.validation_prompts_file, "r") as f:
        validation_prompts = f.read().splitlines()
    # validation_prompts = [""] * 16
    # import jsonlines
    # prompts = []
    # with jsonlines.open("./valid_anno_repath.jsonl") as reader:
    #     for line in tqdm(reader):
    #         k = line['img_path'].split('./')[-1].split('/')[0] + '/' + \
    #             line['img_path'].split('./')[-1].split('/')[1].split('_')[-1].split('.')[0]
    #         # print(line['prompt'])
    #         prompts.append(line['Task2']['Caption'])

    # import ipdb
    # ipdb.set_trace()
    for step in tqdm(range(0, len(validation_prompts), 32)):
        # if step == 50:
        #     break
        prompts = validation_prompts[step:step + 32]
    # for step in tqdm(range(0, 50)):
        mask_token_id = config.model.transformer.vocab_size - 1
        image_tokens = torch.ones((len(prompts), config.model.transformer.num_vq_tokens), dtype=torch.long,
                                  device=device) * mask_token_id
        print(image_tokens.shape)
        input_ids, _ = uni_prompting((prompts, image_tokens), 't2i_gen')

        # set the cfg guidance
        config.training.guidance_scale = 1.75
        config.training.generation_timesteps = 12
        if config.training.guidance_scale > 0:
            uncond_input_ids, _ = uni_prompting(([''] * len(prompts), image_tokens), 't2i_gen')
            attention_mask = create_attention_mask_predict_next(torch.cat([input_ids, uncond_input_ids], dim=0),
                                                                pad_id=int(uni_prompting.sptids_dict['pad']),
                                                                soi_id=int(uni_prompting.sptids_dict['soi']),
                                                                eoi_id=int(uni_prompting.sptids_dict['eoi']),
                                                                rm_pad_in_image=True)
        else:
            attention_mask = create_attention_mask_predict_next(input_ids,
                                                                pad_id=int(uni_prompting.sptids_dict['pad']),
                                                                soi_id=int(uni_prompting.sptids_dict['soi']),
                                                                eoi_id=int(uni_prompting.sptids_dict['eoi']),
                                                                rm_pad_in_image=True)
            uncond_input_ids = None
        # attention_mask = attention_mask.to(dtype=torch.bfloat16)
        if config.get("mask_schedule", None) is not None:
            schedule = config.mask_schedule.schedule
            args = config.mask_schedule.get("params", {})
            mask_schedule = get_mask_chedule(schedule, **args)
        else:
            mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))
        topk = 0
        topp = 1.0
        with torch.no_grad():
            # Generate images
            gen_token_ids = model.generate3(
                input_ids=input_ids,
                uncond_input_ids=uncond_input_ids,
                attention_mask=attention_mask,
                # encoder_hidden_states=encoder_hidden_states,
                # cond_embeds=clip_embeds,
                # empty_embeds=empty_embeds,
                # empty_cond_embeds=empty_clip_embeds,
                # micro_conds=micro_conds,
                guidance_scale=config.training.guidance_scale,
                # guidance_scale=0,
                temperature=config.training.get("generation_temperature", 1.0),
                timesteps=config.training.generation_timesteps,
                noise_schedule=mask_schedule,
                noise_type=config.training.get("noise_type", "mask"),
                predict_all_tokens=config.training.get("predict_all_tokens", False),
                seq_len=config.model.transformer.num_vq_tokens,
                uni_prompting=uni_prompting,
                config=config,
                topk=topk,
                topp=topp
            )
        # In the beginning of training, the model is not fully trained and the generated token ids can be out of range
        # so we clamp them to the correct range.
        # import ipdb
        # ipdb.set_trace()
        # import ipdb
        # ipdb.set_trace()
        # minus 10 special tokens to match vq token indices
        # gen_token_ids = gen_token_ids - torch.tensor(10).to(gen_token_ids.device)
        # gen_token_ids = gen_token_ids - torch.tensor(len(uni_prompting.text_tokenizer)).to(gen_token_ids.device)
        gen_token_ids = torch.clamp(gen_token_ids, max=model.config.codebook_size - 1, min=0)

        images = vq_model.decode_code(gen_token_ids)

        # Convert to PIL images
        # method 1
        # images = 2.0 * images - 1.0
        # images = torch.clamp(images, -1.0, 1.0)
        # images = (images + 1.0) / 2.0
        # images *= 255.0
        # method 2
        # images = images.mul(255).add_(0.5).clamp_(0, 255)
        # save_image(images, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))
        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        images *= 255.0


        images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        pil_images = [Image.fromarray(image) for image in images]

        wandb_images = [wandb.Image(image, caption=prompts[i]) for i, image in enumerate(pil_images)]
        wandb.log({"generated_images": wandb_images}, step=step)

        # path = "/mnt/bn/vgfm2/test_mlx/xavier/code/3062/open_muse/generation_result"
        # for i, image in tqdm(enumerate(pil_images)):
        #     image.save(f"{path}/{i}_cfg_{config.training.guidance_scale}_tp_{config.training.generation_timesteps}_topk_{topk}_topp_{topp}_linear_guidance.jpg")
            # image.save(f"{path}/{i}_cfg_{config.training.guidance_scale}_w_filtering.jpg")

