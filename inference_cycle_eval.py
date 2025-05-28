# coding=utf-8
# Copyright 2024 NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import wandb
from models import Showo, MAGVITv2, get_mask_chedule
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next
from training.prompting_utils import UniversalPrompting, create_attention_mask_for_mmu, create_attention_mask_for_mmu_vit
from training.utils import get_config, flatten_omega_conf, image_transform
from transformers import AutoTokenizer
import torch.nn.functional as F
from models.training_utils import set_seed
import json

# set_seed(0)

def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")

if __name__ == '__main__':

    config = get_config()

    resume_wandb_run = config.wandb.resume
    run_id = config.wandb.get("run_id", None)
    if run_id is None:
        resume_wandb_run = False
        run_id = wandb.util.generate_id()
        config.wandb.run_id = run_id

    # wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}

    # wandb.init(
    #     project="demo",
    #     name=config.experiment.name + '_t2i' + f'_{config.mode}',
    #     config=wandb_config,
    # )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side="left")

    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)

    vq_model = get_vq_model_class(config.model.vq_model.type)
    vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()

    model = Showo.from_pretrained(config.model.showo.pretrained_model_path).to(device)
    model.eval()

    # import pdb; pdb.set_trace()
    # checkpoint = torch.load("/mnt/bn/vgfm2/test_dit/weijia/checkpoints/showo.bin",map_location = "cpu")
    # model.load_state_dict(checkpoint,strict = False)

    mask_token_id = model.config.mask_token_id

    # load from users passed arguments
    if config.get("validation_prompts_file", None) is not None:
        config.dataset.params.validation_prompts_file = config.validation_prompts_file
    config.training.batch_size = 1
    config.training.guidance_scale = config.guidance_scale
    config.training.generation_timesteps = config.generation_timesteps
    # load from users passed arguments
    # import pdb; pdb.set_trace()

    # with open("/mnt/bn/vgfm2/test_dit/weijia/Image-Generation-CoT/evaluation_metadata.jsonl", "r") as f:
    #     validation_prompts = f.read().splitlines()
    # with open("/mnt/bn/vgfm2/test_dit/geneval/results.jsonl", "r") as f:
    #     validation_prompts = f.read().splitlines()
    # import pdb; pdb.set_trace()

    file_path = "/mnt/bn/vgfm2/test_dit/llava_v1_5_mix665k_new.json"
    
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    

    # import pdb; pdb.set_trace()

    # validation_prompts = ["2 trees and 1 person"]
    # question = ["How many items are there in the image?"]
    # question = ["Give me the caption of the image"]
    # question = ["Do you think the image is good or bad? Output the thinking process."]
    # question = [validation_prompts[0]['question']]
    top_k = 1

    for step in tqdm(range(0, len(validation_prompts))):
        # prompts = validation_prompts[step:step + config.training.batch_size]
        # import pdb; pdb.set_trace()
        data = json.loads(validation_prompts[step])
        question = [data['question']]
        prompts = [data['prompt']]
        answer =  data['answer']
        # import pdb; pdb.set_trace()
        top_k = 1

        image_tokens = torch.ones((len(prompts), config.model.showo.num_vq_tokens),
                                    dtype=torch.long, device=device) * mask_token_id

        input_ids, _ = uni_prompting((prompts, image_tokens), 't2i_gen')

        if config.training.guidance_scale > 0:
            uncond_input_ids, _ = uni_prompting(([''] * len(prompts), image_tokens), 't2i_gen')
            attention_mask = create_attention_mask_predict_next(torch.cat([input_ids, uncond_input_ids], dim=0),
                                                                pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                rm_pad_in_image=True)
        else:
            attention_mask = create_attention_mask_predict_next(input_ids,
                                                                pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                rm_pad_in_image=True)
            uncond_input_ids = None

        if config.get("mask_schedule", None) is not None:
            schedule = config.mask_schedule.schedule
            args = config.mask_schedule.get("params", {})
            mask_schedule = get_mask_chedule(schedule, **args)
        else:
            mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))
        
        with torch.no_grad():
            # import pdb; pdb.set_trace()
            gen_token_ids = model.t2i_generate(
                input_ids=input_ids,
                uncond_input_ids=uncond_input_ids,
                attention_mask=attention_mask,
                guidance_scale=config.training.guidance_scale,
                temperature=config.training.get("generation_temperature", 1.0),
                timesteps=config.training.generation_timesteps,
                noise_schedule=mask_schedule,
                noise_type=config.training.get("noise_type", "mask"),
                seq_len=config.model.showo.num_vq_tokens,
                uni_prompting=uni_prompting,
                config=config,
            )

        
        gen_token_ids = torch.clamp(gen_token_ids, max=config.model.showo.codebook_size - 1, min=0)
        images = vq_model.decode_code(gen_token_ids)
        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        images *= 255.0
        images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        pil_images = [Image.fromarray(image) for image in images]

        # wandb_images = [wandb.Image(image, caption=prompts[i]) for i, image in enumerate(pil_images)]
        # wandb.log({"generated_images": wandb_images}, step=step)
        pil_images[0].save("./img{}.png".format(step))
        # import pdb; pdb.set_trace()

        input_ids = uni_prompting.text_tokenizer(['USER: \n' + question[0] + ' ASSISTANT:'])[
                    'input_ids']
        input_ids = torch.tensor(input_ids).to(device)
        gen_token_ids+=len(uni_prompting.text_tokenizer)

        input_ids = torch.cat([
            (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|mmu|>']).to(device),
            (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|soi|>']).to(device),
            gen_token_ids,
            (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|eoi|>']).to(device),
            (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|sot|>']).to(device),
            input_ids
        ], dim=1).long()

        attention_mask = create_attention_mask_for_mmu(input_ids.to(device),
                                                        eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']))

        cont_toks_list = model.mmu_generate(input_ids, attention_mask=attention_mask,
                                    max_new_tokens=100, top_k=top_k,
                                    eot_token=uni_prompting.sptids_dict['<|eot|>'])
        
        cont_toks_list = torch.stack(cont_toks_list).squeeze()[None]

        text = uni_prompting.text_tokenizer.batch_decode(cont_toks_list, skip_special_tokens=True)
        
        print(text)
        print(answer)
        import pdb; pdb.set_trace()


        # responses[i] += f'User: ' + question + f'\n Answer : ' + text[0] + '\n'


        # images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        # images *= 255.0
        # images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        # pil_images = [Image.fromarray(image) for image in images]

        # # wandb_images = [wandb.Image(image, caption=prompts[i]) for i, image in enumerate(pil_images)]
        # # wandb.log({"generated_images": wandb_images}, step=step)
        # pil_images[0].save("./img{}.png".format(step))
        import pdb; pdb.set_trace()
