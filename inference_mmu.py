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
    # checkpoint = torch.load("/mnt/bn/vgfm2/test_dit/weijia/outputs12/show-o-training-journey_drpo_beta_10_only_journeydb/checkpoint-401/unwrapped_model/pytorch_model.bin",map_location = "cpu")
    # /mnt/bn/vgfm2/test_dit/weijia/outputs12/show-o-training-journey_drpo_beta_0_2/checkpoint-401/unwrapped_model/pytorch_model.bin
    checkpoint = torch.load(config.experiment.load_ckpt, map_location = "cpu")
    model.load_state_dict(checkpoint,strict = True)

    mask_token_id = model.config.mask_token_id

    # load from users passed arguments
    if config.get("validation_prompts_file", None) is not None:
        config.dataset.params.validation_prompts_file = config.validation_prompts_file
    config.training.batch_size = 1
    # load from users passed arguments
    # import pdb; pdb.set_trace()

    # with open("/mnt/bn/vgfm2/test_dit/weijia/Image-Generation-CoT/evaluation_metadata.jsonl", "r") as f:
    #     validation_prompts = f.read().splitlines()
    # with open("/mnt/bn/vgfm2/test_dit/geneval/results.jsonl", "r") as f:
    #     validation_prompts = f.read().splitlines()
    # import pdb; pdb.set_trace()
    top_k = 1
    # image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs10/data3/01094/samples/00016.png"
    # image_path = "/mnt/bn/vgfm2/test_dit/weijia/HermesFlow/datasets/journeydb/imgs/989.png"
    # image_path = "/mnt/bn/vgfm2/test_dit/weijia/HermesFlow/datasets/journeydb/imgs/1747.png"
    # image_path = "/mnt/bn/vgfm2/test_dit/weijia/HermesFlow/datasets/journeydb/imgs/453.png"
    # image_path = "/mnt/bn/vgfm2/test_dit/weijia/HermesFlow/datasets/journeydb/imgs/1451.png"
    # image_path = "/mnt/bn/vgfm2/test_dit/weijia/outputs13/output_all_types_beta_0_2_counting/00464/samples/00000.png"
    # image_path =  "/mnt/bn/vgfm2/test_dit/weijia/outputs15/output_grpo_beta_0_2_bs6/00205/samples/00001.png"
    # image_path = "/mnt/bn/vgfm2/test_dit/weijia/code_cycle/img0.png"
    # image_path = "/mnt/bn/vgfm2/test_dit/weijia/code_cycle/flux-dev.png"
    # image_path = "/mnt/bn/vgfm2/test_dit/weijia/Emu3/img.png"
    image_path = "/mnt/bn/vgfm2/test_dit/weijia/code_2stage_mod/ood_1/img10.png"
    
    images = []
    # question = ["What is position relationship between these items?"]
    # question = ["What is position relationship between these items?"]
    # question = ['is this a woman?']
    # question = ['Is this a woman? Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags.']
    # question = ["Question:who is driving the train? Choice:['conductor', 'engineer', 'driver', 'pilot']"]
    # question = ["who is driving the train?"]
    # question = ['is this Theoretical Discretization and tell me the reason?']
    # question = ["Question:is this Theoretical Discretization? Choice:['yes', 'no']"]
    # question = ["How many items in this image?"]
    question = ["Give the cpation for this image."]
    
    # import pdb; pdb.set_trace()

    for step in tqdm(range(0, 6)):
        # prompts = validation_prompts[step:step + config.training.batch_size]
        # import pdb; pdb.set_trace()
        # data = json.loads(validation_prompts[step])
        # question = [data['question']]
        # prompts = [data['prompt']]
        # answer =  data['answer']
        # import pdb; pdb.set_trace()
        # question = ["What is position relationship between these items?"]
        top_k = 50
        # pil_image = Image.open(pil_image)
        image_ori = Image.open(image_path).convert("RGB")
        image = image_transform(image_ori, resolution=config.dataset.params.resolution).to(device)
        image = image.unsqueeze(0)
        images.append(image)
        # pixel_values = clip_image_processor.preprocess(image_ori, return_tensors="pt")["pixel_values"][0]
        image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)   

        input_ids = uni_prompting.text_tokenizer(['USER: \n' + question[0] + ' ASSISTANT:'])[
                    'input_ids']
        input_ids = torch.tensor(input_ids).to(device)
        # gen_token_ids+=len(uni_prompting.text_tokenizer)

        input_ids = torch.cat([
            (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|mmu|>']).to(device),
            (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|soi|>']).to(device),
            image_tokens,
            (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|eoi|>']).to(device),
            (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|sot|>']).to(device),
            input_ids
        ], dim=1).long()

        attention_mask = create_attention_mask_for_mmu(input_ids.to(device),
                                                        eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']))

        cont_toks_list = model.mmu_generate(input_ids, attention_mask=attention_mask, max_new_tokens=200, top_k=top_k,  temperature = 1,  eot_token=uni_prompting.sptids_dict['<|eot|>'])
        
        cont_toks_list = torch.stack(cont_toks_list).squeeze()[None]

        text = uni_prompting.text_tokenizer.batch_decode(cont_toks_list, skip_special_tokens=True)
        
        print(text)
        # print(answer)


        # responses[i] += f'User: ' + question + f'\n Answer : ' + text[0] + '\n'


        # images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        # images *= 255.0
        # images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        # pil_images = [Image.fromarray(image) for image in images]

        # # wandb_images = [wandb.Image(image, caption=prompts[i]) for i, image in enumerate(pil_images)]
        # # wandb.log({"generated_images": wandb_images}, step=step)
        # pil_images[0].save("./img{}.png".format(step))
        import pdb; pdb.set_trace()
