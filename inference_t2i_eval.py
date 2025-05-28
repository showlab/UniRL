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

from xfinders import examples
from xfinders.helpers import DataProcessor
from xfinders.modules import Comparator, Extractor
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 

from evaluate_imgs import evaluate_image
import numpy as np

# set_seed(0)

def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")




torch.random.manual_seed(0) 
model = AutoModelForCausalLM.from_pretrained( 
    "microsoft/Phi-3-mini-128k-instruct",  
    device_map="cuda",  
    torch_dtype="auto",  
    trust_remote_code=True,  
) 

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct") 

pipe = pipeline( 
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
) 


def run_example(question,gt,llm_output):

    messages = [ 
        {"role": "system", "content": "You are a helpful AI assistant."}, 
        {"role": "user", "content": f"Please help to judge the two answers same or not.The question is {question} One is the standard answer, another is model's output. The standard answer is {gt} and the model's output is {llm_output}. Please only use one word to answer me same or not same"}, 
    ] 
    
    generation_args = { 
        "max_new_tokens": 50, 
        "return_full_text": False, 
        "temperature": 0.0, 
        "do_sample": False, 
    } 

    output = pipe(messages, **generation_args)
    result = output[0]['generated_text'].strip()
    return result


def t2i_generate(prompts, model, mask_token_id, device, config):
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
    # import pdb; pdb.set_trace()
    # evaluate_image()
    # total_pil_images.append(pil_images[0])
    # pil_images[0].save("./img{}.png".format(step))
    # import pdb; pdb.set_trace()
    return gen_token_ids, pil_images

def mmu_generate(model, uni_prompting, question, device, top_k, gen_token_ids):
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
    return text
    


if __name__ == '__main__':

    config = get_config()

    resume_wandb_run = config.wandb.resume
    run_id = config.wandb.get("run_id", None)
    if run_id is None:
        resume_wandb_run = False
        run_id = wandb.util.generate_id()
        config.wandb.run_id = run_id

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
    final_result = []

    mask_token_id = model.config.mask_token_id

    # load from users passed arguments
    if config.get("validation_prompts_file", None) is not None:
        config.dataset.params.validation_prompts_file = config.validation_prompts_file
    config.training.batch_size = 1
    config.training.guidance_scale = config.guidance_scale
    config.training.generation_timesteps = config.generation_timesteps

    with open("/mnt/bn/vgfm2/test_dit/weijia/code_cycle/validation_prompts/evaluation_metadata_qa.jsonl") as  f:
        validation_prompts = f.read().splitlines()
    top_k = 1
    correct_num = 0
    total_num = 0
    sample_count = 0
    total_num = len(validation_prompts)
    split_num = total_num//8
    if config.num_device<7:
        validation_prompts = validation_prompts[config.num_device*split_num:config.num_device*split_num+split_num]
    else:
        validation_prompts = validation_prompts[config.num_device*split_num:]
    total_result = []

    # import pdb; pdb.set_trace()

    for step in tqdm(range(0, len(validation_prompts))):
        prompts = validation_prompts[step:step + config.training.batch_size]
        data = json.loads(validation_prompts[step])
        prompts = [data['prompt']]
        top_k = 1
        total_pil_images = []

        outpath = os.path.join(config.outdir, f"{step+config.num_device*split_num:0>5}")
        os.makedirs(outpath, exist_ok=True)
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(data, fp)

        sample_count = 0
        #  for sample_count in range(4):
        for _ in range(90):
            # image_result = {}

            gen_token_ids,pil_images = t2i_generate(prompts, model, mask_token_id, device, config)
            # import pdb; pdb.set_trace()
            result = evaluate_image(np.array(pil_images[0]), data)
            is_correct = result['correct']
            if is_correct:
                img_path = os.path.join(os.path.join(sample_path, f"{sample_count:05}.png"))
                pil_images[0].save(img_path)
                sample_count+=1
            # import pdb; pdb.set_trace()
            # if sample_count==4:
            if sample_count==30:
                break
        
        if sample_count<30:
            while(1):
                gen_token_ids,pil_images = t2i_generate(prompts, model, mask_token_id, device, config)
                img_path = os.path.join(os.path.join(sample_path, f"{sample_count:05}.png"))
                pil_images[0].save(img_path)
                sample_count+=1
                if sample_count ==30:
                    break
        print(sample_count)
            

       