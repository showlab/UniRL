# CUDA_VISIBLE_DEVICES=0 python3 inference_generation_coco_eval.py config=configs/phi_6x8_gpus_pretraining_stage_openimages.yaml device_id=0 num_devices=8 &
# CUDA_VISIBLE_DEVICES=1 python3 inference_generation_coco_eval.py config=configs/phi_6x8_gpus_pretraining_stage_openimages.yaml device_id=1 num_devices=8 &
# CUDA_VISIBLE_DEVICES=2 python3 inference_generation_coco_eval.py config=configs/phi_6x8_gpus_pretraining_stage_openimages.yaml device_id=2 num_devices=8 &
# CUDA_VISIBLE_DEVICES=3 python3 inference_generation_coco_eval.py config=configs/phi_6x8_gpus_pretraining_stage_openimages.yaml device_id=3 num_devices=8 &
# CUDA_VISIBLE_DEVICES=4 python3 inference_generation_coco_eval.py config=configs/phi_6x8_gpus_pretraining_stage_openimages.yaml device_id=4 num_devices=8 &
# CUDA_VISIBLE_DEVICES=5 python3 inference_generation_coco_eval.py config=configs/phi_6x8_gpus_pretraining_stage_openimages.yaml device_id=5 num_devices=8 &
# CUDA_VISIBLE_DEVICES=6 python3 inference_generation_coco_eval.py config=configs/phi_6x8_gpus_pretraining_stage_openimages.yaml device_id=6 num_devices=8 &
# CUDA_VISIBLE_DEVICES=7 python3 inference_generation_coco_eval.py config=configs/phi_6x8_gpus_pretraining_stage_openimages.yaml device_id=7 num_devices=8
# python3 coco_evaluations/test.py
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 resnet50.py


CONFIG_FILE="configs/showo_gen_eval.yaml"
OUTPUT_FILE='/mnt/bn/vgfm2/test_dit/weijia/outputs4/geneval_result_whole_coco_diversity_0_8'

# CUDA_VISIBLE_DEVICES="1" torchrun --nproc_per_node=1 --master_port=2599  inference_t2i_mod_coco.py config=${CONFIG_FILE} num_device=0 batch_size=4 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=0 generation_timesteps=16 outdir=${OUTPUT_FILE} 


CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1 --master_port=24599  inference_t2i_mod_coco.py config=${CONFIG_FILE} num_device=0 batch_size=12 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=2 generation_timesteps=16 outdir=${OUTPUT_FILE} &
CUDA_VISIBLE_DEVICES="1" torchrun --nproc_per_node=1 --master_port=24600  inference_t2i_mod_coco.py config=${CONFIG_FILE} num_device=1 batch_size=12 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=2 generation_timesteps=16 outdir=${OUTPUT_FILE} &
CUDA_VISIBLE_DEVICES="2" torchrun --nproc_per_node=1 --master_port=24601  inference_t2i_mod_coco.py config=${CONFIG_FILE} num_device=2 batch_size=12 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=2 generation_timesteps=16 outdir=${OUTPUT_FILE} &
CUDA_VISIBLE_DEVICES="3" torchrun --nproc_per_node=1 --master_port=24602  inference_t2i_mod_coco.py config=${CONFIG_FILE} num_device=3 batch_size=12 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=2 generation_timesteps=16 outdir=${OUTPUT_FILE} &
CUDA_VISIBLE_DEVICES="4" torchrun --nproc_per_node=1 --master_port=24603  inference_t2i_mod_coco.py config=${CONFIG_FILE} num_device=4 batch_size=12 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=2 generation_timesteps=16 outdir=${OUTPUT_FILE} &
CUDA_VISIBLE_DEVICES="5" torchrun --nproc_per_node=1 --master_port=24604  inference_t2i_mod_coco.py config=${CONFIG_FILE} num_device=5 batch_size=12 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=2 generation_timesteps=16 outdir=${OUTPUT_FILE} &
CUDA_VISIBLE_DEVICES="6" torchrun --nproc_per_node=1 --master_port=24605  inference_t2i_mod_coco.py config=${CONFIG_FILE} num_device=6 batch_size=12 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=2 generation_timesteps=16 outdir=${OUTPUT_FILE} &
CUDA_VISIBLE_DEVICES="7" torchrun --nproc_per_node=1 --master_port=24606  inference_t2i_mod_coco.py config=${CONFIG_FILE} num_device=7 batch_size=12 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=2 generation_timesteps=16 outdir=${OUTPUT_FILE} &

