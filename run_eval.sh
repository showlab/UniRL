


CONFIG_FILE="configs/showo_gen_eval_cycle_512.yaml"
OUTPUT_FILE='/mnt/bn/vgfm2/test_dit/weijia/outputs_0514/output_grpo_mmu_t2i_no_ckpt'
FILE_PATH="/mnt/bn/vgfm2/test_dit/weijia/outputs_0514/output_grpo_mmu_t2i_no_ckpt.jsonl"


CUDA_VISIBLE_DEVICES="0" python3 -m torch.distributed.run --nproc_per_node=1 --master_port=14599 inference_cycle.py config=${CONFIG_FILE} num_device=0 batch_size=1 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=5 generation_timesteps=50 outdir=${OUTPUT_FILE} file_path=${FILE_PATH}

# CUDA_VISIBLE_DEVICES="0" python3 -m torch.distributed.run --nproc_per_node=1 --master_port=14599 inference_cycle.py config=${CONFIG_FILE} num_device=0 batch_size=1 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=5 generation_timesteps=50 outdir=${OUTPUT_FILE} file_path=${FILE_PATH}&
# CUDA_VISIBLE_DEVICES="1" python3 -m torch.distributed.run --nproc_per_node=1 --master_port=14600 inference_cycle.py config=${CONFIG_FILE} num_device=1 batch_size=1 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=5 generation_timesteps=50 outdir=${OUTPUT_FILE} file_path=${FILE_PATH}&
# CUDA_VISIBLE_DEVICES="2" python3 -m torch.distributed.run --nproc_per_node=1 --master_port=14601 inference_cycle.py config=${CONFIG_FILE} num_device=2 batch_size=1 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=5 generation_timesteps=50 outdir=${OUTPUT_FILE} file_path=${FILE_PATH}&
# CUDA_VISIBLE_DEVICES="3" python3 -m torch.distributed.run --nproc_per_node=1 --master_port=14602 inference_cycle.py config=${CONFIG_FILE} num_device=3 batch_size=1 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=5 generation_timesteps=50 outdir=${OUTPUT_FILE} file_path=${FILE_PATH}&
# CUDA_VISIBLE_DEVICES="4" python3 -m torch.distributed.run --nproc_per_node=1 --master_port=14603 inference_cycle.py config=${CONFIG_FILE} num_device=4 batch_size=1 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=5 generation_timesteps=50 outdir=${OUTPUT_FILE} file_path=${FILE_PATH}&
# CUDA_VISIBLE_DEVICES="5" python3 -m torch.distributed.run --nproc_per_node=1 --master_port=14604 inference_cycle.py config=${CONFIG_FILE} num_device=5 batch_size=1 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=5 generation_timesteps=50 outdir=${OUTPUT_FILE} file_path=${FILE_PATH}&
# CUDA_VISIBLE_DEVICES="6" python3 -m torch.distributed.run --nproc_per_node=1 --master_port=14605 inference_cycle.py config=${CONFIG_FILE} num_device=6 batch_size=1 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=5 generation_timesteps=50 outdir=${OUTPUT_FILE} file_path=${FILE_PATH}&
# CUDA_VISIBLE_DEVICES="7" python3 -m torch.distributed.run --nproc_per_node=1 --master_port=14606 inference_cycle.py config=${CONFIG_FILE} num_device=7 batch_size=1 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=5 generation_timesteps=50 outdir=${OUTPUT_FILE} file_path=${FILE_PATH}



# CUDA_VISIBLE_DEVICES="0" python3 -m torch.distributed.run --nproc_per_node=1 --master_port=14599 inference_cycle.py config=${CONFIG_FILE} num_device=0 batch_size=1 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=2 generation_timesteps=16 outdir=${OUTPUT_FILE} file_path=${FILE_PATH}&
# CUDA_VISIBLE_DEVICES="1" python3 -m torch.distributed.run --nproc_per_node=1 --master_port=14600 inference_cycle.py config=${CONFIG_FILE} num_device=1 batch_size=1 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=2 generation_timesteps=16 outdir=${OUTPUT_FILE} file_path=${FILE_PATH}&
# CUDA_VISIBLE_DEVICES="2" python3 -m torch.distributed.run --nproc_per_node=1 --master_port=14601 inference_cycle.py config=${CONFIG_FILE} num_device=2 batch_size=1 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=2 generation_timesteps=16 outdir=${OUTPUT_FILE} file_path=${FILE_PATH}&
# CUDA_VISIBLE_DEVICES="3" python3 -m torch.distributed.run --nproc_per_node=1 --master_port=14602 inference_cycle.py config=${CONFIG_FILE} num_device=3 batch_size=1 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=2 generation_timesteps=16 outdir=${OUTPUT_FILE} file_path=${FILE_PATH}&
# CUDA_VISIBLE_DEVICES="4" python3 -m torch.distributed.run --nproc_per_node=1 --master_port=14603 inference_cycle.py config=${CONFIG_FILE} num_device=4 batch_size=1 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=2 generation_timesteps=16 outdir=${OUTPUT_FILE} file_path=${FILE_PATH}&
# CUDA_VISIBLE_DEVICES="5" python3 -m torch.distributed.run --nproc_per_node=1 --master_port=14604 inference_cycle.py config=${CONFIG_FILE} num_device=5 batch_size=1 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=2 generation_timesteps=16 outdir=${OUTPUT_FILE} file_path=${FILE_PATH}&
# CUDA_VISIBLE_DEVICES="6" python3 -m torch.distributed.run --nproc_per_node=1 --master_port=14605 inference_cycle.py config=${CONFIG_FILE} num_device=6 batch_size=1 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=2 generation_timesteps=16 outdir=${OUTPUT_FILE} file_path=${FILE_PATH}&
# CUDA_VISIBLE_DEVICES="7" python3 -m torch.distributed.run --nproc_per_node=1 --master_port=14606 inference_cycle.py config=${CONFIG_FILE} num_device=7 batch_size=1 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=2 generation_timesteps=16 outdir=${OUTPUT_FILE} file_path=${FILE_PATH}

