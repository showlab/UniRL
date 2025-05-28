

CONFIG_FILE="configs/showo_gen_eval_cycle_512.yaml"
OUTPUT_FILE='/mnt/bn/vgfm2/test_dit/weijia/outputs_0514/showo_mmu_t2i_original_512'
FILE_PATH="/mnt/bn/vgfm2/test_dit/weijia/outputs_0514/showo_mmu_t2i_original_512.jsonl"


# CUDA_VISIBLE_DEVICES="0" python3 -m torch.distributed.run --nproc_per_node=1 --master_port=34599 inference_mmu_t2i.py config=${CONFIG_FILE} num_device=0 batch_size=1 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=5 generation_timesteps=50 outdir=${OUTPUT_FILE} file_path=${FILE_PATH}&
# CUDA_VISIBLE_DEVICES="1" python3 -m torch.distributed.run --nproc_per_node=1 --master_port=35599 inference_mmu_t2i.py config=${CONFIG_FILE} num_device=1 batch_size=1 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=5 generation_timesteps=50 outdir=${OUTPUT_FILE} file_path=${FILE_PATH}&
# CUDA_VISIBLE_DEVICES="2" python3 -m torch.distributed.run --nproc_per_node=1 --master_port=46599 inference_mmu_t2i.py config=${CONFIG_FILE} num_device=2 batch_size=1 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=5 generation_timesteps=50 outdir=${OUTPUT_FILE} file_path=${FILE_PATH}&
# CUDA_VISIBLE_DEVICES="3" python3 -m torch.distributed.run --nproc_per_node=1 --master_port=21599 inference_mmu_t2i.py config=${CONFIG_FILE} num_device=3 batch_size=1 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=5 generation_timesteps=50 outdir=${OUTPUT_FILE} file_path=${FILE_PATH}&
# CUDA_VISIBLE_DEVICES="4" python3 -m torch.distributed.run --nproc_per_node=1 --master_port=18599 inference_mmu_t2i.py config=${CONFIG_FILE} num_device=4 batch_size=1 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=5 generation_timesteps=50 outdir=${OUTPUT_FILE} file_path=${FILE_PATH}&
# CUDA_VISIBLE_DEVICES="5" python3 -m torch.distributed.run --nproc_per_node=1 --master_port=39599 inference_mmu_t2i.py config=${CONFIG_FILE} num_device=5 batch_size=1 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=5 generation_timesteps=50 outdir=${OUTPUT_FILE} file_path=${FILE_PATH}&
# CUDA_VISIBLE_DEVICES="6" python3 -m torch.distributed.run --nproc_per_node=1 --master_port=40599 inference_mmu_t2i.py config=${CONFIG_FILE} num_device=6 batch_size=1 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=5 generation_timesteps=50 outdir=${OUTPUT_FILE} file_path=${FILE_PATH}&
# CUDA_VISIBLE_DEVICES="7" python3 -m torch.distributed.run --nproc_per_node=1 --master_port=61599 inference_mmu_t2i.py config=${CONFIG_FILE} num_device=7 batch_size=1 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=5 generation_timesteps=50 outdir=${OUTPUT_FILE} file_path=${FILE_PATH}


CUDA_VISIBLE_DEVICES="0" python3 -m torch.distributed.run --nproc_per_node=1 --master_port=34599 inference_mmu_t2i.py config=${CONFIG_FILE} num_device=0 batch_size=1 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=2 generation_timesteps=18 outdir=${OUTPUT_FILE} file_path=${FILE_PATH}&
CUDA_VISIBLE_DEVICES="1" python3 -m torch.distributed.run --nproc_per_node=1 --master_port=35599 inference_mmu_t2i.py config=${CONFIG_FILE} num_device=1 batch_size=1 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=2 generation_timesteps=18 outdir=${OUTPUT_FILE} file_path=${FILE_PATH}&
CUDA_VISIBLE_DEVICES="2" python3 -m torch.distributed.run --nproc_per_node=1 --master_port=46599 inference_mmu_t2i.py config=${CONFIG_FILE} num_device=2 batch_size=1 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=2 generation_timesteps=18 outdir=${OUTPUT_FILE} file_path=${FILE_PATH}&
CUDA_VISIBLE_DEVICES="3" python3 -m torch.distributed.run --nproc_per_node=1 --master_port=21599 inference_mmu_t2i.py config=${CONFIG_FILE} num_device=3 batch_size=1 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=2 generation_timesteps=18 outdir=${OUTPUT_FILE} file_path=${FILE_PATH}&
CUDA_VISIBLE_DEVICES="4" python3 -m torch.distributed.run --nproc_per_node=1 --master_port=18599 inference_mmu_t2i.py config=${CONFIG_FILE} num_device=4 batch_size=1 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=2 generation_timesteps=18 outdir=${OUTPUT_FILE} file_path=${FILE_PATH}&
CUDA_VISIBLE_DEVICES="5" python3 -m torch.distributed.run --nproc_per_node=1 --master_port=39599 inference_mmu_t2i.py config=${CONFIG_FILE} num_device=5 batch_size=1 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=2 generation_timesteps=18 outdir=${OUTPUT_FILE} file_path=${FILE_PATH}&
CUDA_VISIBLE_DEVICES="6" python3 -m torch.distributed.run --nproc_per_node=1 --master_port=40599 inference_mmu_t2i.py config=${CONFIG_FILE} num_device=6 batch_size=1 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=2 generation_timesteps=18 outdir=${OUTPUT_FILE} file_path=${FILE_PATH}&
CUDA_VISIBLE_DEVICES="7" python3 -m torch.distributed.run --nproc_per_node=1 --master_port=61599 inference_mmu_t2i.py config=${CONFIG_FILE} num_device=7 batch_size=1 validation_prompts_file=validation_prompts/evaluation_metadata.jsonl guidance_scale=2 generation_timesteps=18 outdir=${OUTPUT_FILE} file_path=${FILE_PATH}


python3 test2.py
