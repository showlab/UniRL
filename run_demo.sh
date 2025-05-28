
# CONFIG_FILE="configs/showo_gen_eval_cycle.yaml"
# OUTPUT_FILE='/mnt/bn/vgfm2/test_dit/weijia/outputs_0506/hermesflow_mmu_t2i_256'
# FILE_PATH="/mnt/bn/vgfm2/test_dit/weijia/outputs_0506/hermesflow_mmu_t2i_256.jsonl"

CONFIG_FILE="configs/showo_gen_eval_cycle_512.yaml"
OUTPUT_FILE='/mnt/bn/vgfm2/test_dit/weijia/outputs_0508/showo_mmu_t2i_sft_512_test'
FILE_PATH="/mnt/bn/vgfm2/test_dit/weijia/outputs_0508/showo_mmu_t2i_sft_512_test.jsonl"

# CUDA_VISIBLE_DEVICES="0" python3 -m torch.distributed.run --nproc_per_node=1 --master_port=64599 inference_t2i.py config=${CONFIG_FILE}  batch_size=4  guidance_scale=2 generation_timesteps=16 
CUDA_VISIBLE_DEVICES="0" python3 -m torch.distributed.run --nproc_per_node=1 --master_port=64599 inference_mmu.py config=${CONFIG_FILE}  batch_size=4  guidance_scale=2 generation_timesteps=16 