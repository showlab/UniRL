
sudo chmod a+wr /mnt/bn/vgfm2/test_dit/weijia/outputs
sudo chmod a+wr /mnt/bn/vgfm2/test_dit/weijia/outputs/wandb
sudo chmod a+wr /mnt/bn/vgfm2/test_dit/MoE-LLaVA/train_image_video
sudo chmod a+wr /mnt/bn/vgfm2/test_dit/weijia/outputs2
sudo chmod a+wr /mnt/bn/vgfm2/test_dit/weijia/code_2stage_mod/
sudo chmod a+wr /mnt/bn/vgfm2/test_dit/weijia/code_2stage_mod/cruise_data.yaml

# accelerate launch --config_file accelerate_configs/1_gpu.yaml --main_process_port=8888 training/train_moe.py config=configs/showo_pretraining_stage1.yaml
# accelerate launch --config_file accelerate_configs/1_gpu_deepspeed_zero2.yaml --main_process_port=8888 training/train_moe.py config=configs/showo_pretraining_stage1.yaml


# accelerate launch --config_file accelerate_configs/8_gpus_deepspeed_zero2.yaml --main_process_port=8888 training/train_moe.py config=configs/showo_instruction_tuning_1.yaml

# accelerate launch --config_file accelerate_configs/8_gpus_deepspeed_zero2.yaml --main_process_port=8888 training/train.py config=configs/showo_instruction_tuning_2.yaml


# accelerate launch --config_file accelerate_configs/8_gpus_deepspeed_zero2.yaml --main_process_port=8888 training/train_moe.py config=configs/showo_instruction_tuning_2.yaml



# accelerate launch --config_file "$CONFIG_FILLED" --main_process_port=8889  training/train_mod.py config=configs/showo_pretraining_stage2_w_t2i_parquet.yaml

# accelerate launch --config_file accelerate_configs/1_gpu.yaml --main_process_port=8888 training/train_mod.py config=configs/showo_pretraining_stage2_w_t2i_parquet.yaml


# accelerate launch --config_file accelerate_configs/1_gpu.yaml --main_process_port=8888 training/train_mod_aux.py config=configs/showo_pretraining_stage3_mod_aux.yaml

# accelerate launch --config_file  accelerate_configs/1_gpu.yaml --main_process_port=8889  training/train_mod.py config=configs/showo_pretraining_stage3_mod_512.yaml

# 
# accelerate launch --config_file accelerate_configs/8_gpus_deepspeed_zero2.yaml --main_process_port=8888 training/train_mod.py config=configs/showo_pretraining_stage3_mod.yaml


# accelerate launch --config_file accelerate_configs/1_gpu.yaml --main_process_port=8888 training/train_mod_split.py config=configs/showo_pretraining_stage3_mod_512.yaml

# accelerate launch --config_file accelerate_configs/8_gpus_deepspeed_zero2.yaml --main_process_port=8889 training/train_cycle.py config=configs/showo_cycle.yaml


# accelerate launch --config_file accelerate_configs/8_gpus_deepspeed_zero2.yaml --main_process_port=8888 training/train_cycle_verify.py config=configs/showo_cycle.yaml

# accelerate launch --config_file accelerate_configs/8_gpus_deepspeed_zero2.yaml --main_process_port=8888 training/train_evolve.py config=configs/showo_cycle.yaml

# accelerate launch --config_file accelerate_configs/8_gpus_deepspeed_zero2.yaml --main_process_port=8888 training/train_evolve_lora.py config=configs/showo_cycle.yaml

# accelerate launch --config_file accelerate_configs/8_gpus_deepspeed_zero2.yaml --main_process_port=8888 training/train_counting.py config=configs/showo_cycle.yaml

# accelerate launch --config_file accelerate_configs/1_gpu_deepspeed_zero2.yaml --main_process_port=8888 training/train_all_types.py config=configs/showo_cycle.yaml

# accelerate launch --config_file accelerate_configs/8_gpus_deepspeed_zero2.yaml --main_process_port=8888 training/train_all_types.py config=configs/showo_cycle_512.yaml

# accelerate launch --config_file accelerate_configs/8_gpus_deepspeed_zero2.yaml --main_process_port=8888 training/train_all_types_lora.py config=configs/showo_cycle.yaml


# accelerate launch --config_file accelerate_configs/8_gpus_deepspeed_zero2.yaml --main_process_port=8889 training/train_all_types_drpo.py config=configs/showo_cycle_drpo.yaml

# accelerate launch --config_file accelerate_configs/8_gpus_deepspeed_zero2.yaml --main_process_port=8889 training/train_all_types_drpo_counting_old.py config=configs/showo_cycle_drpo.yaml

# accelerate launch --config_file accelerate_configs/8_gpus_deepspeed_zero2.yaml --main_process_port=8889 training/train_all_types_drpo_counting.py config=configs/showo_cycle_drpo.yaml


# accelerate launch --config_file accelerate_configs/8_gpus_deepspeed_zero2.yaml --main_process_port=8889 training/train_all_types_drpo_multi.py config=configs/showo_cycle_drpo.yaml

# accelerate launch --config_file accelerate_configs/8_gpus_deepspeed_zero2.yaml --main_process_port=8889 training/train_all_types_drpo_multi_sample.py config=configs/showo_cycle_drpo.yaml


# accelerate launch --config_file accelerate_configs/8_gpus_deepspeed_zero2.yaml --main_process_port=8889 training/train_all_types_drpo_multi.py config=configs/showo_cycle_drpo_512.yaml

# accelerate launch --config_file accelerate_configs/8_gpus_deepspeed_zero2.yaml --main_process_port=8889 training/train_all_types_drpo_multi_sample.py config=configs/showo_cycle_drpo_512.yaml


# accelerate launch --config_file accelerate_configs/8_gpus_deepspeed_zero2.yaml --main_process_port=8889 training/train_all_types_drpo_position.py config=configs/showo_cycle_drpo.yaml

# accelerate launch --config_file accelerate_configs/8_gpus_deepspeed_zero2.yaml --main_process_port=8889 training/train_all_types_drpo_mmu.py config=configs/showo_cycle_drpo.yaml


# accelerate launch --config_file accelerate_configs/1_gpu_deepspeed_zero2.yaml --main_process_port=8889 training/train_check_reward.py config=configs/showo_cycle_drpo.yaml


accelerate launch --config_file accelerate_configs/8_gpus_deepspeed_zero2.yaml --main_process_port=8889 training/train_all_types_drpo_mmu_t2i.py config=configs/showo_cycle_drpo_512.yaml

# accelerate launch --config_file accelerate_configs/8_gpus_deepspeed_zero2.yaml --main_process_port=8889 training/train_all_types_drpo_t2i.py config=configs/showo_cycle_drpo.yaml


# accelerate launch --config_file accelerate_configs/8_gpus_deepspeed_zero2.yaml --main_process_port=8889 training/train_all_types_drpo_t2i_reward.py config=configs/showo_cycle_drpo.yaml


# accelerate launch --config_file accelerate_configs/8_gpus_deepspeed_zero2.yaml --main_process_port=8888 training/train_all_types_vit.py config=configs/showo_cycle_vit.yaml

# accelerate launch --config_file accelerate_configs/8_gpus_deepspeed_zero2.yaml --main_process_port=8888 training/train_all_types_vit_grpo_t2i.py config=configs/showo_cycle_vit.yaml


# accelerate launch --config_file accelerate_configs/8_gpus_deepspeed_zero2.yaml --main_process_port=8888 training/train_all_types_sample.py config=configs/showo_cycle.yaml

python3 test2.py



