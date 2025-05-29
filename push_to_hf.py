# """
# Script which can be used to push and reload the VQ and Show-o model to the Hugging Face hub.
# """

# import torch
# from models import Showo, MAGViTv2
# from training.utils import get_config


# def get_vq_model_class(model_type):
#     if model_type == "magvitv2":
#         return MAGViTv2
#     else:
#         raise ValueError(f"model_type {model_type} not supported.")

# if __name__ == '__main__':

#     config = get_config()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # load VQ model
#     # vq_model = get_vq_model_class(config.model.vq_model.type)().to(device)
#     # vq_model.load_state_dict(torch.load(config.model.vq_model.pretrained))
#     # vq_model.requires_grad_(False)
#     # vq_model.eval()
#     #
#     # # push VQ to the hub
#     vq_model_name = "showlab/magvitv2"
#     # vq_model.push_to_hub(vq_model_name)

#     # verify reloading
#     vq_model = MAGViTv2.from_pretrained(vq_model_name)

#     # load Show-o model
#     # model = Showo(**config.model.showo).to(device)
#     # state_dict = torch.load("./checkpoints/showo.bin")
#     # model.load_state_dict(state_dict, strict=True)
#     # model.eval()

#     # push Show-o to the hub
#     showo_model_name = "showlab/show-o"
#     # model.push_to_hub(showo_model_name)

#     # verify reloading
#     model = Showo.from_pretrained(showo_model_name)

#     showo_model_name = "showlab/show-o-w-clip-vit"
#     # model.push_to_hub(showo_model_name)

#     # verify reloading
#     model = Showo.from_pretrained(showo_model_name)
#     import ipdb
#     ipdb.set_trace()
#     print()

# # python3 push_to_hf.py config=configs/showo_demo.yaml --pretrained_model_path=./checkpoints/showo.bin


"""
# Script which can be used to push and reload the VQ and Show-o model to the Hugging Face hub.
# """

import torch
from models import Showo
from training.utils import get_config

if __name__ == '__main__':

    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load Show-o model
    model = Showo(**config.model.showo).to(device)
    state_dict = torch.load("/mnt/bn/vgfm2/test_dit/weijia/outputs_0513/show-o-training-journey_drpo_beta_0_2_bs6_all_kinds_beta_0_2_512_no_ckpt_again/checkpoint-3001/unwrapped_model/pytorch_model.bin")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    import pdb; pdb.set_trace()

    # push Show-o to the hub
    showo_model_name = "benzweijia/UniRL"
    model.push_to_hub(showo_model_name)
    # import pdb; pdb.set_trace()

    # verify reloading
    model = Showo.from_pretrained(showo_model_name)

    showo_model_name = "showlab/show-o-w-clip-vit"
    # model.push_to_hub(showo_model_name)

    # verify reloading
    model = Showo.from_pretrained(showo_model_name)
    import ipdb
    ipdb.set_trace()
    print()

# python3 push_to_hf.py config=configs/showo_demo.yaml --pretrained_model_path=./checkpoints/showo.bin