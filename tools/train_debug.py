"""
Main Training Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""
from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.train import TRAINERS
from pointcept.engines.launch import launch
import torch
from pointcept.utils.slconfig import DictAction, SLConfig

def main_worker(cfg):
    cfg = default_setup(cfg)
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()


def main():
    args = default_argument_parser().parse_args()
    args.config_file = '/nas2/YJ/git/Fusion/configs/nmg/ptv3_nmg_base.py'
    args.num_gpus = 1
    args.options = {'save_path' : '/nas2/YJ/git/Fusion/exp/debug'}

    ########3 To Del
    torch.autograd.set_detect_anomaly(True)
    ############

    cfg = default_config_parser(args.config_file, args.options)
    cfg.num_worker = 0

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()
