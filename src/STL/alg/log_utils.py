import wandb

from .logger import Logger

def create_logger(run_name, args):
    return Logger(
        metrics=args.metrics,
        out_fname='metrics',
        out_dir=run_name,
        args=args
    )

def init_wandb(run_name, args):
    wandb.init(
        name=run_name,
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        save_code=args.wandb_code,
        config=vars(args)
    )

