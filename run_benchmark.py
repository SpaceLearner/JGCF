import argparse
from logging import getLogger

from recbole.config import Config
from recbole.data import create_dataset
from recbole.data.utils import get_dataloader
from recbole.utils import init_logger, init_seed, get_model, get_trainer, set_color
from recbole.sampler import RepeatableSampler


def get_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="JGCF",
        help="Model for session-based rec.",
    )
    
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="citeulike",
        help="Benchmarks for session-based rec.",
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="Whether evaluating on validation set (split from train set), otherwise on test set.",
    )
    parser.add_argument(
        "--valid_portion", type=float, default=0.05, help="ratio of validation set."
    )
    return parser.parse_known_args()[0]

if __name__ == "__main__":
    
    args = get_args()
    
    config = Config(
        model=args.model, dataset=f"{args.dataset}", config_file_list=["configs/config_pinterest.yaml"]
    )
    init_seed(config["seed"], config["reproducibility"])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(args)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_dataset, test_dataset = dataset.build()
    if args.validation:
        train_dataset.shuffle()
        new_train_dataset, new_test_dataset = train_dataset.split_by_ratio(
            [1 - args.valid_portion, args.valid_portion]
        )
        train_data = get_dataloader(config, "train")(
            config, new_train_dataset, sampler=RepeatableSampler(None, new_train_dataset), shuffle=True
        )
        test_data = get_dataloader(config, "test")(
            config, new_test_dataset, sampler=RepeatableSampler(None, new_test_dataset), shuffle=False
        )
    else:
        train_data = get_dataloader(config, "train")(
            config, train_dataset, sampler=RepeatableSampler(None, train_dataset), shuffle=True
        )
        test_data = get_dataloader(config, "test")(
            config, test_dataset, sampler=RepeatableSampler(None, test_dataset), shuffle=False
        )
        
    model = get_model(config["model"])(config, train_data.dataset).to(config["device"])
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    # model training and evaluation
    test_score, test_result = trainer.fit(
        train_data, test_data, saved=True, show_progress=config["show_progress"]
    )

    logger.info(set_color("test result", "yellow") + f": {test_result}")