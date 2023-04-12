import argparse
from recbole.quick_start import run_recbole

def main(args):
    
    run_recbole(model=args.model, dataset=args.dataset, config_file_list=["configs/config_{}.yaml".format(args.dataset.lower())])
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="JGCF", help="choose the model to use. ")
    parser.add_argument("--dataset", default="gowalla", help="choose the dataset to use. ")
    args = parser.parse_args()
    
    main(args)
