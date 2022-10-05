import os
import argparse
import torch
from tqdm import tqdm

def merge_models(base_model: str,mix_models: str = [],alphas: float = [],output="merged", *,load_all: bool = False,force_overwrite: bool = False): 

    # Distribute undefined alpha weights
    if len(alphas) < len(mix_models):
        remaining = len(mix_models) - len(alphas)
        alpha_div = (1 - sum(alphas))/(remaining + 1)
        alphas = alphas + ([alpha_div] * remaining)

    # Preprocess stepwise alpha weighting
    base_alpha = 1 - sum(alphas)
    step_alphas = []
    for i,alpha in enumerate(alphas):
        step_alphas.append(alpha/(base_alpha + sum(alphas[:i+1])))

    alpha_string = '-'.join(str(round(e,2))[2:].ljust(2, '0') for e in [base_alpha] + alphas)
    output_file = f'{output}-{alpha_string}.ckpt'

    # check if output file already exists, ask to overwrite
    if not force_overwrite and os.path.isfile(output_file):
        print("Output file already exists. Overwrite? (y/n)")
        while True:
            overwrite = input()
            if overwrite == "y":
                break
            elif overwrite == "n":
                print("Exiting...")
                exit()
            else:
                print("Please enter y or n")

    print(f"Creating {output_file}", flush=True)

    print(f"Loading {base_model}", flush=True)
    model_base = torch.load(base_model)
    theta_base = model_base["state_dict"]

    # Load all mixing models at once
    if load_all:
        theta_dict = {}
        for model in mix_models:
            model_load = torch.load(model)
            theta_dict[model] = model_load["state_dict"]

    number_of_steps = len(mix_models)*2

    # Evaluate and mix overlapping keys per model
    for step,(model,alpha) in enumerate(zip(mix_models,step_alphas), start=1):
        print(f'Merging {model}',flush=True)
        if load_all:
            for key in tqdm(theta_base.keys(), desc=f'Stage {2*step-1}/{str(number_of_steps)}'):
                if "model" in key and key in theta_dict[model]:
                    theta_base[key] = (1 - alpha) * theta_base[key] + alpha * theta_dict[model][key]
            for key in tqdm(theta_dict[model].keys(), desc=f'Stage {2*step}/{str(number_of_steps)}'):
                if "model" in key and key not in theta_base:
                    theta_base[key] = theta_dict[model][key]
        else:
            # Overwrite previously loaded models
            model_load = torch.load(model)
            theta_load = model_load["state_dict"]

            for key in tqdm(theta_base.keys(), desc=f'Stage {2*step-1}/{str(number_of_steps)}'):
                if "model" in key and key in theta_load:
                    theta_base[key] = (1 - alpha) * theta_base[key] + alpha * theta_load[key]
            for key in tqdm(theta_load.keys(), desc=f'Stage {2*step}/{str(number_of_steps)}'):
                if "model" in key and key not in theta_base:
                    theta_base[key] = theta_load[key]

    print("Saving...",flush=True)

    torch.save({"state_dict": theta_base}, output_file)

    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge multiple models",add_help=True)
    parser.add_argument("base_model", type=str, help="Path to base model")

    parser.add_argument("-m","--models", type=str, nargs='+', help="Paths to additional input models", required=True)
    parser.add_argument("-a","--alphas", type=float, nargs='+', default=[], help="Array of alpha values, optional, evened weights for unassigned", required=False)
    parser.add_argument("-o","--output", type=str, help="Output file name, without extension", default="merged", required=False)
    parser.add_argument("--load-all", action='store_true', help="Load input models all at the same time (WARNING: RAM HEAVY)")
    parser.add_argument("--force-overwrite", action='store_true', help="Automatically accept the overwrite confirmation")

    args = parser.parse_args()

    merge_models(args.base_model,args.models,args.alphas,args.output,load_all=args.load_all,force_overwrite=args.force_overwrite)