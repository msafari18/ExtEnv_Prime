import argparse
from clustering_new import run_models

ENVS = ["Temperature", "pH"]
NUM_CLUSTERS = {"Temperature": 4, "pH": 2}
# PATH = "/home/m4safari/projects/def-lila-ab/m4safari/ext1prime/data"
PATH = "/content/drive/MyDrive/anew"


def run_pipeline(args):
    fragement_length = args["fragement_length"]
    k = args["k-mer"]
    if args["exp_type"] == "parametric":
        pass

    elif args["exp_type"] == "non-parametric":
        for env in ENVS:
            run_models(env, PATH, fragement_length, k)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_type', action='store', type=str)
    parser.add_argument('--k-mer', action='store', type=int)
    parser.add_argument('--fragement_length', action='store_true')
    args = vars(parser.parse_args())

    run_pipeline(args)

if __name__ == '__main__':
    main()