import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import os

from Build_Signature_Dataset_v2 import run_fragment_builder
from SupervisedModels import run_supervised_classification

from SupervisedModels_Challenging import run_supervised_classification_challenging

ENVS = ["Temperature", "pH"]
NUM_CLUSTERS = {"Temperature": 4,
                "pH": 2}
FRAGMENT_LENGTHS = [10000, 50000, 100000, 250000, 500000, 1000000]
PATH = "/home/m4safari/projects/def-lila-ab/m4safari/ext1prime/data"
# PATH = "/content/drive/MyDrive/anew"
NUM_EXP = 10


def run_pipeline(args):
    # running 10 times to check if the signature is pervasive

    if args["exp_type"] == "exp1":
        classifiers = {
            "SVM": (SVC, {'kernel': 'rbf', 'class_weight': 'balanced', 'C': 10})}
        for env in ENVS:

            for exp in range(NUM_EXP):
                for fragment_length in FRAGMENT_LENGTHS:
                    print("\n Running the pipeline is started:")
                    # building the fragments
                    fragment_file = f"{args['exp_type']}/{exp}/fragments_{fragment_length}"
                    print(f"\n Building fragment with length {fragment_length} is started.")
                    run_fragment_builder(PATH, fragment_file, fragment_length, args['whole_genome'], env)
                    print(f"\n Fragment with length {fragment_length} has been created.")

                    # run the supervised classification under the first scenario (not challenging)
                    result_folder = f"{PATH}/{args['exp_type']}/{exp}/fragments_{fragment_length}"
                    fasta_file = os.path.join(result_folder, env, f'Extremophiles_{env}.fas')
                    print(f"\n Classification is started (scenario 1).")
                    run_supervised_classification(fasta_file, args.max_k, result_folder, env, exp, classifiers)
                    print(f"\n Classification ended (scenario 1).")

                    # run the supervised classification under the 2nd scenario (challenging)
                    print(f"\n Classification is started (scenario 1).")
                    run_supervised_classification_challenging(fasta_file, args.max_k, result_folder, env, exp,
                                                              classifiers)
                    print(f"\n Classification ended (scenario 2).")

    # Trying different k and different length to find the optimal one with different models
    elif args.exp_type == "exp2":

        classifiers = {
            "SVM": (SVC, {'kernel': 'rbf', 'class_weight': 'balanced', 'C': 10}),
            "Random Forest": (RandomForestClassifier, {}),
            "ANN": (MLPClassifier, {'hidden_layer_sizes': (256, 64), 'solver': 'adam',
                                    'activation': 'relu', 'alpha': 1, 'learning_rate_init': 0.001,
                                    'max_iter': 300, 'n_iter_no_change': 10})
        }

        for env in ENVS:

            exp = args['exp_type']
            for fragment_length in FRAGMENT_LENGTHS:
                # building the fragments
                fragment_file = f"{args['exp_type']}/{exp}/fragments_{fragment_length}"
                run_fragment_builder(PATH, fragment_file, fragment_length, args['whole_genome'], env)
                print(f"\n Fragment with length {fragment_length} has been created.")

                # run the supervised classification under the first scenario (not challenging)
                result_folder = f"{PATH}/{args['exp_type']}/{exp}/fragments_{fragment_length}"
                fasta_file = os.path.join(result_folder, env, f'Extremophiles_{env}.fas')
                print(f"\n Classification is started (scenario 1).")
                run_supervised_classification(fasta_file, args.max_k, result_folder, env, exp, classifiers)
                print(f"\n Classification ended (scenario 1).")

                # run the supervised classification under the 2nd scenario (challenging)
                print(f"\n Classification is started (scenario 2).")
                run_supervised_classification_challenging(fasta_file, args.max_k, result_folder, env, exp, classifiers)
                print(f"\n Classification ended (scenario 2).")


def main():
    parser = argparse.ArgumentParser()

    ######## experiment args
    parser.add_argument('--exp_type', action='store', type=str)
    ######## supervised/supervised-challenging args
    parser.add_argument('--max_k', action='store', type=int)
    ######## supervised/supervised-challenging args
    parser.add_argument('--whole_genome', action='store_true')
    args = vars(parser.parse_args())

    run_pipeline(args)


if __name__ == '__main__':
    main()