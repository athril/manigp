# manigp
Manifold Learning with GP

## How to run the software
The command needs to contain the word Classifer in ML method.

python evaluate_model.py -ml NameClassifier /source/to/dataset -save_file /source/to/result/file -seed number


An example launching command looks as follows:
python evaluate_model.py -ml GPClassifier datasets.txt -save_file results.txt -seed 1393

The hyperparameters for each classifier are located in methods/NameOfClassifier.py. The file requires definition of hyper_params, as well as initialization of a classifier as est.

The tests were run on the datasets from PMLB repository: https://github.com/EpistasisLab/penn-ml-benchmarks/