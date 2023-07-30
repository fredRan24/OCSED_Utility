To use OCSED_util you must first create a virtual environment with the appropriate packages installes.
1. Follow the steps here to create an environment and load the requirements.txt: 
https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/

OCSED_util is a simple python class which you can implement to:
 1. Preprocess data into binary matrix format for one-class SED purposes.
 2. Train models
 3. Evaluate models using the DESED evaluation set.

The DESED dataset is not included here, so you need to download it from here: 
1. https://project.inria.fr/desed/
2. https://zenodo.org/record/3550599#.ZECNxnbMKHs	

You should try to achieve the follwing file structure to work with OCSED_util as is:

.
└── BARKSED/
    ├── DESED/
    │   └── dataset/
    │       ├── audio/
    │       │   ├── eval/
    │       │   │   └── public/
    │       │   │       ├── 1.wav
    │       │   │       ├── 2.wav
    │       │   │       └── ...
    │       │   ├── train/
    │       │   │   └── synthetic21_train/
    │       │   │       └── soundscapes/
    │       │   │           ├── 1.wav
    │       │   │           ├── 2.wav
    │       │   │           └── ...
    │       │   └── validation/
    │       │       └── synthetic21_validation/
    │       │           └── soundscapes/
    │       │               ├── 1.wav
    │       │               ├── 2.wav
    │       │               └── ...
    │       └── metadata/
    │           ├── eval/
    │           │   └── public.tsv
    │           ├── train/
    │           │   └── synthetic21_train/
    │           │       └── soundscapes.tsv
    │           └── validation/
    │               └── synthetic21_validation/
    │                   └── soundscapes.tsv
    ...


Please see example.ipynb for examples of how to load, save, preprocess, train and evaluate your model/data