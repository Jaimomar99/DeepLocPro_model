# DeepLocPro
DeepLocPro predicts the subcellular localization of prokaryotic proteins. It can differentiate between 6 different localizations: Cytoplasm, Cytoplasmic membrane, Periplasm, Outer membrane, Cell wall and surface, and extracellular space.

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
**This repository contains the Python implementation and training code for the DeepLocPro model.** If you are looking for the prediction service to process protein sequences, please go to https://services.healthtech.dtu.dk/services/DeepLocPro-1.0/ or https://biolib.com/KU/DeepLocPro/.

Please cite:

### Predicting the subcellular location of prokaryotic proteins.

Jaime Moreno, Henrik Nielsen, Ole Winther, Felix Teufel.
Biorxiv 2024.01.04.574157; doi: https://doi.org/10.1101/2024.01.04.574157

**Abstract**

Protein subcellular location prediction is a widely explored task in bioinformatics because of its importance in proteomics research. We propose DeepLocPro, an extension to the popular method DeepLoc, tailored specifically to archaeal and bacterial organisms. DeepLocPro is a multiclass subcellular location prediction tool for prokaryotic proteins, trained on experimentally verified data curated from UniProt and PSORTdb. DeepLocPro compares favorably to the PSORTb 3.0 ensemble method, surpassing its performance across multiple metrics on our benchmark experiment.

**Summary DeepLocPro**:

    1. Get data from psort and uniprot
    
    2. Run graph-part for homology partitioning: [Github](https://github.com/graph-part/graph-part)
    
    3. Create embeddings using pre-trained language model (esm)
    
    4. Run model finding best hyperparameters
    
    5. Full cross validation (20 models)

    6. Calculate metrics
    
**Scripts**

- Get data

    - ePsort_data

    - Uniprot_data

    - merging_db

    - create_fasta

    - make_embeddings_fsdp_v2

    - compute_dataset

- Run model

    - single_model

    - Training_validation_cross

    - Test_cross

- Evaluate performance

    - get_final_fasta

    - Metrics

- Helping functions
    - single_useful_functions

**Data**
- example.fasta: Example fasta sequences
- graph_part_example.csv: Example output of graph_part
- example_data.csv: Example input to the model
- embeddings: embeddings of the example fasta data

**env file**
* DeepLocPro_model.yaml : packages needed


**Pipeline**

To run the training, you will need:

    1. Clone the repository
    2. Create conda env by: conda create env -f DeepLocPro_model.yaml
    3. Generate fasta file
    4. Create a csv file with the same format as data/example_data/csv (use bin/merging_db.ipnyb as guide)
    5. Output of graph_part in a csv format
    6. Run ./bin/main.sh to:
        - Generate embeddings
        - Train & validate the model
        - Calculate metrics
data/example will only show the format need, the pipeline will not run b/c too few sequences.

