# DeepLocPro
DeepLocPro predicts the subcellular localization of prokaryotic proteins. It can differentiate between 6 different localizations: Cytoplasm, Cytoplasmic membrane, Periplasm, Outer membrane, Cell wall and surface, and extracellular space.

Please cite:

### Predicting the subcellular location of prokaryotic proteins.

Jaime Moreno, Henrik Nielsen, Ole Winther, Felix Teufel.
Biorxiv 2024.01.04.574157; doi: https://doi.org/10.1101/2024.01.04.574157

**Abstract**

Protein subcellular location prediction is a widely explored task in bioinformatics because of its importance in proteomics research. We propose DeepLocPro, an extension to the popular method DeepLoc, tailored specifically to archaeal and bacterial organisms. DeepLocPro is a multiclass subcellular location prediction tool for prokaryotic proteins, trained on experimentally verified data curated from UniProt and PSORTdb. DeepLocPro compares favorably to the PSORTb 3.0 ensemble method, surpassing its performance across multiple metrics on our benchmark experiment.

**Summary DeepLocPro**:

    1. Get data from psort and uniprot
    
    2. Run graph-part for homology partitioning
    
    3. Create embeddings using pre-trained language model
    
    4. Run model finding best hyperparameters
    
    5. Full cross validation (20 models)

    6. Calculate metrics
    
**Scripts**

- Get data

    - ePsort_data

    - Uniprot_data

    - merging_db

    - create_fasta

    - make_embeddings_fsdp_v2: the first version was loading the model in a deterministic random way

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


**Contact information**

Questions on the scientific aspects of the DeepLocPro 1.0 method should go to Henrik Nielsen, hennin@dtu.dk.
