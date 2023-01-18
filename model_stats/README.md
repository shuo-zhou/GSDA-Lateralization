# Statistical analysis on 1st and 2nd-order coefficients

Each `.mat` file contains the mean and standard deviation of coefficients for a group (1000) of model.

**Naming:** `L` denotes the $\lambda$, and `G` denotes the subjects' gender (0 for male and 1 for female) of the subset used for training.

## First-order models

The models were trained to classify right (positive class) and left brain (negative class).

## Second-order models

The models were trained to classify two group of models. 

**Example**: "L2G0_vs_L2G1.mat"

- mean and standard deviation of models (logistic regression)to classify models trained respectively on males and females, both with $\lambda=2$
- "L2G0" was labelled as class 1 and "L2G1" was labelled as class 0
