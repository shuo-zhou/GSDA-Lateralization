# Statistical analysis on 1st and 2nd-order model weights and univariate test

## Processing first-order model weights

First-order models were trained to classify right (labelled as 1) and left (labelled as 0) brain hemispheres.

`process_first_order_weights.py` processes first-order model weight (.pt) files by:

1. Loading weight vectors learned from multiple random splits (seeds).
2. Combining all weights into a single .npz file for archival and future use.
3. Computing the mean and standard deviation across splits.
4. Saving the aggregated results as a .mat file.

Sample first-order (.pt) and second-order (.joblib) model files are available on Hugging Face Model Hub: [https://huggingface.co/shuo-zhou/brain_network_lateralization](https://huggingface.co/shuo-zhou/brain_network_lateralization), DOI: [10.57967/hf/2996](https://doi.org/10.57967/hf/2996).

Each `.mat` file contains the mean and standard deviation of coefficients for a group (1000) of model.

**Naming:** `L` denotes the $\lambda$, and `G` denotes the subjects' gender (0 for male and 1 for female) of the subset used for training.

## Processing second-order model weights

Second-order models were trained to classify two group of first-order models (male-specific vs female-specific).

`process_second_order_weights.py` processes second-order model weight (.joblib) files by:

1. Loading weight vectors learned from multiple random splits (seeds).
2. Computing the mean and standard deviation across splits.
3. Saving the aggregated results as a .mat file.

**Example output**: "L2G0_vs_L2G1.mat"

- mean and standard deviation of models (logistic regression)to classify models trained respectively on males and females, both with $\lambda=2$
- "L2G0" was labelled as class 1 and "L2G1" was labelled as class 0

The output figures are available in the [figures](./figures) folder.

## Univariate test and analysis

- Plot the connectivity with significant weights as a chord diagram
- Plot radar diagrams to represent the intra- and inter-lobe statistics of connections with significant weights

**Input files:**

Sample first-order (.pt) and second-order (.joblib) model files, along with statistical analysis results (.mat), are available on Hugging Face Model Hub: [https://huggingface.co/shuo-zhou/brain_network_lateralization](https://huggingface.co/shuo-zhou/brain_network_lateralization), DOI: [10.57967/hf/2996](https://doi.org/10.57967/hf/2996).

The Brainnetome atlas (BNA) is available at [http://atlas.brainnetome.org/](http://atlas.brainnetome.org/). [BNA_subregions.xlsx](BNA_subregions.xlsx) for the information about the lobes and gyrus is provided by the author and originally available at [https://pan.cstcloud.cn/web/share.html?hash=6eRCJ0zDTFk](https://pan.cstcloud.cn/web/share.html?hash=6eRCJ0zDTFk).

**Output files:**

The output figures are available in the [figures](./figures) folder.

Author: [Shuo Zhou](https://github.com/shuo-zhou) and [Junhao Luo](https://github.com/junhaols)

Contact: <shuo.zhou@sheffield.ac.uk> adn <junhaol@mail.bnu.edu.cn>
