# Analysis of 1st, 2nd-order, and univariate test models and results with visualisation

Sample first-order (.pt) and second-order (.joblib) model files are available on Hugging Face Model Hub: [https://huggingface.co/shuo-zhou/brain_network_lateralization](https://huggingface.co/shuo-zhou/brain_network_lateralization), DOI: [10.57967/hf/2996](https://doi.org/10.57967/hf/2996).

The Brainnetome atlas (BNA) is available at [http://atlas.brainnetome.org/](http://atlas.brainnetome.org/). [BNA_subregions.xlsx](BNA_subregions.xlsx) for the information about the lobes and gyrus is provided by the author and originally available at [https://pan.cstcloud.cn/web/share.html?hash=6eRCJ0zDTFk](https://pan.cstcloud.cn/web/share.html?hash=6eRCJ0zDTFk).

## Processing model weights

### First-order classification

First-order models were trained to classify right (labelled as 1) and left (labelled as 0) brain hemispheres.

`process_first_order_weights.py` processes first-order model weight (.pt) files by:

1. Loading weight vectors learned from multiple random splits (seeds).
2. Combining all weights into a single .npz file for archival and future use.
3. Computing the mean and standard deviation across splits.
4. Saving the aggregated results as a .mat file.

Each `.mat` file contains the mean and standard deviation of coefficients for a group (1000) of models.

**Naming:** `L` denotes the $\lambda$, and `G` denotes the subjects' gender (0 for male and 1 for female) of the subset used for training.

### Second-order classification

Second-order models were trained to classify two group of first-order models (male-specific vs female-specific).

`process_second_order_weights.py` processes second-order model weight (.joblib) files by:

1. Loading weight vectors learned from multiple random splits (seeds).
2. Combining all weights into a single .npz file for archival and future use.
3. Computing the mean and standard deviation across splits.
4. Saving the aggregated results as a .mat file.

**Example output**: "L2G0_vs_L2G1.mat"

- mean and standard deviation of models (logistic regression) to classify models trained respectively on males and females, both with $\lambda=2$
- "L2G0" was labelled as class 1 and "L2G1" was labelled as class 0

## Visualisation

`plot_classification_results.py` visualises the classification results of first-order classification, which reproduces the Figure 2, S1, and S2 in the GigaScience paper.

`plot_weight_corr_lines.py` visualises the first-order and univariate test model weights correlation lines, which reproduces the Figure 3B, S3A, and S3B in the GigaScience paper.

`plot_weight_corr_heatmap.py` visualises the first-order and univariate test model weights correlation heatmap, which reproduces the Figure 3A, 3C, 3D, S3C, and S3D in the GigaScience paper.

`plot_chord.py` visualises a mask derived from second-order classification, and masked top first-order model weights as chord diagrams, which reproduces the Figure 4A and S4 in the GigaScience paper.

`rader-specific.R` visualises the intra- and inter-lobe statistics of connections with significant weights as radar diagrams, which reproduces the Figure 7 and S5 in the GigaScience paper.

## Contact

Author: [Shuo Zhou](https://github.com/shuo-zhou) and [Junhao Luo](https://github.com/junhaols)

Contact: <shuo.zhou@sheffield.ac.uk> adn <junhaol@mail.bnu.edu.cn>
