# Meeting notes and the script for the chord plot and spec_rate

## Main conclusions

<strike>

  - Sex-specific patterns with covariate dependent classification algorithm: accuracy gap and correlation vs control
  - Multivariate control vs univariate control: similarity
  - Lateralization patterns (general, male, and female)

</strike>

UPDATE (03.09.2023)

- Sex-specific patterns learned by group-specific discriminant analysis (GSDA), validated by the accuracy gap (diverged accuracy)
- GSDA learned distinct patterns, demonstrated by the correlation coefficients heatmap, where the univariate and multivariate control models are in a cluster, GSDA models ($\lambda=5 $) are in another cluster
- The patterns learned by GSDA are consistent over the results obtained from HCP and GSP
- [ ] Neuroscience findings, e.g. lateraliztion hub? common sex-specific lateralization pattern? @junhaols

## Meeting notes (29.01.2023)

- First-order for left/right classification
- Second-order for male/female classification
- Overlap between first- and second-order coefficients:
  - Sort first-order coefficients by absolute values
  - Sort second-order coefficients by absolute values
  - Take overlap
  - Show first-order coefficients values
- Justification for second-order classification: select gender-related (statistically) coefficients from first-order coefficients

## Meeting notes (03.07.2023)

- [x] correlation between univariate models and multivariate models:
  - [x] univariate control vs multivariate control
        - HCP, r = 0.918, p = 0.0
        - GSP, r = 0.940, p = 0.0
  - [x] univariate male vs multivariate male
        - HCP, r = 0.472, p = 0.0
        - GSP, r = 0.484, p = 0.0
  - [x] univariate female vs multivariate female
        - HCP, r = 0.490, p = 0.0
        - GSP, r = 0.505, p = 0.0

- [x] core results of laterality patterns:
  - [x] univariate: 并集 - 交集
  - [x] multivariate: mask generated from second-order coefficients

- [x] Remove UK-biobank results

No need to quantify laterality patterns unless there are conclusions to be made about.

## Meeting notes (12.07.2023)

### To do (12.07.2023)

- [x] Figures:
  - [x] Workflow to Figure 1
  - [x] correlation between CoDLR models and control models across $\lambda$s
  - [x] correlation heatmap:
    - univariate control hcp
    - univariate male HCP
    - univariate female HCP
    - univariate control GSP
    - univariate male GSP
    - univariate female GSP
    - multivariate control HCP
    - multivariate male ($\lambda=5$) HCP
    - multivariate female ($\lambda=5$) HCP
    - multivariate control GSP
    - multivariate male ($\lambda=5$) GSP
    - multivariate female ($\lambda=5$) GSP
  - [ ] Masks of sex-specific patterns: univariate vs multivariate (.svg @junhaols)
  - [ ] Multivariate male and female patterns (overlap of first- (5%) and second-order (5%), with values (colorbar), .svg) @junhaols
- [x] @junhaols Share univariate t-values

### Thoughts on the high correlation between male- and female- models

Male and female have similar lateralization patterns but have differences in the density of the lateralized brain network connections

sex-specific lateralization patterns = lateralized brain network connection with significant difference of density between male and female

## Meeting notes (03.09.2023)

### Summary of Figure 4

- More negative weights (blue chords) in male-specific model, more positive weights (orange/red chords) in female-specific model
- More interregional connections in male-specific model, more intraregional connections in female-specific model
- The magnitude of negative interregional connections are higher in male-specific model, the magnitude of positive connections and intraregional connections are higher in female-specific model
- [ ] @junhaols Female-specific patterns: frontal-frontal connections
- [ ] @junhaols Male-specific patterns: frontal-parietal connections
- [ ] @junhaols Potential hubs: frontal MFG 7, frontal MFG 11

### To do (03.09.2023)

- [x] @junhaols Share second-order mask (Fig. 4 (a)) .csv file
- [ ] @shuo-zhou hold out subjects for testing
- [x] Quantitative analysis for the mask
