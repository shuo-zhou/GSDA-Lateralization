# The script for the chord plot and spec_rate

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

### To do

- [ ] Figures:
  - [ ] Workflow to Figure 1
  - [ ] correlation between CoDLR models and control models across $\lambda$s
  - [ ] correlation heatmap:
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
- [ ] @junhaols Share univariate t-values

### Main conclusions

- Sex-specific patterns with covariate dependent classification algorithm: accuracy gap and correlation vs control
- Multivariate control vs univariate control: similarity
- Lateralization patterns (general, male, and female)

### Thoughts on the high correlation between male- and female- models

Male and female have similar lateralization patterns but have differences in the density of the lateralized brain network connections

sex-specific lateralization patterns = lateralized brain network connection with significant difference of density between male and female
