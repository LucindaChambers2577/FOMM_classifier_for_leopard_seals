# Leopard seal individual identity from call sequences

MATLAB code for testing whether individual leopard seals (*Hydrurga leptonyx*) can be identified from the order of sounds in their underwater calling, using first-order Markov models (FOMMs).

**Author:** Lucinda Chambers, UNSW Sydney

## Overview

Each seal's vocal sequence is coded as a string of call types. A first-order Markov model, which gives the probability of each call type following each other call type, is fitted to every seal. Bouts (or clusters of consecutive bouts) are then held out one at a time and attributed to whichever seal's model gives them the highest likelihood. Classification accuracy is reported for cluster sizes of 1 to 10 bouts.

Significance is assessed with a permutation test. Bouts are shuffled among individuals, keeping each individual's number of bouts, and the full classification procedure is repeated to build a null distribution of accuracy.

## Repository contents

```
├── Individual_Identity.m                     main analysis
├── FOMM_null_distribution.m                  permutation test
├── FirstOrderMarkov_IndividualIdentity_v2.m  functions
├── FOMM_Identity_classifier_sequential.m
├── FOMM_Identity_classifier_random.m
├── RelativeEntropy.m
├── data/                                     one .txt sequence file per seal
└── results/                                  created when the scripts are run
```

### Main scripts

| File | Description |
|---|---|
| `Individual_Identity.m` | Main analysis. Fits a FOMM to each seal, calculates bias-corrected Markov entropy, runs the sequential and random bout classifiers, calculates relative entropy between seals, and exports results tables and figures. |
| `FOMM_null_distribution.m` | Permutation test. Shuffles bouts among individuals, refits the models and reruns the sequential classifier to build a null distribution of classification accuracy. |

### Functions

| File | Description |
|---|---|
| `FirstOrderMarkov_IndividualIdentity_v2.m` | Fits a first-order Markov model to a sequence over a fixed alphabet. Returns symbol (IID) probabilities, joint and conditional transition probabilities, and the first-order Markov entropy estimate. |
| `FOMM_Identity_classifier_sequential.m` | Leave-out classifier that holds out clusters of *k* consecutive bouts in the order they were produced. |
| `FOMM_Identity_classifier_random.m` | Leave-out classifier that holds out *k* randomly selected bouts, used to compare against the sequential test. |
| `RelativeEntropy.m` | Calculates the relative entropy (Kullback–Leibler divergence rate) between every pair of seals' Markov models and plots a heatmap. |

## Method details

### Input data

Each seal is one plain-text (`.txt`) file containing its call sequence. Each call is coded as a single letter from the alphabet:

```
D  H  L  M  O  Z
```

`Z` marks the end of a bout and is required, because bouts are identified by splitting the sequence at `Z`. All non-alphabetic characters (spaces, line breaks, punctuation) are removed on import. The seal ID is taken from the file name, for example `92A.txt`.

### Markov model and entropy

For each sequence, the code estimates:

- **IID probabilities** *p(x)*: the proportion of each call type.
- **Conditional transition probabilities** *P(y | x)*: the number of *x→y* transitions divided by the count of *x*.
- **First-order Markov entropy** *H₁ = −Σ p(x, y) log₂ P(y | x)*, bias-corrected as *H₁ + 0.72(|A|² − |A|)/N*, where |A| is the observed alphabet size and *N* is the sequence length.

### Classification

For each seal and each cluster of *k* bouts:

1. The cluster is removed from that seal's sequence and the seal's Markov model is refitted on the remaining bouts (leave-out model), so a bout is never scored against a model that was trained on it.
2. The log-likelihood of the cluster is calculated under the leave-out model and under every other seal's full model. The first call uses the IID probability, and each following call uses the conditional transition probability.
3. The cluster is attributed to the seal with the highest log-likelihood.

The **sequential classifier** slides through each seal's bouts in recorded order. The **random classifier** draws *k* bouts at random for each test and is repeated over bootstrap iterations to give a mean and 95% interval.

### Permutation test (null distribution)

All bouts from all seals are pooled, randomly permuted and dealt back out, so each "seal" keeps its real number of bouts but receives bouts drawn at random from the whole population. New Markov models are fitted and the sequential classifier is rerun for every cluster size. Repeating this many times gives the distribution of classification accuracy expected when there is no individual signal. P-values can then be calculated as:

```
p = (1 + number of permutations with accuracy ≥ observed) / (number of permutations + 1)
```

The p-value calculation is done separately and is not included in these scripts.

## Requirements

- MATLAB R2019b or later. The functions use `arguments` blocks, and the scripts use name–value syntax such as `Color='r'`, which requires R2021a or later.
- Statistics and Machine Learning Toolbox (`confusionchart`, `prctile`).
- Parallel Computing Toolbox (optional). Bootstrap and permutation loops use `parfor`. Without the toolbox they run as ordinary `for` loops.

## Usage

1. Put the seal sequence `.txt` files in a folder called `data` next to the scripts. All scripts and functions should be in the same folder (or on the MATLAB path).
2. In MATLAB, change the current folder to the repository folder. If your data are somewhere else, edit the paths in the INPUTS section:
   ```matlab
   data_path    = 'data';      % folder containing the .txt files
   results_path = "results";   % where results are saved
   ```
3. Set the test parameters:
   ```matlab
   n_bootstraps    = 1000;   % bootstrap or permutation iterations
   cluster_options = 1:10;   % cluster sizes (number of bouts) to test
   ```
4. Run the script. `Individual_Identity.m` creates a time-stamped folder inside `results_path`; `FOMM_null_distribution.m` saves to `results/null_distribution`.

### Running on an HPC cluster (PBS)

The analyses were run on UNSW's Katana cluster. The scripts read the number of CPUs from the `NCPUS` environment variable and start one parallel worker per CPU. On a computer without `NCPUS`, they use all available cores. An example PBS job script:

```bash
#!/bin/bash
#PBS -N seal_identity
#PBS -l select=1:ncpus=16:mem=48gb
#PBS -l walltime=12:00:00
#PBS -j oe

cd $PBS_O_WORKDIR
module load matlab
matlab -batch "Individual_Identity"
```

Random numbers are seeded (`rng(1)` and `RandStream` substreams inside `parfor`), so results are reproducible regardless of how many workers are used.

## Outputs

### `Individual_Identity.m`

| Output | Contents |
|---|---|
| `sequential_classifier_results.csv` | Per seal: sequence length, number of bouts, observed alphabet size, bias-corrected entropy, and % correctly classified for each cluster size (sequential test). |
| `full_classifier_results.csv` | Per cluster size: overall % correct for the sequential test, and mean and 95% interval for the random test. |
| `random_classifier_perseal_results.csv` | Per seal: mean and SD of % correct across random bootstraps, for each cluster size. |
| `random_perseal_all.mat` | Raw per-bootstrap, per-seal results (bootstrap × seal × cluster size). |
| `confusion_matrix_n<k>.png` | Confusion chart for the sequential test at cluster size *k*. |
| `RelativeEntropy.png` | Heatmap of relative entropy between seals. |

### `FOMM_null_distribution.m`

| Output | Contents |
|---|---|
| `null_distribution_boutwise.csv` | Overall % correct for each permutation (rows) and cluster size (columns). |
| `null_distribution_perseal.csv` | Mean per-seal % correct for each permutation and cluster size. |
| `null_distribution.mat` | The above, plus the permutation vectors used. |


## Licence

MIT Licence
