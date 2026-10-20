% Leopard seal Markov Model Individual Identity - Null distribution (permutation test)
% Version: 1.0
% Author: Lucinda Chambers - UNSW Sydney
% Last updated: 20th October 2026
%
% Builds a null distribution of classification accuracy. All bouts from all
% seals are pooled, randomly permuted and dealt back out, so each "seal"
% keeps its real number of bouts but receives bouts drawn at random from all
% seals. New Markov models are fitted and the sequential classifier is rerun
% for every cluster size. Repeating this gives the accuracy expected when
% there is no individual signal.
%
% Associated functions (keep in the same folder as this script, or on the MATLAB path)
% FirstOrderMarkov_IndividualIdentity_v2.m
% FOMM_Identity_classifier_sequential.m

clear all
clc

%% INPUTS

% folder containing one .txt sequence file per seal
data_path = 'data';
% create directory of txt files of leopard seal sequences
leopard_seals = dir(fullfile(data_path, '*.txt'));
% define alphabet ('Z' marks the end of each bout)
input_alphabet = ["D" "H" "L" "M" "O" "Z"];

% TEST PARAMETERS
n_bootstraps = 1000; % number of permutations to run
cluster_options = 1:10; % range of bout cluster lengths to test
n_tests = length(cluster_options); % number of bout scenarios to test
rng(1) % reproducible permutations

% PARALLEL COMPUTING
% Start parallel workers (one per CPU requested on an HPC cluster)
n_workers = str2double(getenv('NCPUS'));   % the ncpus= value from a PBS job script
if isnan(n_workers)                         % not on a cluster (e.g. a desktop computer)
    n_workers = feature('numcores');        % use however many cores this machine has
end
if isempty(gcp('nocreate'))                 % only start a pool if one isn't already running
    parpool('local', n_workers);
end

% folder in which the results folder will be created
results_path = "results";

%% PRE-PROCESSING
% Imports each seal's sequence, removes non-alphabetic characters and
% counts bouts

% Import data
filenames = {leopard_seals.name};
leopard_seals_n = length(filenames); % Number of seals

for q = 1:leopard_seals_n
    % Full path to the current file
    file_path = fullfile(leopard_seals(q).folder, leopard_seals(q).name);
    sample_data = string(fileread(file_path)); % import sequence
    observation = regexprep(sample_data,'[^a-zA-Z]',''); % Remove any non-alphabetic characters
    leopard_seals(q).Full_Sequence = observation;
    % number of bouts
    leopard_seals(q).n_bouts = count(observation,'Z');
end

%% Create results folder
results_folder = fullfile(results_path, "null_distribution");
mkdir(results_folder)

%% RANDOMISATION
tic
% total number of bouts
n_bouts_per_seal = [leopard_seals.n_bouts];
n_bouts_total = sum(n_bouts_per_seal);
% Join all sequences into one
full_set = string();
for l = 1:leopard_seals_n
    full_set = append(full_set,leopard_seals(l).Full_Sequence);
end
% split into bouts at 'Z'
bouts_split = split(full_set,'Z');
% remove empty bout after the final 'Z', so the number of bouts is consistent
bouts_split = bouts_split(1:n_bouts_total);

% Create the set of random bout orders (one per permutation)
randomisation_vectors = cell(1,n_bootstraps);
for r = 1:n_bootstraps
    randomisation_vectors{r} = randperm(n_bouts_total,n_bouts_total);
end

% First and last bout positions belonging to each seal
bout_end   = cumsum(n_bouts_per_seal);
bout_start = bout_end - n_bouts_per_seal + 1;

%% PERMUTATION TEST
% Pre-allocate results (one row per permutation)
null_boutwise = zeros(n_bootstraps,n_tests);
null_perseal  = zeros(n_bootstraps,n_tests);

% Each permutation runs on a different worker
parfor b_t = 1:n_bootstraps

    % Shuffle the bouts using this permutation's pre-generated order
    bouts_randomised = bouts_split(randomisation_vectors{b_t});

    % Deal the shuffled bouts back out: each seal gets the same number
    % of bouts it really had, but drawn at random from all seals
    New_Sequences = cell(1,leopard_seals_n);
    for q = 1:leopard_seals_n
        new_bouts = bouts_randomised(bout_start(q):bout_end(q));
        new_bouts = join(append(new_bouts,'Z'));
        New_Sequences{q} = erase(new_bouts," ");
    end

    % Build a Markov model for each new (randomised) seal
    random_IID_dists  = cell(1,leopard_seals_n);
    random_cond_dists = cell(1,leopard_seals_n);
    for q = 1:leopard_seals_n
        [~,~,~,~,~,random_IID_dists{q},~,random_cond_dists{q},~] = ...
            FirstOrderMarkov_IndividualIdentity_v2(New_Sequences{q},input_alphabet);
    end

    % Run the sequential classifier for every cluster size
    boutwise_row = zeros(1,n_tests);   % this permutation's results
    perseal_row  = zeros(1,n_tests);
    for k = cluster_options
        [~,~,n_clusters_assessed,n_bouts_correct,correct_percentage] = ...
            FOMM_Identity_classifier_sequential(leopard_seals_n,k,New_Sequences,...
            random_cond_dists,random_IID_dists,input_alphabet,false); % false = no figure
        boutwise_row(k) = 100*sum(n_bouts_correct)/n_clusters_assessed;
        perseal_row(k)  = mean(correct_percentage);
    end

    % Store this permutation's results as one row of the main arrays
    null_boutwise(b_t,:) = boutwise_row;
    null_perseal(b_t,:)  = perseal_row;
end
toc

%% OUTPUTS

% Save the null distribution and the permutations used
save(fullfile(results_folder,"null_distribution.mat"), ...
    "null_boutwise","null_perseal","randomisation_vectors","cluster_options")

% Column names, e.g. "cluster_1", "cluster_2", ... "cluster_10"
column_names = compose("cluster_%d", cluster_options);

% Null distribution of overall (boutwise) accuracy: rows = permutations, columns = cluster sizes
null_boutwise_table = array2table(null_boutwise, 'VariableNames', column_names);
writetable(null_boutwise_table, fullfile(results_folder, "null_distribution_boutwise.csv"));

% Null distribution of mean per-seal accuracy (each seal weighted equally)
null_perseal_table = array2table(null_perseal, 'VariableNames', column_names);
writetable(null_perseal_table, fullfile(results_folder, "null_distribution_perseal.csv"));
