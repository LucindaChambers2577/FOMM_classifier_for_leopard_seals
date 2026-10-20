% Leopard seal Markov Model Individual Identity Classifier
% Version: 5.0
% Author: Lucinda Chambers - UNSW Sydney
% Last updated: 20th October 2026
%
% Fits a first-order Markov model (FOMM) to each seal's call sequence, then
% tests whether held-out bouts can be attributed to the correct seal, using
% clusters of 1 to 10 bouts. Bouts are held out in recorded order
% (sequential test) and at random (bootstrap test). Also calculates the
% relative entropy between seals' models.
%
% Associated functions (keep in the same folder as this script, or on the MATLAB path)
% FirstOrderMarkov_IndividualIdentity_v2.m
% FOMM_Identity_classifier_sequential.m
% FOMM_Identity_classifier_random.m
% RelativeEntropy.m

clear all;
clc;
close all

%% INPUTS

% folder containing one .txt sequence file per seal
data_path = 'data';
% create directory of txt files of leopard seal sequences
leopard_seals = dir(fullfile(data_path, '*.txt'));
% define alphabet ('Z' marks the end of each bout)
input_alphabet = ["D" "H" "L" "M" "O" "Z"];

% TEST PARAMETERS
n_bootstraps = 1000; % number of bootstrap randomisations to run
cluster_options = 1:10; % range of bout cluster lengths to test
n_tests = length(cluster_options); % number of bout scenarios to test
rng(1) % reproducible bootstraps

% PARALLEL COMPUTING
% Start parallel workers (one per CPU requested on an HPC cluster)
n_workers = str2double(getenv('NCPUS'));   % the ncpus= value from a PBS job script
if isnan(n_workers)                         % not on a cluster (e.g. a desktop computer)
    n_workers = feature('numcores');        % use however many cores this machine has
end
if isempty(gcp('nocreate'))                 % only start a pool if one isn't already running
    parpool('local', n_workers);
end

% folder in which a time-stamped results folder will be created
results_path = "results";

%% PRE-PROCESSING
% Imports each seal's sequence, removes non-alphabetic characters, counts
% bouts, fits a first-order Markov model and calculates bias-corrected entropy

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
    % First-order Markov model
    [leopard_seals(q).H_1,leopard_seals(q).SequenceLength,leopard_seals(q).SequenceAlphabet,leopard_seals(q).ObservedAlphabetSize,~,leopard_seals(q).IIDProbabilityDistribution,...
        leopard_seals(q).JointProbabilityDistribution,leopard_seals(q).ConditionalProbabilityDistribution,...
        leopard_seals(q).paired_symbols] = FirstOrderMarkov_IndividualIdentity_v2(observation,input_alphabet);
    % Bias correct Markov entropy
    leopard_seals(q).H_1_BC = leopard_seals(q).H_1 + 0.72*(leopard_seals(q).ObservedAlphabetSize^2 - ...
    leopard_seals(q).ObservedAlphabetSize)/leopard_seals(q).SequenceLength;
end

%% Create results folder
process_time = string(datetime("now", "Format", "uuuuMMdd'T'HHmmss"));
results_folder = fullfile(results_path, process_time);
mkdir(results_folder)

%% SEQUENTIAL TEST
% Tests the bouts in the order they were produced
tic
% Pre-allocate arrays
sequential_confusion_matrices = cell(1,n_tests);
sequential_confusion_matrix_figures = cell(1,n_tests);
sequential_results_boutwise = cell(1,n_tests);
sequential_correct_classifications_boutwise = zeros(1,n_tests); % % of all clusters correct for each cluster size
mean_correct_classifications_per_seal = zeros(1,n_tests); % mean of per-seal % correct (each seal weighted equally)

for k = cluster_options % number of bouts per cluster

    [sequential_confusion_matrices{k},sequential_confusion_matrix_figures{k},n_clusters_assessed,n_bouts_correct,correct_percentage] =...
        FOMM_Identity_classifier_sequential(leopard_seals_n,k,{leopard_seals.Full_Sequence},...
        {leopard_seals.ConditionalProbabilityDistribution},{leopard_seals.IIDProbabilityDistribution},...
        input_alphabet);
    sequential_correct_classifications_boutwise(k) = 100*sum(n_bouts_correct)/n_clusters_assessed;
    sequential_results_boutwise{k} = correct_percentage;
    mean_correct_classifications_per_seal(k) = mean(correct_percentage);
    figure_name = append("confusion_matrix_n",string(k),'.png');
    saveas(sequential_confusion_matrix_figures{k},fullfile(results_folder,figure_name));
    close(sequential_confusion_matrix_figures{k}) % free memory

end
toc

%% BOOTSTRAP ANALYSIS
% Tests randomly selected bouts, repeated n_bootstraps times
tic
% Pre-allocate arrays
random_correct_classifications_boutwise = zeros(n_bootstraps,n_tests);
random_mean_correct_classifications_per_seal = zeros(n_bootstraps,n_tests);
% every seal's % correct, for every bootstrap and cluster size
% dimensions: bootstrap x seal x cluster size
random_perseal_all = zeros(n_bootstraps, leopard_seals_n, n_tests);

% Extract the data from the struct before the parfor loop, so each worker
% receives only these cell arrays rather than the whole leopard_seals struct
full_sequences = {leopard_seals.Full_Sequence};
cond_dists     = {leopard_seals.ConditionalProbabilityDistribution};
iid_dists      = {leopard_seals.IIDProbabilityDistribution};

for k = cluster_options % number of bouts per cluster

    % Temporary results for this cluster size: one entry per bootstrap
    boutwise_k = zeros(n_bootstraps,1);
    perseal_k  = zeros(n_bootstraps,1);
    perseal_matrix_k = zeros(n_bootstraps, leopard_seals_n);

    parfor b_t = 1:n_bootstraps % bootstraps run in parallel on different workers

        % Give each bootstrap its own reproducible random numbers
        stream = RandStream('mrg32k3a','Seed',1);
        stream.Substream = (k-1)*n_bootstraps + b_t;
        RandStream.setGlobalStream(stream);

        [~,~,n_clusters_assessed,n_bouts_correct,correct_percentage] =...
            FOMM_Identity_classifier_random(leopard_seals_n,k,full_sequences,...
            cond_dists,iid_dists,input_alphabet);

        boutwise_k(b_t) = 100*sum(n_bouts_correct)/n_clusters_assessed;
        perseal_k(b_t)  = mean(correct_percentage);
        perseal_matrix_k(b_t,:) = correct_percentage;
    end

    % Copy this cluster size's results into the main arrays
    random_correct_classifications_boutwise(:,k) = boutwise_k;
    random_mean_correct_classifications_per_seal(:,k) = perseal_k;
    random_perseal_all(:,:,k) = perseal_matrix_k;
end
toc

% 95% interval and mean of the bootstrap results
CI_correct_classifications_boutwise = [prctile(random_correct_classifications_boutwise,2.5,1);prctile(random_correct_classifications_boutwise,97.5,1)];
mean_random_correct_classifications_boutwise = mean(random_correct_classifications_boutwise,1);

%% RELATIVE ENTROPY
RelEntropy_fig = figure;
[RelEntropy_matrix,RelEntropy_figure] = RelativeEntropy(leopard_seals_n,{leopard_seals.ConditionalProbabilityDistribution});
saveas(RelEntropy_fig,fullfile(results_folder,"RelativeEntropy.png"))

%% OUTPUTS

% SEQUENTIAL RESULTS PER SEAL
% Variables: seal ID, sequence length, number of bouts, observed alphabet
% size, bias-corrected entropy, % correct for each cluster size
output_variable_names = ["Seal_ID" "Sequence_Length" "n_bouts" "|A|" "H_1_BC" "Bout cluster lengths"];
sequential_results_table = table({leopard_seals.name}',[leopard_seals.SequenceLength]',[leopard_seals.n_bouts]',...
    [leopard_seals.ObservedAlphabetSize]',[leopard_seals.H_1_BC]', [cell2mat(sequential_results_boutwise')]',...
    'VariableNames',output_variable_names);
writetable(sequential_results_table,fullfile(results_folder,"sequential_classifier_results.csv"));

% SEQUENTIAL VS RANDOM RESULTS PER CLUSTER SIZE
comparative_variable_names = {'Cluster_length'; 'Sequential_correct'; 'Random_mean_correct'; 'Random_CI_lower';...
    'Random_CI_upper'};
comparative_results = table(cluster_options',sequential_correct_classifications_boutwise',mean_random_correct_classifications_boutwise', ...
    CI_correct_classifications_boutwise(1,:)',CI_correct_classifications_boutwise(2,:)',...
    'VariableNames',comparative_variable_names);
writetable(comparative_results,fullfile(results_folder,"full_classifier_results.csv"));

% RANDOM BOOTSTRAP RESULTS PER SEAL (mean and SD across bootstraps)
% mean/std along dimension 1 = across bootstraps; squeeze gives seal x cluster size
perseal_mean = squeeze(mean(random_perseal_all, 1));     % seals x cluster sizes
perseal_sd   = squeeze(std(random_perseal_all, 0, 1));   % seals x cluster sizes

% Interleave columns so each cluster size has its mean next to its SD:
% cluster_1_mean, cluster_1_sd, cluster_2_mean, cluster_2_sd, ...
perseal_data = zeros(leopard_seals_n, 2*n_tests);
perseal_data(:,1:2:end) = perseal_mean;
perseal_data(:,2:2:end) = perseal_sd;
perseal_names = strings(1, 2*n_tests);
perseal_names(1:2:end) = compose("cluster_%d_mean", cluster_options);
perseal_names(2:2:end) = compose("cluster_%d_sd", cluster_options);

% Add a column of seal IDs (file names without .txt) at the front
seal_IDs = erase(string({leopard_seals.name}'), ".txt");
random_perseal_table = [table(seal_IDs, 'VariableNames', "Seal_ID"), ...
    array2table(perseal_data, 'VariableNames', perseal_names)];
writetable(random_perseal_table, fullfile(results_folder, "random_classifier_perseal_results.csv"));

% Raw per-bootstrap values
save(fullfile(results_folder, "random_perseal_all.mat"), "random_perseal_all", "cluster_options", "seal_IDs")
