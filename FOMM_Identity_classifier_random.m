function [confusion_matrix,confusion_matrix_fig,n_clusters_assessed,n_bouts_correct,correct_percentage]...
    = FOMM_Identity_classifier_random(N_samples,cluster_size,full_sequences,...
    cond_probability_dists,IID_probability_dists,input_alphabet)
% FOMM_Identity_classifier_random
% Leave-out individual identity classifier using first-order Markov models,
% with bouts selected at random. For each test, cluster_size bouts are drawn
% at random from a seal's sequence and held out. The seal's model is refitted
% without them, and the cluster is attributed to the seal whose model gives
% it the highest log-likelihood. Run repeatedly (bootstraps) to compare with
% the sequential classifier.
arguments (Input)
    N_samples % number of seals in test sample
    cluster_size % number of randomly selected bouts to cluster together for testing
    full_sequences % cell array of sequences as strings
    cond_probability_dists % cell array of conditional probability distribution matrices
    IID_probability_dists % cell array of IID probability distribution vectors
    input_alphabet % string array of possible symbols
end

arguments (Output)
    confusion_matrix % proportion of each seal's clusters attributed to each seal (row-normalised)
    confusion_matrix_fig % always empty (no figure is made, to save time and memory in bootstraps)
    n_clusters_assessed % total number of clusters analysed
    n_bouts_correct % number of clusters correctly classified per seal
    correct_percentage % percentage of each seal's clusters classified correctly
end

% number of symbols in alphabet
n_symbols = length(input_alphabet);
% alphabet as one character array
alphabet_chars = char(join(input_alphabet,""));

% Pre-allocate outputs
n_bouts_correct = zeros(1,N_samples);
correct_percentage = zeros(1,N_samples);

confusion_matrix = zeros(N_samples); % Confusion matrix to show attributions
clusters_per_seal = zeros(1,N_samples);

    % Run for each leopard seal
    for seal_index = 1:N_samples
        % Index values of all seals excluding 'unknown' seal
        known_seal_index_vector = 1:N_samples;
        known_seal_index_vector(seal_index) = [];

        % Extract calling sequence of 'unknown' seal
        unknown_sequence = full_sequences{seal_index};

        % Split sequence into bouts
        unknown_bouts = split(unknown_sequence,'Z');

        % number of bouts
        n_bouts = count(unknown_sequence,'Z');

        % Number of bout clusters (same number of tests as the sequential classifier)
        n_clusters = n_bouts - cluster_size+1;
        clusters_per_seal(seal_index) = n_clusters;

        % Set up matrix of likelihood values (log-likelihood of each cluster
        % of the 'unknown' seal originating from each seal)
        Likelihood_matrix = zeros(n_clusters,N_samples);

        % Vector of maximum likelihood for each cluster
        Max_Likelihood = zeros(1,n_clusters);

        % Attributed seal - index of maximum likelihood for each cluster
        Position_of_Max_Likelihood = zeros(1,n_clusters);

        % Loop for each test cluster
        for bout_number = 1:n_clusters

            % Randomly select bouts to hold out
            random_bout_vector = randperm(n_bouts,cluster_size);

            % Extract 'unknown' cluster of bouts
            unknown_bout = join(append(unknown_bouts(random_bout_vector),'Z'));
            unknown_bout = erase(unknown_bout," ");

            % Reconstruct seal sequence without 'unknown' cluster
            unknown_seal_reconstructed = unknown_bouts;
            % Remove unknown cluster
            unknown_seal_reconstructed(random_bout_vector) = [];
            % Put bouts back together with Z between
            unknown_seal_reconstructed = join(append(unknown_seal_reconstructed,'Z'));
            % Remove extra spaces
            unknown_seal_reconstructed = erase(unknown_seal_reconstructed," ");
            % Remove extra Z on end
            unknown_seal_reconstructed = extractBefore(unknown_seal_reconstructed,strlength(unknown_seal_reconstructed));

            % Calculate new Markov model for reconstructed sequence (leave-out model)
            [~,~,~,~,~,New_IIDProbabilityDistribution,~,New_ConditionalProbabilityDistribution,~]...
                = FirstOrderMarkov_IndividualIdentity_v2(unknown_seal_reconstructed,input_alphabet);

            % Count the transitions in the cluster once, then reuse the
            % counts for every seal
            % bout_idx(k) = position of symbol k in the alphabet (1 = D ... 6 = Z)
            [~, bout_idx] = ismember(char(unknown_bout), alphabet_chars);
            % bout_pair_counts(x,y) = number of times x is followed by y in the cluster
            bout_pair_counts = accumarray([bout_idx(1:end-1)' bout_idx(2:end)'], ...
                1, [n_symbols n_symbols]);
            % first symbol of the cluster (scored with the IID probability)
            bout_first_symbol = bout_idx(1);

            Likelihood_vector = zeros(1,N_samples);

            % Vocalising seal (uses its leave-out model)
            Likelihood_vector(seal_index) = bout_log_likelihood(bout_pair_counts, bout_first_symbol, ...
                New_ConditionalProbabilityDistribution, New_IIDProbabilityDistribution);

            % All other seals (full models)
            for q = known_seal_index_vector
                Likelihood_vector(q) = bout_log_likelihood(bout_pair_counts, bout_first_symbol, ...
                    cond_probability_dists{q}, IID_probability_dists{q});
            end
            Likelihood_matrix(bout_number,1:N_samples) = Likelihood_vector;
            [Max_Likelihood(bout_number), Position_of_Max_Likelihood(bout_number)] = max(Likelihood_vector);
        end

    % Outputs
    n_bouts_correct(seal_index) = sum(Position_of_Max_Likelihood == seal_index);
    correct_percentage(seal_index) = 100*n_bouts_correct(seal_index)/n_clusters;

    % Add to confusion matrix: clusters attributed to each seal as a
    % proportion of this seal's total clusters
    [attribution_counts,~] = hist(Position_of_Max_Likelihood,1:N_samples);
    confusion_matrix(seal_index,1:N_samples) = attribution_counts./n_clusters;
    end

    n_clusters_assessed = sum(clusters_per_seal);

    % No confusion chart is made here: building a figure on every bootstrap
    % is slow and uses a lot of memory
    confusion_matrix_fig = [];

end

function log_likelihood = bout_log_likelihood(pair_counts, first_symbol, cond_P, iid_P)
% Log-likelihood of a cluster under one seal's Markov model:
% log p(first symbol) + sum over pairs of count(x,y) * log P(y|x).
% Only transitions that occur in the cluster are included (avoids 0*log(0)).
    used = pair_counts > 0;
    log_likelihood = log(iid_P(first_symbol)) + sum(pair_counts(used) .* log(cond_P(used)));
end
