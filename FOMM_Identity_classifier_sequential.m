function [confusion_matrix,confusion_matrix_fig,n_clusters_assessed,n_bouts_correct,correct_percentage] =...
    FOMM_Identity_classifier_sequential(N_samples,cluster_size,full_sequences,...
    cond_probability_dists,IID_probability_dists,input_alphabet,make_figure)
% FOMM_Identity_classifier_sequential
% Leave-out individual identity classifier using first-order Markov models.
% For each seal, clusters of cluster_size consecutive bouts are held out in
% the order they were produced. The seal's model is refitted without the
% held-out cluster, and the cluster is attributed to the seal whose model
% gives it the highest log-likelihood.
arguments (Input)
    N_samples % number of seals in test sample
    cluster_size % number of consecutive bouts to cluster together for testing
    full_sequences % cell array of sequences as strings
    cond_probability_dists % cell array of conditional probability distribution matrices
    IID_probability_dists % cell array of IID probability distribution vectors
    input_alphabet % string array of possible symbols
    make_figure (1,1) logical = true   % false = skip the confusion chart
end

arguments (Output)
    confusion_matrix % proportion of each seal's clusters attributed to each seal (row-normalised)
    confusion_matrix_fig % confusion chart figure (empty if make_figure is false)
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
attributed_seal = [];

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

        % Number of bout clusters
        n_clusters = n_bouts - cluster_size+1;
        clusters_per_seal(seal_index) = n_clusters;

        % Set up matrix of likelihood values (log-likelihood of each cluster
        % of the 'unknown' seal originating from each seal)
        Likelihood_matrix = zeros(n_clusters,N_samples);

        % Vector of maximum likelihood for each cluster
        Max_Likelihood = zeros(1,n_clusters);

        % Attributed seal - index of maximum likelihood for each cluster
        Position_of_Max_Likelihood = zeros(1,n_clusters);

        % Loop for each cluster of bouts
        for bout_number = 1:n_clusters

            % Extract 'unknown' cluster of bouts
            unknown_bout = join(append(unknown_bouts(bout_number:(bout_number+cluster_size-1)),'Z'));
            unknown_bout = erase(unknown_bout," ");

            % Reconstruct seal sequence without 'unknown' cluster
            unknown_seal_reconstructed = unknown_bouts;
            % Remove unknown cluster
            unknown_seal_reconstructed(bout_number:(bout_number+cluster_size-1)) = [];
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

    attributed_seal = horzcat(attributed_seal,Position_of_Max_Likelihood);

    % Outputs
    n_bouts_correct(seal_index) = sum(Position_of_Max_Likelihood == seal_index);
    correct_percentage(seal_index) = 100*n_bouts_correct(seal_index)/n_clusters;

    % Add to confusion matrix: clusters attributed to each seal as a
    % proportion of this seal's total clusters
    [attribution_counts,~] = hist(Position_of_Max_Likelihood,1:N_samples);
    confusion_matrix(seal_index,1:N_samples) = attribution_counts./n_clusters;
    end

    n_clusters_assessed = sum(clusters_per_seal);

    % Confusion matrix figure
    if make_figure
        % vector of the true (vocalising) seal for every cluster
        responsible_seal = [];
        for w = 1:N_samples
            responsible_seal = horzcat(responsible_seal, w.*ones(1,clusters_per_seal(w)));
        end
        % Create confusion chart
        confusion_matrix_fig = figure;
        confusion_matrix_fig.Visible = 'off';
        confusion_matrix_fig.Position = [282,86,1000,600];
        cc = confusionchart(responsible_seal,attributed_seal);
        cc.RowSummary = 'row-normalized';
        cc.XLabel = 'Seal ID (attributed seal)';
        cc.YLabel = 'Seal ID (vocalising seal)';
    else
        confusion_matrix_fig = [];
    end

end

function log_likelihood = bout_log_likelihood(pair_counts, first_symbol, cond_P, iid_P)
% Log-likelihood of a cluster under one seal's Markov model:
% log p(first symbol) + sum over pairs of count(x,y) * log P(y|x).
% Only transitions that occur in the cluster are included (avoids 0*log(0)).
    used = pair_counts > 0;
    log_likelihood = log(iid_P(first_symbol)) + sum(pair_counts(used) .* log(cond_P(used)));
end
