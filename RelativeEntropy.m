function [RelEntropy_matrix,RelEntropy_figure] = RelativeEntropy(N_samples,cond_probability_dists)
% RelativeEntropy
% Calculates the relative entropy (Kullback-Leibler divergence rate) between
% the first-order Markov models of every pair of seals, and plots the result
% as a heatmap.
%
% D(p||q) = sum over i,j of mu_p(i) * P_p(i,j) * log2( P_p(i,j) / P_q(i,j) )
% where mu_p is the stationary distribution of seal p's transition matrix.
% Values are Inf where seal p uses a transition that seal q never uses.
arguments (Input)
    N_samples % number of seals in test sample
    cond_probability_dists % cell array of conditional probability distribution matrices
end

arguments (Output)
    RelEntropy_matrix % N_samples x N_samples matrix; row = base model (p), column = comparison model (q)
    RelEntropy_figure % handle to the heatmap image
end

N_symbols = size(cond_probability_dists{1},1);

% Stationary distribution of each seal's Markov chain
LS_mu = cell(1,N_samples);

for q = 1:N_samples
    [V,D] = eig(cond_probability_dists{q}'); % Transpose P to get left eigenvectors
    [~, ind] = max(diag(D)); % Find the eigenvalue 1 (or close to it)
    LS_mu{q} = V(:, ind) / sum(V(:, ind)); % Normalise
end

% Matrix of relative entropy between each pair of seals
RelEntropy_matrix = zeros(N_samples,N_samples);

for p = 1:N_samples
    for q = 1:N_samples
        % Relative entropy contribution of each transition for this
        % seal-vs-seal comparison
        RelEntropy = zeros(N_symbols,N_symbols);
        for i = 1:N_symbols
            for j = 1:N_symbols
                % D(i,j) = mu(i)P(i,j)*log2(P(i,j)/Q(i,j))
                if cond_probability_dists{p}(i,j) == 0
                    RelEntropy(i,j) = 0;
                else
                    RelEntropy(i,j) = LS_mu{p}(i)*cond_probability_dists{p}(i,j)*...
                        log2(cond_probability_dists{p}(i,j)/...
                        cond_probability_dists{q}(i,j));
                end
            end
        end

        % Sum over all transitions to get the relative entropy between the two seals
        RelEntropy_matrix(p,q) = sum(sum(RelEntropy));
    end
end

% Heatmap
data = RelEntropy_matrix;
% White background layer, so that NaN and Inf values appear white
infRGB = reshape([1 1 1],1,1,3);
infimgdata = repmat(infRGB, size(data,1), size(data,2));
image(infimgdata, 'alphadata', ~isnan(data));
hold on
% Heatmap of finite values
RelEntropy_figure = imagesc(data,'alphadata', ~(isnan(data)|isinf(data)));
colormap(parula)
c = colorbar;
clim([0 1])
xticks(1:N_samples)
yticks(1:N_samples)
ylabel(c,"Relative Entropy (bits)")
xlabel("Seal ID")
ylabel("Seal ID (base model)")
title("")
hold off

end
