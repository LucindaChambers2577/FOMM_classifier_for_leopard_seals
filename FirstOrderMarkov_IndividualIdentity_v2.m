function [EntropyEstimate,SequenceLength,SequenceAlphabet,alphabet_size,...
    NumOccurrences,IIDProbabilityDistribution,JointProbabilityDistribution,...
    ConditionalProbabilityDistribution,paired_symbols] = FirstOrderMarkov_IndividualIdentity_v2(Markov_Sequence,defined_alphabet)
% FirstOrderMarkov_IndividualIdentity_v2
% Fits a first-order Markov model to a symbol sequence over a pre-defined
% alphabet and estimates its first-order Markov entropy.
%
% Inputs:
%   Markov_Sequence  - sequence of symbols as a string (spaces and
%                      non-symbol characters removed first), e.g. "DHHLZ..."
%   defined_alphabet - string array of possible symbols, e.g. ["D" "H" "L" "M" "O" "Z"]
%
% Outputs:
%   EntropyEstimate                    - first-order Markov entropy estimate (bits)
%   SequenceLength                     - number of symbols in the sequence
%   SequenceAlphabet                   - unique symbols observed in the sequence
%   alphabet_size                      - number of unique symbols observed
%   NumOccurrences                     - count of each alphabet symbol
%   IIDProbabilityDistribution         - proportion of each alphabet symbol
%   JointProbabilityDistribution       - p(x,y): proportion of all pairs that are x followed by y
%   ConditionalProbabilityDistribution - P(y|x): probability that x is followed by y
%   paired_symbols                     - matrix of pair labels, e.g. "DH"

% total number of observed symbols
SequenceLength = strlength(Markov_Sequence);
% sequence and alphabet as character arrays
sequence_chars = char(Markov_Sequence);
alphabet_chars = char(join(defined_alphabet,""));
% symbols found in sequence
SequenceAlphabet = cellstr(unique(sequence_chars)')';
% position of each symbol in the alphabet (1 = first letter, 0 = not in alphabet)
[isKnown, idx] = ismember(sequence_chars, alphabet_chars);
% frequency of each alphabet symbol
NumOccurrences = histcounts(idx(isKnown), 0.5:1:numel(defined_alphabet)+0.5);
% number of symbols in possible alphabet
n_symbols = length(defined_alphabet);
% size of observed alphabet
alphabet_size = length(SequenceAlphabet);
% total number of observed pairs (duplets) in sequence
total_pairs = SequenceLength-1;

% IID probability distribution
IIDProbabilityDistribution = NumOccurrences./SequenceLength;

% labels for all possible alphabet pairs
paired_symbols = defined_alphabet' + defined_alphabet;

% count every pair in one pass through the sequence
% first_symbol(k) and second_symbol(k) are the two halves of the k-th pair
first_symbol  = idx(1:end-1);
second_symbol = idx(2:end);
both_known = first_symbol > 0 & second_symbol > 0;
% tally: for every pair (x,y) add 1 to element (x,y)
no_occurences_FOM = accumarray([first_symbol(both_known)' second_symbol(both_known)'], ...
    1, [n_symbols n_symbols]);

% joint probability p(x,y)
joint_probability_dist_array_FOM = no_occurences_FOM/total_pairs;

% conditional probability P(y|x) = count(x,y) / count(x); 0 where the pair never occurs
cond_probability_dist_array_FOM = zeros(n_symbols,n_symbols);
occurs = no_occurences_FOM > 0;
row_totals = repmat(NumOccurrences', 1, n_symbols);   % count(x) in every column
cond_probability_dist_array_FOM(occurs) = no_occurences_FOM(occurs)./row_totals(occurs);

% entropy terms -p(x,y)*log2(P(y|x)); 0 where the pair never occurs
H_1_array = zeros(n_symbols,n_symbols);
H_1_array(occurs) = -joint_probability_dist_array_FOM(occurs).*log2(cond_probability_dist_array_FOM(occurs));

JointProbabilityDistribution = joint_probability_dist_array_FOM;
ConditionalProbabilityDistribution = cond_probability_dist_array_FOM;

EntropyEstimate = sum(H_1_array,'all');

end
