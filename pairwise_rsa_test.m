function [accuracies] = pairwise_rsa_test(test_matrix, training_matrix)

%% Prep some basic values

% Number of stimulus classes to compare
number_classes = size(test_matrix,1);

% Number of comparisons to make (same as number of unique off-diagnoal
% values in the test or training matrix)
number_comparisons = (numel(test_matrix)-number_classes)/2;

% Generate a list of every pairwise comparison and the results array
list_of_comparisons = combnk([1:number_classes],2);
results_of_comparisons = nan(number_comparisons,1);

if sum(isnan(test_matrix(:)))<numel(test_matrix),
    
    %% Loop through all comparisons and test
    for this_comp = 1:number_comparisons
        
        test_classes = list_of_comparisons(this_comp,:);
        
        % Define which classes to keep in the RSA structures (non-test classes)
        non_test_classes = logical(ones(number_classes,1));
        non_test_classes(test_classes) = 0;
        
        % Rebuild the RSA matrices without the test classes in them
        test_matrix_temp = test_matrix(non_test_classes,:);
        training_matrix_temp = training_matrix(non_test_classes,:);
        
        % Extract the two vectors of interest from test and train matrices
        testA = test_matrix_temp(:,test_classes(1));
        testB = test_matrix_temp(:,test_classes(2));
        trainA = training_matrix_temp(:,test_classes(1));
        trainB = training_matrix_temp(:,test_classes(2));
        
        % Compare the correlations
        if (corr(testA,trainA)+corr(testB,trainB) > corr(testB,trainA)+corr(testA,trainB)),
            results_of_comparisons(this_comp) = 1;
        else
            results_of_comparisons(this_comp) = 0;
        end
        
    end
end

accuracies = results_of_comparisons;

