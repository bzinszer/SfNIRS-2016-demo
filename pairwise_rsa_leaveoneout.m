function accuracies = pairwise_rsa_leaveoneout(test_subjs_mat)

subjs = 1:size(test_subjs_mat,3);
acc = nan(length(subjs),1);

for this_sub = subjs,
    
    acc(this_sub) = mean(pairwise_rsa_test(test_subjs_mat(:,:,this_sub),nanmean(test_subjs_mat(:,:,subjs(subjs~=this_sub)),3)));
    
end

accuracies = acc;