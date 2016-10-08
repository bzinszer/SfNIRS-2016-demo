
conditions = {'kitty' 'bunny' 'dog' 'bear' 'foot' 'hand' 'mouth' 'nose'};
num_subjs = length(subj_struct);

% Semantic model based on COMPOSES model (Baroni et al., 2014)
semantic_sim_struct = atanh(corr(semantic_matrix));
semantic_tri = semantic_sim_struct(logical(tril(semantic_sim_struct+1,-1)));

% Initialize the various permutations to use
num_trials = 5000;
all_perms = perms(1:8);
perms_to_use = randsample(1:size(all_perms,1),num_trials.*num_subjs);
perms_to_use = reshape(mat2cell(all_perms(perms_to_use,:),ones(num_trials*num_subjs,1),8),num_trials,num_subjs);

%% Perform permutation test for between-subjects decoding using RSA
btw_subj_acc = nan(num_trials,num_subjs);
tic
for this_trial = 1:num_trials,
    if ~mod(this_trial,100), fprintf('Trial %g: %.0f sec\n',this_trial,toc); end
    % Using only the stable channels (keep_chans) for each subject, build the
    % RSA matrices for each subject and save them in a 3-D matrix of dimensions
    % NumClasses X NumClasses X NumSubjects
    sim_struct = nan(length(conditions),length(conditions),length(subj_struct));
    for i = 1:length(subj_struct),
        sim_struct(:,:,i) = atanh(corr(subj_struct(i).rsavecs(subj_struct(i).keep_chans,:),'rows','pairwise'));
        sim_struct(:,:,i) = sim_struct(perms_to_use{this_trial,i},perms_to_use{this_trial,i},i);
    end
    % Run all pairwise comparisons between the 8 classes for each subject,
    % decoding based on the group model of all remaining subjects.
    btw_subj_acc(this_trial,:) = pairwise_rsa_leaveoneout(sim_struct);
    
end
subplot(1,3,1)
hist(mean(btw_subj_acc,2))
fprintf('Stable Channel Semantic Decoding\n')
sig_acc = prctile(mean(btw_subj_acc,2),95);
p_val = sum(mean(btw_subj_acc,2)>=0.6518) / size(btw_subj_acc,1);
fprintf('95th %%ile accuracy: %0.2f; p(Acc=0.6518)=%0.3f\n\n',sig_acc,p_val)

%% Decoding each subject based on semantic model
sem_model_acc = nan(num_trials,num_subjs);
tic
for this_trial = 1:num_trials,
    if ~mod(this_trial,100), fprintf('Trial %g: %.0f sec\n',this_trial,toc); end
    % Using only the stable channels (keep_chans) for each subject, build the
    % RSA matrices for each subject and save them in a 3-D matrix of dimensions
    % NumClasses X NumClasses X NumSubjects
    for i = 1:length(subj_struct),
        neural_sim_struct = atanh(corr(subj_struct(i).rsavecs(subj_struct(i).keep_chans,:),'rows','pairwise'));
        permed_semantic_sim_struct = semantic_sim_struct(perms_to_use{this_trial,i},perms_to_use{this_trial,i});
        sem_model_acc(this_trial,i) = mean(pairwise_rsa_test(neural_sim_struct,permed_semantic_sim_struct));    
    end
end
subplot(1,3,2)
hist(mean(sem_model_acc,2))
fprintf('Stable Channel Semantic Decoding\n')
sig_acc = prctile(mean(sem_model_acc,2),95);
p_val = sum(mean(sem_model_acc,2)>=0.6339) / size(sem_model_acc,1);
fprintf('95th %%ile accuracy: %0.2f; p(Acc=0.6339)=%0.3f\n\n',sig_acc,p_val)

%% Decoding each subject based on best semantic chans and semantic model
sem_model_subset_acc = nan(num_trials,num_subjs);
tic
for this_trial = 1:num_trials,
    if ~mod(this_trial,100), fprintf('Trial %g: %.0f sec\n',this_trial,toc); end
    % Using only the semantic_channels for each subject, based on the group
    % subsetting analysis. Build the RSA matrices for each subject and save
    % them in a 3-D matrix of dimensions NumClasses X NumClasses X NumSubjects
    for i = 1:length(subj_struct),
        neural_sim_struct = atanh(corr(subj_struct(i).rsavecs(subj_struct(i).semantic_channels,:),'rows','pairwise'));
        permed_semantic_sim_struct = semantic_sim_struct(perms_to_use{this_trial,i},perms_to_use{this_trial,i});
        sem_model_subset_acc(this_trial,i) = mean(pairwise_rsa_test(neural_sim_struct,permed_semantic_sim_struct));    
    end
end
subplot(1,3,3)
hist(mean(sem_model_subset_acc,2))
fprintf('Semantic Channel Decoding\n')
sig_acc = prctile(mean(sem_model_subset_acc,2),95);
p_val = sum(mean(sem_model_subset_acc,2)>=0.6920) / size(sem_model_subset_acc,1);
fprintf('95th %%ile accuracy: %0.2f; p(Acc=0.6920)=%0.3f\n\n',sig_acc,p_val)