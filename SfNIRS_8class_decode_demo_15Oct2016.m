%%% Demonstration of fNIRS decoding for 8 stimulus classes
%  Society for fNIRS biennial meeting, Oct 2016, Paris
% Benjamin Zinszer & Laurie Bayet
% 06 June 2016
% rev 08 Oct 2016
%%%

%% Find files and define a few parameters
clear all;

% Check in the directories Probe1 and Probe2 for *.nirs files of
% participants to include in the analysis. The search will be guided by the
% contents of the Probe1 directory.
Probe1List = dir('Probe1/*.nirs');
Probe2List = dir('Probe2/*.nirs');

% The eight stimulus classes listed in order of appearance in the *.nirs
% file (excluding the fireworks).
conditions = {'kitty' 'bunny' 'dog' 'bear' 'foot' 'hand' 'mouth' 'nose'};

% A visual inspection of the data suggested that the most information for 
% decoding existed in a window around 6.5 to 9 s after stimulus 
% presentation. This corresponds to the rise of the hemodynamic response 
% and the end of the trial.
scan_window = [65 90]; % In scans (collected at 10 Hz)

% Initialize a struct for storing each subject's data.
subj_struct = struct('ID',num2cell(1:length(Probe1List)));


%% Define channel stability
% Voxel stability is a typical pre-decoding analysis in functional MRI to 
% identify voxels that behave similarly across runs, with no a priori
% assumptions about how voxels should respond to given stimuli (and thus no
% double-dipping). Here, we implement channel stability to the same ends.

% Initialize an array to store channel stability data
chan_selection = nan(46,length(subj_struct),2);

% Set the cutoff for stable channels. If this value is between -1 and 1, it
% will be a minimum threshold for channel stability. If this value is
% between 1 and 100, it will be a percentile cutoff (e.g., top 50th
% percentile of channels) for the given subject.
chan_stable_cut = 50;


%% Extract the NIRS data necessary for analysis

% Loops through each subject in the list of discovered files
for subjnum = 1:length(Probe1List),
    
    % Define files to look for
    fprintf('Subject %s:\n',Probe1List(subjnum).name);
    p1_file_path = ['Probe1/' Probe1List(subjnum).name];
    p2_file_path = ['Probe2/' Probe2List(subjnum).name];
    
    
    % This calls a function "extract_vecs_1subj" which parses the *.nirs
    % file, searches it for the various stimulus onsets, clips out the
    % epoched time series data, and saves it in a structure.
    subj_struct(subjnum).Condition = extract_vecs_1subj(p1_file_path, p2_file_path, conditions, scan_window);
    fprintf('Extracted subject %g data\n',subjnum);

    
    % Take that condition data average it down into an array with the
    % dimensions Channel X Condition
    for condnum = 1:length(subj_struct(subjnum).Condition),
        subj_struct(subjnum).rsavecs(:,condnum) = subj_struct(subjnum).Condition(condnum).ts_average;
    end
    fprintf('Saved response vectors in one array\n');
    
    
    % Use channel stability to select channels for inclusion. This calls
    % the function channel_stability_calculator which wraps another
    % function called StatMap_ST designed for voxel stability analysis. A
    % list of "stable channels" (defined by the cutoff value from line 44)
    % is returned, along with the stability estimates for each channel.
    subj_struct(subjnum).all_chans = 1:size(subj_struct(subjnum).rsavecs,1);
    [subj_struct(subjnum).keep_chans, stabilites]= channel_stability_calculator(subj_struct(subjnum),chan_stable_cut);
    
    chan_selection(:,subjnum,1) = stabilites;
    chan_selection(subj_struct(subjnum).keep_chans,subjnum,2) = stabilites(subj_struct(subjnum).keep_chans);


end

%% Plot the channel stability data
% Figure 1 shows all channels for each participant and their respective
% stability values
figure
plot_tiled_data(chan_selection(:,:,1),'bar','All Channel Stability, Subject','Channel #');
% Figure 2 shows only the stable channels for each participant and their 
% respective stability values
figure
plot_tiled_data(chan_selection(:,:,2),'bar','Retained Stable Channels, Subject','Channel #');

%% Saving output
% Optionally, save the output so far for performing future analyses.
%save(sprintf('subjData_forDemo_%s.mat',date),'subj_struct')

%% Perform between-subjects decoding using RSA
% Using only the stable channels (keep_chans) for each subject, build the
% RSA matrices for each subject and save them in a 3-D matrix of dimensions
% NumClasses X NumClasses X NumSubjects
sim_struct = nan(length(conditions),length(conditions),length(subj_struct));
for i = 1:length(subj_struct),
    sim_struct(:,:,i) = atanh(corr(subj_struct(i).rsavecs(subj_struct(i).keep_chans,:),'rows','pairwise'));
end

% Run all pairwise comparisons between the 8 classes for each subject,
% decoding based on the group model of all remaining subjects.
% Method for this pairwise comparison is described in:
% (Anderson, Zinszer, & Raizada, 2015, Neuroimage) and on the SfNIRS poster

% This calls the function pairwise_rsa_leaveoneout which iterates through
% the subjects in an n-fold cross-validation and performs all pairwise
% comparisons for each participant (see pairwise_rsa_test)
pairwise_acc = pairwise_rsa_leaveoneout(sim_struct);

% This is a quick-and-dirty parametric test of significance, which does NOT
% properly apply to cross-validation, but is convenient for the moment. We
% will run the randomization test to get a real significance test later.
[~, p , ~, stat] = ttest(pairwise_acc, 0.5);
fprintf('----------------------\nQUICK AND DIRTY T-test:\n');
fprintf('Mean: %0.2f, T(%g)=%1.1f, p=%0.2f\n',nanmean(pairwise_acc),stat.df,stat.tstat,p);
fprintf('DO NOT USE FOR SIGNIFICANCE TESTING\n----------------------\n\n');

% Figure 3 plots each subject's mean pairwise decoding accuracy against the
% group (other n-1 subjects)
figure
bar(pairwise_acc)
title(sprintf('Between-subjects decoding accuracy, Mean: %0.2f',nanmean(pairwise_acc)));
xlabel('Subject ID');


%% Load and transform the external semantic model
% The semantic model is a 300 dimensional vector for each word from Baroni 
% et al's (2014) COMPOSES corpus model. This model is state-of-the-art in 
% corpus-based semantic representation and has proven powerful for fMRI 
% decoding.

% Semantic model based on COMPOSES model (Baroni et al., 2014)
load('SfNIRS_semantic_vectors_01June2016.mat','semantic_matrix');
semantic_sim_struct = atanh(corr(semantic_matrix));
semantic_tri = semantic_sim_struct(logical(tril(semantic_sim_struct+1,-1)));

% Figure 4 displays the semantic similarity structure
figure
imagesc(semantic_sim_struct);
title('Semantic Similarity (COMPOSES)')
set(gca,'YTick',[1:1:8])
set(gca,'XTick',[1:1:8])
set(gca,'YTickLabel',conditions)
set(gca,'XTickLabel',conditions)
colormap('hot')
colorbar;


%% Decoding each subject based on semantic model
% Here the external RSA model for semantic representation is applied to
% attempt decoding of the fNIRS RSA representations. Only the stable
% channels identified in the foregoing sections are used here.

subject_decoding_results = nan(length(subj_struct),1);
for subjnum = 1:length(subj_struct),
    tic;
    fprintf('Decoding subject %g... ',subjnum);
    
    % Limit channels to the top 50% stable channels (keep_chans)
    chan_set = subj_struct(subjnum).keep_chans;
    % Build neural similarity struct based on the stable channels
    neural_sim_struct = atanh(corr(subj_struct(subjnum).rsavecs(chan_set,:),'rows','pairwise'));
    
    % Attempt decoding of each subject's neural data based on the semantic model
    subject_decoding_results(subjnum) = mean(pairwise_rsa_test(neural_sim_struct,semantic_sim_struct));
    toc;
    
end

% Figure 5 plots each subject's mean pairwise decoding accuracy against the
% semantic model (COMPOSES)
figure
bar(subject_decoding_results)
title(sprintf('Model-based decoding accuracy, Mean: %0.2f',nanmean(subject_decoding_results)));
xlabel('Subject ID');




%% Use the channel-subset procedure to find the best channels for each
% model type, independently for each subject. Select the best channels (top
% 50% of stable channels) and perform decoding again. See Emberson,
% Zinszer, Raizada, & Aslin (2016) for details on channel subsetting.

% Channels are evaluated in subsets of 3, which previous research showed
% was a sufficient characterization of channel informativeness to pick the
% most informative channels.
search_subset_size = 3;

% Initialize results matrices
subject_channel_accuracy = nan(length(subj_struct),46,1);
subject_decoding_results_chanSel = nan(length(subj_struct),1);

% Loop through each subject, performing the channel subsetting procedure to
% determine the relative informativeness of each channel.
for subjnum = 1:length(subj_struct),
    
    fprintf('Evaluating Subj %g with %g channels\n',subj_struct(subjnum).ID, length(subj_struct(subjnum).keep_chans))
    
    % Search subsets of the retained channels, and test their correlations
    % to the RSA model of interest.
    chan_set_list = combnk(subj_struct(subjnum).keep_chans, search_subset_size);
    chan_set_results = nan(size(chan_set_list,1),1);
    
    fprintf('Will now search %g channel subsets\n',size(chan_set_list,1));
    disp([datestr(clock) ' Beginning...'])
    for set_num = 1:size(chan_set_list,1),
        
        chan_set = chan_set_list(set_num,:);
        neural_sim_struct = atanh(corr(subj_struct(subjnum).rsavecs(chan_set,:),'rows','pairwise'));
        neural_sim_struct(isnan(neural_sim_struct)) = 0;        

        % Each channel triad (assuming search_subset_size = 3) is tested
        % for its decoding accuracy, and these results are saved for later
        % aggregation.
        chan_set_results(set_num) = mean(pairwise_rsa_test(neural_sim_struct,semantic_sim_struct));
        
    end
    disp([datestr(clock) ' Done.'])
    
    % Compute the mean informativeness for each channel based on the
    % average decoding accuracy achieved across all subsets that the
    % channel participated in.
    mean_chan_corr = nan(length(subj_struct(subjnum).all_chans),1);
    for curr_chan = subj_struct(subjnum).keep_chans;
        sets_incl_curr_chan = logical(sum(chan_set_list==curr_chan));
        mean_chan_corr(curr_chan) = nanmean(chan_set_results(sets_incl_curr_chan));
    end
    subject_channel_accuracy(subjnum,:) = mean_chan_corr(:);
end
    
    
% Loop through each subject, performing the semantic decoding based on the
% channels which are most informative in the rest of the group (avoid
% double-dipping by NOT looking at the test subject's best channels). Since
% we are now selecting channels based on the other subjects' semantic 
% decoding accuracy, the channel stability threshold is not applied.
for subjnum = 1:length(subj_struct),
    
    % Compute the mean channel informativeness based on the other n-1
    % subjects' data
    group_subjs = find((1:length(subj_struct))~=subjnum);
    mean_chan_inform = nanmean(subject_channel_accuracy(group_subjs,:));
    
    % Select the 50th percentile most-informative-channels from the group
    % data and use these channels in the test subject (calling them
    % "semantic channels" / sem_chans
    threshold = prctile(mean_chan_inform,50);
    sem_chans = find(mean_chan_inform>=threshold);
    RSA_from_sem_chans = atanh(corr(subj_struct(subjnum).rsavecs(sem_chans,:),'rows','pairwise'));
    subject_decoding_results_chanSel(subjnum) = mean(pairwise_rsa_test(RSA_from_sem_chans,semantic_sim_struct));
    subj_struct(subjnum).semantic_channels = sem_chans;    
end

figure
bar(subject_decoding_results_chanSel)
title(sprintf('Channel-search, Model-based decoding accuracy, Mean: %0.2f',nanmean(subject_decoding_results_chanSel)));
xlabel('Subject ID');

% This is a quick-and-dirty parametric test of significance, which does NOT
% properly apply to cross-validation, but is convenient for the moment. We
% will run the randomization test to get a real significance test later.
[~, p, ~, stat] = ttest(subject_decoding_results_chanSel, 0.5);
fprintf('----------------------\nQUICK AND DIRTY T-test:\n');
fprintf('Mean: %0.2f, T(%g)=%1.1f, p=%0.2f\n',nanmean(subject_decoding_results_chanSel),stat.df,stat.tstat,p);
fprintf('DO NOT USE FOR SIGNIFICANCE TESTING\n----------------------\n\n');