function subj_data = extract_vecs_1subj(Probe1_path, Probe2_path, cond_name_list, scan_window)

%% Load in Windowed Data

% Load the NIRS data from Probe Set 1 (Channels 1-24)
load(Probe1_path,'-mat')
array1 = procResult.dc(:,1,:);

% Load the NIRS data from Probe Set 2 (Channels 25-46)
load(Probe2_path,'-mat')
array2 = procResult.dc(:,1,:);

% Save the NIRS data from both probes to a single matrix
array1 = reshape(array1,size(array1,1),size(array1,3));
array2 = reshape(array2,size(array2,1),size(array2,3));
full_array = [array1 array2];

% Save the event onsets from the aux timeseries data
% cond_name_list = {'kitty' 'bunny' 'dog' 'bear' 'foot' 'hand' 'mouth' 'nose'};
condition = struct('name',cond_name_list);
for cond_num = 1:length(cond_name_list),
    condition(cond_num).onsets = find(aux(:,cond_num+1));
    condition(cond_num).onsets = condition(cond_num).onsets(1:2:end);
end

% Store windowed timeseries data for each condition

scan_onset = scan_window(1);
scan_offset = scan_window(2);
window_length = scan_offset-scan_onset+1;

for cond_num = 1:length(cond_name_list),
    % Create a 3-d matrix ( TIME x CHAN x EVENT )
    win_data = nan(window_length,size(full_array,2),length(condition(cond_num).onsets));
    
    for i = 1:length(condition(cond_num).onsets),
        
        % Set the onset and offset for this trial (align scan window to the
        % trial onset marker)
        this_onset = condition(cond_num).onsets(i) + scan_onset;
        this_offset = condition(cond_num).onsets(i) + scan_offset;
        
        % Extract the windowed data of interest for this trial
        win_data(:,:,i) = full_array(this_onset:this_offset,:) ;
        
        % Re-zero the windowed data based on the onset value. Importantly,
        % there is a decision to make here about using the actual trial
        % onset or the onset of the scan window. Since the scan window is a
        % bit arbitrary, I'm concerned we might cut off an actual rise in
        % the response prior to onset. The principled approach, IMO is to
        % use the trial onset which might result in a set of data all above
        % zero, but that is representative of the real response.
        win_data(:,:,i) = win_data(:,:,i) - repmat(full_array(condition(cond_num).onsets(i),:),window_length,1);
    end
    
    % Save the windowed data for all trials of this condition to the struct
    condition(cond_num).windowed_data = win_data;
    % Average across all trials to get the average time series for this
    % condition in each channel
    condition(cond_num).window_averages = nanmean(win_data,3);
end

%% Compute some sort of summary statistic for the multi-channel patterns

for cond_num = 1:length(cond_name_list), 
    condition(cond_num).ts_average = nanmean(condition(cond_num).window_averages)';
    condition(cond_num).second_deriv = nanmean(diff(diff(condition(cond_num).window_averages)))';
end

subj_data = condition;

%save(['subjdata/' Probe1_path(7:(end-16)) '.mat'])

