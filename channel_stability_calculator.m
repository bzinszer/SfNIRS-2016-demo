function [stable_chans, v] = channel_stability_calculator(subj_struct, P)

% Get info about subject
conditions = {'kitty' 'bunny' 'dog' 'bear' 'foot' 'hand' 'mouth' 'nose'};
num_cond = length(conditions);
num_blocks = size(subj_struct.Condition(1).windowed_data,3);

% Prep variables for the stability estimate
voxel_stability_data = nan(length(subj_struct(1).all_chans),12*8);
labels = repmat(conditions,1,12);

for block = 1:12,
    for condition = 1:num_cond,
        col_index = (block-1)*num_cond+condition;
        voxel_stability_data(:,col_index) = mean(subj_struct(1).Condition(condition).windowed_data(:,:,block)*10000,1)';
    end
end

% Run Pearson correlation-based channel stability estimate
v=StatMap_ST(voxel_stability_data',labels,'Pearson');


if P>1 && P<100,
    % Find the best P-th percentile of channels if P is >1
    stability_cutoff = prctile(v,P);
elseif P<1 && P>-1,
    % Use the raw value of P if P is a correlation
    stability_cutoff = P;
else
    % Otherwise (if P=<-1 or P>=100, just return all channels
    disp(['WARNING: P=' P ' is not a valid stability threshold.'])
    disp(['For absolute threshold, use -1<P<1. For percentile, use 1<P<100.'])
    stability_cutoff = min(v);
end
stable_chans = find(v>=stability_cutoff);

