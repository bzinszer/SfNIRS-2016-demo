function plot_tiled_data(data_array,type,titletext,xlabtext)

%% Configure for plotting the channels
display_dims = get(0,'MonitorPosition');
num_panels = size(data_array,2);

vert_tiles = ceil(sqrt(num_panels / (display_dims(3)/display_dims(4))));
horiz_tiles = ceil(num_panels / vert_tiles);


switch type
    case 'bar'
        myplot=@(plotdat) bar(plotdat);
    case 'dot'
        myplot=@(plotdat) plot(plotdat,'o');
    otherwise
        myplot=@(plotdat) plot(plotdat);
end

for curr_panel = 1:size(data_array,2);
    subplot(vert_tiles,horiz_tiles,curr_panel)    
    myplot(data_array(:,curr_panel))
    title(sprintf('%s #%g',titletext,curr_panel))
    xlabel(xlabtext)
    ylim([min(data_array(:)) max(data_array(:))]);
    xlim([0 size(data_array,1)]);
end