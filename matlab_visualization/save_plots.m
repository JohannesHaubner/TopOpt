loc = '/home/joha/Documents/TopOpt/Output/matlab_controls_%d_%d.mat';
saveloc_3d =  '/home/joha/Documents/TopOpt/Output/3d_d_%d=%d';
saveloc_2d =  '/home/joha/Documents/TopOpt/Output/2d_d_%d=%d';

n = 5 % number of files

global nx ny

nx= 61; 
ny= 41; 
gentri_nx_ny;

for i= 1:n
    location = sprintf(loc,ny-1, i);
    sloc3d = sprintf(saveloc_3d, ny-1, i);
    sloc2d = sprintf(saveloc_2d, ny-1, i);
    l_x0 = load(location);
    l_x0 = l_x0.data';
    f1 = figure;
    plot_rho_DG0(l_x0);
    saveas(gcf,sloc3d, 'png');
    view(0,90);
    set(gca,'visible','off');
    saveas(gcf,sloc2d, 'png');
end