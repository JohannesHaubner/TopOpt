%
% (C) Johannes Haubner, 2020
%
% plot control corresponding to dof x

function plot_rho_DG0(x)

global v t 

    c = d2c(x);
    vx = v(:,1)';
    vy = v(:,2)';
    tt = t';
    vtt = [vx(tt(:)); vy(tt(:))];
    vttT = reshape(1:size(vtt,2), [3 size(vtt,2)/3]);
    tmp = [c'; c'; c'];
    trisurf(vttT',vtt(1,:),vtt(2,:),tmp(:));
    %options for plots
    colormap_tum = 1/255* [0 0 0
        0 51 89
        0 101 189
        218 215 213
        227 114 34];
    colormap_tum_1 = [linspace(colormap_tum(1,1), colormap_tum(2,1), 25)' ...
        linspace(colormap_tum(1,2), colormap_tum(2,2), 25)' ...
        linspace(colormap_tum(1,3), colormap_tum(2,3), 25)'];
    colormap_tum_2 = [linspace(colormap_tum(2,1), colormap_tum(3,1), 25)' ...
        linspace(colormap_tum(2,2), colormap_tum(3,2), 25)' ...
        linspace(colormap_tum(2,3), colormap_tum(3,3), 25)'];
    colormap_tum_3 = [linspace(colormap_tum(3,1), colormap_tum(4,1), 25)' ...
        linspace(colormap_tum(3,2), colormap_tum(4,2), 25)' ...
        linspace(colormap_tum(3,3), colormap_tum(4,3), 25)'];
    colormap_tum_4 = [linspace(colormap_tum(4,1), colormap_tum(5,1), 25)' ...
        linspace(colormap_tum(4,2), colormap_tum(5,2), 25)' ...
        linspace(colormap_tum(4,3), colormap_tum(5,3), 25)'];
    colormap_tum_all = [colormap_tum_1;
        colormap_tum_2(2:end,:);
        colormap_tum_3(2:end,:);
        colormap_tum_4(2:end,:)];
    colormap(colormap_tum_all);
    caxis([-2 2]);
    zlim([-2 2]);
end

