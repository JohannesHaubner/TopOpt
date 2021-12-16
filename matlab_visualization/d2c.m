function c = d2c(x)
%
% (C) Johannes Haubner, 2020
% 
% d2c: dofs to control
% define dofs x as degrees of freedom of a DG0 function on a uniform 
% rectangular mesh and interpolate it to a DG0 function on a uniform
% triangular mesh

c = repelem(x,2);
%c = x;
end

