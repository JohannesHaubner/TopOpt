% (C) Michael Ulbrich, TUM, 2020

% (ny-1)* (ny-1)*                                    ny*nx
% *nx+1    *nx+2                                     
%  :                                                 :
%  :                                                 :
% nx+1    nx+2    nx+3                               2nx
%   1       2       3       4       5       6   ...   nx

global t v nx ny

lx=1.5;
ly=1;

hx=lx/(nx-1);
hy=ly/(ny-1);
x=hx*[0:nx-1]';
y=hy*[0:ny-1]';
X=repmat(x,1,ny);
Y=repmat(y',nx,1);
v=[X(:),Y(:)];

t=zeros(2*(nx-1)*(ny-1),3);
z=repmat(nx*[0:ny-2],nx-1,1)+repmat([1:nx-1]',1,ny-1);
t(1:2:end,1)=z(:);
t(1:2:end,2)=z(:)+1;
t(1:2:end,3)=z(:)+nx;
t(2:2:end,1)=z(:)+1;
t(2:2:end,2)=z(:)+nx+1;
t(2:2:end,3)=z(:)+nx;
