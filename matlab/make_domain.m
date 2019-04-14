function domain = make_domain(im,xlim,ylim,m)
%% make_domain - Laplace operator eigendecomposition in a 2D mask domain
% 
% Syntax:
%   domain = make_domain(im,xlim,ylim,m)
%
% In:
%   im    - Grayscale mask image (128 cutoff)
%   xlim  - Physical image dimensions in x-dim
%   ylim  - Physical image dimensions in y-ydim
%   m     - Number of basis functions
%      
% Out (struct with fields):
%   eigenval - Function handle: eigenval(n)
%   eigenfun - Function handle: eigenfun(n,x)
%   NN       - Indices to evaluate the handles at
%   x1       - Discretization in x-dim
%   x2       - Discretization in y-dim
%   poly     - Domain boundary polygon (for visualization)
%   mask     - Domain mask (for visualization)
%   S_h      - Sparse stencil matrix (for debugging)
% 
% Description:
%   This code returns the eigendecomposition of the (negative) Laplacian in
%   a 2D domain specified by a grayscale bitmap mask image. The image is
%   interpreted such that black areas are inside the domain and white areas
%   are outside.
%
% References:
%   [1] Arno Solin and Manon Kok (2019). Know your boundaries: Constraining 
%       Gaussian processes by variational harmonic features. Proceedings 
%       of the 22nd International Conference on Artificial Intelligence and 
%       Statistics (AISTATS). Naha, Okinawa, Japan.
%
% Copyright 2018-2019 Manon Kok and Arno Solin
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
% http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

%%
    
  % Discretization in x
  Nx = size(im,2);
  hx = (xlim(2)-xlim(1))/(Nx-1);

  % Discretization in y
  Ny = size(im,1); 
  hy = (ylim(2)-ylim(1))/(Ny-1);
    
  % Discretize
  x1 = linspace(xlim(1),xlim(2),Nx);
  x2 = linspace(ylim(1),ylim(2),Ny);
  
  % Check that hx and hy are equal
  if hx~=hy, error('Discretization in x and y should be equal.'), end

  % Check which are inside
  u = double(im<127);
  
  % Get contour
  [cc, hh] = contour(x1, x2, im, [127 127], 'k'); % Overlay contour line
  cc(:,cc(1,:)==127) = nan;
  poly = [cc(1,2:end); cc(2, 2:end)];

  % These can be useful
  domain.x1 = x1;
  domain.x2 = x2;
  domain.poly = poly;
  domain.m = m;
  domain.mask = u;
  
  
%% Create operator    
  
  % Laplacian: The composition of the stencil matrix is based on the
  % 9-point rule as explained here: 
  % https://uk.mathworks.com/help/symbolic/examples/eigenvalues-of-the-laplace-operator.html
  [I,J] = find(u ~= 0);
  n = length(I);
  S_h = sparse(n,n);
  for k=1:n
      i = I(k);
      j = J(k);
      S_h(k,I==i+1 & J==j+1)=  1/6; %#ok
      S_h(k,I==i+1 & J== j )=  2/3; %#ok
      S_h(k,I==i+1 & J==j-1)=  1/6; %#ok
      S_h(k,I== i  & J==j+1)=  2/3; %#ok
      S_h(k,I== i  & J== j )=-10/3; %#ok
      S_h(k,I== i  & J==j-1)=  2/3; %#ok
      S_h(k,I==i-1 & J==j+1)=  1/6; %#ok
      S_h(k,I==i-1 & J== j )=  2/3; %#ok
      S_h(k,I==i-1 & J==j-1)=  1/6; %#ok
  end
  S_h = S_h./hx^2;
  
  % Here, S_h is the (sparse) stencil matrix. Use eigs that handles sparse 
  % matrices to compute the three largest eigenvalues.
  [V,D,FLAG] = eigs(S_h,m,'la');

  % Better approximations of the eigenvalues
  mu = diag(D);
  hlambda = 2*mu ./ (sqrt(1 + mu*hx^2/3) + 1);
  
  % Address scaling issues
  V = V * 1/hx;
    
  % The eigenvectors (functions) in the size of the domain
  Vsquare = nan(numel(u),m);
  Vsquare(u(:)~=0,:) = V;

  % Ignore nans
  Vsquare(isnan(Vsquare)) = 0;

  % Store
  domain.S_h = S_h;
  
%% Set up the interpolation schemes

  % Eigenfunction ranks
  domain.NN = (1:m)';

  % Define eigenvalues of the negative Laplacian 
  domain.eigenval = @(n) return_eigenval(n,hlambda);
  
  % Define eigenfunctions of the negative Laplacian 
  domain.eigenfun = @(n,x) return_eigenfun(n,x,Vsquare,x1,x2);
  
end
  
function [v]=return_eigenfun(n,x,Vsquare,x1,x2)

  % Allocate space
  v = ones(size(x,1),size(n,1));

  % For each
  for j=1:size(n,1)

    % The eigenfunction of the
    ind = n(j,1);
      
    % Interpolate to neares evaluation points
    v(:,j) = interp2(x1,x2,reshape(Vsquare(:,ind),numel(x1),[]),x(:,1),x(:,2),'nearest');

  end
  
  v(isnan(v)) = 0;
  
end

function [lambda]=return_eigenval(n,hlambda)

  % Just return
  lambda = -hlambda(n(:,1));
  
end