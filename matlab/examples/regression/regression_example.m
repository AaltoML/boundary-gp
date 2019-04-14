%% Example of Gaussian process regression in a star-shaped domain
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

  addpath ../../
  clear

  
%% Make domain

  % Read image
  im = rgb2gray(imread('../shapes/star.png'));
  im = imresize(im,.25);
  foo = 255*ones(size(im)+2);
  foo(2:end-1,2:end-1) = im;
  
  % Make domain
  domain = make_domain(foo,[0 1],[0 1],256);
  
  % For visualization
  boundary = domain.poly(:,1:1:end)';
  
  
%% Load data  

  % Load training inputs and outputs
  x = load('star_train_X');
  y = load('star_train_Y');
  
  % Set test points used in visualization
  [X1,X2] = meshgrid(linspace(domain.x1(1),domain.x1(end),1000), ...
                     linspace(domain.x2(1),domain.x2(end),1000));
   xt = [X1(:) X2(:)];
  
   
%% Set up the GP prior

  % Intial parameters
  param = [.1^2 1 1];
  nu = 3/2;
  d = 2;

  % The spectral density of the Matern covariance function
  S = @(w,p) p(1) * (2^d * pi^(d/2) * gamma(nu + d/2) * (2*nu)^nu / (gamma(nu) * p(2)^(2*nu))) * ...
        (2*nu/p(2)^2 + w.^2).^(-nu-d/2);
  
  % Derivative of Matern kernel w.r.t. magnSigma2  
  dS{1} = @(w,p) S(w,p)/p(1);
  
  % Derivative of Matern kernel w.r.t. lengthScale  
  dS{2} = @(w,p) -2*nu/p(2)*S(w,p) + ...
                4*nu/p(2)^3*( nu + d/2 ) ./ ( 2*nu/p(2)^2 + w.^2 ).*...
                S(w,p);

            
%% Solve GP regression problem            
            
  % Optimize hyperparameters
  opts = optimset('GradObj','on','display','iter');
  
  % Optimize hyperparameters w.r.t. log marginal likelihood
  warning off
  [w,ll] = fminunc(@(w) gpr_domain_fast(w,x(:,:,1),y(:,1),S,dS,domain), log(param),opts);
  warning on
  
  % Do GPR
  tic
  [Eft,Varft] = gpr_domain_fast(w,x(:,:,1),y(:,1),S,xt,domain);
  toc

  
%% Visualize results

  % Reshape
  f = reshape(Eft,size(X1));

  % Make mask
  mask = (Varft(:)~=0);  
  
  % Color limits
  clims = [-1 1]*max(abs(f(:))); % Adaptive
  
  % Crop away the area outside the hexagon (for nicer plots)
  RGB = ind2rgb(ceil(64*(f-min(clims))/(max(clims)-min(clims))),parula(64));
  RGB = reshape(RGB,[],3);
  ind = ~mask(:);
  RGB([ind ind ind]) = ones(sum(ind),3);
  RGB = reshape(RGB,size(f,1),size(f,2),3);

  figure(1); clf; hold on
    h1=imagesc(domain.x1,domain.x2,RGB);

    h2=scatter(x(:,1),x(:,2),50,y,'filled','MarkerEdgeColor','k');
    h3=plot(domain.poly(1,[1:3:end end]), ...
            domain.poly(2,[1:3:end end]),'-k','LineWidth',1);
    box on
    set(gca,'layer','top')
    axis equal square tight
    legend([ h2 h3],'Noisy observations','Boundary')


    
    