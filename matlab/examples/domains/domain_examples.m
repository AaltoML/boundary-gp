%% Examples of the eigenbasis in different domains
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

  % List example domains
  filename{1} = 'star.png';
  filename{2} = 'car.png';
  filename{3} = 'pika.png';
  filename{4} = 'aalto.png';
  filename{5} = 'japan.png';
  filename{6} = 'steam.png';
  filename{7} = 'cheese.png';
  filename{8} = 'fish.png';
  filename{9} = 'mug.png';
  filename{10} = 'crab.png';
  filename{11} = 'christmas.png';  

  
%% Make domain

  % Read image
  im = rgb2gray(imread(['../shapes/' filename{1}]));
  im = imresize(im,.25);
  foo = 255*ones(size(im)+2);
  foo(2:end-1,2:end-1) = im;
  
  % Make domain
  domain = make_domain(foo,[0 1],[0 1],256);
  
  
%% Visualize the 25 first basis functions

  % Evaluate basis functions
  [X1,X2] = meshgrid(domain.x1,domain.x2);  
  V = domain.eigenfun(domain.NN,[X1(:) X2(:)]);

  % Visualize the results
  figure(1); clf
  
  for j=1:25
    
    subplot(5,5,j); hold on

    % Pick the values
    v = V(:,j);
    u = reshape(v,size(X1));
    
    % Crop away the area outside the hexagon (for nicer plots)
    RGB = ind2rgb(ceil(64*(u-min(u(:)))/(max(u(:))-min(u(:)))),parula(64));
    RGB = reshape(RGB,[],3);
    ind = ~u(:);
    RGB([ind ind ind]) = ones(sum(ind),3);
    RGB = reshape(RGB,size(u,1),size(u,2),3);
    
    % Visualize
    imagesc(domain.x1,domain.x2,RGB)
    plot(domain.poly(1,:),domain.poly(2,:),'-k')
    caxis([-1 1]*max(abs(v)))
    axis ij square equal tight off
    
  end
  
  set(gcf,'Color','w')

  
%% Visualize the stencil matrix

  figure(1);clf
  spy(domain.S_h,2,'k')
  frame = getframe(gca);
  axis equal square

  
%% Make a lot of figures

  % A two-dimensional domain
  d = 2;

  % The spectral density of the squared exponential covariance function
  Sse = @(w,p) p(1)*sqrt(2*pi)^d*p(2)^d*exp(-w.^2*p(2)^2/2);   
  
  % The spectral density of the Matern
  Smat = @(w,p,nu) p(1) * (2^d * pi^(d/2) * gamma(nu + d/2) * (2*nu)^nu / (gamma(nu) * p(2)^(2*nu))) * ...
          (2*nu/p(2)^2 + w.^2).^(-nu-d/2);  
  
  % Different covariance functions (their spectral densities)
  S = {@(w,p) Smat(w,p,1/2), ...
       @(w,p) Smat(w,p,3/2), ...
       @(w,p) Smat(w,p,5/2), ...
       @(w,p) Sse(w,p)};
 
  % Hyperparameters
  lengthScales = [.01 .1];
  magnSigma2 = 1;
  
  Snames = {'mat12','mat32','mat52','sexp'};
  lennames = {'short','long'};
  
  % Evaluate basis functions
  [X1,X2] = meshgrid(domain.x1,domain.x2);  

  % For each domain
  for i=1:length(filename)
    
    figure(i+1); clf
    set(gcf,'color','w')
      
    % Read image
    im = rgb2gray(imread(['../shapes/' filename{i}]));
    im = imresize(im,.25);
    foo = 255*ones(size(im)+2);
    foo(2:end-1,2:end-1) = im;
  
    % Make domain
    domain = make_domain(foo,[0 1],[0 1],256);
      
    % Evaluate eigenfunctions
    V = domain.eigenfun(domain.NN,[X1(:) X2(:)]);
    
    clf
    
    % For each kernel
    for j=1:length(S)
        
      % For each lengthScale
      for k=1:length(lengthScales) 
        
        % Fix seed
        rng(i+j+k,'twister')
        
        % Draw sample
        lambda = domain.eigenval(domain.NN);
        kk = S{j}(sqrt(lambda),[magnSigma2 lengthScales(k)]);
        f = V*diag(sqrt(kk))*randn(domain.m,1);
        f = reshape(f,size(X1));
        
        % Color limits
        clims = [-1 1]*max(abs(f(:))); % Adaptive
        
        % Crop away the area outside the hexagon (for nicer plots)
        RGB = ind2rgb(ceil(64*(f-min(clims))/(max(clims)-min(clims))),parula(64));
        RGB = reshape(RGB,[],3);
        ind = ~domain.mask(:);
        RGB([ind ind ind]) = ones(sum(ind),3);
        RGB = reshape(RGB,size(f,1),size(f,2),3);
        
        % Visualize
        subplot(length(lengthScales),length(S),(k-1)*length(S)+j)
        hold on
        imagesc(domain.x1,domain.x2,RGB)
        plot(domain.poly(1,:),domain.poly(2,:),'-k','LineWidth',2)
        caxis(clims)
        text(.5,1.1,Snames{j},'HorizontalAlign','center')
        axis ij equal tight off        
        drawnow
        pause(.5)
        
      end
    end
  end
  
