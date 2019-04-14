function [varargout] = gpr_domain_fast(w,x,y,S,xt,domain)
% GPR_SOLVE - Solve GP regression problem
%
% Syntax:
%   [...] = gpr_domain_fast(w,x,y,S,xt,domain)
%
% In:
%   w      - Log-parameters (sigma2, theta)
%   x      - Training inputs
%   y      - Training outputs
%   S      - Spectral density function, @(w,theta) ...
%   xt/dS  - Test inputs / Spectral density derivatives
%   domain - The domain struct in the form make_domain gives it
%
% Out (if xt is empty or a cell array of funvtion handles):
%
%   e     - Negative log marginal likelihood
%   eg    - ... and its gradient
%
% Out (if xt is not empty):
%
%   Eft   - Predicted mean
%   Varft - Predicted marginal variance
%   Covft - Predicted joint covariance matrix
%   lb    - 95% confidence lower bound
%   ub    - 95% confidence upper bound
%
% Description:
%   Consider the following GP regression [1] problem:
%
%       f ~ GP(0,k(x,x')), x \in \Omega
%       f(x) = 0,          x \in \partial\Omega
%     y_i = f(x_i) + e_i,  i=1,2,...,n,
%
%   where k(x,x') = k_theta(x,x') and e_i ~ N(0,sigma2). The domain \Omega
%   is known a priori. The covariance function k(.,.) is required to be
%   stationary, and instead of the kernel, its spectral density function is
%   passed to this method in form of a function handle.
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
  
  
%% Set up basis functions

  % The eigenbasis
  eigenval = domain.eigenval;
  eigenfun = domain.eigenfun;
  NN       = domain.NN;
  
  % The eigenvalues
  lambda = eigenval(NN);
  
  % Evaluate Phi for the observations
  Phi = eigenfun(NN,x);

  % Pre-calculate basis functions
  PhiPhi = Phi'*Phi;        % O(nm^2)
  Phiy   = Phi'*y(:);
  

%% Optimize hyper-parameters  
  
  if iscell(xt)

    dS = xt;
    [e,eg] = opthyperparams(w,y,lambda,Phiy,PhiPhi,S,dS);
    varargout = {e,eg};
    return
    
  end
  
  
%% Solve the GP regression problem
  
  % Evaluate the Phi for test inputs
  Phit   = eigenfun(NN,xt);

  % Extract parameters
  sigma2 = exp(w(1));
  theta = exp(w(2:end));

  % Solve GP with optimized hyperparameters and 
  % return predictive mean and variance
  k = S(sqrt(lambda),theta);
  L = chol(PhiPhi + diag(sigma2./k),'lower');
  foo = (L'\(L\Phiy));
  Eft = Phit*foo;
  Varft = sigma2*sum((Phit/L').^2,2);

  % Return
  varargout = {Eft,Varft};
  
end
  
function [e,eg] = opthyperparams(w,y,lambda,Phiy,PhiPhi,S,dS)

  % Initialize
  e  = nan;
  eg = nan(1,numel(w));
  
  % Extract parameters
  sigma2 = exp(w(1));
  theta = exp(w(2:end));
  
  % Evaluate the spectral density
  k = S(sqrt(lambda),theta);
  
  % Number of n=observations and m=basis functions
  n = numel(y);
  m = size(Phiy,1);
  
  % Calculate the Cholesky factor
  [L,p] = chol(PhiPhi + diag(sigma2./k),'lower');  % O(m^3)
  
  % Check if pos. def
  if p>0, return; end
  
  % Evaluate all parts
  v = L\Phiy; % Phiy = (Phi'*y);
  yiQy = (y'*y - v'*v)/sigma2;
  logdetQ = (n-m)*log(sigma2) + sum(log(k)) + 2*sum(log(diag(L)));
  
  % Return approx. negative log marginal likelihood
  e = .5*yiQy + .5*logdetQ + .5*n*log(2*pi);
  
  % Precalculate
  vv = L'\v;
  LLk = L'\(L\diag(1./k)); % O(m^3)
  
  % Return if no derivatives requested
  if nargout==1 || isnan(e), return; end
  
  % For the covariance function hyperparameters
  for j=2:numel(w)
    
    % Should we skip this?
    %if opt(j)==false, continue; end
    
    % Evaluate the partial derivative
    dk = dS{j-1}(sqrt(lambda),theta);
        
    % Evaluate parts
    dlogdetQ = sum(dk./k) - sigma2*sum(diag(LLk).*dk./k);
    dyiQy = -vv'*diag(dk./k.^2)*vv;
        
    % The partial derivative
    eg(j) = .5*dlogdetQ + .5*dyiQy;
    
  end
  
  % Gradient of noise magnitude, sigma2
  
  % Evaluate parts
  dlogdetQ = (n-m)/sigma2 + sum(diag(LLk));
  dyiQy    = vv'*diag(1./k)*vv/sigma2 - yiQy/sigma2;
  
  % For the measurement noise
  eg(1)  = .5*dlogdetQ + .5*dyiQy;
    
  % Remove those gradient that was not calculated
  eg(isnan(eg)) = [];
  
  % Account for the log-transformed values
  eg(:) = exp(w(:)).*eg(:);
  
end