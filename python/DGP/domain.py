# Copyright 2018-2019 Manon Kok and Arno Solin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

class gp_domain:

    def __init__(self, mask, xlim, ylim, m):
        """
        Make domain from mask.
        """

        # Assert that the mask represents a square area
        assert mask.shape[0]==mask.shape[1]
        assert xlim==ylim

        # Composition of the stencil matrix is based on the 9-point rule
        I,J = np.where(mask);

        # Values for constructing sparse array
        row = np.array(1);
        col = np.array(1);
        val = np.array(1);

        # Define the operator
        OP = np.array([[1/6,2/3,1/6],[2/3,-10/3,2/3],[1/6,2/3,1/6]])

        for k in range(0,len(I)):
          for di in [-1,0,1]:
            for dj in [-1,0,1]:
              i = np.where((I==I[k]+di) & (J==J[k]+dj))
              row = np.append(row,k*np.ones(i[0].shape))
              col = np.append(col,i[0])
              val = np.append(val,OP[di+1,dj+1]*np.ones(i[0].shape))

        # Discard placeholders
        row = row[1:]
        col = col[1:]
        val = val[1:]

        # Scale by step size
        h = (xlim[1]-xlim[0])/(mask.shape[1]+1)
        val /= h**2

        # Construct the pencil matrix
        S_h = coo_matrix((val, (row, col)), shape=(len(I), len(I)))

        # Solve eigenvalue problem
        mu,V = eigsh(S_h, k=m, which='LA')

        # Better approximations of the eigenvalues
        self.hlambda = np.flipud(2*mu / (np.sqrt(1 + mu*h**2/3) + 1));
  
        # Address scaling issues
        V = V * 1/h;

        # Expand size to match mask
        Vsquare = np.zeros((mask.shape[0]*mask.shape[1],m))
        ind, = np.where(mask.flatten())
        for i in range(len(ind)):
            Vsquare[ind[i],:] = V[i,:]

        # Store eigenvectors and mask
        self.V = np.fliplr(Vsquare)
        self.mask = mask
        self.x1 = np.linspace(xlim[0],xlim[1],mask.shape[1])
        self.x2 = np.linspace(ylim[0],ylim[1],mask.shape[0])
        self.m = m
        self.S_h = S_h

    def eigenfun(self,x):
        """
        Evaluate eigenfunctions.
        """
        foo = self.V.reshape((self.mask.shape[0],self.mask.shape[1],self.m))
        U = np.zeros((x.shape[0],self.m))
        for k in range(x.shape[0]):
            i = np.abs(self.x1-x[k,0]).argmin()
            j = np.abs(self.x2-x[k,1]).argmin()
            U[k,:] = foo[j,i,:].flatten()
        return U

    def eigenval(self):
        """
        Evaluate eigenvalues.
        """
        return -self.hlambda
