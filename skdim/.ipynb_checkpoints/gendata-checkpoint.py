#
# BSD 3-Clause License
#
# Copyright (c) 2020, Jonathan Bac
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
import numpy as np
from sklearn.utils.validation import check_random_state
from ._commonfuncs import randball as hyperBall

def hyperSphere(n_points, n_dim, center=[], random_state=None):
    """
    Generates a sample from a uniform distribution on an hypersphere surface
    """
    random_state_ = check_random_state(random_state)
    vec = random_state_.randn(n_points, n_dim)
    vec /= np.linalg.norm(vec, axis=1)[:,None]
    return vec

def hyperTwinPeaks(n_points, n_dim = 2, height = 1, random_state=None):
    """ 
    Translated from Kerstin Johnsson's R package intrinsicDimension
    """
    random_state_ = check_random_state(random_state)
    base_coord = random_state_.uniform(size=(n_points, n_dim))
    _height = height * np.prod(np.sin(2 * np.pi * base_coord),axis=1,keepdims=1)
    return np.hstack((base_coord, _height))

def swissRoll3Sph(Ns, Nsph, a = 1, b = 2, nturn = 1.5, h = 4, random_state=None):
    '''
    Generates a sample from a uniform distribution on a Swiss roll-surface, 
    possibly together with a sample from a uniform distribution on a 3-sphere
    inside the Swiss roll. Translated from Kerstin Johnsson's R package intrinsicDimension

    Parameters
    ----------

    Ns : int 
        Number of data points on the Swiss roll.

    Nsph : int
        Number of data points on the 3-sphere.

    a : int or float, default=1
        Minimal radius of Swiss roll and radius of 3-sphere.

    b : int or float, default=2
        Maximal radius of Swiss roll.

    nturn : int or float, default=1.5
        Number of turns of the surface. 

    h : int or float, default=4
        Height of Swiss roll.

    Returns
    -------
    
    np.array, (npoints x ndim)
    '''
    random_state_ = check_random_state(random_state)

    if Ns > 0:
        omega = 2 * np.pi * nturn
        dl = lambda r: np.sqrt(b**2 + omega**2 * (a + b * r)**2)
        ok = np.zeros(1)
        while sum(ok) < Ns:
            r_samp = random_state_.uniform(size = 3 * Ns)
            ok = random_state_.uniform(size = 3 * Ns) < dl(r_samp)/dl(1)

        r_samp = r_samp[ok][:Ns]
        x = (a + b * r_samp) * np.cos(omega * r_samp)
        y = (a + b * r_samp) * np.sin(omega * r_samp)
        z = random_state_.uniform(-h, h, size = Ns)
        w = np.zeros(Ns)

    else:
        x = y = z = w = np.array([])

    if Nsph > 0:
        sph = hyperSphere(Nsph, 4, random_state = random_state_) * a
        x = np.concatenate((x, sph[:, 0]))
        y = np.concatenate((y, sph[:, 1]))
        z = np.concatenate((z, sph[:, 2]))
        w = np.concatenate((w, sph[:, 3]))
    
    return np.hstack((x[:,None], y[:,None], z[:,None], w[:,None]))

def lineDiskBall(n_points, random_state=None):
    """ 
    Generates a sample from a uniform distribution on a line, an oblong disk and an oblong ball
    Translated from ldbl function in Hideitsu Hino's package
    """
    random_state_ = check_random_state(random_state)

    line = np.hstack((np.repeat(0, 5 * n_points)[:,None], np.repeat(0, 5 * n_points)[:,None], random_state_.uniform(-0.5, 0,size=5 * n_points)[:,None]))
    disc = np.hstack((random_state_.uniform(-1, 1,(13 * n_points,2)), np.zeros(13 * n_points)[:,None]))
    disc = disc[~(np.sqrt(np.sum(disc**2,axis=1)) > 1),:]
    disc = disc[:, [0, 2, 1]]
    disc[:, 2] = disc[:, 2] - min(disc[:, 2]) + max(line[:, 2])

    fb = random_state_.uniform(-0.5, 0.5,size=(n_points * 100,3))
    rmID = np.where(np.sqrt(np.sum(fb**2,axis=1)) > 0.5)[0]

    if len(rmID) > 0:
        fb = fb[~(np.sqrt(np.sum(fb**2,axis=1)) > 0.5),:]

    fb = np.hstack((fb[:,:2], fb[:, [2]] + 0.5))
    fb[:, 2] = fb[:, 2] - min(fb[:, 2]) + max(disc[:, 2])

#     if _sorted:
#         fb = fb[order(fb[:, 2]),:]

    line2 = np.hstack((np.repeat(0, 5 * n_points)[:,None], np.repeat(0, 5 * n_points)[:,None], random_state_.uniform(-0.5, 0,size=5 * n_points)[:,None]))
    line2[:, 2] = line2[:, 2] - min(line2[:, 2]) + max(fb[:, 2])
    lineID = np.repeat(1, len(line))
    discID = np.repeat(2, len(disc))
    fbID = np.repeat(3, len(fb))
    line2ID = np.repeat(1, len(line2))
    x = np.vstack((line, disc, fb, line2))
    useID = np.sort(random_state_.choice(len(x), n_points,replace=False))
    x = x[useID,:]

    return x, np.concatenate((lineID, discID, fbID, line2ID),axis=0)[useID]