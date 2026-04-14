#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 08:22:43 2026

@author: nhunterjr
"""

from cva_run import CVARun
run = CVARun.load('s01d11_cva.h5')

from cva_phi_cluster import CVAPhiCluster
clust = CVAPhiCluster(run)
clust.fit()
clust.plot_all()