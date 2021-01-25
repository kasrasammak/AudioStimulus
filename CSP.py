#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 16:39:39 2020

Common Spatial Pattern Algorithm for Filter Construction

CSP is a statistical algorithm for construction of spatial filters.  These filters may be used in constructing feature vectors, or other analyses, where it is useful to remove as much noise as possible from a signal.  For example, in a system that classifies between two movements, a given reading will include information from the current task-related activity as well as noise from the other, non-task-related activity.  The classifier will be able to distinguish between the two classes better if a spatial filter is used to reduce noise from the non-task-related activity.

This implementation is a written in python.  To use, simply call CSP(C1,C2,..,Cn) where Cx has the shape (number of samples x feature vector).  That is, each class Cx passed to CSP should be an array where each row is a unique test case/sample.  Each sample will then be a feature vector of some given dimension.  As with other classifiers, the feature vector does not have a required dimension, but it must be the same across all classes in order for the filters to be constructed.

For anyone curious to read more about the algorithm, I based the code on the following paper:

Wang, Yunhua, Patrick Berg, and Michael Scherg. "Common spatial subspace decomposition applied to analysis of brain responses under multiple task conditions: a simulation study." Clinical Neurophysiology 110.4 (1999): 604-614. (available online through Elsevier at https://www.sciencedirect.com/science/article/pii/S138824579800056X)

@author: Seth Polsley
"""

# Common Spatial Pattern implementation in Python, used to build spatial filters for identifying task-related activity.
import numpy as np
import scipy.linalg as la

# CSP takes any number of arguments, but each argument must be a collection of trials associated with a task
# That is, for N tasks, N arrays are passed to CSP each with dimensionality (# of trials of task N) x (feature vector)
# Trials may be of any dimension, provided that each trial for each task has the same dimensionality,
# otherwise there can be no spatial filtering since the trials cannot be compared
def CSP(*tasks):
	if len(tasks) < 2:
		print("Must have at least 2 tasks for filtering.")
		return (None,) * len(tasks)
	else:
		filters = ()
		# CSP algorithm
		# For each task x, find the mean variances Rx and not_Rx, which will be used to compute spatial filter SFx
		iterator = range(0,len(tasks))
		for x in iterator:
			# Find Rx
			Rx = covarianceMatrix(tasks[x][0])
			for t in range(1,len(tasks[x])):
				Rx += covarianceMatrix(tasks[x][t])
			Rx = Rx / len(tasks[x])

			# Find not_Rx
			count = 0
			not_Rx = Rx * 0
			for not_x in [element for element in iterator if element != x]:
				for t in range(0,len(tasks[not_x])):
					not_Rx += covarianceMatrix(tasks[not_x][t])
					count += 1
			not_Rx = not_Rx / count

			# Find the spatial filter SFx
			SFx = spatialFilter(Rx,not_Rx)
			filters += (SFx,)

			# Special case: only two tasks, no need to compute any more mean variances
			if len(tasks) == 2:
				filters += (spatialFilter(not_Rx,Rx),)
				break
		return filters

# covarianceMatrix takes a matrix A and returns the covariance matrix, scaled by the variance
def covarianceMatrix(A):
	Ca = np.dot(A,np.transpose(A))/np.trace(np.dot(A,np.transpose(A)))
	return Ca

# spatialFilter returns the spatial filter SFa for mean covariance matrices Ra and Rb
def spatialFilter(Ra,Rb):
	R = Ra + Rb
	E,U = la.eig(R)

	# CSP requires the eigenvalues E and eigenvector U be sorted in descending order
	ord = np.argsort(E)
	ord = ord[::-1] # argsort gives ascending order, flip to get descending
	E = E[ord]
	U = U[:,ord]

	# Find the whitening transformation matrix
	P = np.dot(np.sqrt(la.inv(np.diag(E))),np.transpose(U))

	# The mean covariance matrices may now be transformed
	Sa = np.dot(P,np.dot(Ra,np.transpose(P)))
	Sb = np.dot(P,np.dot(Rb,np.transpose(P)))

	# Find and sort the generalized eigenvalues and eigenvector
	E1,U1 = la.eig(Sa,Sb)
	ord1 = np.argsort(E1)
	ord1 = ord1[::-1]
	E1 = E1[ord1]
	U1 = U1[:,ord1]

	# The projection matrix (the spatial filter) may now be obtained
	SFa = np.dot(np.transpose(U1),P)
	return SFa.astype(np.float32)