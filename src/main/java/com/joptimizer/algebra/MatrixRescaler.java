/*
 * Copyright 2011-2014 JOptimizer
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 */
package com.joptimizer.algebra;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

/**
 * Calculate the row and column scaling matrices R and T relative to a given
 * matrix A (scaled A = R.A.T). 
 * They may be used, for instance, to scale the matrix prior to solving a
 * corresponding set of linear equations.
 *  
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public interface MatrixRescaler {

	/**
	 * Calculates the R and T scaling factors (matrices) for a generic matrix A so that  A'(=scaled A) = R.A.T
	 * @return array with R,T 
	 */
	public DoubleMatrix1D[] getMatrixScalingFactors(DoubleMatrix2D A);

	/**
	 * Calculates the R and T scaling factors (matrices) for a symmetric matrix A so that  A'(=scaled A) = R.A.T
	 * @return array with R,T 
	 */
	public DoubleMatrix1D getMatrixScalingFactorsSymm(DoubleMatrix2D A);

	/**
	 * Check if the scaling algorithm returned proper results.
	 * @param AOriginal the ORIGINAL (before scaling) matrix
	 * @param U the return of the scaling algorithm
	 * @param V the return of the scaling algorithm
	 * @param base
	 * @return
	 */
	public boolean checkScaling(final DoubleMatrix2D AOriginal, final DoubleMatrix1D U, final DoubleMatrix1D V);

}