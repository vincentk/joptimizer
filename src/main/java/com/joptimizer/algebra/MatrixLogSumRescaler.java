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

import org.apache.commons.lang3.ArrayUtils;
import org.apache.log4j.Logger;

import cern.colt.function.IntIntDoubleFunction;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;

/**
 * Calculates the matrix rescaling factors so that the scaled
 * matrix has its entries near to unity in the sense that the sum of the
 * squares of the logarithms of the entries is minimized.
 * 
 * @TODO: add documentation
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 * @see Xin Huang, Preprocessing and Postprocessing in Linear Optimization
 * @see A. Chang and J.K. Reid. On the automatic scaling of matrices for Gaussian elimination, 
 * 		Journal of the Institute of Mathematics and Its Applications 10 (1972)
 * @see Gajulapalli, Lasdon "Scaling Sparse Matrices for Optimization Algorithms"
 */
public class MatrixLogSumRescaler implements MatrixRescaler {

	private double base = 10;
	private Logger logger = Logger.getLogger(this.getClass().getName());
	
	public MatrixLogSumRescaler(){
	}
	
	public MatrixLogSumRescaler(double base){
		this.base = base;
	}
	
	/**
	 * Gauss-Seidel scaling for a sparse matrix: 
	 * <br>AScaled = U.A.V, with A mxn matrix, U, V diagonal.
	 * Returns the two scaling matrices
	 * <br>U[i,i] = base^x[i], i=0,...,m
	 * <br>V[i,i] = base^y[i], i=0,...,n
	 * 
	 * @see Gajulapalli, Lasdon "Scaling Sparse Matrices for Optimization Algorithms", algorithms 1 and 2
	 */
	public DoubleMatrix1D[] getMatrixScalingFactors(DoubleMatrix2D A) {
		int m = A.rows();
		int n = A.columns();
		final double log10_b = Math.log10(base);

		//Setup for Gauss-Seidel Iterations
		final int[] R = new int[m];
		final int[] C = new int[n];
		final double[] t = new double[1];
		final double[] a = new double[m];
		final double[] b = new double[n];
		final boolean[][] Z = new boolean[m][n];
		A.forEachNonZero(new IntIntDoubleFunction() {
			public double apply(int i, int j, double aij) {
				R[i] = R[i] + 1;
				C[j] = C[j] + 1;
				Z[i][j] = true;
				t[0] = -(Math.log10(Math.abs(aij)) / log10_b + 0.5);// log(b, x) = log(k, x) / log(k, b)
				a[i] = a[i] + t[0];
				b[j] = b[j] + t[0];
				return aij;
			}
		});
		
		for(int i=0; i<m; i++){
			a[i] = a[i] / R[i];
		}
		for(int j=0; j<n; j++){
			b[j] = b[j] / C[j];
		}
		
//		log.debug("a: " + ArrayUtils.toString(a));
//		log.debug("b: " + ArrayUtils.toString(b));
//		log.debug("R: " + ArrayUtils.toString(R));
//		log.debug("C: " + ArrayUtils.toString(C));
		
		int[] xx = new int[m];
		int[] yy = new int[n];
		int[] previousXX = null;
		int[] previousYY = null;
		boolean stopX = false;
		boolean stopY = false;
		int maxIteration = 3;
		int l=0;
		for(l=0; l<=maxIteration && !(stopX && stopY); l++){
			double[] tt = new double[m];
			System.arraycopy(a, 0, tt, 0, m);
			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {
					boolean[] ZI = Z[i]; 
					if(ZI[j]){
						tt[i] = tt[i] - ((double) yy[j]) / R[i];
					}
				}
			}
			for (int k = 0; k < m; k++) {
				xx[k] = (int)Math.round(tt[k]);
			}
			if(previousXX == null){
				previousXX = xx;
			}else{
				boolean allEquals = true;
				for (int k = 0; k < m && allEquals; k++) {
					allEquals = (xx[k] == previousXX[k]);
				}
				stopX = allEquals;
				previousXX = xx;
			}
			
			tt = new double[n];
			System.arraycopy(b, 0, tt, 0, n);
			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {
					if(Z[i][j]){
						tt[j] = tt[j] - ((double) xx[i]) / C[j];
					}
				}
			}
			for (int k = 0; k < n; k++) {
				yy[k] = (int)Math.round(tt[k]);
			}
			if(previousYY == null){
				previousYY = yy;
			}else{
				boolean allEquals = true;
				for (int k = 0; k < n && allEquals; k++) {
					allEquals = (yy[k] == previousYY[k]);
				}
				stopY = allEquals;
				previousYY = yy;
			}
		}
		
		if(l == maxIteration){
			//@TODO: just for test, remove this
			//throw new RuntimeException("max iterations reached");
		}
		
		logger.debug("xx: " + ArrayUtils.toString(xx));
		logger.debug("yy: " + ArrayUtils.toString(yy));
		
		DoubleMatrix1D u = new DenseDoubleMatrix1D(m);
		for (int k = 0; k < m; k++) {
			u.setQuick(k, Math.pow(base, xx[k]));
		}
		DoubleMatrix1D v = new DenseDoubleMatrix1D(n);
		for (int k = 0; k < n; k++) {
			v.setQuick(k, Math.pow(base, yy[k]));
		}
		DoubleMatrix1D[] ret = new DoubleMatrix1D[] { u, v };
		return ret;
	}
	
	/**
	 * Symmetry preserving scale factors
	 * @see Gajulapalli, Lasdon "Scaling Sparse Matrices for Optimization Algorithms", algorithm 3
	 */
	public DoubleMatrix1D getMatrixScalingFactorsSymm(DoubleMatrix2D A) {
		int n = A.rows();
		final double log10_b = Math.log10(base);
		
		final int[] x = new int[n];
		final double[] cHolder = new double[1];
		final double[] tHolder = new double[1];
		final int[] currentColumnIndexHolder = new int[] { -1 };
		IntIntDoubleFunction myFunct = new IntIntDoubleFunction() {
			public double apply(int i, int j, double pij) {
				int currentColumnIndex = currentColumnIndexHolder[0];
				// we take into account only the lower left subdiagonal part of Q (that is symmetric)
				if(i == currentColumnIndex){
				  //diagonal element
					//log.debug("i:" + i + ", j:" + currentColumnIndex + ": " + pij);
					tHolder[0] = tHolder[0] - 0.5 * (Math.log10(Math.abs(pij))/log10_b + 0.5);//log(b, x) = log(k, x) / log(k, b)
					cHolder[0] = cHolder[0] + 1;
				}else if (i > currentColumnIndex) {
				  //sub-diagonal elements
					//log.debug("i:" + i + ", j:" + currentColumnIndex + ": " + pij);
					tHolder[0] = tHolder[0] - 2 * (Math.log10(Math.abs(pij))/log10_b + 0.5) -2*x[i];//log(b, x) = log(k, x) / log(k, b)
					cHolder[0] = cHolder[0] + 2;//- 2*x[i]
				}
				return pij;
			}
		};
		
		//view A column by column
		for (int currentColumnIndex = n - 1; currentColumnIndex >= 0; currentColumnIndex--) {
			//log.debug("currentColumnIndex:" + currentColumnIndex);
			cHolder[0] = 0;//reset
			tHolder[0] = 0;//reset
			currentColumnIndexHolder[0] = currentColumnIndex;
			DoubleMatrix2D P = A.viewPart(0, currentColumnIndex, n, 1);
			P.forEachNonZero(myFunct);
			if(cHolder[0] > 0){
				x[currentColumnIndex] = (int)Math.round(tHolder[0] / cHolder[0]);
			}
		}
		
		//log.debug("x: " + ArrayUtils.toString(x));
		
		DoubleMatrix1D u = new DenseDoubleMatrix1D(n);
		for (int k = 0; k < n; k++) {
			u.setQuick(k, Math.pow(base, x[k]));
		}
		return u;
	}
	
	/**
	 * Check if the scaling algorithm returned proper results.
	 * Note that the scaling algorithm is for minimizing a given objective function of the original matrix elements, and
	 * the check will be done on the value of this objective function. 
	 * @param A the ORIGINAL (before scaling) matrix
	 * @param U the return of the scaling algorithm
	 * @param V the return of the scaling algorithm
	 * @param base
	 * @return
	 */
	public boolean checkScaling(final DoubleMatrix2D A, 
			final DoubleMatrix1D U, final DoubleMatrix1D V){
		
		final double log10_2 = Math.log10(base);
		final double[] originalOFValue = {0};
		final double[] scaledOFValue = {0};
		final double[] x = new double[A.rows()];
		final double[] y = new double[A.columns()];
		
		A.forEachNonZero(new IntIntDoubleFunction() {
			public double apply(int i, int j, double aij) {
				double v = Math.log10(Math.abs(aij)) / log10_2 + 0.5;
				originalOFValue[0] = originalOFValue[0] + Math.pow(v, 2);
				double xi = Math.log10(U.getQuick(i)) / log10_2;
				double yj = Math.log10(V.getQuick(j)) / log10_2;
				scaledOFValue[0] = scaledOFValue[0] + Math.pow(xi + yj + v, 2);
				x[i] = xi;
				y[j] = yj;
				return aij;
			}
		});
		
		originalOFValue[0] = 0.5 * originalOFValue[0];
		scaledOFValue[0]   = 0.5 * scaledOFValue[0];
		
		logger.debug("x: " + ArrayUtils.toString(x));
		logger.debug("y: " + ArrayUtils.toString(y));
		logger.debug("originalOFValue: " + originalOFValue[0]);
		logger.debug("scaledOFValue  : " + scaledOFValue[0]);
		return !(originalOFValue[0] < scaledOFValue[0]);
	}

}
