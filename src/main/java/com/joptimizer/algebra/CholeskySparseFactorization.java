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
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;

import cern.colt.function.IntIntDoubleFunction;
import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;

import com.joptimizer.util.ColtUtils;
import com.joptimizer.util.Utils;

/**
 * Cholesky factorization and inverse for symmetric and positive sparse matrix 
 * Q = L.L[T], with L left-lower triangular.
 * 
 * NOTE: this class allows to factorize a matrix the is filled only in its subdiagonal lower left part.
 * 
 * @see "Computing the Cholesky Factorization of Sparse Matrices" online free available pdf
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class CholeskySparseFactorization {
	private int dim;
	private SparseDoubleMatrix2D Q;
	private MatrixRescaler rescaler = null;
	private DoubleMatrix1D U;
	protected Algebra ALG = Algebra.DEFAULT;
	protected DoubleFactory2D F2 = DoubleFactory2D.dense;
	protected DoubleFactory1D F1 = DoubleFactory1D.dense;
	private double[][] LcolumnsValues = null;
	private double[] Svalues = null;
	private DoubleMatrix2D L;
	private DoubleMatrix2D LT;
	private Log log = LogFactory.getLog(this.getClass().getName());

	public long factorizeTime = 0l;
	public long foreachTime = 0l;

	public CholeskySparseFactorization(SparseDoubleMatrix2D Q) {
		this(Q, null);
	}
	
	/**
	 * @param Q sparse symmetric positive definite matrix. Only its subdiagonal lower left part is relevant.
	 */
	public CholeskySparseFactorization(SparseDoubleMatrix2D Q, MatrixRescaler rescaler) {
		//ColtUtils.dumpSparseMatrix(Q);
		//log.debug(org.apache.commons.lang3.ArrayUtils.toString(Q.toArray()));
		this.dim = Q.rows();
		this.Q = Q;
		this.rescaler = rescaler;
	}

	/**
	 * Q = L.L[T], L lower-left triangular. Construction of the matrix L.
	 */
	public void factorize() throws Exception {
		long t0 = System.currentTimeMillis();
		this.LcolumnsValues = new double[dim][];
		
		if(this.rescaler != null){
			double[] cn_00_original = null;
			double[] cn_2_original = null;
			double[] cn_00_scaled = null;
			double[] cn_2_scaled = null;
			if(log.isDebugEnabled()){
				cn_00_original = ColtUtils.getConditionNumberRange(new Array2DRowRealMatrix(ColtUtils.fillSubdiagonalSymmetricMatrix(Q).toArray()), Integer.MAX_VALUE);
				log.debug("cn_00_original Q before scaling: " + ArrayUtils.toString(cn_00_original));
				cn_2_original = ColtUtils.getConditionNumberRange(new Array2DRowRealMatrix(ColtUtils.fillSubdiagonalSymmetricMatrix(Q).toArray()), 2);
				log.debug("cn_2_original Q before scaling : " + ArrayUtils.toString(cn_2_original));
			}
			//scaling the Q matrix, we have:
			//Q1 = U.Q.U[T] = U.L.L[T].U[T] = (U.L).(U.L)[T] 
			//and because U is diagonal it preserves the triangular form of U.L, so
			//Q1 = U.Q.U[T] = L1.L1[T] is the new Cholesky decomposition  
			DoubleMatrix1D Uv = rescaler.getMatrixScalingFactorsSymm(Q);
			if(log.isDebugEnabled()){
				boolean checkOK = rescaler.checkScaling(ColtUtils.fillSubdiagonalSymmetricMatrix(Q), Uv, Uv);
				if(!checkOK){
					log.warn("Scaling failed (checkScaling = false)");
				}
			}
			this.U = Uv;
			this.Q = (SparseDoubleMatrix2D)ColtUtils.diagonalMatrixMult(Uv, Q, Uv);
			if(log.isDebugEnabled()){
				cn_00_scaled = ColtUtils.getConditionNumberRange(new Array2DRowRealMatrix(ColtUtils.fillSubdiagonalSymmetricMatrix(Q).toArray()), Integer.MAX_VALUE);
				log.debug("cn_00_scaled Q after scaling : " + ArrayUtils.toString(cn_00_scaled));
				cn_2_scaled = ColtUtils.getConditionNumberRange(new Array2DRowRealMatrix(ColtUtils.fillSubdiagonalSymmetricMatrix(Q).toArray()), 2);
				log.debug("cn_2_scaled Q after scaling  : " + ArrayUtils.toString(cn_2_scaled));
				
				if(cn_00_original[0] < cn_00_scaled[0] || cn_2_original[0] < cn_2_scaled[0]){
					//log.info("Q: " + ArrayUtils.toString(ColtUtils.fillSubdiagonalSymmetricMatrix(Q).toArray()));
					log.warn("Problematic scaling");
					//throw new RuntimeException("Scaling failed");
				}
			}
		}
		
		final int[] currentColumnIndexHolder = new int[] { -1 };
		
		IntIntDoubleFunction myFunct = new IntIntDoubleFunction() {
			public double apply(int i, int s, double pis) {
				int step = currentColumnIndexHolder[0];
				//log.debug("i:" + i + ", step:" + step + ": " + pis);
				if (i >= step) {
					// at this point, step is the row index and j is the column index,
					// but we take into account only the lower left subdiagonal part of Q (that is symmetric)
					//log.debug("i:" + i + ", step:" + step + ": " + pis);
					Svalues[i - step] = pis;
				}
				return pis;
			}
		};
		
		//view Q column by column
		for (int step = 0; step < dim; step++) {
			//log.debug("column:" + step);
			
			// reset and fill initial values
			Svalues = new double[dim - step];
			DoubleMatrix2D P = Q.viewPart(0, step, dim, 1);
			currentColumnIndexHolder[0] = step;
			P.forEachNonZero(myFunct);
			doStep(step);
		}
		
		this.factorizeTime += (System.currentTimeMillis() - t0);
	}
	
	private void doStep(int step) throws Exception {
		// subtract L[j:n,1:j-n]L[j,1:j-1][T] from S (j=step+1)
		for (int k = 0; k < step; k++) {
			int j = step;//just for easy reading
			double[] LcolumnsValuesK = LcolumnsValues[k];
			double LJK = LcolumnsValuesK[j - k];// array of variable dimension (dim-colIndex)
			if (Double.compare(LJK, 0.) == 0) {
				continue;
			}
			// subtract L[j:n,k]*L[k,j][T]=L[j,k]*L[j:n,k]
			for (int i = j; i < dim; i++) {
				double LIK = LcolumnsValuesK[i - k];// array of variable dimension (dim-colIndex)
				double LJKLIK = LJK * LIK;
				Svalues[i - step] -= LJKLIK;
			}
		}

		// compute L[j:n,j]
		if (!(Svalues[0] > Utils.getDoubleMachineEpsilon())) {
			throw new Exception("not positive definite matrix");
		}
		double evStep = Math.sqrt(Svalues[0]);
		double[] LcolumnsValuesS = new double[dim - step];// max length of SValues
		LcolumnsValuesS[0] = evStep;
		for (int i = 1; i < Svalues.length; i++) {
			double SvaluesR = Svalues[i];
			if (Double.compare(SvaluesR, 0.) != 0) {
				SvaluesR = SvaluesR / evStep;
				LcolumnsValuesS[i] = SvaluesR;
			}
		}
		LcolumnsValues[step] = LcolumnsValuesS;

		if (log.isDebugEnabled()) {
			log.debug("step " + step + " situation:");
			log.debug("LcolumnsValues: " + ArrayUtils.toString(LcolumnsValues));
			log.debug("Svalues:        " + ArrayUtils.toString(Svalues));
		}
	}

	/**
	 * Solves Q.x = b
	 */
	public DoubleMatrix1D solve(DoubleMatrix1D b) {
		if (b.size() != dim) {
			log.error("wrong dimension of vector b: expected " + dim + ", actual " + b.size());
			throw new RuntimeException("wrong dimension of vector b: expected "	+ dim + ", actual " + b.size());
		}
		
	// with scaling, we must solve U.Q.U.z = U.b, after that we have x = U.z
		if (this.rescaler != null) {
			// b = ALG.mult(this.U, b);
			b = ColtUtils.diagonalMatrixMult(this.U, b);
		}

		final double[] y = new double[dim];// copy
		System.arraycopy(b.toArray(), 0, y, 0, dim);

		// Solve L.y = b
		for (int j = 0; j < dim; j++) {
			final double[] LTJ = LcolumnsValues[j];
			y[j] /= LTJ[0];// the diagonal of the matrix L
			final double yJ = y[j];
			for (int i = j + 1; i < dim; i++) {
				y[i] -= yJ * LTJ[i - j];
			}
		}

		// Solve L[T].x = y
		final DoubleMatrix1D x = F1.make(dim);
		for (int i = dim - 1; i > -1; i--) {
			final double[] LTI = LcolumnsValues[i];
			double sum = 0;
			for (int j = dim - 1; j > i; j--) {
				sum += LTI[j - i] * x.getQuick(j);
			}
			x.setQuick(i, (y[i] - sum) / LTI[0]);
		}
		
		if (this.rescaler != null) {
			// return ALG.mult(this.U, x);
			return ColtUtils.diagonalMatrixMult(this.U, x);
		} else {
			return x;
		}
	}

	/**
	 * Solves Q.X = B
	 */
	public DoubleMatrix2D solve(DoubleMatrix2D B) {
		if (B.rows() != dim) {
			log.error("wrong dimension of vector b: expected " + dim +", actual " + B.rows());
			throw new RuntimeException("wrong dimension of vector b: expected " + dim +", actual " + B.rows());
		}
		
	// with scaling, we must solve U.Q.U.z = U.b, after that we have x = U.z
		if (this.rescaler != null) {
			// B = ALG.mult(this.U, B);
			B = ColtUtils.diagonalMatrixMult(this.U, B);
		}
	  
    int nOfColumns = B.columns(); 

    // copy
		final double[][] Y = B.copy().toArray();

		// Solve LY = B (same as L.Yc = Bc for every column of Y and B)
		for (int j = 0; j < dim; j++) {
			final double[] LTJ = LcolumnsValues[j];
			for (int col = 0; col < nOfColumns; col++) {
				Y[j][col] /= LTJ[0];// the diagonal of the matrix L
				final double YJCol = Y[j][col];
				if(Double.compare(YJCol, 0.)!=0){
					for (int i = j + 1; i < dim; i++) {
						Y[i][col] -= YJCol * LTJ[i - j];
					}
				}
			}
		}

		// Solve L[T].X = Y (same as L[T].Xc = Yc for every column of X and Y)
		final DoubleMatrix2D X = F2.make(dim, nOfColumns);
		for (int i = dim - 1; i > -1; i--) {
			final double[] LTI = LcolumnsValues[i];
			double[] sum = new double[nOfColumns];
			for (int col = 0; col < nOfColumns; col++) {
				for (int j = dim - 1; j > i; j--) {
					sum[col] += LTI[j - i] * X.getQuick(j, col);
				}
				X.setQuick(i, col, (Y[i][col] - sum[col]) / LTI[0]);
			}
		}

		if (this.rescaler != null) {
			// return ALG.mult(this.U, X);
			return ColtUtils.diagonalMatrixMult(this.U, X);
		} else {
			return X;
		}
	}

	public DoubleMatrix2D getL() {
		if(this.L == null){
			this.L = ALG.transpose(getLT());
		}
		return this.L;
	}

	//@TODO: check this
	public DoubleMatrix2D getLT() {
		if (this.LT == null) {
			double[][] myLT = new double[dim][];
			for (int i = 0; i < dim; i++) {
				double[] LTI = new double[dim];
				double[] valuesI = LcolumnsValues[i];
				for (int j = i; j < dim; j++) {
					LTI[j] = valuesI[j - i];
				}
				myLT[i] = LTI;
			}
			if (this.rescaler != null) {
				//Q = UInv.Q1.UInv[T] = UInv.L1.L1[T].UInv[T] = (UInv.L1).(UInv.L1)[T]
				//so 
				//L = UInv.L1
				DoubleMatrix1D UInv = F1.make(dim);
				for(int i=0; i<dim; i++){
					UInv.setQuick(i, 1. / U.getQuick(i));
				}
				this.LT = ColtUtils.diagonalMatrixMult(DoubleFactory2D.sparse.make(myLT), UInv);
			} else {
				this.LT = DoubleFactory2D.sparse.make(myLT);
			}
		}
		return this.LT;
	}
}
