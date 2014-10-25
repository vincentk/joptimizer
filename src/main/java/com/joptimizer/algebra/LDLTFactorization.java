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

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import cern.colt.matrix.linalg.Property;

import com.joptimizer.util.ColtUtils;
import com.joptimizer.util.Utils;

/**
 * L.D.L[T] factorization for symmetric not singular matrices:
 * 
 * Q = L.D.L[T], L lower-triangular, D diagonal
 * 
 * NB: D are NOT the eigenvalues of Q
 * 
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class LDLTFactorization {

	private int dim;
	private DoubleMatrix2D Q;
	private MatrixRescaler rescaler = null;
	private DoubleMatrix1D U;//the rescaling factor
	protected Algebra ALG = Algebra.DEFAULT;
	protected DoubleFactory2D F2 = DoubleFactory2D.dense;
	protected DoubleFactory1D F1 = DoubleFactory1D.dense;
	private double[][] LData;
	private double[] DData;
	private DoubleMatrix2D L;
	private DoubleMatrix2D LT;
	private DoubleMatrix2D D;
	private Log log = LogFactory.getLog(this.getClass().getName());

	public LDLTFactorization(DoubleMatrix2D Q) throws Exception {
		this(Q, null);
	}
	
	public LDLTFactorization(DoubleMatrix2D Q, MatrixRescaler rescaler) throws Exception {
		this.dim = Q.rows();
		this.Q = Q;
		this.rescaler = rescaler;
	}

	public void factorize() throws Exception {
		factorize(false);
	}

	/**
	 * Cholesky factorization L of psd matrix, Q = L.LT
	 */
	public void factorize(boolean checkSymmetry) throws Exception {
		if (checkSymmetry && !Property.TWELVE.isSymmetric(Q)) {
			throw new Exception("Matrix is not symmetric");
		}

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
			this.Q = ColtUtils.diagonalMatrixMult(Uv, Q, Uv);
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
		
		double threshold = Utils.getDoubleMachineEpsilon();
		this.LData = new double[dim][];
		this.DData = new double[dim];

		for (int i = 0; i < dim; i++) {
			LData[i] = new double[i + 1];
			double[] LI = LData[i];
			// j < i
			for (int j = 0; j < i; j++) {
				double[] LJ = LData[j];
				double sum = 0.0;
				for (int k = 0; k < j; k++) {
					sum += LI[k] * LJ[k] * DData[k];
				}
				LI[j] = 1.0 / DData[j] * (Q.getQuick(i, j) - sum);
			}
			// j==i
			double sum = 0.0;
			for (int k = 0; k < i; k++) {
				sum += Math.pow(LI[k], 2) * DData[k];
			}
			double dii = Q.getQuick(i, i) - sum;
			if(!(Math.abs(dii) > threshold)){	
				throw new Exception("singular matrix");
			}
			DData[i] = dii;
			LI[i] = 1.;
		}
	}

	/**
	 * Solves Q.x = b
	 * @param b vector
	 * @return the solution x
	 * @see "S.Boyd and L.Vandenberghe, Convex Optimization, p. 672"
	 */
	public DoubleMatrix1D solve(DoubleMatrix1D b) {
		if (b.size() != dim) {
			log.error("wrong dimension of vector b: expected " + dim + ", actual "	+ b.size());
			throw new RuntimeException("wrong dimension of vector b: expected " + dim	+ ", actual " + b.size());
		}

		// with scaling, we must solve U.Q.U.z = U.b, after that we have x = U.z
		if (this.rescaler != null) {
			// b = ALG.mult(this.U, b);
			b = ColtUtils.diagonalMatrixMult(this.U, b);
		}

		// Solve L.y = b
		final double[] y = new double[dim];
		for (int i = 0; i < dim; i++) {
			double[] LI = LData[i];
			double sum = 0;
			for (int j = 0; j < i; j++) {
				sum += LI[j] * y[j];
			}
			y[i] = (b.getQuick(i) - sum) / LI[i];
		}

		// logger.debug("b: " + ArrayUtils.toString(b));
		// logger.debug("L.y: " + ArrayUtils.toString(getL().operate(y)));

		// Solve D.z = y
		final double[] z = new double[dim];
		for (int i = 0; i < dim; i++) {
			z[i] = y[i] / DData[i];
		}

		// Solve L[T].x = z
		final DoubleMatrix1D x = F1.make(dim);
		for (int i = dim - 1; i > -1; i--) {
			double sum = 0;
			for (int j = dim - 1; j > i; j--) {
				sum += LData[j][i] * x.getQuick(j);
			}
			x.setQuick(i, (z[i] - sum) / LData[i][i]);
		}

		if (this.rescaler != null) {
			// return ALG.mult(this.U, x);
			return ColtUtils.diagonalMatrixMult(this.U, x);
		} else {
			return x;
		}
	}
	
	/**
	 * @TODO: implement this method
	 */
	public DoubleMatrix2D solve(DoubleMatrix2D B) {
		  if (B.rows() != dim) {
				log.error("wrong dimension of vector b: expected " + dim +", actual " + B.rows());
				throw new RuntimeException("wrong dimension of vector b: expected " + dim +", actual " + B.rows());
			}
		  throw new RuntimeException("not yet implemented");
	}

	public DoubleMatrix2D getL() {
		if (this.L == null) {
			double[][] myL = new double[dim][dim];
			for (int i = 0; i < dim; i++) {
				double[] LDataI = LData[i];
				double[] myLI = myL[i];
				for (int j = 0; j < i + 1; j++) {
					myLI[j] = LDataI[j];
				}
			}
			if (this.rescaler != null) {
				//Q = UInv.Q1.UInv[T] = UInv.L1.L1[T].UInv[T] = (UInv.L1).(UInv.L1)[T]
				//so 
				//L = UInv.L1
				DoubleMatrix1D UInv = F1.make(dim);
				for(int i=0; i<dim; i++){
					UInv.setQuick(i, 1. / U.getQuick(i));
				}
				this.L = ColtUtils.diagonalMatrixMult(UInv, DoubleFactory2D.sparse.make(myL));
			} else {
				this.L = F2.make(myL);
			}
		}
		return this.L;
	}

	public DoubleMatrix2D getLT() {
		if(this.LT == null){
			this.LT = ALG.transpose(getL());
		}
		return this.LT;
	}

	public DoubleMatrix2D getD() {
		if (this.D == null) {
			this.D = F2.diagonal(F1.make(this.DData));
		}
		return this.D;
	}

}