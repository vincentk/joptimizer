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
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import cern.colt.matrix.linalg.Property;

import com.joptimizer.util.ColtUtils;
import com.joptimizer.util.Utils;

/**
 * Cholesky L.L[T] factorization for symmetric, positive and almost diagonal matrix:
 * 
 * Q = L.L[T], L lower-triangular
 * 
 * <br>Q is expected to be diagonal in its upper left corner of dimension <i>diagonalLength</i>.
 * 
 * @author <a href="mailto:alberto.trivellato@gmail.com">alberto trivellato</a>
 */
public class CholeskyUpperDiagonalFactorization {

	private int dim;
	private int diagonalLength;
	private MatrixRescaler rescaler = null;
	private SparseDoubleMatrix2D Q;
	private DoubleMatrix1D U;
	protected Algebra ALG = Algebra.DEFAULT;
	protected DoubleFactory2D F2 = DoubleFactory2D.dense;
	protected DoubleFactory1D F1 = DoubleFactory1D.dense;
	private double[][] LData;
	private DoubleMatrix2D L;
	private DoubleMatrix2D LT;
	private Log log = LogFactory.getLog(this.getClass().getName());
	
	public CholeskyUpperDiagonalFactorization(SparseDoubleMatrix2D Q, int diagonalLength) throws Exception{
		this(Q, diagonalLength, null);
	}
	
	/**
	 * 
	 * @param Q sparse matrix diagonal in its left upper corner
	 * @param digonalLength the number of diagonal elements
	 * @throws Exception
	 */
	public CholeskyUpperDiagonalFactorization(SparseDoubleMatrix2D Q, int diagonalLength, MatrixRescaler rescaler) throws Exception{
		this.dim = Q.rows();
		this.Q = Q;
		if(diagonalLength < 0){
			throw new IllegalArgumentException("DiagonalLength cannot be < 0");
		}
		if(dim < diagonalLength){
			throw new IllegalArgumentException("Dimension cannot be less than diagonalLength");
		}
		this.diagonalLength = diagonalLength;
		this.rescaler = rescaler;
	}
	
	public void factorize() throws Exception{
		factorize(false);
	}
	
	/**
	 * Cholesky factorization L of psd matrix, Q = L.LT.
	 * Construction of the matrix L.
	 */
	public void factorize(boolean checkSymmetry) throws Exception{
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
			this.Q = (SparseDoubleMatrix2D) ColtUtils.diagonalMatrixMult(Uv, Q, Uv);
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
		//elements of L are stored in a single array for i<diagonalLength, 
		//and with arrays of dimension i+1 for i>=diagonalLength
		this.LData = new double[dim - diagonalLength + 1][];
		double[] LData0 = new double[diagonalLength];
		LData[0] = LData0;
		
		if(dim - diagonalLength == 1 && Double.compare(Q.getQuick(dim-1, dim-1), 1.)==0){
			//TODO: the second condition can always be true if the matrix is normalized on the last element
			//closed form decomposition
			//@see "S.Boyd and L.Vandenberghe, Convex Optimization, p. 671"
			for (int i = 0; i < dim-1; i++) {
				double d = Q.getQuick(i, i);
				if (d < Utils.getDoubleMachineEpsilon()) {
					throw new Exception("not positive definite matrix");
				}
				LData0[i] = Math.sqrt(d);
			}
			//now the last row of L
			double[] LData1 = new double[dim];
			for(int j=0; j<dim - 1; j++){
				LData1[j] = Q.getQuick(dim-1, j) / LData0[j];
			}
			double d = 0;
			for(int k=0; k<diagonalLength; k++){
				d += Math.pow(Q.getQuick(dim-1, k), 2) / Q.getQuick(k, k);
			}
			LData1[dim-1] = Math.sqrt(1 - d);
			LData[1] = LData1;
		}else{
			for (int i = 0; i < dim; i++) {
			    if (i < diagonalLength) {
					double d = Q.getQuick(i, i);
					if(!(d > threshold)){
						throw new Exception("not positive definite matrix");
					}
					LData0[i] = Math.sqrt(d);
				} else {

					LData[i - diagonalLength + 1] = new double[i + 1];
					double[] LDataI = LData[i - diagonalLength + 1];

					// j < i
					for (int j = 0; j < diagonalLength; j++) {
						// here LData is not null only in its diagonal
						LDataI[j] = 1.0 / LData[0][j] * (Q.getQuick(i, j));
					}
					for (int j = diagonalLength; j < i; j++) {
						double[] LDataJ = LData[j - diagonalLength + 1];
						double sum = 0.0;
						for (int k = 0; k < j; k++) {
							sum += LDataI[k] * LDataJ[k];
						}
						LDataI[j] = 1.0 / LDataJ[j] * (Q.getQuick(i, j) - sum);
					}
					// j==i
					double sum = 0.0;
					for (int k = 0; k < i; k++) {
						sum += Math.pow(LDataI[k], 2);
					}
					double d = Q.getQuick(i, i) - sum;
					if (d < Utils.getDoubleMachineEpsilon()) {
						throw new Exception("not positive definite matrix");
					}
					LDataI[i] = Math.sqrt(d);
				}
			}
		}
	}
		
	/**
	 * Solve Q.x = b
	 * @param b vector
	 * @return the solution vector x
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
		// L is upper left diagonal
		for (int i = 0; i < diagonalLength; i++) {
			double LII = LData[0][i];
			y[i] = b.getQuick(i) / LII;
		}
		for (int i = diagonalLength; i < dim; i++) {
			double[] LI = LData[i - diagonalLength + 1];
			double LII = LI[i];
			double sum = 0;
			for (int j = 0; j < i; j++) {
				sum += LI[j] * y[j];
			}
			y[i] = (b.getQuick(i) - sum) / LII;
		}

		// logger.debug("b: " + ArrayUtils.toString(b));
		// logger.debug("L.y: " + ArrayUtils.toString(getL().operate(y)));

		// Solve L[T].x = y
		final DoubleMatrix1D x = F1.make(dim);
		// for (int i = dim-1; i > -1; i--) {
		// double sum = 0;
		// int ll = Math.max(i, diagonalLength-1);
		// for (int j = dim-1; j > ll; j--) {
		// sum += LData[j][i] * x.getQuick(j);
		//
		// }
		// x.setQuick(i, (y[i] - sum)/ LData[i][i]);
		// }
		for (int i = dim - 1; i > diagonalLength - 1; i--) {
			double sum = 0;
			for (int j = dim - 1; j > i; j--) {
				sum += LData[j - diagonalLength + 1][i] * x.getQuick(j);

			}
			x.setQuick(i, (y[i] - sum) / LData[i - diagonalLength + 1][i]);
		}
		for (int i = diagonalLength - 1; i > -1; i--) {
			double sum = 0;
			for (int j = dim - 1; j > diagonalLength - 1; j--) {
				sum += LData[j - diagonalLength + 1][i] * x.getQuick(j);

			}
			x.setQuick(i, (y[i] - sum) / LData[0][i]);
		}

		if (this.rescaler != null) {
			// return ALG.mult(this.U, x);
			return ColtUtils.diagonalMatrixMult(this.U, x);
		} else {
			return x;
		}
	}
	  
	/**
	 * solve Q.X = B
	 * @param B matrix
	 * @return the solution matrix X
	 */
	public DoubleMatrix2D solve(DoubleMatrix2D B) {
		if (B.rows() != dim) {
			log.error("wrong dimension of vector b: expected " + dim + ", actual "	+ B.rows());
			throw new RuntimeException("wrong dimension of vector b: expected " + dim	+ ", actual " + B.rows());
		}

	// with scaling, we must solve U.Q.U.z = U.b, after that we have x = U.z
		if (this.rescaler != null) {
			// B = ALG.mult(this.U, B);
			B = ColtUtils.diagonalMatrixMult(this.U, B);
		}

		int nOfColumns = B.columns();

		// Solve LY = B (the same as L.Yc = Bc for each column Yc e Bc)
		final double[][] Y = new double[dim][nOfColumns];
		for (int i = 0; i < diagonalLength; i++) {
			double LII = LData[0][i];
			double[] YI = Y[i];
			DoubleMatrix1D BI = B.viewRow(i);
			for (int col = 0; col < nOfColumns; col++) {
				YI[col] = BI.getQuick(col) / LII;
			}
		}
		for (int i = diagonalLength; i < dim; i++) {
			double[] LI = LData[i - diagonalLength + 1];
			double LII = LI[i];
			double[] sum = new double[nOfColumns];
			for (int j = 0; j < i; j++) {
				double LIJ = LI[j];
				double[] YJ = Y[j];
				for (int col = 0; col < nOfColumns; col++) {
					sum[col] += LIJ * YJ[col];
				}
			}
			double[] YI = Y[i];
			DoubleMatrix1D BI = B.viewRow(i);
			for (int col = 0; col < nOfColumns; col++) {
				YI[col] = (BI.getQuick(col) - sum[col]) / LII;
			}
		}

		// Solve L[T].X = Y (the same as L[T].Xc = Yc for each column)
		final DoubleMatrix2D X = F2.make(dim, nOfColumns);
		for (int i = dim - 1; i > diagonalLength - 1; i--) {
			double LII = LData[i - diagonalLength + 1][i];
			double[] sum = new double[nOfColumns];
			for (int j = dim - 1; j > i; j--) {
				double[] LJ = LData[j - diagonalLength + 1];
				double LJI = LJ[i];
				DoubleMatrix1D XJ = X.viewRow(j);
				for (int col = 0; col < nOfColumns; col++) {
					sum[col] += LJI * XJ.getQuick(col);
				}
			}
			DoubleMatrix1D XI = X.viewRow(i);
			double[] YI = Y[i];
			for (int col = 0; col < nOfColumns; col++) {
				XI.setQuick(col, (YI[col] - sum[col]) / LII);
			}
		}
		for (int i = diagonalLength - 1; i > -1; i--) {
			double LII = LData[0][i];
			double[] sum = new double[nOfColumns];
			for (int j = dim - 1; j > diagonalLength - 1; j--) {
				double[] LJ = LData[j - diagonalLength + 1];
				double LJI = LJ[i];
				DoubleMatrix1D XJ = X.viewRow(j);
				for (int col = 0; col < nOfColumns; col++) {
					sum[col] += LJI * XJ.getQuick(col);
				}
			}
			DoubleMatrix1D XI = X.viewRow(i);
			double[] YI = Y[i];
			for (int col = 0; col < nOfColumns; col++) {
				XI.setQuick(col, (YI[col] - sum[col]) / LII);
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
		if(this.L==null){
			double[][] myL = new double[dim][dim];
			for (int i = 0; i < diagonalLength; i++) {
				myL[i][i] = LData[0][i];
			}
			for (int i = diagonalLength; i < dim; i++) {
				double[] LDataI = LData[i-diagonalLength+1];
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

}
