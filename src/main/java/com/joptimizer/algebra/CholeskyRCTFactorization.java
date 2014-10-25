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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import cern.colt.matrix.linalg.Property;

import com.joptimizer.util.Utils;

/**
 * Cholesky L.L[T] factorization for symmetric and positive matrix.
 * L is stores in a Row-Compressed way as a triangular matrix.
 * 
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 * @TODO: implement the solve method
 */
public class CholeskyRCTFactorization {

	private int dim;
	private DoubleMatrix2D Q;
	double[] LData;
	private DoubleMatrix2D L;
	private DoubleMatrix2D LT;
	protected Algebra ALG = Algebra.DEFAULT;
	protected DoubleFactory2D F2 = DoubleFactory2D.dense;
	protected DoubleFactory1D F1 = DoubleFactory1D.dense;
	private Log log = LogFactory.getLog(this.getClass().getName());
	
	public CholeskyRCTFactorization(DoubleMatrix2D Q) throws Exception{
		this.dim = Q.rows();
		this.Q = Q;
	}
	
	public void factorize() throws Exception{
		factorize(false);
	}
	
	/**
	 * Cholesky factorization L of psd matrix, Q = L.LT
	 */
	public void factorize(boolean checkSymmetry) throws Exception{
		if (checkSymmetry && !Property.TWELVE.isSymmetric(Q)) {
			throw new Exception("Matrix is not symmetric");
		}
		
		double threshold = Utils.getDoubleMachineEpsilon();
		this.LData = new double[(dim+1)*dim/2];

		for (int i = 0; i < dim; i++) {
			int iShift = (i+1)*i/2;
			for (int j = 0; j < i+1; j++) {
				int jShift = (j+1)*j/2;
				double sum = 0.0;
				for (int k = 0; k < j; k++) {
					sum += LData[jShift + k] * LData[iShift + k];
				}
				if (i == j){
					double d = Q.getQuick(i, i) - sum;
					if(!(d > threshold)){
						throw new Exception("not positive definite matrix");
					}
					LData[iShift + i] = Math.sqrt(d);
				} else {
					LData[iShift + j] = 1.0 / LData[jShift + j] * (Q.getQuick(i, j) - sum);
				}
			}
		}
	}
	
	/**
	 * 
	 * @deprecated use the solve() methods instead
	 */
	@Deprecated
	public DoubleMatrix2D getInverse() {

		//QInv = LTInv * LInv, but for symmetry (QInv=QInvT)
		//QInv = LInvT * LTInvT = LInvT * LInv, so
		//LInvT = LTInv, and we calculate
		//QInv = LInvT * LInv

		// LTInv calculation (it will be x)
		// NB: LInv is lower-triangular
		double[] LInv = new double[(dim+1)*dim/2];
//		for(int i=0; i<dim; i++){
//			//diagonal filling
//			LInv[(i+1)*i/2 + i] = 1.;
//		}
		for (int j = 0; j < dim; j++) {
			int jShift = (j+1)*j/2;//diagonal filling
			LInv[jShift + j] = 1.;
			final double lTJJ = LData[jShift + j];
			for (int k = 0; k < j+1; ++k) {
				LInv[jShift + k] /= lTJJ;
			}
			for (int i = j + 1; i < dim; i++) {
				int iShift = (i+1)*i/2;
				final double lTJI = LData[iShift + j];
				for (int k = 0; k < j+1; ++k) {
					LInv[iShift + k] -= LInv[jShift + k] * lTJI;
				}
			}
		}
		
		//log.debug("LInv: " + ArrayUtils.toString(LInv));

		// QInv
		// NB: LInvT is upper-triangular, so LInvT[i][j]=0 if i>j
		final DoubleMatrix2D QInvData = F2.make(dim, dim);
		for (int row = 0; row < dim; row++) {
			//final double[] LInvTDataRow = LInvTData[row];
			final DoubleMatrix1D QInvDataRow = QInvData.viewRow(row);
			for (int col = row; col < dim; col++) {// symmetry of QInv
				//final double[] LInvTDataCol = LInvTData[col];
				double sum = 0;
				for (int i = col; i < dim; i++) {// upper triangular
					sum += LInv[(i+1)*i/2 + row] * LInv[(i+1)*i/2 + col];
				}
				QInvDataRow.setQuick(col, sum);
				QInvData.setQuick(col, row, sum);// symmetry of QInv
			}
		}

		return QInvData;
	}
	
	/**
	 * @TODO: implement this method
	 */
	public DoubleMatrix1D solve(DoubleMatrix1D b) {
		if (b.size() != dim) {
			log.error("wrong dimension of vector b: expected " + dim +", actual " + b.size());
			throw new RuntimeException("wrong dimension of vector b: expected " + dim +", actual " + b.size());
		}
		
		throw new RuntimeException("not yet implemented");
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
		if(L == null){
			double[][] myL = new double[dim][dim];
			for(int i=0; i<dim; i++){
				double[] myLI = myL[i];
				int iShift = (i+1)*i/2;
				for(int j=0; j<i+1; j++){
					myLI[j] = LData[iShift + j];
				}
			}
			this.L = F2.make(myL);
		}
		
		return L;
	}
	
	public DoubleMatrix2D getLT() {
		if(this.LT == null){
			this.LT = ALG.transpose(getL());
		}
		return this.LT;
	}
	
}
