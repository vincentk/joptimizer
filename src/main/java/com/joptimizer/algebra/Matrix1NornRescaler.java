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

import com.joptimizer.util.ColtUtils;

import cern.colt.function.IntIntDoubleFunction;
import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;


/**
 * Calculates the matrix rescaling factors so that the 1-norm of each row and each column of the scaled matrix
 * asymptotically converges to one.
 * 
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 * @see Daniel Ruiz, "A scaling algorithm to equilibrate both rows and columns norms in matrices"
 * @see Philip A. Knight, Daniel Ruiz, Bora Ucar "A Symmetry Preserving Algorithm for Matrix Scaling"
 */
public final class Matrix1NornRescaler implements MatrixRescaler{
	
	private double eps = 1.e-3;
	private static final Log log = LogFactory.getLog(Matrix1NornRescaler.class.getName());

	public Matrix1NornRescaler(){
	}
	
	public Matrix1NornRescaler(double eps){
		this.eps = eps;
	}
	
	/**
	 * Scaling factors for not singular matrices.
	 * @see Daniel Ruiz, "A scaling algorithm to equilibrate both rows and columns norms in matrices"
	 * @see Philip A. Knight, Daniel Ruiz, Bora Ucar "A Symmetry Preserving Algorithm for Matrix Scaling"
	 */
	@Override
	public DoubleMatrix1D[] getMatrixScalingFactors(DoubleMatrix2D A){
		DoubleFactory1D F1 = DoubleFactory1D.dense;
		Algebra ALG = Algebra.DEFAULT;
		int r = A.rows();
		int c = A.columns();
		DoubleMatrix1D D1 = F1.make(r, 1);
		DoubleMatrix1D D2 = F1.make(c, 1);
		DoubleMatrix2D AK = A.copy();
		DoubleMatrix1D DR = F1.make(r, 1);
		DoubleMatrix1D DC = F1.make(c, 1);
		DoubleMatrix1D DRInv = F1.make(r);
		DoubleMatrix1D DCInv = F1.make(c);
		log.debug("eps  : " + eps);
		int maxIteration = 50;
		for(int k=0; k<=maxIteration; k++){
			double normR = -Double.MAX_VALUE;
			double normC = -Double.MAX_VALUE;
			for(int i=0; i<r; i++){
				double dri = ALG.normInfinity(AK.viewRow(i));
				DR.setQuick(i, Math.sqrt(dri));
				DRInv.setQuick(i, 1./Math.sqrt(dri));
				normR = Math.max(normR, Math.abs(1-dri));
			}
			for(int j=0; j<c; j++){
				double dci = ALG.normInfinity(AK.viewColumn(j));
				DC.setQuick(j, Math.sqrt(dci));
				DCInv.setQuick(j, 1./Math.sqrt(dci));
				normC = Math.max(normC, Math.abs(1-dci));
			}
			
			log.debug("normR: " + normR);
			log.debug("normC: " + normC);
			if(normR < eps && normC < eps){
				break;
			}
			
			//D1 = ALG.mult(D1, DRInv);
			for(int i=0; i<r; i++){
				double prevD1I = D1.getQuick(i);
				double newD1I = prevD1I * DRInv.getQuick(i);
				D1.setQuick(i, newD1I);
			}
			//D2 = ALG.mult(D2, DCInv);
			for(int j=0; j<c; j++){
				double prevD2J = D2.getQuick(j);
				double newD2J = prevD2J * DCInv.getQuick(j);
				D2.setQuick(j, newD2J);
			}
			//log.debug("D1: " + ArrayUtils.toString(D1.toArray()));
			//log.debug("D2: " + ArrayUtils.toString(D2.toArray()));
			
			if(k==maxIteration){
				log.warn("max iteration reached");
			}
			
			//AK = ALG.mult(DRInv, ALG.mult(AK, DCInv));
			AK = ColtUtils.diagonalMatrixMult(DRInv, AK, DCInv);
		}
		
		return new DoubleMatrix1D[]{D1, D2};
	}

	/**
	 * Scaling factors for symmetric (not singular) matrices.
	 * Just the subdiagonal elements of the matrix are required.
	 * @see Daniel Ruiz, "A scaling algorithm to equilibrate both rows and columns norms in matrices"
	 * @see Philip A. Knight, Daniel Ruiz, Bora Ucar "A Symmetry Preserving Algorithm for Matrix Scaling"
	 */
	@Override
	public DoubleMatrix1D getMatrixScalingFactorsSymm(DoubleMatrix2D A) {
		DoubleFactory1D F1 = DoubleFactory1D.dense;
		DoubleFactory2D F2 = DoubleFactory2D.sparse;
		int dim = A.columns();
		DoubleMatrix1D D1 = F1.make(dim, 1);
		DoubleMatrix2D AK = A.copy();
		DoubleMatrix2D DR = F2.identity(dim);
		DoubleMatrix1D DRInv = F1.make(dim);
		int maxIteration = 50;
		for(int k=0; k<=maxIteration; k++){
			double normR = -Double.MAX_VALUE;
			for(int i=0; i<dim; i++){
				double dri = getRowInfinityNorm(AK, i);
				DR.setQuick(i, i, Math.sqrt(dri));
				DRInv.setQuick(i, 1./Math.sqrt(dri));
				normR = Math.max(normR, Math.abs(1-dri));
				if(Double.isNaN(normR)){
					throw new IllegalArgumentException("matrix is singular");
				}
			}
			
			if(normR < eps){
				break;
			}
			
			for(int i=0; i<dim; i++){
				double prevD1I = D1.getQuick(i);
				double newD1I = prevD1I * DRInv.getQuick(i);
				D1.setQuick(i, newD1I);
			}
			
			if(k==maxIteration){
				log.warn("max iteration reached");
			}
			
			AK = ColtUtils.diagonalMatrixMult(DRInv, AK, DRInv);
		}
		
		return D1;
	}
	
	/**
	 * Check if the scaling algorithm returned proper results.
	 * Note that AOriginal cannot be only subdiagonal filled, because this check
	 * is for both symm and bath notsymm matrices.
	 * @param AOriginal the ORIGINAL (before scaling) matrix
	 * @param U the return of the scaling algorithm
	 * @param V the return of the scaling algorithm
	 * @param base
	 * @return
	 */
	@Override
	public boolean checkScaling(final DoubleMatrix2D AOriginal, final DoubleMatrix1D U, final DoubleMatrix1D V){
		
		int c = AOriginal.columns();
		int r = AOriginal.rows();
		final double[] maxValueHolder = new double[]{-Double.MAX_VALUE}; 
		
		IntIntDoubleFunction myFunct = new IntIntDoubleFunction() {
			@Override
			public double apply(int i, int j, double pij) {
				maxValueHolder[0] = Math.max(maxValueHolder[0], Math.abs(pij));
				return pij;
			}
		};
		
		DoubleMatrix2D AScaled = ColtUtils.diagonalMatrixMult(U, AOriginal, V);
		
		//view A row by row
		boolean isOk = true;
		for (int i = 0; isOk && i < r; i++) {
			maxValueHolder[0] = -Double.MAX_VALUE;
			DoubleMatrix2D P = AScaled.viewPart(i, 0, 1, c);
			P.forEachNonZero(myFunct);
			isOk = Math.abs(1. - maxValueHolder[0]) < eps; 
		}
		//view A col by col
		for (int j = 0; isOk && j < c; j++) {
			maxValueHolder[0] = -Double.MAX_VALUE;
			DoubleMatrix2D P = AScaled.viewPart(0, j, r, 1);
			P.forEachNonZero(myFunct);
			isOk = Math.abs(1. - maxValueHolder[0]) < eps;
		}
		return isOk;
	}
	
	/**
	 * 
	 * @param ASymm symm matrix filled in its subdiagonal elements
	 * @param r the index of the row
	 * @return
	 */
	public static double getRowInfinityNorm(final DoubleMatrix2D ASymm, final int r){
		
		final double[] maxValueHolder = new double[]{-Double.MAX_VALUE}; 
		
		IntIntDoubleFunction myFunct = new IntIntDoubleFunction() {
			@Override
			public double apply(int i, int j, double pij) {
				maxValueHolder[0] = Math.max(maxValueHolder[0], Math.abs(pij));
				return pij;
			}
		};
		
		//view A row from starting element to diagonal
		DoubleMatrix2D AR = ASymm.viewPart(r, 0, 1, r+1);
		AR.forEachNonZero(myFunct);
	  //view A col from diagonal to final element
		DoubleMatrix2D AC = ASymm.viewPart(r, r, ASymm.rows()-r, 1);
		AC.forEachNonZero(myFunct);
		
		return maxValueHolder[0];
	}

}
