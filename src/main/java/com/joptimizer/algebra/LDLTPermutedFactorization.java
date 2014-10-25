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

import com.joptimizer.util.ColtUtils;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import cern.colt.matrix.linalg.Property;


/**
 * Symmetric indefinite factorization for symmetric not singular matrices:
 * 
 * P.Q.P[T] = L.D.L[T], P permutation, L lower-triangular, D block-diagonal (1x1 or 2x2 blocks)
 * 
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 * @see "J.D. Hogg, High Performance Cholesky and Symmetric Indefinite Factorizations with Applications" 
 * @see "N.J. Higham, STABILITY OF THE DIAGONAL PIVOTING METHOD WITH PARTIAL PIVOTING"
 * @TODO: implement sparsity and leverage Q symmetry
 */
public class LDLTPermutedFactorization {

	private int dim;
	private DoubleMatrix2D Q;
	private int mode;
	private MatrixRescaler rescaler = null;
	private DoubleMatrix1D U;//the rescaling factor
	public static final int DIAGONAL_PIVOLTING = 0;
	public static final int DIAGONAL_PIVOLTING_WITH_PARTIAL_PIVOTING = 1;
	protected Algebra ALG = Algebra.DEFAULT;
	protected DoubleFactory2D F2 = DoubleFactory2D.dense;
	protected DoubleFactory1D F1 = DoubleFactory1D.dense;
	private DoubleMatrix2D P;
	private DoubleMatrix2D D;
	private DoubleMatrix2D L;
	private Log log = LogFactory.getLog(this.getClass().getName());

	public LDLTPermutedFactorization(DoubleMatrix2D Q) throws Exception {
		this(Q, DIAGONAL_PIVOLTING_WITH_PARTIAL_PIVOTING, null);
	}
	
	public LDLTPermutedFactorization(DoubleMatrix2D Q, int mode) throws Exception {
		this(Q, mode, null);
	}
	
	public LDLTPermutedFactorization(DoubleMatrix2D Q, MatrixRescaler rescaler) throws Exception {
		this(Q, DIAGONAL_PIVOLTING_WITH_PARTIAL_PIVOTING, rescaler);
	}
	
	public LDLTPermutedFactorization(DoubleMatrix2D Q, int mode, MatrixRescaler rescaler) throws Exception {
		this.dim = Q.rows();
		this.Q = Q;
		this.mode = mode;
		this.rescaler = rescaler;
	}

	public void factorize() throws Exception {
		factorize(false);
	}

	/**
	 * Factorization of symmetric matrix Q, P.Q.P[T] = L.LT
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

		switch (this.mode) {

		case DIAGONAL_PIVOLTING:
			pldltpt();
			break;

		case DIAGONAL_PIVOLTING_WITH_PARTIAL_PIVOTING:
			pldltptBK();
			break;

		default:
			throw new IllegalArgumentException("unknown mode " + mode);
		}
		
	}
	
	/**
	 * Diagonal pivoting method.
	 * @see "STABILITY OF THE DIAGONAL PIVOTING METHOD WITH PARTIAL PIVOTING" 1.1
	 */
	private  void pldltpt() throws Exception{
		//matrix S will be changed by the factorization, ad we do not want to change the matrix passed in by the client
		DoubleMatrix2D S = (rescaler==null)? this.Q.copy() : this.Q;
		int n = S.rows();
		DoubleMatrix2D A = S.copy();
		this.P = DoubleFactory2D.sparse.identity(n);
		this.D = DoubleFactory2D.sparse.make(n, n);
		this.L = DoubleFactory2D.sparse.make(n, n);
		DoubleMatrix2D LT = ALG.transpose(L);//remove and work only with L
		int s = 0;
		for(int j=0; j<n; j++){
			//log.debug("j: " + j);
			DoubleMatrix2D LPart = null;
			DoubleMatrix2D LTPart = null;
			DoubleMatrix2D APart = null;
			DoubleMatrix2D DPart = null;
			double ajj = A.getQuick(j, j);
			if(Math.abs(ajj) > 1.e-16){
				//1 x pivot
				D.setQuick(j, j, ajj);
				s = 1;
			}else{
				//ajj = 0, so the 2x2 matrix with ajj in its upper left position
				//is non singular if a(j+1, j)=a(j, j+1) !=0 
				int k=-1;
				for (int r = j + 1; r < n; r++) {
					if (Math.abs(A.getQuick(r, j)) > 1.e-16) {
						k = r;
						break;
					}
				}
				if(k<0){
					throw new Exception("singular matrix");
				}
				//Symmetrically permute row/column k to position j + 1.
				A = ColtUtils.symmPermutation(A, k, j+1);
				P.setQuick(k, k, 0);
				P.setQuick(j + 1, j + 1, 0);
				P.setQuick(k, j + 1, 1);
				P.setQuick(j + 1, k, 1);
				//Choose a 2 � 2 pivot,
				//D(j:j+1)(j:j+1) =  A(j:j+1)(j:j+1)
				D.setQuick(j, j, A.getQuick(j, j));
				D.setQuick(j, j+1, A.getQuick(j, j+1));
				D.setQuick(j+1, j, A.getQuick(j+1, j));
				D.setQuick(j+1, j+1, A.getQuick(j+1, j+1));
				s=2;
			}
			//log.debug("s: " + s);
			
			// L(j:n)(j:j+s-1) = A(j:n)(j:j+s-1).DInv(j:j+s-1)(j:j+s-1)
			APart = A.viewPart(j, j, n - j, 1 + s - 1);
			DPart = ALG.inverse(D.viewPart(j, j, 1 + s - 1, 1 + s - 1));
			LPart = L.viewPart(j, j, n - j, 1 + s - 1);
			DoubleMatrix2D AD = ALG.mult(APart, DPart);
			for (int r = 0; r < LPart.rows(); r++) {
				for (int c = 0; c < LPart.columns(); c++) {
					LPart.setQuick(r, c, AD.getQuick(r, c));
				}
			}
			
			// A(j+s-1:n)(j+s-1:n) = A(j+s-1:n)(j+s-1:n) - L(j+s-1:n)(j:j+s-1).D(j:j+s-1)(j:j+s-1).LT(j:j+s-1)(j+s-1:n)
			LPart = L.viewPart(j + s - 1, j, n - (j + s - 1), 1 + s - 1);
			DPart = D.viewPart(j, j, 1 + s - 1, 1 + s - 1);
			LTPart = LT.viewPart(j, j + s - 1, s, n - (j + s - 1));
			APart = A.viewPart(j + s - 1, j + s - 1, n - (j + s - 1), n - (j + s - 1));
			DoubleMatrix2D LDLT = ALG.mult(LPart, ALG.mult(DPart, LTPart));
			for (int r = 0; r < APart.rows(); r++) {
				for (int c = 0; c < APart.columns(); c++) {
					APart.setQuick(r, c, APart.getQuick(r, c) - LDLT.getQuick(r, c));
				}
			}
			//logger.debug("L: " + ArrayUtils.toString(L.toArray()));
			//logger.debug("A: " + ArrayUtils.toString(A.toArray()));
			
			j = j + s - 1;
		}
	}
	
	/**
	 * Diagonal pivoting method with the partial pivoting strategy of Bunch and Kaufman.
	 * @see "STABILITY OF THE DIAGONAL PIVOTING METHOD WITH PARTIAL PIVOTING" 1.1 and Algorithm 1
	 */
	private void pldltptBK() throws Exception {
		//matrix S will be changed by the factorization, ad we do not want to change the matrix passed in by the client
		DoubleMatrix2D S = (rescaler==null)? this.Q.copy() : this.Q;
		int n = S.rows();
		DoubleMatrix2D A = S.copy();
		this.P = DoubleFactory2D.sparse.identity(n);
		this.D = DoubleFactory2D.sparse.make(n, n);
		this.L = DoubleFactory2D.sparse.make(n, n);
		DoubleMatrix2D LT = ALG.transpose(L);//@TODO: remove and work only with L
		double mu = (1 + Math.sqrt(17)) / 8;
		int s = 0;
		for (int j = 0; j < n; j++) {
			//log.debug("j: " + j);
			DoubleMatrix2D LPart = null;
			DoubleMatrix2D LTPart = null;
			DoubleMatrix2D APart = null;
			DoubleMatrix2D DPart = null;
			
			//Determine largest off-diagonal element in column j
			double lambda = -1;
			int r = -1;
			for(int rr = j+1; rr<n; rr++){
				double d = Math.abs(A.getQuick(rr, j)); 
				if(d > lambda){
					lambda = d;
					r = rr;
				}
			}
			double ajj = A.getQuick(j, j);
			if (Math.abs(ajj) > mu*lambda) {
				// Choose a 1 x 1 pivot,
				D.setQuick(j, j, ajj);
				s = 1;
			} else {
				//Determine largest off-diagonal element in row/column r
				double sigma = -1;
				int p = -1;
				for(int rr = 0; rr<n; rr++){
					if(rr==r){
						continue;
					}
					double d = Math.abs(A.getQuick(rr, r)); 
					if(d > sigma){
						sigma = d;
						p = rr;
					}
				}
				if(sigma * Math.abs(ajj) > mu*lambda*lambda){
					//Choose a 1 x 1 pivot,
					D.setQuick(j, j, A.getQuick(j, j));
					s = 1;
				}else if(Math.abs(A.getQuick(r, r)) > mu*sigma){
					//Symmetrically permute row/column r to position j
				  A = ColtUtils.symmPermutation(A, r, j);
					P.setQuick(r, r, 0);
					P.setQuick(j, j, 0);
					P.setQuick(r, j, 1);
					P.setQuick(j, r, 1);
					//Choose a 1 x 1 pivot,
					D.setQuick(j, j, A.getQuick(j, j));
					s = 1;
				}else{
					//Symmetrically permute rows/columns r and p to positions j and j + 1 respectively
					int t = j+1;
					//int v = j+1;
					A = ColtUtils.symmPermutation(A, r, t);
					P.setQuick(r, r, 0);
					P.setQuick(t, t, 0);
					P.setQuick(r, t, 1);
					P.setQuick(t, r, 1);
//					A = Utils.symmPermutation(A, p, v);
//					P.setQuick(p, p, 0);
//					P.setQuick(v, v, 0);
//					P.setQuick(p, v, 1);
//					P.setQuick(v, p, 1);
					//Choose a 2 x pivot,
					//D(j:j+1)(j:j+1) =  A(j:j+1)(j:j+1)
					D.setQuick(j, j, A.getQuick(j, j));
					D.setQuick(j, j+1, A.getQuick(j, j+1));
					D.setQuick(j+1, j, A.getQuick(j+1, j));
					D.setQuick(j+1, j+1, A.getQuick(j+1, j+1));
					s = 2;
				}
			}
			//log.debug("s: " + s);

			// L(j:n)(j:j+s-1) = A(j:n)(j:j+s-1).DInv(j:j+s-1)(j:j+s-1)
			APart = A.viewPart(j, j, n - j, 1 + s - 1);
//			if(s==2){
//				try{
//					ColtUtils.invert2x2Matrix(D.viewPart(j, j, 1 + s - 1, 1 + s - 1));
//				}catch(Exception e){
//					throw e;
//				}
//			}
			DPart = ALG.inverse(D.viewPart(j, j, 1 + s - 1, 1 + s - 1));
			LPart = L.viewPart(j, j, n - j, 1 + s - 1);
			DoubleMatrix2D AD = ALG.mult(APart, DPart);
			for (int rr = 0; rr < LPart.rows(); rr++) {
				for (int cc = 0; cc < LPart.columns(); cc++) {
					LPart.setQuick(rr, cc, AD.getQuick(rr, cc));
				}
			}

			// A(j+s-1:n)(j+s-1:n) = A(j+s-1:n)(j+s-1:n) -
			// L(j+s-1:n)(j:j+s-1).D(j:j+s-1)(j:j+s-1).LT(j:j+s-1)(j+s-1:n)
			LPart = L.viewPart(j + s - 1, j, n - (j + s - 1), 1 + s - 1);
			DPart = D.viewPart(j, j, 1 + s - 1, 1 + s - 1);
			LTPart = LT.viewPart(j, j + s - 1, s, n - (j + s - 1));
			APart = A.viewPart(j + s - 1, j + s - 1, n - (j + s - 1), n - (j + s - 1));
			DoubleMatrix2D LDLT = ALG.mult(LPart, ALG.mult(DPart, LTPart));
			for (int rr = 0; rr < APart.rows(); rr++) {
				for (int cc = 0; cc < APart.columns(); cc++) {
					APart.setQuick(rr, cc, APart.getQuick(rr, cc) - LDLT.getQuick(rr, cc));
				}
			}
			//logger.debug("L: " + ArrayUtils.toString(L.toArray()));
			//logger.debug("A: " + ArrayUtils.toString(A.toArray()));

			j = j + s - 1;
		}
	}

	/**
	 * Solves Q.x = b
	 * @param b vector
	 * @return the solution x
	 * @see "S.Boyd and L.Vandenberghe, Convex Optimization, p. 672"
	 * @TODO: implement this method
	 */
	public DoubleMatrix1D solve(DoubleMatrix1D b) {
		if (b.size() != dim) {
			log.error("wrong dimension of vector b: expected " + dim + ", actual "	+ b.size());
			throw new RuntimeException("wrong dimension of vector b: expected " + dim	+ ", actual " + b.size());
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
	
	public DoubleMatrix2D getP() {
		if(this.rescaler==null){
			return this.P;
		}
		//@TODO: implement this
		throw new RuntimeException("not yet implemented");
	}

	public DoubleMatrix2D getD() {
		if(this.rescaler==null){
			return this.D;
		}
		//@TODO: implement this
		throw new RuntimeException("not yet implemented");
	}
	
	public DoubleMatrix2D getL() {
		if(this.rescaler==null){
			return this.L;
		}
		//@TODO: implement this
		throw new RuntimeException("not yet implemented");
	}

	public DoubleMatrix2D getLT() {
		if(this.rescaler==null){
			return ALG.transpose(this.L);
		}
		//@TODO: implement this
		throw new RuntimeException("not yet implemented");
	}

}