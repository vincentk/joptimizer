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
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;

import com.joptimizer.util.ColtUtils;
import com.joptimizer.util.Utils;

import edu.emory.mathcs.csparsej.tdouble.Dcs_common.Dcs;
import edu.emory.mathcs.csparsej.tdouble.Dcs_common.Dcsn;
import edu.emory.mathcs.csparsej.tdouble.Dcs_common.Dcss;
import edu.emory.mathcs.csparsej.tdouble.Dcs_happly;
import edu.emory.mathcs.csparsej.tdouble.Dcs_ipvec;
import edu.emory.mathcs.csparsej.tdouble.Dcs_pvec;
import edu.emory.mathcs.csparsej.tdouble.Dcs_qr;
import edu.emory.mathcs.csparsej.tdouble.Dcs_sqr;
import edu.emory.mathcs.csparsej.tdouble.Dcs_usolve;
import edu.emory.mathcs.csparsej.tdouble.Dcs_utsolve;

/**
 * For an <tt>m x n</tt> matrix <tt>A</tt> with <tt>m >= n</tt>, the QR
 * decomposition is an <tt>m x n</tt> orthogonal matrix <tt>Q</tt> and an
 * <tt>n x n</tt> upper triangular matrix <tt>R</tt> so that <tt>A = Q*R</tt>.
 * <br>
 * The QR decompostion always exists, even if the matrix does not have full
 * rank. The primary use of the QR decomposition is in the least squares
 * solution of nonsquare systems of simultaneous linear equations. This will
 * fail if <tt>isFullRank()</tt> returns <tt>false</tt>.
 * 
 * <br>NOTE1: rescaling is not embedded into QR factorization because its not possible 
 * to recover the factorization of the original matrix from the factorization
 * of the rescaled matrix (Q orthogonality will be lost)
 * 
 * <br>NOTE2: this class is a free adaptation of the original class <i>cern.colt.matrix.tdouble.algo.decomposition.SparseDoubleQRDecomposition</i>
 * of the library ParallelColt written by Piotr Wendykier.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class QRSparseFactorization {
	private SparseDoubleMatrix2D A;
	private int order;
	private MatrixRescaler rescaler = null;
	private DoubleMatrix1D U;//the rescaling factor
	private DoubleMatrix1D V;//the rescaling factor
	private Dcss S;
	private Dcsn N;
	private DoubleMatrix2D R;
	private int m, n;
	// private boolean rcMatrix = false;
	protected DoubleFactory1D F1 = DoubleFactory1D.dense;
	private Algebra ALG = Algebra.DEFAULT;
	private Log log = LogFactory.getLog(this.getClass().getName());

	public QRSparseFactorization(SparseDoubleMatrix2D A) throws Exception {
		this(A, 0, null);
	}
	
	public QRSparseFactorization(SparseDoubleMatrix2D A, int order) throws Exception {
		this(A, order, null);
	}
	
	public QRSparseFactorization(SparseDoubleMatrix2D A, MatrixRescaler rescaler) throws Exception {
		this(A, 0, rescaler);
	}

	public QRSparseFactorization(SparseDoubleMatrix2D A, int order, MatrixRescaler rescaler) throws Exception {
		if (order < 0 || order > 3) {
			throw new IllegalArgumentException("order must be a number between 0 and 3");
		}
		this.A = A;
		this.order = order;
		this.rescaler = rescaler;
	}

	/**
	 * Constructs and returns a new QR decomposition object; computed by
	 * Householder reflections; If m < n then then the QR of A' is computed. The
	 * decomposed matrices can be retrieved via instance methods of the returned
	 * decomposition object.
	 * 
	 * @param A
	 *          A rectangular matrix.
	 * @param order
	 *          ordering option (0 to 3); 0: natural ordering, 1: amd(A+A'), 2:
	 *          amd(S'*S), 3: amd(A'*A)
	 * @throws IllegalArgumentException
	 *           if <tt>A</tt> is not sparse
	 * @throws IllegalArgumentException
	 *           if <tt>order</tt> is not in [0,3]
	 */
	public void factorize() throws Exception {
		m = A.rows();
		n = A.columns();
		
		if(this.rescaler != null){
			double[] cn_00_original = null;
			double[] cn_2_original = null;
			double[] cn_00_scaled = null;
			double[] cn_2_scaled = null;
			if(log.isDebugEnabled()){
				cn_00_original = ColtUtils.getConditionNumberRange(new Array2DRowRealMatrix(A.toArray()), Integer.MAX_VALUE);
				log.debug("cn_00_original Q before scaling: " + ArrayUtils.toString(cn_00_original));
				cn_2_original = ColtUtils.getConditionNumberRange(new Array2DRowRealMatrix(A.toArray()), 2);
				log.debug("cn_2_original Q before scaling : " + ArrayUtils.toString(cn_2_original));
			}
			//scaling the A matrix, we have:
			//A1 = U.A.V[T]  
			DoubleMatrix1D[] UV = rescaler.getMatrixScalingFactors(A);
			this.U = UV[0];
			this.V = UV[1];
			if(log.isDebugEnabled()){
				boolean checkOK = rescaler.checkScaling(A, U, V);
				if(!checkOK){
					log.warn("Scaling failed (checkScaling = false)");
				}
			}
			this.A = (SparseDoubleMatrix2D) ColtUtils.diagonalMatrixMult(U, A, V);
			if(log.isDebugEnabled()){
				cn_00_scaled = ColtUtils.getConditionNumberRange(new Array2DRowRealMatrix(A.toArray()), Integer.MAX_VALUE);
				log.debug("cn_00_scaled Q after scaling : " + ArrayUtils.toString(cn_00_scaled));
				cn_2_scaled = ColtUtils.getConditionNumberRange(new Array2DRowRealMatrix(A.toArray()), 2);
				log.debug("cn_2_scaled Q after scaling  : " + ArrayUtils.toString(cn_2_scaled));
				
				if(cn_00_original[0] < cn_00_scaled[0] || cn_2_original[0] < cn_2_scaled[0]){
					log.warn("Problematic scaling");
					//throw new RuntimeException("Scaling failed");
				}
			}
		}
		
		Dcs dcs;
		if (m >= n) {
			dcs = ColtUtils.matrixToDcs(A);
		} else {
			dcs = ColtUtils.matrixToDcs((SparseDoubleMatrix2D) ALG.transpose(A));
		}
		S = Dcs_sqr.cs_sqr(order, dcs, true);
		if (S == null) {
			throw new IllegalArgumentException("Exception occured in cs_sqr()");
		}
		N = Dcs_qr.cs_qr(dcs, S);
		if (N == null) {
			throw new IllegalArgumentException("Exception occured in cs_qr()");
		}
	}

//	/**
//	 * Returns a copy of the Householder vectors v, from the Householder
//	 * reflections H = I - beta*v*v'.
//	 * 
//	 * @return the Householder vectors.
//	 */
//	public DoubleMatrix2D getV() {
//		if (this.V == null) {
//			this.V = ColtUtils.dcsToMatrix(N.L);
//		}
//		return this.V;
//
//		// if (V == null) {
//		// V = new SparseCCDoubleMatrix2D(N.L);
//		// if (rcMatrix) {
//		// V = ((SparseCCDoubleMatrix2D) V).getRowCompressed();
//		// }
//		// }
//		// return V.copy();
//	}

//	/**
//	 * Returns a copy of the beta factors, from the Householder 
//	 * reflections H = I - beta*v*v'.
//	 * 
//	 * @return the beta factors.
//	 */
//	public double[] getBeta() {
//		if (N.B == null) {
//			return null;
//		}
//		double[] beta = new double[N.B.length];
//		System.arraycopy(N.B, 0, beta, 0, N.B.length);
//		return beta;
//	}

	/**
	 * Returns a copy of the upper triangular factor, <tt>R</tt>.
	 * 
	 * @return <tt>R</tt>
	 */
	public DoubleMatrix2D getR() {
		if(this.rescaler==null){
			if (this.R == null) {
				this.R = ColtUtils.dcsToMatrix(N.U);
			}
			return this.R;
		}
		//not able to find the original decomposition A=QR with the appropriate properties of Q and A 
		//from the rescaled entities
		throw new RuntimeException("not implemented");
		
	}
	
	//@TODO: implement this
	public DoubleMatrix2D getQ() {
		if(this.rescaler==null){
			throw new RuntimeException("Not yet implemented");
		}
		//not able to find the original decomposition A=QR with the appropriate properties of Q and A 
		//from the rescaled entities
		throw new RuntimeException("not implemented");
	}

//	/**
//	 * Returns a copy of the symbolic QR analysis object
//	 * 
//	 * @return symbolic QR analysis
//	 */
//	public Dcss getSymbolicAnalysis() {
//		Dcss S2 = new Dcss();
//		S2.cp = S.cp != null ? S.cp.clone() : null;
//		S2.leftmost = S.leftmost != null ? S.leftmost.clone() : null;
//		S2.lnz = S.lnz;
//		S2.m2 = S.m2;
//		S2.parent = S.parent != null ? S.parent.clone() : null;
//		S2.pinv = S.pinv != null ? S.pinv.clone() : null;
//		S2.q = S.q != null ? S.q.clone() : null;
//		S2.unz = S.unz;
//		return S2;
//	}

	/**
	 * Returns whether the matrix <tt>A</tt> has full rank.
	 * NOTE: even when rescaled is used, the response does not change
	 * @return true if <tt>R</tt>, and hence <tt>A</tt>, has full rank.
	 */
	public boolean hasFullRank() {
		// if (R == null) {
		// R = new SparseCCDoubleMatrix2D(N.U);
		// if (rcMatrix) {
		// R = ((SparseCCDoubleMatrix2D) R).getRowCompressed();
		// }
		//
		// }
		int mn = Math.min(m, n);
		double mng = Math.sqrt(m * n);
		// double threshold = ALG.property().tolerance();
		double threshold = mng * Utils.getDoubleMachineEpsilon();
		for (int j = 0; j < mn; j++) {
			// if (R.getQuick(j, j) == 0){
			double rjj = getQuick(N.U, j, j);
			// log.debug("r(" + j + "," + j + "): " + rjj);
			// if (rjj < Utils.getDoubleMachineEpsilon()) {
			// return false;
			// }
			if (rjj < threshold) {
				return false;
			}
		}
		return true;
	}

	/**
	 * Solve a least-squares problem (min ||Ax-b||_2, where A is m-by-n (with m >=
	 * n) or underdetermined system (Ax=b, where m < n).
	 * 
	 * @param b
	 *          right-hand side.
	 * @exception IllegalArgumentException
	 *              if <tt>b.size() != max(A.rows(), A.columns())</tt>.
	 * @exception IllegalArgumentException
	 *              if <tt>!this.hasFullRank()</tt> (<tt>A</tt> is rank
	 *              deficient).
	 */
	public DoubleMatrix1D solve(DoubleMatrix1D b) {
		// if (b.size() != Math.max(m, n)) {
		// throw new
		// IllegalArgumentException("The size b must be equal to max(A.rows(), A.columns()).");
		// }
		if (!this.hasFullRank()) {
			log.error("Matrix is rank deficient: "
					+ ArrayUtils.toString(this.A.toArray()));
			throw new IllegalArgumentException("Matrix is rank deficient");
		}
		
		// with scaling, we must solve U.A.V.z = U.b, after that we have x = V.z
		if (this.rescaler != null) {
			// b = ALG.mult(this.U, b);
			b = ColtUtils.diagonalMatrixMult(this.U, b);
		}
		
		double[] bdata = b.toArray();
		double[] x = new double[this.n];
		System.arraycopy(bdata, 0, x, 0, bdata.length);

		if (m >= n) {
			double[] y = new double[S != null ? S.m2 : 1]; /* get workspace */
			Dcs_ipvec.cs_ipvec(S.pinv, x, y, m); /* y(0:m-1) = b(p(0:m-1) */
			for (int k = 0; k < n; k++) {
				/* apply Householder refl. to x */
				Dcs_happly.cs_happly(N.L, k, N.B[k], y);
			}
			Dcs_usolve.cs_usolve(N.U, y); /* y = R\y */
			Dcs_ipvec.cs_ipvec(S.q, y, x, n); /* x(q(0:n-1)) = y(0:n-1) */
		} else {
			double[] y = new double[S != null ? S.m2 : 1]; /* get workspace */
			Dcs_pvec.cs_pvec(S.q, x, y, m); /* y(q(0:m-1)) = b(0:m-1) */
			Dcs_utsolve.cs_utsolve(N.U, y); /* y = R'\y */
			for (int k = m - 1; k >= 0; k--) {
				/* apply Householder refl. to x */
				Dcs_happly.cs_happly(N.L, k, N.B[k], y);
			}
			Dcs_pvec.cs_pvec(S.pinv, y, x, n); /* x(0:n-1) = y(p(0:n-1)) */
		}
		// log.debug("x: " + ArrayUtils.toString(x));
		// log.debug("b: " + ArrayUtils.toString(b.toArray()));
		
		//return new DenseDoubleMatrix1D(x);
		
		if (this.rescaler != null) {
			// return ALG.mult(this.U, x);
			return ColtUtils.diagonalMatrixMult(this.V, F1.make(x));
		} else {
			return F1.make(x);
		}
	}

	private double getQuick(Dcs dcs, int row, int column) {
		int k = searchFromTo(dcs.i, row, dcs.p[column], dcs.p[column + 1] - 1);
		double v = 0;
		if (k >= 0) {
			v = dcs.x[k];
		}
		return v;
	}

	private int searchFromTo(int[] list, int key, int from, int to) {
		while (from <= to) {
			if (list[from] == key) {
				return from;
			} else {
				from++;
				continue;
			}
		}
		return -(from + 1); // key not found.
	}

}
