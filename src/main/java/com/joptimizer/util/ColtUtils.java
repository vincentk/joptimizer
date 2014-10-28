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
package com.joptimizer.util;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.apache.commons.math3.linear.RealMatrix;

import cern.colt.function.IntIntDoubleFunction;
import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix1D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import edu.emory.mathcs.csparsej.tdouble.Dcs_common;

/**
 * Support class for recurrent algebra with Colt.
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class ColtUtils {

	/**
	 * Matrix-vector multiplication with diagonal matrix.
	 * @param diagonalM diagonal matrix M, in the form of a vector of its diagonal elements
	 * @param vector
	 * @return M.x
	 */
	public static final DoubleMatrix1D diagonalMatrixMult(DoubleMatrix1D diagonalM, DoubleMatrix1D vector){
		int n = diagonalM.size();
		DoubleMatrix1D ret = DoubleFactory1D.dense.make(n);
		for(int i=0; i<n; i++){
			ret.setQuick(i, diagonalM.getQuick(i) * vector.getQuick(i));
		}
		return ret;
	}

	/**
	 * Return diagonalU.A with diagonalU diagonal.
	 * @param diagonal matrix U, in the form of a vector of its diagonal elements
	 * @return U.A
	 */
	public static final DoubleMatrix2D diagonalMatrixMult(final DoubleMatrix1D diagonalU, DoubleMatrix2D A){
		int r = diagonalU.size();
		int c = A.columns();
		final DoubleMatrix2D ret;
		if (A instanceof SparseDoubleMatrix2D) {
			ret = DoubleFactory2D.sparse.make(r, c);
			A.forEachNonZero(new IntIntDoubleFunction() {
				@Override
				public double apply(int i, int j, double aij) {
					ret.setQuick(i, j, aij * diagonalU.getQuick(i));
					return aij;
				}
			});
		} else {
			ret = DoubleFactory2D.dense.make(r, c);
			for (int i = 0; i < r; i++) {
				for (int j = 0; j < c; j++) {
					ret.setQuick(i, j, A.getQuick(i, j) * diagonalU.getQuick(i));
				}
			}
		}

		return ret;
	}

	/**
	 * Return A.diagonalU with diagonalU diagonal.
	 * @param diagonal matrix U, in the form of a vector of its diagonal elements
	 * @return U.A
	 */
	public static final DoubleMatrix2D diagonalMatrixMult(DoubleMatrix2D A, final DoubleMatrix1D diagonalU){
		int r = diagonalU.size();
		int c = A.columns();
		final DoubleMatrix2D ret;
		if (A instanceof SparseDoubleMatrix2D) {
			ret = DoubleFactory2D.sparse.make(r, c);
			A.forEachNonZero(new IntIntDoubleFunction() {
				@Override
				public double apply(int i, int j, double aij) {
					ret.setQuick(i, j, aij * diagonalU.getQuick(j));
					return aij;
				}
			});
		} else {
			ret = DoubleFactory2D.dense.make(r, c);
			for (int i = 0; i < r; i++) {
				for (int j = 0; j < c; j++) {
					ret.setQuick(i, j, A.getQuick(i, j) * diagonalU.getQuick(j));
				}
			}
		}

		return ret;
	}

	/**
	 * Return diagonalU.A.diagonalV with diagonalU and diagonalV diagonal.
	 * @param diagonalU diagonal matrix U, in the form of a vector of its diagonal elements
	 * @param diagonalV diagonal matrix V, in the form of a vector of its diagonal elements
	 * @return U.A.V
	 */
	public static final DoubleMatrix2D diagonalMatrixMult(final DoubleMatrix1D diagonalU, DoubleMatrix2D A, final DoubleMatrix1D diagonalV){
		int r = A.rows();
		int c = A.columns();
		final DoubleMatrix2D ret;
		if (A instanceof SparseDoubleMatrix2D) {
			ret = DoubleFactory2D.sparse.make(r, c);
			A.forEachNonZero(new IntIntDoubleFunction() {
				@Override
				public double apply(int i, int j, double aij) {
					ret.setQuick(i, j, aij * diagonalU.getQuick(i) * diagonalV.getQuick(j));
					return aij;
				}
			});
		} else {
			ret = DoubleFactory2D.dense.make(r, c);
			for (int i = 0; i < r; i++) {
				for (int j = 0; j < c; j++) {
					ret.setQuick(i, j, A.getQuick(i, j) * diagonalU.getQuick(i)	* diagonalV.getQuick(j));
				}
			}
		}

		return ret;
	}

	/**
	 * Return the sub-diagonal result of the multiplication.
	 * If A is sparse, returns a sparse matrix (even if, generally speaking, 
	 * the multiplication of two sparse matrices is not sparse) because the result
	 * is at least 50% (aside the diagonal elements) sparse.  
	 */
	public static DoubleMatrix2D subdiagonalMultiply(final DoubleMatrix2D A, final DoubleMatrix2D B){
		final int r = A.rows();
		final int rc = A.columns();
		final int c = B.columns();

		if(r != c){
			throw new IllegalArgumentException("The result must be square");
		}

		boolean useSparsity = A instanceof SparseDoubleMatrix2D;
		DoubleFactory2D F2 = (useSparsity)? DoubleFactory2D.sparse : DoubleFactory2D.dense; 
		final DoubleMatrix2D ret = F2.make(r, c);

		if(useSparsity){
			IntIntDoubleFunction myFunct = new IntIntDoubleFunction() {
				@Override
				public double apply(int t, int s, double pts) {
					int i = t;
					for (int j = 0; j < i + 1; j++) {
						ret.setQuick(i, j, ret.getQuick(i, j) + pts * B.getQuick(s, j));
					}
					return pts;
				}
			};

			//view A row by row
			A.forEachNonZero(myFunct);
		}else{
			for (int i = 0; i < r; i++) {
				for (int j = 0; j < i + 1; j++) {
					double s = 0;
					for (int k = 0; k < rc; k++) {
						s += A.getQuick(i, k) * B.getQuick(k, j);
					}
					ret.setQuick(i, j, s);
				}
			}
		}

		return ret;
	}

	/**
	 * Returns v = beta * A.b.
	 * Useful in avoiding the need of the copy() in the colt api.
	 */
	public static final DoubleMatrix1D zMult(final DoubleMatrix2D A, final DoubleMatrix1D b, final double beta){
		if(A.columns() != b.size()){
			throw new IllegalArgumentException("wrong matrices dimensions");
		}
		final DoubleMatrix1D ret = DoubleFactory1D.dense.make(A.rows());

		if(A instanceof SparseDoubleMatrix2D){
			//sparse matrix
			A.forEachNonZero(new IntIntDoubleFunction() {
				@Override
				public double apply(int i, int j, double Aij) {
					double vi = 0;
					vi += Aij * b.getQuick(j);
					ret.setQuick(i, ret.getQuick(i) + beta * vi);
					return Aij;
				}
			});
		}else{
			//dense matrix
			for(int i=0; i<A.rows(); i++){
				double vi = 0;
				for(int j=0; j<A.columns(); j++){
					vi += A.getQuick(i, j) * b.getQuick(j);
				}
				ret.setQuick(i, beta * vi);
			}
		}

		return ret;
	}

	/**
	 * Returns v = A.a + beta*b.
	 * Useful in avoiding the need of the copy() in the colt api.
	 */
	public static final DoubleMatrix1D zMult(final DoubleMatrix2D A, final DoubleMatrix1D a, final DoubleMatrix1D b, final double beta){

		if(A.columns()!=a.size()){
			throw new IllegalArgumentException("Wrong matrix dimensions. Number of columns must be " + a.size() + ", found: " + A.columns());
		}

		if(A.rows()!=b.size()){
			throw new IllegalArgumentException("Wrong matrix dimensions. Number of rows must be " + b.size() + ", found: " + A.rows());
		}

		final DoubleMatrix1D ret = DoubleFactory1D.dense.make(A.rows());

		if(A instanceof SparseDoubleMatrix2D){
			//sparse matrix
			A.forEachNonZero(new IntIntDoubleFunction() {
				@Override
				public double apply(int i, int j, double Aij) {
					ret.setQuick(i, ret.getQuick(i) + Aij * a.getQuick(j));
					return Aij;
				}
			});
			for(int i=0; i<ret.size(); i++){
				ret.setQuick(i, ret.getQuick(i) + beta * b.getQuick(i));
			}
		}else{
			//dense matrix
			for(int i=0; i<A.rows(); i++){
				double vi = beta * b.getQuick(i);
				for(int j=0; j<A.columns(); j++){
					vi += A.getQuick(i, j) * a.getQuick(j);
				}
				ret.setQuick(i, vi);
			}
		}

		return ret;
	}

	/**
	 * Returns v = A[T].a + beta*b.
	 * Useful in avoiding the need of the copy() in the colt api.
	 */
	public static final DoubleMatrix1D zMultTranspose(final DoubleMatrix2D A, final DoubleMatrix1D a, final DoubleMatrix1D b, final double beta){
		if(A.rows()!=a.size() || A.columns()!=b.size()){
			throw new IllegalArgumentException("wrong matrices dimensions");
		}
		final DoubleMatrix1D ret = DoubleFactory1D.dense.make(A.columns());

		if(A instanceof SparseDoubleMatrix2D){
			//if(1==2){	
			A.forEachNonZero(new IntIntDoubleFunction() {
				@Override
				public double apply(int i, int j, double Aij) {
					ret.setQuick(j, ret.getQuick(j) + Aij * a.getQuick(i));
					return Aij;
				}
			});
			if(Double.compare(0. ,beta)!=0){
				for(int i=0; i<ret.size(); i++){
					ret.setQuick(i, ret.getQuick(i) + beta * b.getQuick(i));
				}
			}
		}else{
			for(int i=0; i<A.columns(); i++){
				double vi = beta * b.getQuick(i);
				for(int j=0; j<A.rows(); j++){
					vi += A.getQuick(j, i) * a.getQuick(j);
				}
				ret.setQuick(i, vi);
			}
		}

		return ret;
	}

	/**
	 * Returns C = A + B.
	 * Useful in avoiding the need of the copy() in the colt api.
	 */
	public static final DoubleMatrix2D add(DoubleMatrix2D A, DoubleMatrix2D B){
		if(A.rows()!=B.rows() || A.columns()!=B.columns()){
			throw new IllegalArgumentException("wrong matrices dimensions");
		}
		DoubleMatrix2D ret = DoubleFactory2D.dense.make(A.rows(), A.columns());
		for(int i=0; i<ret.rows(); i++){
			for(int j=0; j<ret.columns(); j++){
				ret.setQuick(i, j, A.getQuick(i, j) + B.getQuick(i, j));
			}
		}

		return ret;
	}

	/**
	 * Returns C = A + beta*B.
	 * Useful in avoiding the need of the copy() in the colt api.
	 */
	public static final DoubleMatrix2D add(DoubleMatrix2D A, DoubleMatrix2D B, double beta){
		if(A.rows()!=B.rows() || A.columns()!=B.columns()){
			throw new IllegalArgumentException("wrong matrices dimensions");
		}
		DoubleMatrix2D ret = DoubleFactory2D.dense.make(A.rows(), A.columns());
		for(int i=0; i<ret.rows(); i++){
			//DoubleMatrix1D AI = A.viewRow(i);
			//DoubleMatrix1D BI = B.viewRow(i);
			//DoubleMatrix1D retI = ret.viewRow(i);
			for(int j=0; j<ret.columns(); j++){
				ret.setQuick(i, j, A.getQuick(i, j) + beta*B.getQuick(i, j));
			}
		}

		return ret;
	}

	/**
	 * Returns v = v1 + v2.
	 * Useful in avoiding the need of the copy() in the colt api.
	 */
	public static final DoubleMatrix1D add(DoubleMatrix1D v1, DoubleMatrix1D v2){
		if(v1.size()!=v2.size()){
			throw new IllegalArgumentException("wrong vectors dimensions");
		}
		DoubleMatrix1D ret = DoubleFactory1D.dense.make(v1.size());
		for(int i=0; i<ret.size(); i++){
			ret.setQuick(i, v1.getQuick(i) + v2.getQuick(i));
		}

		return ret;
	}

	/**
	 * Returns v = v1 + c*v2.
	 * Useful in avoiding the need of the copy() in the colt api.
	 */
	public static final DoubleMatrix1D add(DoubleMatrix1D v1, DoubleMatrix1D v2, double c){
		if(v1.size()!=v2.size()){
			throw new IllegalArgumentException("wrong vectors dimensions");
		}
		DoubleMatrix1D ret = DoubleFactory1D.dense.make(v1.size());
		for(int i=0; i<ret.size(); i++){
			ret.setQuick(i, v1.getQuick(i) + c*v2.getQuick(i));
		}

		return ret;
	}

	/**
	 * Returns v = c * v1.
	 * Useful in avoiding the need of the copy() in the colt api.
	 */
	public static final DoubleMatrix1D scalarMult(DoubleMatrix1D v1, double c){
		DoubleMatrix1D ret = DoubleFactory1D.dense.make(v1.size());
		for(int i=0; i<ret.size(); i++){
			ret.setQuick(i, c * v1.getQuick(i));
		}

		return ret;
	}

	public static final Dcs_common.Dcs matrixToDcs(SparseDoubleMatrix2D A) {

		//m (number of rows):
		int m = A.rows();
		//n (number of columns) 
		int n = A.columns();
		//nz (# of entries in triplet matrix, -1 for compressed-col) 
		int nz = -1;
		//nxmax (maximum number of entries)
		int nzmax = m*n;
		//p (column pointers (size n+1) or col indices (size nzmax))
		final int[] p = new int[n+1];
		//i (row indices, size nzmax)
		final int[] i = new int[nzmax];
		//x (numerical values, size nzmax)
		final double[] x = new double[nzmax];

		final int[] currentColumnIndexHolder = new int[]{-1};

		IntIntDoubleFunction myFunct = new IntIntDoubleFunction() {
			int nzCounter = 0;
			@Override
			public double apply(int r, int c, double prc) {
				//log.debug("r:" + r + ", c:" + currentColumnIndexHolder[0] + ": " + prc);

				i[nzCounter] = r;
				x[nzCounter] = prc;
				nzCounter++;

				p[currentColumnIndexHolder[0]+1] = p[currentColumnIndexHolder[0]+1] + 1;
				//log.debug("p: " + ArrayUtils.toString(p));

				return prc;
			}
		};

		//view A column by column
		for (int c = 0; c < n; c++) {
			//log.debug("column:" + c);
			DoubleMatrix2D P = A.viewPart(0, c, m, 1);
			currentColumnIndexHolder[0] = c;
			p[currentColumnIndexHolder[0]+1] = p[currentColumnIndexHolder[0]];
			P.forEachNonZero(myFunct);
		}

		Dcs_common.Dcs dcs = new Dcs_common.Dcs();
		dcs.m = m;
		dcs.n = n;
		dcs.nz = nz;
		dcs.nzmax = nzmax;
		dcs.p = p;
		dcs.i = i;
		dcs.x = x;
		//log.debug("dcs.p: " + ArrayUtils.toString(dcs.p));
		return dcs;
	}

	public static final SparseDoubleMatrix2D dcsToMatrix(Dcs_common.Dcs dcs) {
		SparseDoubleMatrix2D A = new SparseDoubleMatrix2D(dcs.m, dcs.n);
		final int[] rowIndexes = dcs.i;
		final int[] columnPointers = dcs.p;
		final double values[] = dcs.x;
		//for example
		//rowIndexes      2, 0, 2, 3, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
		//columnPointers  0,    2, 3,    5,    7
		//values          2.0, 1.0, 3.0, 4.0, 2.0, 4.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
		int cnt =0;
		for (int j = 0; j < dcs.n; j++) {
			int colStartIndex = columnPointers[j];//example 5
			int colEndIndex = columnPointers[j+1];//example 7
			for (int pointer = colStartIndex; pointer < colEndIndex; pointer++) {
				int i = rowIndexes[pointer];
				A.setQuick(i, j, values[cnt]);
				cnt++;
			}
		}

		//log.debug("A: " + ArrayUtils.toString(A.toArray()));
		return A;
	}

	/**
	 * Return a new array with all the occurences of oldValue replaced by newValue.
	 */
	public static final DoubleMatrix1D replaceValues(DoubleMatrix1D v, double oldValue,	double newValue) {
		if(v == null){
			return null;
		}
		DoubleFactory1D F1 = (v instanceof SparseDoubleMatrix1D)? DoubleFactory1D.sparse : DoubleFactory1D.dense;   
		DoubleMatrix1D ret = F1.make(v.size());
		for (int i = 0; i < v.size(); i++) {
			double vi = v.getQuick(i);
			if (Double.compare(oldValue, vi) != 0) {
				// no substitution
				ret.setQuick(i, vi);
			} else {
				ret.setQuick(i, newValue);
			}
		}
		return ret;
	}

	/**
	 * inversion of {{a, b},{c, d}}
	 * @param A
	 * @return
	 */
	public static final DoubleMatrix2D invert2x2Matrix(DoubleMatrix2D A) throws Exception{
		if(2!=A.rows() || A.rows() != A.columns()){
			throw new IllegalArgumentException("matrix is not 2x2");
		}
		//ad - bc
		double s = A.getQuick(0, 0)*A.getQuick(1, 1)-A.getQuick(0, 1)*A.getQuick(1, 0);
		if(Math.abs(s) < 1.e-16){
			throw new Exception("Matrix is singular");
		}
		DoubleMatrix2D ret = new DenseDoubleMatrix2D(2, 2);
		ret.setQuick(0, 0, A.getQuick(1, 1) / s);
		ret.setQuick(1, 1, A.getQuick(0, 0) / s);
		ret.setQuick(0, 1, -A.getQuick(1, 0) / s);
		ret.setQuick(1, 0, A.getQuick(0, 1) / s);
		return ret;
	}

	public static final DoubleMatrix2D symmPermutation(DoubleMatrix2D A, int from, int to){
		int n = A.rows();
		int[] rowIndexes = new int[n];
		int[] columnIndexes = new int[n];
		for(int i=0; i<n; i++){
			rowIndexes[i] = i;
			columnIndexes[i] = i;
		}
		rowIndexes[from] = to;
		rowIndexes[to] = from;
		columnIndexes[from] = to;
		columnIndexes[to] = from;
		return Algebra.DEFAULT.permute(A, rowIndexes, columnIndexes);
	}

	/**
	 * Returns a lower and an upper bound for the condition number
	 * <br>kp(A) = Norm[A, p] / Norm[A^-1, p]   
	 * <br>where
	 * <br>		Norm[A, p] = sup ( Norm[A.x, p]/Norm[x, p] , x !=0 )
	 * <br>for a matrix and
	 * <br>		Norm[x, 1]  := Sum[Math.abs(x[i]), i] 				
	 * <br>		Norm[x, 2]  := Math.sqrt(Sum[Math.pow(x[i], 2), i])
	 * <br>   Norm[x, 00] := Max[Math.abs(x[i]), i]
	 * <br>for a vector.
	 *  
	 * @param A matrix you want the condition number of
	 * @param p norm order (2 or Integer.MAX_VALUE)
	 * @return an array with the two bounds (lower and upper bound)
	 * 
	 * @see Ravindra S. Gajulapalli, Leon S. Lasdon "Scaling Sparse Matrices for Optimization Algorithms"
	 */
	public static double[] getConditionNumberRange(RealMatrix A, int p) {
		double infLimit = Double.NEGATIVE_INFINITY;
		double supLimit = Double.POSITIVE_INFINITY;
		List<Double> columnNormsList = new ArrayList<Double>();
		switch (p) {
		case 2:
			for(int j=0; j<A.getColumnDimension(); j++){
				columnNormsList.add(A.getColumnVector(j).getL1Norm());
			}
			Collections.sort(columnNormsList);
			//kp >= Norm[Ai, p]/Norm[Aj, p], for each i, j = 0,1,...,n, Ak columns of A
			infLimit = columnNormsList.get(columnNormsList.size()-1) / columnNormsList.get(0);
			break;

		case Integer.MAX_VALUE:
			double normAInf = A.getNorm();
			for(int j=0; j<A.getColumnDimension(); j++){
				columnNormsList.add(A.getColumnVector(j).getLInfNorm());
			}
			Collections.sort(columnNormsList);
			//k1 >= Norm[A, +oo]/min{ Norm[Aj, +oo], for each j = 0,1,...,n }, Ak columns of A
			infLimit = normAInf / columnNormsList.get(0);
			break;

		default:
			throw new IllegalArgumentException("p must be 2 or Integer.MAX_VALUE");
		}
		return new double[]{infLimit, supLimit};
	}

	/**
	 * Given a symm matrix S that stores just its subdiagonal elements, 
	 * reconstructs the full symmetric matrix.
	 * @FIXME: evitare il doppio setQuick
	 */
	public static final DoubleMatrix2D fillSubdiagonalSymmetricMatrix(DoubleMatrix2D S){

		if(S.rows() != S.columns()){
			throw new IllegalArgumentException("Not square matrix");
		}

		boolean isSparse = S instanceof SparseDoubleMatrix2D;
		DoubleFactory2D F2D = (isSparse)? DoubleFactory2D.sparse: DoubleFactory2D.dense;
		final DoubleMatrix2D SFull = F2D.make(S.rows(), S.rows());

		if (isSparse) {
			S.forEachNonZero(new IntIntDoubleFunction() {
				@Override
				public double apply(int i, int j, double hij) {
					SFull.setQuick(i, j, hij);
					SFull.setQuick(j, i, hij);
					return hij;
				}
			});
		} else {
			for (int i = 0; i < S.rows(); i++) {
				for (int j = 0; j < i + 1; j++) {
					double sij = S.getQuick(i, j);
					SFull.setQuick(i, j, sij);
					SFull.setQuick(j, i, sij);
				}
			}
		}

		return SFull;
	}
}
