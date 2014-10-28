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

import junit.framework.TestCase;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import com.joptimizer.util.ColtUtils;

import cern.colt.function.IntIntDoubleFunction;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import edu.emory.mathcs.csparsej.tdouble.Dcs_common;

public class ColtUtilsTest extends TestCase {

	private Log log = LogFactory.getLog(this.getClass().getName());
	
	/**
	 * Use of a Colt sparse matrix
	 */
	public void testDumpSparseMatrix() {
		log.debug("testDumpSparseMatrix");
		final double[][] A = new double[][]{
				{1, 0, 0, 2},
				{0, 0, 2, 0},
				{2, 3, 0, 0},
				{0, 0, 4, 4}
		};
		
//		SparseCCDoubleMatrix2D S1 = new SparseCCDoubleMatrix2D(4, 4);
//		ColtUtils.dumpSparseMatrix(S1);
		
		SparseDoubleMatrix2D S2 = new SparseDoubleMatrix2D(A);
		//ColtUtils.dumpSparseMatrix(S2);
		S2.forEachNonZero(new IntIntDoubleFunction() {
			@Override
			public double apply(int i, int j, double sij) {
				assertEquals(sij, A[i][j]);
				return sij;
			}
		});
	}
	
	/**
	 * Use of a Colt sparse matrix
	 */
	public void testDumpSparseMatrix2() {
		log.debug("testDumpSparseMatrix2");
		final double[][] A = new double[][]{
				{1, 0, 0, 2},
				{0, 0, 2, 0},
				{2, 3, 0, 0},
				{0, 0, 4, 4}
		};
				
		SparseDoubleMatrix2D S2 = new SparseDoubleMatrix2D(A);
		log.debug("S: " + ArrayUtils.toString(S2.toArray()));
		
		DoubleMatrix2D R = S2.viewPart(0, 0, 1, 4);
		R.forEachNonZero(new IntIntDoubleFunction() {
			@Override
			public double apply(int i, int j, double sij) {
				log.debug("i:" + i + ", j:" + j + ": " + sij);
				return sij;
			}
		});
		R.setQuick(0, 1, 7.0);
		log.debug("S: " + ArrayUtils.toString(S2.toArray()));
		assertEquals(7.0, S2.getQuick(0, 1));//the change on R is also in S2
	}
	
	public void testSubdiagonalMultiply() {
		log.debug("testSubdiagonalMultiply");
		double[][] A = {{1, 2, 3, 4}, {5, 6, 7, 8}, {1, 3, 5, 7}};
		double[][] B = {{1, 2, 3}, {3, 4, 2}, {5, 6, 7}, {7, 8, 9}};
		//double[][] expectedResult = {{50, 60, 64}, {114, 140, 148}, {84, 100, 107}}; 
		double[][] expectedResult = {
				{50, 0, 0}, 
				{114, 140, 0}, 
				{84, 100, 107}};
		
		// with sparsity
		DoubleMatrix2D ASparse = DoubleFactory2D.sparse.make(A);
		DoubleMatrix2D BSparse = DoubleFactory2D.sparse.make(B);
		DoubleMatrix2D ret1 = ColtUtils.subdiagonalMultiply(ASparse, BSparse);
		log.debug("ret1: " + ArrayUtils.toString(ret1.toArray()));
		for (int i = 0; i < expectedResult.length; i++) {
			for (int j = 0; j < expectedResult[i].length; j++) {
				assertEquals(expectedResult[i][j], ret1.getQuick(i, j));
			}
		}

		// with no sparsity
		DoubleMatrix2D ADense = DoubleFactory2D.dense.make(A);
		DoubleMatrix2D BDense = DoubleFactory2D.dense.make(B);
		DoubleMatrix2D ret2 = ColtUtils.subdiagonalMultiply(ADense, BDense);
		log.debug("ret2: " + ArrayUtils.toString(ret2.toArray()));
		for (int i = 0; i < expectedResult.length; i++) {
			for (int j = 0; j < expectedResult[i].length; j++) {
				assertEquals(expectedResult[i][j], ret1.getQuick(i, j));
			}
		}
	}
	
	/**
	 * Manually compose a Dcs representation of a sparse matrix
	 */
	public void testMatrixToDcs() {
		log.debug("testMatrixToDcs");
		final double[][] A = new double[][]{
				{1, 0, 0, 2},
				{0, 0, 2, 0},
				{2, 3, 0, 0},
				{0, 0, 4, 4}
		};
		SparseDoubleMatrix2D S2 = new SparseDoubleMatrix2D(A);
		
		//expected representation edu.emory.mathcs.csparsej.tdouble.Dcs_common.Dcs
		//i (row indices, size nzmax):
		int[] expected_i = new int[]{2, 0, 2, 3, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};//S2.dcs.i
		//m (number of rows):
		int expected_m = 4;//S2.dcs.m
		//n (number of columns) 
		int expected_n = 4;//S2.dcs.n
		//nz (# of entries in triplet matrix, -1 for compressed-col) 
		int expected_nz = -1;//S2.dcs.nz
		//nxmax (maximum number of entries)
		int expected_nzmax = 16;//S2.dcs.nzmax
		//p (column pointers (size n+1) or col indices (size nzmax))
		int[] expected_p = new int[]{0, 2, 3, 5, 7};//S2.dcs.p
		//x (numerical values, size nzmax)
		double[] expected_x = new double[]{2.0, 1.0, 3.0, 4.0, 2.0, 4.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};//S2.dcs.x
		
		SparseDoubleMatrix2D AMatrix = new SparseDoubleMatrix2D(A);
		Dcs_common.Dcs dcs = ColtUtils.matrixToDcs(AMatrix);
		
		//assertions
		assertEquals(expected_i.length, dcs.i.length);
		for(int i=0; i<expected_i.length; i++){
			assertEquals(expected_i[i], dcs.i[i]);
		}
		assertEquals(expected_m, dcs.m);
		assertEquals(expected_n, dcs.n);
		assertEquals(expected_nz, dcs.nz);
		assertEquals(expected_nzmax, dcs.nzmax);
		assertEquals(expected_p.length, dcs.p.length);
		for(int i=0; i<expected_p.length; i++){
			assertEquals(expected_p[i], dcs.p[i]);
		}
		assertEquals(expected_x.length, dcs.x.length);
		for(int i=0; i<expected_x.length; i++){
			assertEquals(expected_x[i], dcs.x[i]);
		}
	}
	
	public void testDcsToMatrix() {
		log.debug("testDcsToMatrix");
		
		Dcs_common.Dcs dcs = new Dcs_common.Dcs();
		dcs.i = new int[]{2, 0, 2, 3, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
		dcs.m = 4;
		dcs.n = 4;
		dcs.nz = -1;
		dcs.nzmax = 16;
		dcs.p = new int[]{0, 2, 3, 5, 7};
		dcs.x = new double[]{2.0, 1.0, 3.0, 4.0, 2.0, 4.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
		
		double[][] expectedA = new double[][]{
				{1, 0, 0, 2},
				{0, 0, 2, 0},
				{2, 3, 0, 0},
				{0, 0, 4, 4}
		};
		DoubleMatrix2D S2 = ColtUtils.dcsToMatrix(dcs);
		
		//assertions
		assertEquals(4, S2.rows());
		assertEquals(4, S2.columns());
		for(int i=0; i<4; i++){
			for(int j=0; j<4; j++){
				assertEquals(expectedA[i][j], S2.getQuick(i, j));
			}
		}
	}
	
	public void testGetConditionNumberRanges() throws Exception {
		log.debug("testConditionNumberRanges");
		double[][] A = new double[][] {
				{1., 0, 0}, 
				{0., 2., 0}, 
				{0., 0., 3.}};
		double kExpected2 = 3;
		double kExpected00 = 3;
		RealMatrix AMatrix = new Array2DRowRealMatrix(A);
		double[] cn_2 = ColtUtils.getConditionNumberRange(AMatrix, 2);
		double[] cn_00 = ColtUtils.getConditionNumberRange(AMatrix, Integer.MAX_VALUE);
		log.debug("cn_2 : " + ArrayUtils.toString(cn_2));
		log.debug("cn_00: " + ArrayUtils.toString(cn_00));
		assertTrue(kExpected2 >= cn_2[0]);
		assertTrue(kExpected00 >= cn_00[0]);
	}
	
	public void testSymmPermutation1() throws Exception {
		log.debug("testSymmPermutation1");
		DoubleFactory2D F2 = DoubleFactory2D.dense;
		Algebra ALG = Algebra.DEFAULT;
		double[][] A = new double[][] {
				{0.0, 0.1, 0.2}, 
				{1.0, 1.1, 1.2}, 
				{2.0, 2.1, 2.2}};
		double[][] P = new double[][] {
				{1, 0, 0}, 
				{0, 0, 1}, 
				{0, 1, 0}};
		DoubleMatrix2D AMatrix = F2.make(A);
		DoubleMatrix2D PMatrix = F2.make(P);
		DoubleMatrix2D APermuted = ColtUtils.symmPermutation(AMatrix, 1, 2);
		log.debug("APermuted: " + ArrayUtils.toString(APermuted.toArray()));
		DoubleMatrix2D E = ALG.mult(PMatrix, ALG.mult(AMatrix, ALG.transpose(PMatrix)));
		double norm = MatrixUtils.createRealMatrix(E.toArray()).subtract(MatrixUtils.createRealMatrix(APermuted.toArray())).getNorm();
		log.debug("norm: " + norm);
		assertEquals(0, norm, 1.e-15);
	}
	
	public void testSymmPermutation2() throws Exception {
		log.debug("testSymmPermutation2");
		DoubleFactory2D F2 = DoubleFactory2D.dense;
		Algebra ALG = Algebra.DEFAULT;
		double[][] A = new double[][] {
				{0, 1}, 
				{1, 0}};
		double[][] P = new double[][] {
				{0, 1}, 
				{1, 0}};
		DoubleMatrix2D AMatrix = F2.make(A);
		DoubleMatrix2D PMatrix = F2.make(P);
		DoubleMatrix2D APermuted = ColtUtils.symmPermutation(AMatrix, 0, 1);
		log.debug("APermuted: " + ArrayUtils.toString(APermuted.toArray()));
		DoubleMatrix2D E = ALG.mult(PMatrix, ALG.mult(AMatrix, ALG.transpose(PMatrix)));
		double norm = MatrixUtils.createRealMatrix(E.toArray()).subtract(MatrixUtils.createRealMatrix(APermuted.toArray())).getNorm();
		log.debug("norm: " + norm);
		assertEquals(0, norm, 1.e-15);
	}
}
