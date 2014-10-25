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

import java.io.File;

import junit.framework.TestCase;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.QRDecomposition;
import org.apache.commons.math3.linear.RealMatrix;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import cern.jet.math.Functions;

import com.joptimizer.util.Utils;

public class QRSparseFactorizationTest extends TestCase {
	
	private Log log = LogFactory.getLog(this.getClass().getName());

	public void testDecompose() throws Exception{
		log.debug("testDecompose");
		final double[][] A = new double[][]{
				{1, 0, 0, 2},
				{0, 0, 2, 0},
				{2, 3, 0, 0},
				{0, 0, 4, 4}
		};
		
		double[][] EQ = { 
				{ 0.447214, -0.894427, 0., 0. },
				{ 0., 0., 0.447214, -0.894427 }, 
				{ 0.894427, 0.447214, 0., 0. },
				{ 0., 0., 0.894427, 0.447214 } };
		double[][] ER = { 
				{ 2.23607, 2.68328, 0., 0.894427 },
				{ 0., 1.34164, 0., -1.78885 }, 
				{ 0., 0., 4.47214, 3.57771 },
				{ 0., 0., 0., 1.78885 } };
		
		QRDecomposition dFact = new QRDecomposition(new Array2DRowRealMatrix(A));
		RealMatrix Q = dFact.getQ();
		RealMatrix R = dFact.getR();
		RealMatrix H = dFact.getH();
		log.debug("Q: " + ArrayUtils.toString(Q.getData()));
		log.debug("R: " + ArrayUtils.toString(R.getData()));
		//log.debug("H: " + ArrayUtils.toString(H.getData()));
		
		SparseDoubleMatrix2D S = new SparseDoubleMatrix2D(A);
		QRSparseFactorization qr = new QRSparseFactorization(S);
		qr.factorize();
		log.debug("R: " + ArrayUtils.toString(qr.getR().toArray()));
		for(int i=0; i<R.getRowDimension(); i++){
			for(int j=0; j<R.getColumnDimension(); j++){
				assertEquals(ER[i][j], qr.getR().getQuick(i, j), 1.e-5);
			}
		}
		assertTrue(qr.hasFullRank());
	}
	
	public void testSolve() throws Exception{
		log.debug("testSolve");
		double[][] A = new double[][]{
				{1.0, 0.0, -1.0, -1.0},
				{0.0, -1.0, 0.5, 1.0}
		};
		double[] b = new double[]{1.0, 2.0};
		
		SparseDoubleMatrix2D AMatrix = new SparseDoubleMatrix2D(A);
		DoubleMatrix1D bVector = new DenseDoubleMatrix1D(b);
		
		QRSparseFactorization qr = new QRSparseFactorization(AMatrix);
		qr.factorize();
		assertTrue(qr.hasFullRank());
		DoubleMatrix1D x = qr.solve(bVector);		
		double scaledResidual = Utils.calculateScaledResidual(AMatrix, x, bVector);
		log.debug("scaledResidual: " + scaledResidual);
		assertTrue(scaledResidual < Utils.getDoubleMachineEpsilon());
	}
	
	public void testSolveWithScaling() throws Exception{
		log.debug("testSolveWithScaling");
		double[][] A = new double[][]{
				{1.0, 0.0, -1.0, -1.0},
				{0.0, -1.0, 0.5, 1.0}
		};
		double[] b = new double[]{1.0, 2.0};
		
		SparseDoubleMatrix2D AMatrix = new SparseDoubleMatrix2D(A);
		DoubleMatrix1D bVector = new DenseDoubleMatrix1D(b);
		
		//a bit more complicated
		AMatrix.assign(Functions.mult(1.2345e7));
		bVector.assign(Functions.mult(1.2345e7));
		
		QRSparseFactorization qr = new QRSparseFactorization(AMatrix, new Matrix1NornRescaler());
		qr.factorize();
		assertTrue(qr.hasFullRank());
		DoubleMatrix1D x = qr.solve(bVector);
		double scaledResidual = Utils.calculateScaledResidual(AMatrix, x, bVector);
		log.debug("scaledResidual: " + scaledResidual);
		assertTrue(scaledResidual < Utils.getDoubleMachineEpsilon());		
	}
	
	public void testSolve9() throws Exception {
		log.debug("testSolve9");
		double[][] A = Utils.loadDoubleMatrixFromFile("factorization" + File.separator	+ "matrix9.txt", " ".charAt(0));
		double[] b = Utils.loadDoubleArrayFromFile("factorization" + File.separator	+ "vector9.txt");
		
		SparseDoubleMatrix2D AMatrix = new SparseDoubleMatrix2D(A);
		DoubleMatrix1D bVector = new DenseDoubleMatrix1D(b);
		
		QRSparseFactorization qr = new QRSparseFactorization(AMatrix);
		qr.factorize();
		assertTrue(qr.hasFullRank());
		DoubleMatrix1D x = qr.solve(bVector);		
		double scaledResidual = Utils.calculateScaledResidual(AMatrix, x, bVector);
		log.debug("scaledResidual: " + scaledResidual);
		assertTrue(scaledResidual < Utils.getDoubleMachineEpsilon());
	}
	
	/**
	 * This test is with a rank-deficient matrix, and must reveal it.
	 * It is a 381x694 matrix with rank=379.
	 */
	public void testSolve16() throws Exception {
		log.debug("testSolve16");
		double[][] A = Utils.loadDoubleMatrixFromFile("factorization" + File.separator	+ "matrix16.txt", " ".charAt(0));
		double[] b = Utils.loadDoubleArrayFromFile("factorization" + File.separator	+ "vector16.txt");
		
		//JAMA
//		Matrix M1 = new Matrix(A);
//		Jama.QRDecomposition qr1 = new Jama.QRDecomposition(M1);
//		assertFalse(qr1.isFullRank());//this is OK
		
//		//COMMONS-MATH 3
//		//NOTE: decomposing A[T] gives the opposite result
//		RealMatrix M2 = MatrixUtils.createRealMatrix(A);
//		org.apache.commons.math3.linear.QRDecomposition qr2 = new org.apache.commons.math3.linear.QRDecomposition(M2);
//		assertFalse(qr2.getSolver().isNonSingular());//this is OK
//		
//		//COLT
//		SparseDoubleMatrix2D M3 = new SparseDoubleMatrix2D(A);
//		cern.colt.matrix.linalg.QRDecomposition qr3 = new cern.colt.matrix.linalg.QRDecomposition(Algebra.DEFAULT.transpose(M3));
//		assertFalse(qr3.hasFullRank());//this fails
//		
//		//PARALLEL-COLT
//		SparseCCDoubleMatrix2D M4 = new SparseCCDoubleMatrix2D(A);
//		cern.colt.matrix.tdouble.algo.decomposition.SparseDoubleQRDecomposition qr4 = new cern.colt.matrix.tdouble.algo.decomposition.SparseDoubleQRDecomposition(M4, 0);
//		assertFalse(qr4.hasFullRank());//this fails
		
		//JOptimizer
		SparseDoubleMatrix2D M5 = new SparseDoubleMatrix2D(A);
		QRSparseFactorization qr5 = new QRSparseFactorization(M5);
		qr5.factorize();
		assertFalse(qr5.hasFullRank());
		
	}
	
	/**
	 * This test is the same as testSolve16, with its transpose matrix 
	 * It is a 694x381 matrix with rank=379.
	 */
	public void testSolve16Transpose() throws Exception {
		log.debug("testSolve16Transpose");
		double[][] A = Utils.loadDoubleMatrixFromFile("factorization" + File.separator	+ "matrix16.txt", " ".charAt(0));
		
//		//COMMONS-MATH3-3.2
//		RealMatrix M2 = MatrixUtils.createRealMatrix(A);
//		org.apache.commons.math3.linear.SingularValueDecomposition sv2 = new org.apache.commons.math3.linear.SingularValueDecomposition(M2);
//		assertEquals(379, sv2.getRank());//this is OK
//		org.apache.commons.math3.linear.QRDecomposition qr2 = new org.apache.commons.math3.linear.QRDecomposition(M2);
//		assertFalse(qr2.getSolver().isNonSingular());//this fails
//		
//		//COLT
//		SparseDoubleMatrix2D M3 = new SparseDoubleMatrix2D(A);
//		cern.colt.matrix.linalg.QRDecomposition qr3 = new cern.colt.matrix.linalg.QRDecomposition(M3);
//		assertFalse(qr3.hasFullRank());//this fails
//		
//		//PARALLEL-COLT
//		SparseCCDoubleMatrix2D M4 = new SparseCCDoubleMatrix2D(A);
//		cern.colt.matrix.tdouble.algo.decomposition.SparseDoubleQRDecomposition qr4 = new cern.colt.matrix.tdouble.algo.decomposition.SparseDoubleQRDecomposition(M4, 0);
//		assertFalse(qr4.hasFullRank());//this fails
		
		//JOptimizer
		SparseDoubleMatrix2D M5 = (SparseDoubleMatrix2D) Algebra.DEFAULT.transpose(new SparseDoubleMatrix2D(A));
		QRSparseFactorization qr5 = new QRSparseFactorization(M5);
		qr5.factorize();
		assertFalse(qr5.hasFullRank());
	}
}
