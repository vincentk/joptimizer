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
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import cern.jet.math.Functions;

import com.joptimizer.util.ColtUtils;
import com.joptimizer.util.Utils;

/**
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class CholeskySparseFactorizationTest extends TestCase {

	protected Algebra ALG = Algebra.DEFAULT;
	protected DoubleFactory2D F2 = DoubleFactory2D.dense;
	protected DoubleFactory1D F1 = DoubleFactory1D.dense;
	private Log log = LogFactory.getLog(this.getClass().getName());

	public void testSimple1_norescaling() throws Exception {
		log.debug("testSimple1_norescaling");
		double[][] A = new double[][] { 
				{ 4,  0,  2,  2 }, 
				{ 0,  4,  2, -2 }, 
				{ 2,  2,  6,  0 }, 
				{ 2, -2,  0,  6 } };
		//expected L
		double[][] EL = new double[][] { 
				{ 2,  0,  0,  0 }, 
				{ 0,  2,  0,  0 }, 
				{ 1,  1,  2,  0 }, 
				{ 1, -1,  0,  2 } };
		
		SparseDoubleMatrix2D ASparse = new SparseDoubleMatrix2D(A);
		CholeskySparseFactorization cs = new CholeskySparseFactorization(ASparse);
		cs.factorize();
		DoubleMatrix2D L = cs.getL();
		DoubleMatrix2D LT = cs.getLT();
		log.debug("L : " + ArrayUtils.toString(L.toArray()));
		log.debug("LT: " + ArrayUtils.toString(LT.toArray()));
		
		RealMatrix ELMatrix = MatrixUtils.createRealMatrix(EL);
		RealMatrix LMatrix = MatrixUtils.createRealMatrix(L.toArray());
		RealMatrix LTMatrix = MatrixUtils.createRealMatrix(LT.toArray());
		assertEquals((ELMatrix.subtract(LMatrix).getNorm()), 0.);
		assertEquals((ELMatrix.subtract(LTMatrix.transpose()).getNorm()), 0.);
		
		//A.x = b
		double[] b = new double[]{1, 2, 3, 4};
		double[] x = cs.solve(F1.make(b)).toArray();
		
		//check the norm ||A.x-b||
		double norm = new Array2DRowRealMatrix(A).operate(new ArrayRealVector(x)).subtract(new ArrayRealVector(b)).getNorm();
		log.debug("norm: " + norm);
		assertEquals(0., norm, Utils.getDoubleMachineEpsilon());
		
		//check the scaled residual
		double residual = Utils.calculateScaledResidual(A, x, b);
		log.debug("residual: " + residual);
		assertEquals(0., residual, Utils.getDoubleMachineEpsilon());
	}
	
	public void testSimple1_rescaling() throws Exception {
		log.debug("testSimple1_rescaling");
		double[][] A = new double[][] { 
				{ 4,  0,  2,  2 }, 
				{ 0,  4,  2, -2 }, 
				{ 2,  2,  6,  0 }, 
				{ 2, -2,  0,  6 } };
		//expected L
		double[][] EL = new double[][] { 
				{ 2,  0,  0,  0 }, 
				{ 0,  2,  0,  0 }, 
				{ 1,  1,  2,  0 }, 
				{ 1, -1,  0,  2 } };
		
		SparseDoubleMatrix2D ASparse = new SparseDoubleMatrix2D(A);
		CholeskySparseFactorization cs = new CholeskySparseFactorization(ASparse, new Matrix1NornRescaler());
		cs.factorize();
		DoubleMatrix2D L = cs.getL();
		DoubleMatrix2D LT = cs.getLT();
		log.debug("L : " + ArrayUtils.toString(L.toArray()));
		log.debug("LT: " + ArrayUtils.toString(LT.toArray()));
		
		RealMatrix ELMatrix = MatrixUtils.createRealMatrix(EL);
		RealMatrix LMatrix = MatrixUtils.createRealMatrix(L.toArray());
		RealMatrix LTMatrix = MatrixUtils.createRealMatrix(LT.toArray());
		assertEquals((ELMatrix.subtract(LMatrix).getNorm()), 0., Utils.getDoubleMachineEpsilon());
		assertEquals((ELMatrix.subtract(LTMatrix.transpose()).getNorm()), 0., Utils.getDoubleMachineEpsilon());
		
		//A.x = b
		double[] b = new double[]{1, 2, 3, 4};
		double[] x = cs.solve(F1.make(b)).toArray();
		
		//check the norm ||A.x-b||
		double norm = new Array2DRowRealMatrix(A).operate(new ArrayRealVector(x)).subtract(new ArrayRealVector(b)).getNorm();
		log.debug("norm: " + norm);
		assertEquals(0., norm, 1.e-12);
		
		//check the scaled residual
		double residual = Utils.calculateScaledResidual(A, x, b);
		log.debug("residual: " + residual);
		assertEquals(0., residual, Utils.getDoubleMachineEpsilon());
	}
	
	public void testSimple2() throws Exception {
		log.debug("testSimple2");
		double[][] A = new double[][] { 
				{ 4,  0,  0,  1 }, 
				{ 0,  4,  0, -1 }, 
				{ 0,  0,  6,  1 }, 
				{ 1, -1,  1,  6 } };
		
		CholeskySparseFactorization cs = new CholeskySparseFactorization(new SparseDoubleMatrix2D(A));
		cs.factorize();
		DoubleMatrix2D L = cs.getL();
		DoubleMatrix2D LT = cs.getLT();
		log.debug("L : " + ArrayUtils.toString(L.toArray()));
		log.debug("LT: " + ArrayUtils.toString(LT.toArray()));
		
		//check the norm ||A.x-b||
		double[] b = new double[]{1, 2, 3, 4};
		double[] x = cs.solve(F1.make(b)).toArray();
		double norm = new Array2DRowRealMatrix(A).operate(new ArrayRealVector(x)).subtract(new ArrayRealVector(b)).getNorm();
		log.debug("norm: " + norm);
		assertEquals(0., norm, 1.e-15);
		
		//check the scaled residual
		double residual = Utils.calculateScaledResidual(A, x, b);
		log.debug("residual: " + residual);
		assertEquals(0., residual, Utils.getDoubleMachineEpsilon());
	}
	
	/**
	 * This test shows that the correct check of the inversion accuracy must be done with
	 * the scaled residual, not with the simple norm ||A.x-b||
	 */
	public void testScaledResidual() throws Exception{
		log.debug("testScaledResidual");
		double[][] A = Utils.loadDoubleMatrixFromFile("factorization" + File.separator	+ "matrix1.csv");
		RealMatrix Q = MatrixUtils.createRealMatrix(A);
		int dim = Q.getRowDimension();
	
		RealVector b = new ArrayRealVector(new double[]{1,2,3,4,5,6,7,8,9,10});
		
		CholeskySparseFactorization cs = new CholeskySparseFactorization(new SparseDoubleMatrix2D(Q.getData()));
		cs.factorize();
		RealVector x = new ArrayRealVector(cs.solve(F1.make(b.toArray())).toArray());
		
		//scaledResidual = ||Ax-b||_oo/( ||A||_oo . ||x||_oo + ||b||_oo )
		// with ||x||_oo = max(x[i])
		double residual = Utils.calculateScaledResidual(A, x.toArray(), b.toArray());
		log.debug("residual: " + residual);
		assertTrue(residual < Utils.getDoubleMachineEpsilon());
		
		//b - Q.x
		//checking the simple norm, this will fail
		double n1 = b.subtract(Q.operate(x)).getNorm();
		log.debug("||b - Q.x||: " + n1);
		//assertTrue(n1 < 1.E-8);
	}
	
	/**
	 * Tests vector and matrix solve method.
	 */
	public void testFromFile3() throws Exception{
		log.debug("testFromFile3");
		String matrixId = "3";
		double[][] G = Utils.loadDoubleMatrixFromFile("factorization" + File.separator + "matrix" + matrixId + ".csv");
		RealMatrix Q = MatrixUtils.createRealMatrix(G);
		int dim = Q.getRowDimension();
	
		CholeskySparseFactorization myc = new CholeskySparseFactorization(new SparseDoubleMatrix2D(G));
		myc.factorize();
		
		//solve for a vector
		RealVector b = new ArrayRealVector(Utils.randomValuesVector(dim, -0.5, 0.5, new Long(dim)).toArray());
		RealVector x = new ArrayRealVector(myc.solve(F1.make(b.toArray())).toArray());
		
		//b - Q.x
		double n1 = b.subtract(Q.operate(x)).getNorm();
		double sr1 = Utils.calculateScaledResidual(G, x.toArray(), b.toArray());
		log.debug("||b - Q.x||: " + n1);
		log.debug("scaled res : " + sr1);
		assertTrue(n1  < 1.E-8);
		assertTrue(sr1 < Utils.getDoubleMachineEpsilon());
		
		//solve for a matrix
		RealMatrix B = new Array2DRowRealMatrix(Utils.randomValuesMatrix(dim, 10, -0.5, 0.5, new Long(dim)).toArray());
		RealMatrix X = new Array2DRowRealMatrix(myc.solve(F2.make(B.getData())).toArray());
		
		//B - Q.X
		double n2 = B.subtract(Q.multiply(X)).getNorm();
		double sr2 = Utils.calculateScaledResidual(G, X.getData(), B.getData());
		log.debug("||B - Q.X||: " + n2);
		log.debug("scaled res : " + sr2);
		assertTrue(n2 < 1.E-8);
		assertTrue(sr2 < Utils.getDoubleMachineEpsilon());	
	}	
	
	/**
	 * The matrix6 has a regular Cholesky factorization (as given by Mathematica) 
	 * This test shows how rescaling a matrix can help its factorization.
	 */
	public void testScale6() throws Exception {
		log.debug("testScale6");
		DoubleFactory2D F2 = DoubleFactory2D.sparse;
		DoubleFactory1D F1 = DoubleFactory1D.sparse;
		Algebra ALG = Algebra.DEFAULT;
		
		String matrixId = "6";
		double[][] A = Utils.loadDoubleMatrixFromFile("factorization" + File.separator + "matrix" + matrixId + ".txt", " ".charAt(0));
		SparseDoubleMatrix2D AMatrix = (SparseDoubleMatrix2D) F2.make(A);
		int dim = AMatrix.rows();
		
		CholeskySparseFactorization cs;
		try{
			cs = new CholeskySparseFactorization(AMatrix);
			cs.factorize();
		}catch(Exception e){
			log.debug("numeric problem, try to rescale the matrix");
			MatrixRescaler rescaler = new Matrix1NornRescaler();
			DoubleMatrix1D Uv = rescaler.getMatrixScalingFactorsSymm(AMatrix);
			DoubleMatrix2D U = F2.diagonal(Uv);
			
			assertTrue(rescaler.checkScaling(ColtUtils.fillSubdiagonalSymmetricMatrix(AMatrix), Uv, Uv));
			
			DoubleMatrix2D AScaled = ColtUtils.diagonalMatrixMult(Uv, AMatrix, Uv);
			cs = new CholeskySparseFactorization((SparseDoubleMatrix2D)AScaled);
			cs.factorize();
			
			//NOTE: with scaling, we must solve U.A.U.z = U.b, after that we have x = U.z
			
			//solve Q.x = b
			DoubleMatrix1D b = Utils.randomValuesVector(dim, -1, 1, 12345L);
			DoubleMatrix1D x = cs.solve(ALG.mult(U, b));
			double scaledResidualx = Utils.calculateScaledResidual(AMatrix, ALG.mult(U, x), b);
			log.debug("scaledResidualx: " + scaledResidualx);
			assertTrue(scaledResidualx < Utils.getDoubleMachineEpsilon());
			
			//solve Q.X = B
			DoubleMatrix2D B = Utils.randomValuesMatrix(dim, 5, -1, 1, 12345L);
			DoubleMatrix2D X = cs.solve(ALG.mult(U, B));
			double scaledResidualX = Utils.calculateScaledResidual(AMatrix, ALG.mult(U, X), B);
			log.debug("scaledResidualX: " + scaledResidualX);
			assertTrue(scaledResidualX < Utils.getDoubleMachineEpsilon());
		}
	}
	
	/**
	 * The matrix10 has a regular Cholesky factorization (as given by Mathematica)
	 */
	public void testSolve10() throws Exception {
		log.debug("testSolve");
		DoubleFactory2D F2 = DoubleFactory2D.sparse;
		DoubleFactory1D F1 = DoubleFactory1D.sparse;
		Algebra ALG = Algebra.DEFAULT;
		
		String matrixId = "10";
		double[][] A = Utils.loadDoubleMatrixFromFile("factorization" + File.separator + "matrix" + matrixId + ".txt", " ".charAt(0));
		SparseDoubleMatrix2D AMatrix = (SparseDoubleMatrix2D) F2.make(A);
		int dim = AMatrix.rows();
		
		CholeskySparseFactorization cs = new CholeskySparseFactorization(AMatrix);
		cs.factorize();
			
		//solve Q.x = b
		DoubleMatrix1D b = Utils.randomValuesVector(dim, -1, 1, 12345L);
		DoubleMatrix1D x = cs.solve(b);
		double scaledResidualx = Utils.calculateScaledResidual(AMatrix, x, b);
		log.debug("scaledResidualx: " + scaledResidualx);
		assertTrue(scaledResidualx < 1.e-12);
		
		//solve Q.X = B
		DoubleMatrix2D B = Utils.randomValuesMatrix(dim, 5, -1, 1, 12345L);
		DoubleMatrix2D X = cs.solve(B);
		double scaledResidualX = Utils.calculateScaledResidual(AMatrix, X, B);
		log.debug("scaledResidualX: " + scaledResidualX);
		assertTrue(scaledResidualX < 1.e-12);
	}
}
