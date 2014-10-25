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
import org.apache.commons.math3.linear.MatrixUtils;

import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import cern.colt.matrix.linalg.Property;

import com.joptimizer.util.TestUtils;

public class LDLTPermutedFactorizationTest extends TestCase {

	private Log logger = LogFactory.getLog(this.getClass().getName());
	
	/**
	 * Simple test positive
	 */
	public void testPldltpt() throws Exception {
		logger.debug("testPldltpt");
		DoubleFactory2D F2 = DoubleFactory2D.dense;
		Algebra ALG = Algebra.DEFAULT;
		double[][] A = new double[][] {
				{3, 0, 0}, 
				{0, 2, 0},
				{0, 0, 1}};
		DoubleMatrix2D AMatrix = F2.make(A);
		LDLTPermutedFactorization fact = new LDLTPermutedFactorization(AMatrix, LDLTPermutedFactorization.DIAGONAL_PIVOLTING);
		fact.factorize();
		DoubleMatrix2D P = fact.getP();
		DoubleMatrix2D D = fact.getD();
		DoubleMatrix2D L = fact.getL();
		DoubleMatrix2D LDLT = ALG.mult(L, ALG.mult(D, ALG.transpose(L)));
		DoubleMatrix2D PLDLTPT = ALG.mult(P, ALG.mult(LDLT, ALG.transpose(P)));
		double norm = MatrixUtils.createRealMatrix(A).subtract(MatrixUtils.createRealMatrix(PLDLTPT.toArray())).getNorm();
		logger.debug("norm: " + norm);
		assertEquals(0, norm, 1.e-15);
	}
	
	/**
	 * Simple test positive
	 */
	public void testPldltptBK() throws Exception {
		logger.debug("testPldltptBK");
		DoubleFactory2D F2 = DoubleFactory2D.dense;
		Algebra ALG = Algebra.DEFAULT;
		double[][] A = new double[][] {
				{3, 0, 0}, 
				{0, 2, 0},
				{0, 0, 1}};
		DoubleMatrix2D AMatrix = F2.make(A);
		LDLTPermutedFactorization fact = new LDLTPermutedFactorization(AMatrix, LDLTPermutedFactorization.DIAGONAL_PIVOLTING_WITH_PARTIAL_PIVOTING);  
		fact.factorize();
		DoubleMatrix2D P = fact.getP();
		DoubleMatrix2D D = fact.getD();
		DoubleMatrix2D L = fact.getL();
		DoubleMatrix2D LDLT = ALG.mult(L, ALG.mult(D, ALG.transpose(L)));
		DoubleMatrix2D PLDLTPT = ALG.mult(P, ALG.mult(LDLT, ALG.transpose(P)));
		double norm = MatrixUtils.createRealMatrix(A).subtract(MatrixUtils.createRealMatrix(PLDLTPT.toArray())).getNorm();
		logger.debug("norm: " + norm);
		assertEquals(0, norm, 1.e-15);
	}
	
	/**
	 * Simple test indefinite
	 */
	public void testPldltpt2() throws Exception {
		logger.debug("testPldltpt2");
		DoubleFactory2D F2 = DoubleFactory2D.dense;
		Algebra ALG = Algebra.DEFAULT;
		double[][] A = new double[][] {
				{3,  0, 0}, 
				{0, -2, 0},
				{0,  0, 1}};
		DoubleMatrix2D AMatrix = F2.make(A);
		LDLTPermutedFactorization fact = new LDLTPermutedFactorization(AMatrix, LDLTPermutedFactorization.DIAGONAL_PIVOLTING);  
		fact.factorize();
		DoubleMatrix2D P = fact.getP();
		DoubleMatrix2D D = fact.getD();
		DoubleMatrix2D L = fact.getL();
		DoubleMatrix2D LDLT = ALG.mult(L, ALG.mult(D, ALG.transpose(L)));
		DoubleMatrix2D PLDLTPT = ALG.mult(P, ALG.mult(LDLT, ALG.transpose(P)));
		double norm = MatrixUtils.createRealMatrix(A).subtract(MatrixUtils.createRealMatrix(PLDLTPT.toArray())).getNorm();
		logger.debug("norm: " + norm);
		assertEquals(0, norm, 1.e-15);
	}
	
	/**
	 * Simple test indefinite
	 */
	public void testPldltpt2BK() throws Exception {
		logger.debug("testPldltpt2BK");
		DoubleFactory2D F2 = DoubleFactory2D.dense;
		Algebra ALG = Algebra.DEFAULT;
		double[][] A = new double[][] {
				{3,  0, 0}, 
				{0, -2, 0},
				{0,  0, 1}};
		DoubleMatrix2D AMatrix = F2.make(A);
		LDLTPermutedFactorization fact = new LDLTPermutedFactorization(AMatrix, LDLTPermutedFactorization.DIAGONAL_PIVOLTING_WITH_PARTIAL_PIVOTING);  
		fact.factorize();
		DoubleMatrix2D P = fact.getP();
		DoubleMatrix2D D = fact.getD();
		DoubleMatrix2D L = fact.getL();
		DoubleMatrix2D LDLT = ALG.mult(L, ALG.mult(D, ALG.transpose(L)));
		DoubleMatrix2D PLDLTPT = ALG.mult(P, ALG.mult(LDLT, ALG.transpose(P)));
		double norm = MatrixUtils.createRealMatrix(A).subtract(MatrixUtils.createRealMatrix(PLDLTPT.toArray())).getNorm();
		logger.debug("norm: " + norm);
		assertEquals(0, norm, 1.e-15);
	}
	
	/**
	 * Test positive
	 */
	public void testPldltpt3() throws Exception {
		logger.debug("testPldltpt3");
		DoubleFactory2D F2 = DoubleFactory2D.dense;
		Algebra ALG = Algebra.DEFAULT;
		double[][] A = new double[][] { 
				{ 1, .12, .13, .14, .15 },
				{ .12, 2, .23, .24, .25 }, 
				{ .13, .23, 3, 0, 0 },
				{ .14, .24, 0, 4, 0 }, 
				{ .15, .25, 0, 0, 5 } };
		DoubleMatrix2D AMatrix = F2.make(A);
		LDLTPermutedFactorization fact = new LDLTPermutedFactorization(AMatrix, LDLTPermutedFactorization.DIAGONAL_PIVOLTING);  
		fact.factorize();
		DoubleMatrix2D P = fact.getP();
		DoubleMatrix2D D = fact.getD();
		DoubleMatrix2D L = fact.getL();
		DoubleMatrix2D LDLT = ALG.mult(L, ALG.mult(D, ALG.transpose(L)));
		DoubleMatrix2D PLDLTPT = ALG.mult(P, ALG.mult(LDLT, ALG.transpose(P)));
		double norm = MatrixUtils.createRealMatrix(A).subtract(MatrixUtils.createRealMatrix(PLDLTPT.toArray())).getNorm();
		logger.debug("norm: " + norm);
		assertEquals(0, norm, 1.e-14);
	}
	
	/**
	 * Test positive
	 */
	public void testPldltpt3BK() throws Exception {
		logger.debug("testPldltpt3BK");
		DoubleFactory2D F2 = DoubleFactory2D.dense;
		Algebra ALG = Algebra.DEFAULT;
		double[][] A = new double[][] { 
				{ 1, .12, .13, .14, .15 },
				{ .12, 2, .23, .24, .25 }, 
				{ .13, .23, 3, 0, 0 },
				{ .14, .24, 0, 4, 0 }, 
				{ .15, .25, 0, 0, 5 } };
		DoubleMatrix2D AMatrix = F2.make(A);
		LDLTPermutedFactorization fact = new LDLTPermutedFactorization(AMatrix, LDLTPermutedFactorization.DIAGONAL_PIVOLTING_WITH_PARTIAL_PIVOTING);  
		fact.factorize();
		DoubleMatrix2D P = fact.getP();
		DoubleMatrix2D D = fact.getD();
		DoubleMatrix2D L = fact.getL();
		DoubleMatrix2D LDLT = ALG.mult(L, ALG.mult(D, ALG.transpose(L)));
		DoubleMatrix2D PLDLTPT = ALG.mult(P, ALG.mult(LDLT, ALG.transpose(P)));
		double norm = MatrixUtils.createRealMatrix(A).subtract(MatrixUtils.createRealMatrix(PLDLTPT.toArray())).getNorm();
		logger.debug("norm: " + norm);
		assertEquals(0, norm, 1.e-14);
	}
	
	/**
	 * Test permute
	 * @see pag 31 di "High performance..."
	 */
	public void testPldltpt4() throws Exception {
		logger.debug("testPldltpt4");
		DoubleFactory2D F2 = DoubleFactory2D.dense;
		Algebra ALG = Algebra.DEFAULT;
		double[][] A = new double[][] { 
				{ 0, 1 }, 
				{ 1, 0 } };
		DoubleMatrix2D AMatrix = F2.make(A);
		LDLTPermutedFactorization fact = new LDLTPermutedFactorization(AMatrix, LDLTPermutedFactorization.DIAGONAL_PIVOLTING);  
		fact.factorize();
		DoubleMatrix2D P = fact.getP();
		DoubleMatrix2D D = fact.getD();
		DoubleMatrix2D L = fact.getL();
		DoubleMatrix2D LDLT = ALG.mult(L, ALG.mult(D, ALG.transpose(L)));
		DoubleMatrix2D PLDLTPT = ALG.mult(P, ALG.mult(LDLT, ALG.transpose(P)));
		double norm = MatrixUtils.createRealMatrix(A).subtract(MatrixUtils.createRealMatrix(PLDLTPT.toArray())).getNorm();
		logger.debug("norm: " + norm);
		assertEquals(0, norm, 1.e-14);
	}
	
	/**
	 * Test permute
	 * @see pag 31 di "High performance..."
	 */
	public void testPldltpt4BK() throws Exception {
		logger.debug("testPldltpt4BK");
		DoubleFactory2D F2 = DoubleFactory2D.dense;
		Algebra ALG = Algebra.DEFAULT;
		double[][] A = new double[][] { 
				{ 0, 1 }, 
				{ 1, 0 } };
		DoubleMatrix2D AMatrix = F2.make(A);
		LDLTPermutedFactorization fact = new LDLTPermutedFactorization(AMatrix, LDLTPermutedFactorization.DIAGONAL_PIVOLTING_WITH_PARTIAL_PIVOTING);  
		fact.factorize();
		DoubleMatrix2D P = fact.getP();
		DoubleMatrix2D D = fact.getD();
		DoubleMatrix2D L = fact.getL();
		DoubleMatrix2D LDLT = ALG.mult(L, ALG.mult(D, ALG.transpose(L)));
		DoubleMatrix2D PLDLTPT = ALG.mult(P, ALG.mult(LDLT, ALG.transpose(P)));
		double norm = MatrixUtils.createRealMatrix(A).subtract(MatrixUtils.createRealMatrix(PLDLTPT.toArray())).getNorm();
		logger.debug("norm: " + norm);
		assertEquals(0, norm, 1.e-14);
	}
	
	/**
	 * Can permute A so that L is sparse?
	 */
	public void testPldltptDensel() throws Exception {
		logger.debug("testPldltptDensel");
		DoubleFactory2D F2 = DoubleFactory2D.dense;
		Algebra ALG = Algebra.DEFAULT;
		double[][] A = new double[][] {
			  {1, 0.1, 0.1, 0.1},
			  {0.1, 1, 0, 0},
			  {0.1, 0, 1, 0},
			  {0.1, 0, 0, 1}};
		DoubleMatrix2D AMatrix = F2.make(A);
		LDLTPermutedFactorization fact = new LDLTPermutedFactorization(AMatrix, LDLTPermutedFactorization.DIAGONAL_PIVOLTING);  
		fact.factorize();
		DoubleMatrix2D P = fact.getP();
		DoubleMatrix2D D = fact.getD();
		DoubleMatrix2D L = fact.getL();
		logger.debug("L: " + ArrayUtils.toString(L.toArray()));
		DoubleMatrix2D LDLT = ALG.mult(L, ALG.mult(D, ALG.transpose(L)));
		DoubleMatrix2D PLDLTPT = ALG.mult(P, ALG.mult(LDLT, ALG.transpose(P)));
		double norm = MatrixUtils.createRealMatrix(A).subtract(MatrixUtils.createRealMatrix(PLDLTPT.toArray())).getNorm();
		logger.debug("norm: " + norm);
		assertEquals(0, norm, 1.e-14);
	}
	
	/**
	 * Can permute A so that L is sparse?
	 */
	public void testPldltptDenselBK() throws Exception {
		logger.debug("testPldltptDenselBK");
		DoubleFactory2D F2 = DoubleFactory2D.dense;
		Algebra ALG = Algebra.DEFAULT;
		double[][] A = new double[][] {
			  {1, 0.1, 0.1, 0.1},
			  {0.1, 1, 0, 0},
			  {0.1, 0, 1, 0},
			  {0.1, 0, 0, 1}};
		DoubleMatrix2D AMatrix = F2.make(A);
		LDLTPermutedFactorization fact = new LDLTPermutedFactorization(AMatrix, LDLTPermutedFactorization.DIAGONAL_PIVOLTING_WITH_PARTIAL_PIVOTING);  
		fact.factorize();
		DoubleMatrix2D P = fact.getP();
		DoubleMatrix2D D = fact.getD();
		DoubleMatrix2D L = fact.getL();
		logger.debug("L: " + ArrayUtils.toString(L.toArray()));
		DoubleMatrix2D LDLT = ALG.mult(L, ALG.mult(D, ALG.transpose(L)));
		DoubleMatrix2D PLDLTPT = ALG.mult(P, ALG.mult(LDLT, ALG.transpose(P)));
		double norm = MatrixUtils.createRealMatrix(A).subtract(MatrixUtils.createRealMatrix(PLDLTPT.toArray())).getNorm();
		logger.debug("norm: " + norm);
		assertEquals(0, norm, 1.e-14);
	}
	
	/**
	 * Test Singular
	 */
	public void testPldltpt6() throws Exception {
		logger.debug("testPldltpt6");
		DoubleFactory2D F2 = DoubleFactory2D.dense;
		Algebra ALG = Algebra.DEFAULT;
		double[][] A = new double[][] { 
				{ 1,  0, 1 }, 
				{ 0, -1, 0 },
				{ 1,  0, 1 } };
		DoubleMatrix2D AMatrix = F2.make(A);
		try{
			LDLTPermutedFactorization fact = new LDLTPermutedFactorization(AMatrix, LDLTPermutedFactorization.DIAGONAL_PIVOLTING);  
			fact.factorize();
			fail();//is singular, cannot be factorized
		}catch(Exception e){
			//ok
		}
	}
	
	/**
	 * Test Singular
	 */
	public void testPldltpt6BK() throws Exception {
		logger.debug("testPldltpt6BK");
		DoubleFactory2D F2 = DoubleFactory2D.dense;
		Algebra ALG = Algebra.DEFAULT;
		double[][] A = new double[][] { 
				{ 1,  0, 1 }, 
				{ 0, -1, 0 },
				{ 1,  0, 1 } };
		DoubleMatrix2D AMatrix = F2.make(A);
		try{
			LDLTPermutedFactorization fact = new LDLTPermutedFactorization(AMatrix, LDLTPermutedFactorization.DIAGONAL_PIVOLTING_WITH_PARTIAL_PIVOTING);  
			fact.factorize();
			fail();//is singular, cannot be factorized
		}catch(Exception e){
			//ok
		}
	}
	
	/**
	 * Test KKT
	 */
	public void testPldltpt7() throws Exception {
		logger.debug("testPldltpt7");
		DoubleFactory2D F2 = DoubleFactory2D.dense;
		Algebra ALG = Algebra.DEFAULT;
		double[][] H = new double[][] { 
				{ 1,  0, 0 }, 
				{ 0,  1, 0 },
				{ 0,  0, 1 } };
		double[][] A = new double[][] { 
				{ 1,  0, 1 }, 
				{ 0, -1, 0 } };
		DoubleMatrix2D HMatrix = F2.make(H);
		DoubleMatrix2D AMatrix = F2.make(A);
		DoubleMatrix2D[][] parts = {
		   { HMatrix, ALG.transpose(AMatrix)},
		   { AMatrix, null}};
		DoubleMatrix2D KKTMatrix = F2.compose(parts);

		LDLTPermutedFactorization fact = new LDLTPermutedFactorization(KKTMatrix, LDLTPermutedFactorization.DIAGONAL_PIVOLTING);  
		fact.factorize();
		DoubleMatrix2D P = fact.getP();
		DoubleMatrix2D D = fact.getD();
		DoubleMatrix2D L = fact.getL();
		DoubleMatrix2D LDLT = ALG.mult(L, ALG.mult(D, ALG.transpose(L)));
		DoubleMatrix2D PLDLTPT = ALG.mult(P, ALG.mult(LDLT, ALG.transpose(P)));
		double norm = MatrixUtils.createRealMatrix(KKTMatrix.toArray()).subtract(MatrixUtils.createRealMatrix(PLDLTPT.toArray())).getNorm();
		logger.debug("norm: " + norm);
		assertEquals(0, norm, 1.e-14);
	}
	
	/**
	 * Test KKT
	 */
	public void testPldltpt7BK() throws Exception {
		logger.debug("testPldltpt7BK");
		DoubleFactory2D F2 = DoubleFactory2D.dense;
		Algebra ALG = Algebra.DEFAULT;
		double[][] H = new double[][] { 
				{ 1,  0, 0 }, 
				{ 0,  1, 0 },
				{ 0,  0, 1 } };
		double[][] A = new double[][] { 
				{ 1,  0, 1 }, 
				{ 0, -1, 0 } };
		DoubleMatrix2D HMatrix = F2.make(H);
		DoubleMatrix2D AMatrix = F2.make(A);
		DoubleMatrix2D[][] parts = {
		   { HMatrix, ALG.transpose(AMatrix)},
		   { AMatrix, null}};
		DoubleMatrix2D KKTMatrix = F2.compose(parts);

		LDLTPermutedFactorization fact = new LDLTPermutedFactorization(KKTMatrix, LDLTPermutedFactorization.DIAGONAL_PIVOLTING_WITH_PARTIAL_PIVOTING);  
		fact.factorize();
		DoubleMatrix2D P = fact.getP();
		DoubleMatrix2D D = fact.getD();
		DoubleMatrix2D L = fact.getL();
		DoubleMatrix2D LDLT = ALG.mult(L, ALG.mult(D, ALG.transpose(L)));
		DoubleMatrix2D PLDLTPT = ALG.mult(P, ALG.mult(LDLT, ALG.transpose(P)));
		double norm = MatrixUtils.createRealMatrix(KKTMatrix.toArray()).subtract(MatrixUtils.createRealMatrix(PLDLTPT.toArray())).getNorm();
		logger.debug("norm: " + norm);
		assertEquals(0, norm, 1.e-14);
	}
	
	/**
	 * Test growth factor
	 * @see pag 31 di "High performance..."
	 */
	public void testPldltpt8() throws Exception {
		logger.debug("testPldltpt8");
		DoubleFactory2D F2 = DoubleFactory2D.dense;
		Algebra ALG = Algebra.DEFAULT;
		double e = 1.e-6; 		
		double[][] A = new double[][] { 
				{ e, 1 }, 
				{ 1, 0 } };
		DoubleMatrix2D AMatrix = F2.make(A);
		LDLTPermutedFactorization fact = new LDLTPermutedFactorization(AMatrix, LDLTPermutedFactorization.DIAGONAL_PIVOLTING);  
		fact.factorize();
		DoubleMatrix2D P = fact.getP();
		DoubleMatrix2D D = fact.getD();
		DoubleMatrix2D L = fact.getL();
		logger.debug("P: " + ArrayUtils.toString(P.toArray()));
		logger.debug("D: " + ArrayUtils.toString(D.toArray()));
		logger.debug("L: " + ArrayUtils.toString(L.toArray()));
		DoubleMatrix2D LDLT = ALG.mult(L, ALG.mult(D, ALG.transpose(L)));
		DoubleMatrix2D PLDLTPT = ALG.mult(P, ALG.mult(LDLT, ALG.transpose(P)));
		double norm = MatrixUtils.createRealMatrix(A).subtract(MatrixUtils.createRealMatrix(PLDLTPT.toArray())).getNorm();
		logger.debug("norm: " + norm);
		assertEquals(0, norm, 1.e-14);
	}
	
	/**
	 * Test growth factor
	 * @see pag 31 di "High performance..."
	 */
	public void testPldltpt8BK() throws Exception {
		logger.debug("testPldltpt8BK");
		DoubleFactory2D F2 = DoubleFactory2D.dense;
		Algebra ALG = Algebra.DEFAULT;
		double e = 1.e-1; 		
		
		double[][] A = new double[][] { 
				{ e, 1 }, 
				{ 1, 0 } };
		
//		double[][] A = new double[][] {
//				{e, 1, 0, 1},
//				{1, 0, 0, 0},
//				{0, 0, e, 1},
//				{1, 0, 1, 0}};
		DoubleMatrix2D AMatrix = F2.make(A);
		assertTrue(Property.ZERO.isSymmetric(AMatrix));
		LDLTPermutedFactorization fact = new LDLTPermutedFactorization(AMatrix, LDLTPermutedFactorization.DIAGONAL_PIVOLTING_WITH_PARTIAL_PIVOTING);  
		fact.factorize();
		DoubleMatrix2D P = fact.getP();
		DoubleMatrix2D D = fact.getD();
		DoubleMatrix2D L = fact.getL();
		logger.debug("P: " + ArrayUtils.toString(P.toArray()));
		logger.debug("D: " + ArrayUtils.toString(D.toArray()));
		logger.debug("L: " + ArrayUtils.toString(L.toArray()));
		DoubleMatrix2D LDLT = ALG.mult(L, ALG.mult(D, ALG.transpose(L)));
		DoubleMatrix2D PLDLTPT = ALG.mult(P, ALG.mult(LDLT, ALG.transpose(P)));
		double norm = MatrixUtils.createRealMatrix(A).subtract(MatrixUtils.createRealMatrix(PLDLTPT.toArray())).getNorm();
		logger.debug("norm: " + norm);
		assertEquals(0, norm, 1.e-14);
	}
	
	/**
	 * Test a big matrix
	 */
	public void testPldltpt9BK() throws Exception {
		logger.debug("testPldltpt9BK");
		DoubleFactory2D F2 = DoubleFactory2D.sparse;
		Algebra ALG = Algebra.DEFAULT;
		double[][] G = TestUtils.loadDoubleMatrixFromFile("factorization" + File.separator	+ "matrix6.txt", " ".charAt(0));
		DoubleMatrix2D AMatrix = F2.make(G);
		LDLTPermutedFactorization fact = new LDLTPermutedFactorization(AMatrix, LDLTPermutedFactorization.DIAGONAL_PIVOLTING_WITH_PARTIAL_PIVOTING);  
		fact.factorize();
		DoubleMatrix2D P = fact.getP();
		DoubleMatrix2D D = fact.getD();
		DoubleMatrix2D L = fact.getL();
		DoubleMatrix2D LDLT = ALG.mult(L, ALG.mult(D, ALG.transpose(L)));
		DoubleMatrix2D PLDLTPT = ALG.mult(P, ALG.mult(LDLT, ALG.transpose(P)));
		double norm = MatrixUtils.createRealMatrix(AMatrix.toArray()).subtract(MatrixUtils.createRealMatrix(PLDLTPT.toArray())).getNorm();
		logger.debug("norm: " + norm);
		assertEquals(0, norm, 1.e-14);
	}
}
