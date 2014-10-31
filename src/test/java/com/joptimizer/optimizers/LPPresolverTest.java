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
package com.joptimizer.optimizers;

import java.io.File;

import junit.framework.TestCase;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularValueDecomposition;

import com.joptimizer.util.TestUtils;

/**
 * LP presolving test.
 * 
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class LPPresolverTest extends TestCase {
	private Log log = LogFactory.getLog(this.getClass().getName());

	public void testFromFile1() throws Exception {
		log.debug("testFromFile1");
		
		String problemId = "1";
		
		log.debug("problemId: " + problemId);
		double[] c = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"c"+problemId+".txt");
		double[][] A = TestUtils.loadDoubleMatrixFromFile("lp"+File.separator+"presolving"+File.separator+"A"+problemId+".txt", " ".charAt(0));
		double[] b = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"b"+problemId+".txt");
		double[] lb = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"lb"+problemId+".txt");
		double[] ub = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"ub"+problemId+".txt");
		double s = 0;
		try{
			s = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"s"+problemId+".txt")[0];
		}catch(Exception e){}
		double[] expectedSolution = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"sol"+problemId+".txt");
		double expectedValue = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"value"+problemId+".txt")[0];
		double expectedTolerance = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"tolerance"+problemId+".txt")[0];
		
		expectedTolerance = Math.max(expectedTolerance, MatrixUtils.createRealMatrix(A).operate(MatrixUtils.createRealVector(expectedSolution)).subtract(MatrixUtils.createRealVector(b)).getNorm()); 
		doPresolving(c, A, b, lb, ub, s, expectedSolution, expectedValue,
				expectedTolerance);
	}
	
	/**
	 * This problem has a deterministic solution.
	 */
	public void testFromFile2() throws Exception {
		log.debug("testFromFile2");
		
		String problemId = "2";
		
		double[] c = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"c"+problemId+".txt");
		double[][] A = TestUtils.loadDoubleMatrixFromFile("lp"+File.separator+"presolving"+File.separator+"A"+problemId+".txt", " ".charAt(0));
		double[] b = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"b"+problemId+".txt");
		double[] lb = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"lb"+problemId+".txt");
		double[] ub = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"ub"+problemId+".txt");
		double s = 0;
		try{
			s = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"s"+problemId+".txt")[0];
		}catch(Exception e){}
		double[] expectedSolution = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"sol"+problemId+".txt");
		
		//must be: A pXn with rank(A)=p < n
		RealMatrix AMatrix = MatrixUtils.createRealMatrix(A);
		SingularValueDecomposition dec = new SingularValueDecomposition(AMatrix);
		int rankA = dec.getRank();
		log.debug("p: " + AMatrix.getRowDimension());
		log.debug("n: " + AMatrix.getColumnDimension());
		log.debug("rank: " + rankA);
		
		LPPresolver lpPresolver = new LPPresolver();
		lpPresolver.setNOfSlackVariables((short)s);
		lpPresolver.setExpectedSolution(expectedSolution);//this is just for test!
		lpPresolver.presolve(c, A, b, lb, ub);
		int n = lpPresolver.getPresolvedN();

		//deterministic solution
		assertEquals(0, n);
		assertTrue(lpPresolver.getPresolvedC() == null);
		assertTrue(lpPresolver.getPresolvedA() == null);
		assertTrue(lpPresolver.getPresolvedB() == null);
		assertTrue(lpPresolver.getPresolvedLB() == null);
		assertTrue(lpPresolver.getPresolvedUB() == null);
		assertTrue(lpPresolver.getPresolvedYlb() == null);
		assertTrue(lpPresolver.getPresolvedYub() == null);
		assertTrue(lpPresolver.getPresolvedZlb() == null);
		assertTrue(lpPresolver.getPresolvedZub() == null);
		double[] sol = lpPresolver.postsolve(new double[]{});
		assertEquals(expectedSolution.length, sol.length);
		for(int i=0; i<sol.length; i++){
			//log.debug("i: " + i);
			assertEquals(expectedSolution[i], sol[i], 1.e-9);
		}
	}
	
	public void testFromFile4() throws Exception {
		log.debug("testFromFile4");
		
		String problemId = "4";
		
		log.debug("problemId: " + problemId);
		double[] c = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"c"+problemId+".txt");
		double[][] A = TestUtils.loadDoubleMatrixFromFile("lp"+File.separator+"presolving"+File.separator+"A"+problemId+".txt", " ".charAt(0));
		double[] b = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"b"+problemId+".txt");
		double[] lb = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"lb"+problemId+".txt");
		double[] ub = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"ub"+problemId+".txt");
		double s = 0;
		try{
			s = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"s"+problemId+".txt")[0];
		}catch(Exception e){}
		double[] expectedSolution = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"sol"+problemId+".txt");
		double expectedValue = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"value"+problemId+".txt")[0];
		double expectedTolerance = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"tolerance"+problemId+".txt")[0];
		
		expectedTolerance = Math.max(expectedTolerance, MatrixUtils.createRealMatrix(A).operate(MatrixUtils.createRealVector(expectedSolution)).subtract(MatrixUtils.createRealVector(b)).getNorm()); 
		doPresolving(c, A, b, lb, ub, s, expectedSolution, expectedValue,
				expectedTolerance);
	}
	
	/**
	 * This test involves duplicated columns.
	 */
	public void testFromFile5() throws Exception {
		log.debug("testFromFile5");
		
		String problemId = "5";
		
		log.debug("problemId: " + problemId);
		double[] c = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"c"+problemId+".txt");
		double[][] A = TestUtils.loadDoubleMatrixFromFile("lp"+File.separator+"presolving"+File.separator+"A"+problemId+".txt", " ".charAt(0));
		double[] b = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"b"+problemId+".txt");
		double[] lb = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"lb"+problemId+".txt");
		double[] ub = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"ub"+problemId+".txt");
		double s = 0;
		try{
			s = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"s"+problemId+".txt")[0];
		}catch(Exception e){}
		double[] expectedSolution = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"sol"+problemId+".txt");
		double expectedValue = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"value"+problemId+".txt")[0];
		double expectedTolerance = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"tolerance"+problemId+".txt")[0];
		
		expectedTolerance = Math.max(expectedTolerance, MatrixUtils.createRealMatrix(A).operate(MatrixUtils.createRealVector(expectedSolution)).subtract(MatrixUtils.createRealVector(b)).getNorm()); 
		doPresolving(c, A, b, lb, ub, s, expectedSolution, expectedValue,
				expectedTolerance);
	}
	
	public void testFromFile8() throws Exception {
		log.debug("testFromFile8");
		
		String problemId = "8";
		
		log.debug("problemId: " + problemId);
		double[] c = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"c"+problemId+".txt");
		double[][] A = TestUtils.loadDoubleMatrixFromFile("lp"+File.separator+"presolving"+File.separator+"A"+problemId+".txt", " ".charAt(0));
		double[] b = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"b"+problemId+".txt");
		double[] lb = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"lb"+problemId+".txt");
		double[] ub = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"ub"+problemId+".txt");
		double s = 0;
		try{
			s = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"s"+problemId+".txt")[0];
		}catch(Exception e){}
		double[] expectedSolution = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"sol"+problemId+".txt");
		double expectedValue = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"value"+problemId+".txt")[0];
		double expectedTolerance = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"tolerance"+problemId+".txt")[0];
		
		expectedTolerance = Math.max(expectedTolerance, MatrixUtils.createRealMatrix(A).operate(MatrixUtils.createRealVector(expectedSolution)).subtract(MatrixUtils.createRealVector(b)).getNorm());
		expectedTolerance = 0.0005;
		doPresolving(c, A, b, lb, ub, s, expectedSolution, expectedValue,
				expectedTolerance);
	}
	
	/**
	 * This is the afiro netlib problem presolved with JOptimizer without compareBounds.
	 */
	public void testFromFile10() throws Exception {
		log.debug("testFromFile10");
		
		String problemId = "10";
		
		log.debug("problemId: " + problemId);
		double[] c = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"c"+problemId+".txt");
		double[][] A = TestUtils.loadDoubleMatrixFromFile("lp"+File.separator+"presolving"+File.separator+"A"+problemId+".txt", " ".charAt(0));
		double[] b = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"b"+problemId+".txt");
		double[] lb = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"lb"+problemId+".txt");
		double[] ub = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"ub"+problemId+".txt");
		double s = 0;
		try{
			s = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"s"+problemId+".txt")[0];
		}catch(Exception e){}
		double[] expectedSolution = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"sol"+problemId+".txt");
		double expectedValue = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"value"+problemId+".txt")[0];
		double expectedTolerance = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"tolerance"+problemId+".txt")[0];
		
		expectedTolerance = Math.max(expectedTolerance, MatrixUtils.createRealMatrix(A).operate(MatrixUtils.createRealVector(expectedSolution)).subtract(MatrixUtils.createRealVector(b)).getNorm()); 
		doPresolving(c, A, b, lb, ub, s, expectedSolution, expectedValue,
				expectedTolerance);
	}
	
	/**
	 * This is the presolved (with CPlex) Recipe netlib problem in standard form.
	 */
	public void testFromFile11() throws Exception {
		log.debug("testFromFile11");
		
		String problemId = "11";
		
		log.debug("problemId: " + problemId);
		double[] c = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"c"+problemId+".txt");
		double[][] A = TestUtils.loadDoubleMatrixFromFile("lp"+File.separator+"presolving"+File.separator+"A"+problemId+".txt", " ".charAt(0));
		double[] b = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"b"+problemId+".txt");
		double[] lb = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"lb"+problemId+".txt");
		double[] ub = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"ub"+problemId+".txt");
		double s = 0;
		try{
			s = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"s"+problemId+".txt")[0];
		}catch(Exception e){}
		double[] expectedSolution = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"sol"+problemId+".txt");
		double expectedValue = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"value"+problemId+".txt")[0];
		double expectedTolerance = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"tolerance"+problemId+".txt")[0];
		
		expectedTolerance = Math.max(expectedTolerance, MatrixUtils.createRealMatrix(A).operate(MatrixUtils.createRealVector(expectedSolution)).subtract(MatrixUtils.createRealVector(b)).getNorm());
		expectedTolerance = 1.e-9;
		doPresolving(c, A, b, lb, ub, s, expectedSolution, expectedValue,
				expectedTolerance);
	}
	
	/**
	 * This is the VTP.BASE netlib problem in standard form.
	 */
	public void testFromFile12() throws Exception {
		log.debug("testFromFile12");
		
		String problemId = "12";
		
		log.debug("problemId: " + problemId);
		double[] c = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"c"+problemId+".txt");
		double[][] A = TestUtils.loadDoubleMatrixFromFile("lp"+File.separator+"presolving"+File.separator+"A"+problemId+".txt", " ".charAt(0));
		double[] b = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"b"+problemId+".txt");
		double[] lb = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"lb"+problemId+".txt");
		double[] ub = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"ub"+problemId+".txt");
		double s = 0;
		try{
			s = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"s"+problemId+".txt")[0];
		}catch(Exception e){}
		double[] expectedSolution = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"sol"+problemId+".txt");
		double expectedValue = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"value"+problemId+".txt")[0];
		double expectedTolerance = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"presolving"+File.separator+"tolerance"+problemId+".txt")[0];
		
		expectedTolerance = Math.max(expectedTolerance, MatrixUtils.createRealMatrix(A).operate(MatrixUtils.createRealVector(expectedSolution)).subtract(MatrixUtils.createRealVector(b)).getNorm()); 
		doPresolving(c, A, b, lb, ub, s, expectedSolution, expectedValue,
				expectedTolerance);
	}

	private void doPresolving(double[] c, double[][] A, double[] b, double[] lb,
			double[] ub, double s, double[] expectedSolution, double expectedValue,
			double expectedTolerance) throws Exception{
		
		RealMatrix AMatrix = MatrixUtils.createRealMatrix(A);
		SingularValueDecomposition dec = new SingularValueDecomposition(AMatrix);
		int rankA = dec.getRank();
		log.debug("p: " + AMatrix.getRowDimension());
		log.debug("n: " + AMatrix.getColumnDimension());
		log.debug("rank: " + rankA);
		
		LPPresolver lpPresolver = new LPPresolver();
		lpPresolver.setNOfSlackVariables((short)s);
		lpPresolver.setExpectedSolution(expectedSolution);//this is just for test!
		//lpPresolver.setExpectedTolerance(expectedTolerance);//this is just for test!
		lpPresolver.presolve(c, A, b, lb, ub);
		int n = lpPresolver.getPresolvedN();
		double[] presolvedC = lpPresolver.getPresolvedC().toArray();
		double[][] presolvedA = lpPresolver.getPresolvedA().toArray();
		double[] presolvedB = lpPresolver.getPresolvedB().toArray();
		double[] presolvedLb = lpPresolver.getPresolvedLB().toArray();
		double[] presolvedUb = lpPresolver.getPresolvedUB().toArray();
		double[] presolvedYlb = lpPresolver.getPresolvedYlb().toArray();
		double[] presolvedYub = lpPresolver.getPresolvedYub().toArray();
		double[] presolvedZlb = lpPresolver.getPresolvedZlb().toArray();
		double[] presolvedZub = lpPresolver.getPresolvedZub().toArray();
		log.debug("n  : " + n);
		log.debug("presolvedC  : " + ArrayUtils.toString(presolvedC));
		log.debug("presolvedA  : " + ArrayUtils.toString(presolvedA));
		log.debug("presolvedB  : " + ArrayUtils.toString(presolvedB));
		log.debug("presolvedLb : " + ArrayUtils.toString(presolvedLb));
		log.debug("presolvedUb : " + ArrayUtils.toString(presolvedUb));
		log.debug("presolvedYlb: " + ArrayUtils.toString(presolvedYlb));
		log.debug("presolvedYub: " + ArrayUtils.toString(presolvedYub));
		log.debug("presolvedZlb: " + ArrayUtils.toString(presolvedZlb));
		log.debug("presolvedZub: " + ArrayUtils.toString(presolvedZub));
		
		//check objective function
		double delta = expectedTolerance;
		RealVector presolvedX = MatrixUtils.createRealVector(lpPresolver.presolve(expectedSolution));
		log.debug("presolved value: " + MatrixUtils.createRealVector(presolvedC).dotProduct(presolvedX));
		RealVector postsolvedX = MatrixUtils.createRealVector(lpPresolver.postsolve(presolvedX.toArray()));
		double value = MatrixUtils.createRealVector(c).dotProduct(postsolvedX);
		assertEquals(expectedValue, value, delta);
		
		//check postsolved constraints
		for(int i=0; i<lb.length; i++){
			double di = Double.isNaN(lb[i])? -Double.MAX_VALUE : lb[i];
			assertTrue(di <= postsolvedX.getEntry(i) + delta);
		}
		for(int i=0; i<ub.length; i++){
			double di = Double.isNaN(ub[i])? Double.MAX_VALUE : ub[i];
			assertTrue(di + delta >= postsolvedX.getEntry(i));
		}
		RealVector Axmb = AMatrix.operate(postsolvedX).subtract(MatrixUtils.createRealVector(b));
		assertEquals(0., Axmb.getNorm(), expectedTolerance);
		
		//check presolved constraints
		assertEquals(presolvedLb.length, presolvedX.getDimension());
		assertEquals(presolvedUb.length, presolvedX.getDimension());
		AMatrix = MatrixUtils.createRealMatrix(presolvedA); 
		RealVector bvector = MatrixUtils.createRealVector(presolvedB);
		for(int i=0; i<presolvedLb.length; i++){
			double di = Double.isNaN(presolvedLb[i])? -Double.MAX_VALUE : presolvedLb[i];
			assertTrue(di <= presolvedX.getEntry(i) + delta);
		}
		for(int i=0; i<presolvedUb.length; i++){
			double di = Double.isNaN(presolvedUb[i])? Double.MAX_VALUE : presolvedUb[i];
			assertTrue(di + delta >= presolvedX.getEntry(i));
		}
		Axmb = AMatrix.operate(presolvedX).subtract(bvector);
		assertEquals(0., Axmb.getNorm(), expectedTolerance);
		
		//check rank(A): must be A pXn with rank(A)=p < n
		AMatrix = MatrixUtils.createRealMatrix(presolvedA);
		dec = new SingularValueDecomposition(AMatrix);
		rankA = dec.getRank();
		log.debug("p: " + AMatrix.getRowDimension());
		log.debug("n: " + AMatrix.getColumnDimension());
		log.debug("rank: " + rankA);
		assertEquals(AMatrix.getRowDimension(), rankA);
		assertTrue(rankA < AMatrix.getColumnDimension());
	}
}
