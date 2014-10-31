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
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import com.joptimizer.util.TestUtils;

/**
 * Standard form conversion test.
 * 
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class LPStandardConverterTest extends TestCase {

	private Log log = LogFactory.getLog(this.getClass().getName());
	
	/**
	 * Standardization of a problem on the form:
	 * min(c) s.t.
	 * G.x < h
	 * A.x = b
	 * lb <= x <= ub
	 */
	public void testCGhAbLbUb1() throws Exception {
		log.debug("testCGhAbLbUb1");
		
		String problemId = "1";
		
		double[] c = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"standardization"+File.separator+"c"+problemId+".txt");
		double[][] G = TestUtils.loadDoubleMatrixFromFile("lp"+File.separator+"standardization"+File.separator+"G"+problemId+".txt", " ".charAt(0));
		double[] h = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"standardization"+File.separator+"h"+problemId+".txt");;
		double[][] A = TestUtils.loadDoubleMatrixFromFile("lp"+File.separator+"standardization"+File.separator+"A"+problemId+".txt", " ".charAt(0));
		double[] b = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"standardization"+File.separator+"b"+problemId+".txt");
		double[] lb = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"standardization"+File.separator+"lb"+problemId+".txt");
		double[] ub = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"standardization"+File.separator+"ub"+problemId+".txt");
		double[] expectedSol = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"standardization"+File.separator+"sol"+problemId+".txt");
		double expectedTolerance = MatrixUtils.createRealMatrix(A).operate(MatrixUtils.createRealVector(expectedSol)).subtract(MatrixUtils.createRealVector(b)).getNorm(); 
			
		//standard form conversion
		double unboundedLBValue = Double.NEGATIVE_INFINITY;//this is because in the file the unbounded lb are -Infinity values (not the default value) 
		double unboundedUBValue = Double.POSITIVE_INFINITY;//this is because in the file the unbounded ub are +Infinity values
		LPStandardConverter lpConverter = new LPStandardConverter(unboundedLBValue, unboundedUBValue);
		lpConverter.toStandardForm(c, G, h, A, b, lb, ub);
		
		int n = lpConverter.getStandardN();
		int s = lpConverter.getStandardS();
		c = lpConverter.getStandardC().toArray();
		A = lpConverter.getStandardA().toArray();
		b = lpConverter.getStandardB().toArray();
		lb = lpConverter.getStandardLB().toArray();
		ub = lpConverter.getStandardUB().toArray();
		log.debug("n : " + n);
		log.debug("s : " + s);
		log.debug("c : " + ArrayUtils.toString(c));
		log.debug("A : " + ArrayUtils.toString(A));
		log.debug("b : " + ArrayUtils.toString(b));
		log.debug("lb : " + ArrayUtils.toString(lb));
		log.debug("ub : " + ArrayUtils.toString(ub));
		
	  //check consistency
		assertEquals(G.length, s);
		assertEquals(s + lpConverter.getOriginalN(), n);
		assertEquals(lb.length, n);
		assertEquals(ub.length, n);
		
	//check constraints
		RealMatrix GOrig = new Array2DRowRealMatrix(G);
		RealVector hOrig = new ArrayRealVector(h);
		RealMatrix AStandard = new Array2DRowRealMatrix(A);
		RealVector bStandard = new ArrayRealVector(b);
		RealVector expectedSolVector = new ArrayRealVector(expectedSol);
		RealVector Gxh = GOrig.operate(expectedSolVector).subtract(hOrig);//G.x - h
		RealVector slackVariables = new ArrayRealVector(s);
		for(int i=0; i<s; i++){
			slackVariables.setEntry(i, 0. - Gxh.getEntry(i));//the difference from 0
			assertTrue(slackVariables.getEntry(i) >= 0.);
		}
		RealVector sol = slackVariables.append(expectedSolVector);
		RealVector Axmb = AStandard.operate(sol).subtract(bStandard);
		assertEquals(0., Axmb.getNorm(), expectedTolerance);
		
//		Utils.writeDoubleArrayToFile(new double[]{s}, "target" + File.separator	+ "standardS"+problemId+".txt");
//		Utils.writeDoubleArrayToFile(c, "target" + File.separator	+ "standardC"+problemId+".txt");
//		Utils.writeDoubleMatrixToFile(A, "target" + File.separator	+ "standardA"+problemId+".txt");
//		Utils.writeDoubleArrayToFile(b, "target" + File.separator	+ "standardB"+problemId+".txt");
//		Utils.writeDoubleArrayToFile(lb, "target" + File.separator	+ "standardLB"+problemId+".txt");
//		Utils.writeDoubleArrayToFile(ub, "target" + File.separator	+ "standardUB"+problemId+".txt");
	}
	
	/**
	 * Standardization (to the strictly standard form) of a problem on the form:
	 * min(c) s.t.
	 * G.x < h
	 * A.x = b
	 * lb <= x <= ub
	 * @TODO: the strict conversion is net yet ready.
	 */
	public void xxxtestCGhAbLbUb1Strict() throws Exception {
		log.debug("testCGhAbLbUb1Strict");
		
		String problemId = "1";
		
		double[] c = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"standardization"+File.separator+"c"+problemId+".txt");
		double[][] G = TestUtils.loadDoubleMatrixFromFile("lp"+File.separator+"standardization"+File.separator+"G"+problemId+".txt", " ".charAt(0));
		double[] h = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"standardization"+File.separator+"h"+problemId+".txt");;
		double[][] A = TestUtils.loadDoubleMatrixFromFile("lp"+File.separator+"standardization"+File.separator+"A"+problemId+".txt", " ".charAt(0));
		double[] b = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"standardization"+File.separator+"b"+problemId+".txt");
		double[] lb = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"standardization"+File.separator+"lb"+problemId+".txt");
		double[] ub = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"standardization"+File.separator+"ub"+problemId+".txt");
		double[] expectedSol = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"standardization"+File.separator+"sol"+problemId+".txt");
		double expectedTolerance = MatrixUtils.createRealMatrix(A).operate(MatrixUtils.createRealVector(expectedSol)).subtract(MatrixUtils.createRealVector(b)).getNorm();
		
		int nOfSlackVariables = 0;
		for(int i=0; i<c.length; i++){
			double lbi = lb[i];
			int lbCompare = Double.compare(lbi, 0.); 
			if(lbCompare != 0 && !Double.isNaN(lbi)){
				nOfSlackVariables++;
			}
			if(!Double.isNaN(ub[i])){
				nOfSlackVariables++;
			}
		}
		int expectedS = G.length + nOfSlackVariables;
		
		//standard form conversion
		boolean strictlyStandardForm = true;
		LPStandardConverter lpConverter = new LPStandardConverter(strictlyStandardForm);
		lpConverter.toStandardForm(c, G, h, A, b, lb, ub);
		
		int n = lpConverter.getStandardN();
		int s = lpConverter.getStandardS();
		c = lpConverter.getStandardC().toArray();
		A = lpConverter.getStandardA().toArray();
		b = lpConverter.getStandardB().toArray();
		lb = lpConverter.getStandardLB().toArray();
		ub = (lpConverter.getStandardUB()==null)? null : ub;
		log.debug("n : " + n);
		log.debug("s : " + s);
		log.debug("c : " + ArrayUtils.toString(c));
		log.debug("A : " + ArrayUtils.toString(A));
		log.debug("b : " + ArrayUtils.toString(b));
		log.debug("lb : " + ArrayUtils.toString(lb));
		//log.debug("ub : " + ArrayUtils.toString(ub));
		
		//check consistency
		assertEquals(expectedS, s);
		assertEquals(lb.length, n);
		assertTrue(ub == null);
		
		//check constraints
		RealMatrix AStandard = new Array2DRowRealMatrix(A);
		RealVector bStandard = new ArrayRealVector(b);
		double[] expectedStandardSol = lpConverter.getStandardComponents(expectedSol);
		RealVector expectedStandardSolVector = new ArrayRealVector(expectedStandardSol);
		
		for(int i=0; i<expectedStandardSolVector.getDimension(); i++){
			assertTrue(expectedStandardSolVector.getEntry(i) >= 0.);
		}
		
		RealVector Axmb = AStandard.operate(expectedStandardSolVector).subtract(bStandard);
		assertEquals(0., Axmb.getNorm(), expectedTolerance);
		
		TestUtils.writeDoubleArrayToFile(new double[]{s}, "target" + File.separator	+ "standardS_"+problemId+".txt");
		TestUtils.writeDoubleArrayToFile(c, "target" + File.separator	+ "standardC_"+problemId+".txt");
		TestUtils.writeDoubleMatrixToFile(A, "target" + File.separator	+ "standardA_"+problemId+".txt");
		TestUtils.writeDoubleArrayToFile(b, "target" + File.separator	+ "standardB_"+problemId+".txt");
		TestUtils.writeDoubleArrayToFile(lb, "target" + File.separator	+ "standardLB_"+problemId+".txt");
		//ub is null TestUtils.writeDoubleArrayToFile(ub, "target" + File.separator	+ "standardUB_"+problemId+".txt");
	}
	
	/**
	 * Standardization of a problem on the form:
	 * min(c) s.t.
	 * G.x < h
	 * A.x = b
	 */
	public void testCGhAb2() throws Exception {
		log.debug("testCGhAb2");
		
		String problemId = "2";
		
		double[] c = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"standardization"+File.separator+"c"+problemId+".txt");
		double[][] G = TestUtils.loadDoubleMatrixFromFile("lp"+File.separator+"standardization"+File.separator+"G"+problemId+".txt", " ".charAt(0));
		double[] h = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"standardization"+File.separator+"h"+problemId+".txt");;
		double[][] A = TestUtils.loadDoubleMatrixFromFile("lp"+File.separator+"standardization"+File.separator+"A"+problemId+".txt", " ".charAt(0));
		double[] b = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"standardization"+File.separator+"b"+problemId+".txt");
		double[] expectedSol = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"standardization"+File.separator+"sol"+problemId+".txt");
		double expectedTolerance = MatrixUtils.createRealMatrix(A).operate(MatrixUtils.createRealVector(expectedSol)).subtract(MatrixUtils.createRealVector(b)).getNorm();
		
		//standard form conversion
		double unboundedLBValue = Double.NEGATIVE_INFINITY;
		double unboundedUBValue = Double.POSITIVE_INFINITY;
		LPStandardConverter lpConverter = new LPStandardConverter(unboundedLBValue, unboundedUBValue);
		lpConverter.toStandardForm(c, G, h, A, b, null, null);
		
		int n = lpConverter.getStandardN();
		int s = lpConverter.getStandardS();
		c = lpConverter.getStandardC().toArray();
		A = lpConverter.getStandardA().toArray();
		b = lpConverter.getStandardB().toArray();
		double[] lb = lpConverter.getStandardLB().toArray();
		double[] ub = lpConverter.getStandardUB().toArray();
		log.debug("n : " + n);
		log.debug("s : " + s);
		log.debug("c : " + ArrayUtils.toString(c));
		log.debug("A : " + ArrayUtils.toString(A));
		log.debug("b : " + ArrayUtils.toString(b));
		log.debug("lb : " + ArrayUtils.toString(lb));
		log.debug("ub : " + ArrayUtils.toString(ub));
		
	  //check consistency
		assertEquals(G.length, s);
		assertEquals(A[0].length, n);
		assertEquals(s + lpConverter.getOriginalN(), n);
		assertEquals(lb.length, n);
		assertEquals(ub.length, n);
		
		//check constraints
		RealMatrix GOrig = new Array2DRowRealMatrix(G);
		RealVector hOrig = new ArrayRealVector(h);
		RealMatrix AStandard = new Array2DRowRealMatrix(A);
		RealVector bStandard = new ArrayRealVector(b);
		RealVector expectedSolVector = new ArrayRealVector(expectedSol);
		RealVector Gxh = GOrig.operate(expectedSolVector).subtract(hOrig);//G.x - h
		RealVector slackVariables = new ArrayRealVector(s);
		for(int i=0; i<s; i++){
			slackVariables.setEntry(i, 0. - Gxh.getEntry(i));//the difference from 0
			assertTrue(slackVariables.getEntry(i) >= 0.);
		}
		RealVector sol = slackVariables.append(expectedSolVector);
		RealVector Axmb = AStandard.operate(sol).subtract(bStandard);
		assertEquals(0., Axmb.getNorm(), expectedTolerance);
		
//		Utils.writeDoubleArrayToFile(new double[]{s}, "target" + File.separator	+ "standardS"+problemId+".txt");
//		Utils.writeDoubleArrayToFile(c, "target" + File.separator	+ "standardC"+problemId+".txt");
//		Utils.writeDoubleMatrixToFile(A, "target" + File.separator	+ "standardA"+problemId+".txt");
//		Utils.writeDoubleArrayToFile(b, "target" + File.separator	+ "standardB"+problemId+".txt");
//		Utils.writeDoubleArrayToFile(lb, "target" + File.separator	+ "standardLB"+problemId+".txt");
//		Utils.writeDoubleArrayToFile(ub, "target" + File.separator	+ "standardUB"+problemId+".txt");
	}
	
	/**
	 * Standardization of a problem on the form:
	 * min(c) s.t.
	 * G.x < h
	 * A.x = b
	 */
	public void testCGhAb3() throws Exception {
		log.debug("testCGhAb3");
		
		String problemId = "3";
		
		double[] c = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"standardization"+File.separator+"c"+problemId+".txt");
		double[][] G = TestUtils.loadDoubleMatrixFromFile("lp"+File.separator+"standardization"+File.separator+"G"+problemId+".txt", " ".charAt(0));
		double[] h = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"standardization"+File.separator+"h"+problemId+".txt");;
		double[][] A = TestUtils.loadDoubleMatrixFromFile("lp"+File.separator+"standardization"+File.separator+"A"+problemId+".txt", " ".charAt(0));
		double[] b = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"standardization"+File.separator+"b"+problemId+".txt");
		double[] expectedSol = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"standardization"+File.separator+"sol"+problemId+".txt");
		double expectedTolerance = MatrixUtils.createRealMatrix(A).operate(MatrixUtils.createRealVector(expectedSol)).subtract(MatrixUtils.createRealVector(b)).getNorm();
		
		//standard form conversion
		LPStandardConverter lpConverter = new LPStandardConverter();
		lpConverter.toStandardForm(c, G, h, A, b, null, null);
		
		int n = lpConverter.getStandardN();
		int s = lpConverter.getStandardS();
		c = lpConverter.getStandardC().toArray();
		A = lpConverter.getStandardA().toArray();
		b = lpConverter.getStandardB().toArray();
		double[] lb = lpConverter.getStandardLB().toArray();
		double[] ub = lpConverter.getStandardUB().toArray();
		log.debug("n : " + n);
		log.debug("s : " + s);
		
		//check consistency
		assertEquals(G.length, s);
		assertEquals(A[0].length, n);
		assertEquals(s + lpConverter.getOriginalN(), n);
		assertEquals(lb.length, n);
		assertEquals(ub.length, n);
		
	//check constraints
		RealMatrix GOrig = new Array2DRowRealMatrix(G);
		RealVector hOrig = new ArrayRealVector(h);
		RealMatrix AStandard = new Array2DRowRealMatrix(A);
		RealVector bStandard = new ArrayRealVector(b);
		RealVector expectedSolVector = new ArrayRealVector(expectedSol);
		RealVector Gxh = GOrig.operate(expectedSolVector).subtract(hOrig);//G.x - h
		RealVector slackVariables = new ArrayRealVector(s);
		for(int i=0; i<s; i++){
			slackVariables.setEntry(i, 0. - Gxh.getEntry(i));//the difference from 0
			assertTrue(slackVariables.getEntry(i) >= 0.);
		}
		RealVector sol = slackVariables.append(expectedSolVector);
		RealVector Axmb = AStandard.operate(sol).subtract(bStandard);
		assertEquals(0., Axmb.getNorm(), expectedTolerance);
		
//		Utils.writeDoubleArrayToFile(new double[]{s}, "target" + File.separator	+ "standardS"+problemId+".txt");
//		Utils.writeDoubleArrayToFile(c, "target" + File.separator	+ "standardC"+problemId+".txt");
//		Utils.writeDoubleMatrixToFile(A, "target" + File.separator	+ "standardA"+problemId+".txt");
//		Utils.writeDoubleArrayToFile(b, "target" + File.separator	+ "standardB"+problemId+".txt");
//		Utils.writeDoubleArrayToFile(lb, "target" + File.separator	+ "standardLB"+problemId+".txt");
//		Utils.writeDoubleArrayToFile(ub, "target" + File.separator	+ "standardUB"+problemId+".txt");
	}

	/**
	 * Standardization of a problem on the form:
	 * min(c) s.t.
	 * G.x < h
	 * A.x = b
	 * lb <= x <= ub
	 */
	public void testCGhAbLbUb4() throws Exception {
		log.debug("testCGhAbLbUb4");
		
		String problemId = "4";
		
		double[] c = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"standardization"+File.separator+"c"+problemId+".txt");
		double[][] G = TestUtils.loadDoubleMatrixFromFile("lp"+File.separator+"standardization"+File.separator+"G"+problemId+".txt", " ".charAt(0));
		double[] h = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"standardization"+File.separator+"h"+problemId+".txt");;
		double[][] A = TestUtils.loadDoubleMatrixFromFile("lp"+File.separator+"standardization"+File.separator+"A"+problemId+".txt", " ".charAt(0));
		double[] b = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"standardization"+File.separator+"b"+problemId+".txt");
		double[] lb = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"standardization"+File.separator+"lb"+problemId+".txt");
		double[] ub = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"standardization"+File.separator+"ub"+problemId+".txt");
		double[] expectedSol = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"standardization"+File.separator+"sol"+problemId+".txt");
		double expectedTolerance = MatrixUtils.createRealMatrix(A).operate(MatrixUtils.createRealVector(expectedSol)).subtract(MatrixUtils.createRealVector(b)).getNorm();
		
		int nOsSplittingVariables = 0;
//		for(int i=0; i<lb.length; i++){
//			if(Double.compare(lb[i], 0.) != 0){
//				nOsSplittingVariables++;
//			}
//		}
		
		//standard form conversion
		double unboundedLBValue = Double.NaN;//this is because in the file the unbounded lb are NaN values (and also the default value) 
		double unboundedUBValue = Double.NaN;//this is because in the file the unbounded ub are NaN values
		LPStandardConverter lpConverter = new LPStandardConverter(unboundedLBValue, unboundedUBValue);
		lpConverter.toStandardForm(c, G, h, A, b, lb, ub);
		
		int n = lpConverter.getStandardN();
		int s = lpConverter.getStandardS();
		c = lpConverter.getStandardC().toArray();
		A = lpConverter.getStandardA().toArray();
		b = lpConverter.getStandardB().toArray();
		lb = lpConverter.getStandardLB().toArray();
		ub = lpConverter.getStandardUB().toArray();
		log.debug("n : " + n);
		log.debug("s : " + s);
		
	  //check consistency
		assertEquals(G.length, s);
		assertEquals(s + lpConverter.getOriginalN() + nOsSplittingVariables, n);
		assertEquals(lb.length, n);
		assertEquals(ub.length, n);
		
	  //check constraints
		RealMatrix GOrig = new Array2DRowRealMatrix(G);
		RealVector hOrig = new ArrayRealVector(h);
		RealMatrix AStandard = new Array2DRowRealMatrix(A);
		RealVector bStandard = new ArrayRealVector(b);
		RealVector expectedSolVector = new ArrayRealVector(expectedSol);
		RealVector Gxh = GOrig.operate(expectedSolVector).subtract(hOrig);//G.x - h
		RealVector slackVariables = new ArrayRealVector(s);
		for(int i=0; i<s; i++){
			slackVariables.setEntry(i, 0. - Gxh.getEntry(i));//the difference from 0
			assertTrue(slackVariables.getEntry(i) >= 0.);
		}
		RealVector sol = slackVariables.append(expectedSolVector);
		RealVector Axmb = AStandard.operate(sol).subtract(bStandard);
		assertEquals(0., Axmb.getNorm(), expectedTolerance * 1.001);
		
//		Utils.writeDoubleArrayToFile(new double[]{s}, "target" + File.separator	+ "standardS"+problemId+".txt");
//		Utils.writeDoubleArrayToFile(c, "target" + File.separator	+ "standardC"+problemId+".txt");
//		Utils.writeDoubleMatrixToFile(A, "target" + File.separator	+ "standardA"+problemId+".txt");
//		Utils.writeDoubleArrayToFile(b, "target" + File.separator	+ "standardB"+problemId+".txt");
//		Utils.writeDoubleArrayToFile(lb, "target" + File.separator	+ "standardLB"+problemId+".txt");
//		Utils.writeDoubleArrayToFile(ub, "target" + File.separator	+ "standardUB"+problemId+".txt");
	}
	
	/**
	 * Standardization (to the strictly standard form) of a problem on the form:
	 * min(c) s.t.
	 * A.x = b
	 * lb <= x <= ub
	 * 
	 * This is the presolved (with JOptimizer) pilot4 netlib problem.
	 * @TODO: the strict conversion is net yet ready.
	 */
	public void xxxtestCAbLbUb5Strict() throws Exception {
		log.debug("testCAbLbUb5Strict");
		
		String problemId = "5";
		
		double[] c = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"standardization"+File.separator+"c"+problemId+".txt");
		double[][] A = TestUtils.loadDoubleMatrixFromFile("lp"+File.separator+"standardization"+File.separator+"A"+problemId+".txt", " ".charAt(0));
		double[] b = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"standardization"+File.separator+"b"+problemId+".txt");
		double[] lb = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"standardization"+File.separator+"lb"+problemId+".txt");
		double[] ub = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"standardization"+File.separator+"ub"+problemId+".txt");
		double[] expectedSol = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"standardization"+File.separator+"sol"+problemId+".txt");
		double expectedTol = TestUtils.loadDoubleArrayFromFile("lp"+File.separator+"standardization"+File.separator+"tolerance"+problemId+".txt")[0];
		
		int nOfSlackVariables = 0;
		for(int i=0; i<c.length; i++){
			double lbi = lb[i];
			int lbCompare = Double.compare(lbi, 0.); 
			if(lbCompare != 0 && !Double.isNaN(lbi)){
				nOfSlackVariables++;
			}
			if(!Double.isNaN(ub[i])){
				nOfSlackVariables++;
			}
		}
		int expectedS = nOfSlackVariables;
		
		//standard form conversion
		boolean strictlyStandardForm = true;
		LPStandardConverter lpConverter = new LPStandardConverter(strictlyStandardForm);
		lpConverter.toStandardForm(c, null, null, A, b, lb, ub);
		
		int n = lpConverter.getStandardN();
		int s = lpConverter.getStandardS();
		c = lpConverter.getStandardC().toArray();
		A = lpConverter.getStandardA().toArray();
		b = lpConverter.getStandardB().toArray();
		lb = lpConverter.getStandardLB().toArray();
		ub = (lpConverter.getStandardUB()==null)? null : ub;
		log.debug("n : " + n);
		log.debug("s : " + s);
		log.debug("c : " + ArrayUtils.toString(c));
		log.debug("A : " + ArrayUtils.toString(A));
		log.debug("b : " + ArrayUtils.toString(b));
		log.debug("lb : " + ArrayUtils.toString(lb));
		//log.debug("ub : " + ArrayUtils.toString(ub));
		
		//check consistency
		assertEquals(expectedS, s);
		assertEquals(lb.length, n);
		assertTrue(ub == null);
		
		//check constraints
		RealMatrix AStandard = new Array2DRowRealMatrix(A);
		RealVector bStandard = new ArrayRealVector(b);
		double[] expectedStandardSol = lpConverter.getStandardComponents(expectedSol);
		RealVector expectedStandardSolVector = new ArrayRealVector(expectedStandardSol);
		
		for(int i=0; i<expectedStandardSolVector.getDimension(); i++){
			assertTrue(expectedStandardSolVector.getEntry(i) + 1.E-8 >= 0.);
		}
		
		RealVector Axmb = AStandard.operate(expectedStandardSolVector).subtract(bStandard);
		for(int i=0; i<Axmb.getDimension(); i++){
			assertEquals(0., Axmb.getEntry(i), expectedTol);
		}
		
		TestUtils.writeDoubleArrayToFile(new double[]{s}, "target" + File.separator	+ "standardS_"+problemId+".txt");
		TestUtils.writeDoubleArrayToFile(c, "target" + File.separator	+ "standardC_"+problemId+".txt");
		TestUtils.writeDoubleMatrixToFile(A, "target" + File.separator	+ "standardA_"+problemId+".txt");
		TestUtils.writeDoubleArrayToFile(b, "target" + File.separator	+ "standardB_"+problemId+".txt");
		TestUtils.writeDoubleArrayToFile(lb, "target" + File.separator	+ "standardLB_"+problemId+".txt");
		//ub is null TestUtils.writeDoubleArrayToFile(ub, "target" + File.separator	+ "standardUB_"+problemId+".txt");
	}
	
}
