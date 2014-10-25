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
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import com.joptimizer.util.Utils;

/**
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class LPPrimalDualMethodTest extends TestCase {

	private Log log = LogFactory.getLog(this.getClass().getName());

	public void testDummy() throws Exception {
		log.debug("testDummy");
	}
	
	/**
	 * Simple problem in the form
	 * min(100x + y) s.t.
	 * x - y = 0
	 * 0 <= x <= 1
	 * 0 <= y <= 1
	 * 
	 */
	public void testSimple1() throws Exception {
		log.debug("testSimple1");
		
		double[] c = new double[] { -100, 1 };
		double[][] A = new double[][] {{1, -1}};
		double[] b = new double[] {0};
		double[] lb = new double[] {0, 0};
		double[] ub = new double[] {1, 1};
		double minLb = LPPrimalDualMethod.DEFAULT_MIN_LOWER_BOUND;
		double maxUb = LPPrimalDualMethod.DEFAULT_MAX_UPPER_BOUND;
		
		LPOptimizationRequest or = new LPOptimizationRequest();
		or.setC(c);
		or.setA(A);
		or.setB(b);
		or.setLb(lb);
		or.setUb(ub);
		or.setCheckKKTSolutionAccuracy(true);
		or.setToleranceFeas(1.E-7);
		or.setTolerance(1.E-7);
		or.setDumpProblem(true);
		//or.setPresolvingDisabled(true);
		or.setRescalingDisabled(true);
		
		//optimization
		LPPrimalDualMethod opt = new LPPrimalDualMethod(minLb, maxUb);
		
		opt.setLPOptimizationRequest(or);
		int returnCode = opt.optimize();
		
		if(returnCode == OptimizationResponse.FAILED){
			fail();
		}
		
		LPOptimizationResponse response = opt.getLPOptimizationResponse();
		double[] sol = response.getSolution();
		RealVector cVector = new ArrayRealVector(c);
		RealVector solVector = new ArrayRealVector(sol);
		double value = cVector.dotProduct(solVector);
		log.debug("sol   : " + ArrayUtils.toString(sol));
		log.debug("value : "	+ value);
		
		assertEquals( 2, sol.length);
		assertEquals( 1, sol[0], or.getTolerance());
		assertEquals( 1, sol[1], or.getTolerance());
		assertEquals(-99, value, or.getTolerance());
	}
	
	/**
	 * Simple problem in the form
	 * min(c.x) s.t.
	 * A.x = b
	 * x >=0
	 */
	public void testSimple2() throws Exception {
		log.debug("testSimple2");
		
		double[] c = new double[] { -1, -2 };
		double[][] A = new double[][] {{1, 1}};
		double[] b = new double[] {1};
		
		LPOptimizationRequest or = new LPOptimizationRequest();
		or.setC(c);
		or.setA(A);
		or.setB(b);
		or.setLb(new double[]{0, 0});
		//or.setInitialPoint(new double[] { 0.9, 0.1 });
		//or.setNotFeasibleInitialPoint(new double[] { -0.5, 1.5 });
		or.setCheckKKTSolutionAccuracy(true);
		or.setToleranceFeas(1.E-7);
		or.setTolerance(1.E-7);
		or.setDumpProblem(true);
		//or.setPresolvingDisabled(true);
		
		//optimization
		LPPrimalDualMethod opt = new LPPrimalDualMethod();
		
		opt.setLPOptimizationRequest(or);
		int returnCode = opt.optimize();
		
		if(returnCode == OptimizationResponse.FAILED){
			fail();
		}
		
		LPOptimizationResponse response = opt.getLPOptimizationResponse();
		double[] sol = response.getSolution();
		RealVector cVector = new ArrayRealVector(c);
		RealVector solVector = new ArrayRealVector(sol);
		double value = cVector.dotProduct(solVector);
		log.debug("sol   : " + ArrayUtils.toString(sol));
		log.debug("value : "	+ value);
		
		assertEquals( 2, sol.length);
		assertEquals( 0, sol[0], or.getTolerance());
		assertEquals( 1, sol[1], or.getTolerance());
		assertEquals(-2, value, or.getTolerance());
	}
	
	/**
	 * Simple problem in the form
	 * min(c.x) s.t.
	 * A.x = b
	 * lb <= x <= ub
	 * with a free variable.
	 * This test shows that it is necessary to provide bounds for all the variables in order to avoid singular KKT systems.
	 */
	public void testSimple3() throws Exception {
		log.debug("testSimple3");
		
		double[] c = new double[] { -1, -2, 0 };
		double[][] A = new double[][] {{1, 1, 0}};
		double[] b = new double[] {1};
		//double minLb = LPPrimalDualMethod.DEFAULT_MIN_LOWER_BOUND;
		//double maxUb = LPPrimalDualMethod.DEFAULT_MAX_UPPER_BOUND;
		double minLb = -99;
		double maxUb = +99;
		
		LPOptimizationRequest or = new LPOptimizationRequest();
		or.setC(c);
		or.setA(A);
		or.setB(b);
		or.setLb(new double[]{-1, -1, -100});//this will be limited to minLb
		or.setUb(new double[]{ 1,  1,  100});//this will be limited to maxUb
		or.setCheckKKTSolutionAccuracy(true);
//		or.setToleranceFeas(1.E-7);
//		or.setTolerance(1.E-7);
		or.setDumpProblem(true);
		//or.setPresolvingDisabled(true);
		//or.setRescalingDisabled(true);
		
		//optimization
		LPPrimalDualMethod opt = new LPPrimalDualMethod(minLb, maxUb);
		opt.setLPOptimizationRequest(or);
		int returnCode = opt.optimize();
		if(returnCode == OptimizationResponse.FAILED){
			fail();
		}
		LPOptimizationResponse response = opt.getLPOptimizationResponse();
		double[] sol = response.getSolution();
		RealVector cVector = new ArrayRealVector(c);
		RealVector solVector = new ArrayRealVector(sol);
		double value = cVector.dotProduct(solVector);
		log.debug("sol   : " + ArrayUtils.toString(sol));
		log.debug("value : "	+ value);
		
		assertEquals( 3, sol.length);
		assertEquals( 0, sol[0], or.getTolerance());
		assertEquals( 1, sol[1], or.getTolerance());
		assertEquals(-2, value,  or.getTolerance());
	}
	
	/**
	 * Minimize x subject to 
	 * x+y=4, 
	 * x-y=2. 
	 * Should return (3,1).
	 * This problem is the same as NewtonLEConstrainedISPTest.testOptimize2()
	 * and can be solved only with the use of a linear presolving phase:
	 * if passed directly to the solver, it will fail because JOptimizer
	 * does not want rank-deficient inequalities matrices like that of this problem.
	 */
	public void testSimple4() throws Exception {
		log.debug("testSimple4");
		double[] c = new double[] { 1, 0 };
		double[][] A = new double[][] { { 1.0, 1.0 }, { 1.0, -1.0 } };
		double[] b = new double[] { 4.0, 2.0 };
		
		LPOptimizationRequest or = new LPOptimizationRequest();
		or.setC(c);
		or.setA(A);
		or.setB(b);
		or.setLb(new double[]{-100, -100});
		or.setUb(new double[]{ 100,  100});
		or.setCheckKKTSolutionAccuracy(true);
		or.setDumpProblem(true);
		
		//optimization
		LPPrimalDualMethod opt = new LPPrimalDualMethod();
		opt.setLPOptimizationRequest(or);
		int returnCode = opt.optimize();
		if(returnCode == OptimizationResponse.FAILED){
			fail();
		}
		LPOptimizationResponse response = opt.getLPOptimizationResponse();
		double[] sol = response.getSolution();
		RealVector cVector = new ArrayRealVector(c);
		RealVector solVector = new ArrayRealVector(sol);
		double value = cVector.dotProduct(solVector);
		log.debug("sol   : " + ArrayUtils.toString(sol));
		log.debug("value : "	+ value);
		
		assertEquals( 2, sol.length);
		assertEquals(3.0, sol[0], or.getTolerance());
		assertEquals(1.0, sol[1], or.getTolerance());
	}
	
	/**
	 * Problem in the form
	 * min(c.x) s.t.
	 * G.x < h
	 * A.x = b
	 * 
	 * This is a good for testing with a small size problem.
	 * Submitted 01/09/2013 by Chris Myers.
	 */
	public void testCGhAb1() throws Exception {
		log.debug("testCGhAb1");
		
		String problemId = "1";
		
		//the original problem: ok until precision 1.E-7
		double[] c = Utils.loadDoubleArrayFromFile("lp"+File.separator+"c"+problemId+".txt");
		double[][] G = Utils.loadDoubleMatrixFromFile("lp"+File.separator+"G"+problemId+".txt", " ".charAt(0));
		double[] h = Utils.loadDoubleArrayFromFile("lp"+File.separator+"h"+problemId+".txt");;
		double[][] A = Utils.loadDoubleMatrixFromFile("lp"+File.separator+"A"+problemId+".txt", " ".charAt(0));
		double[] b = Utils.loadDoubleArrayFromFile("lp"+File.separator+"b"+problemId+".txt");
		double[] expectedSol = Utils.loadDoubleArrayFromFile("lp"+File.separator+"sol"+problemId+".txt");
		double expectedvalue = Utils.loadDoubleArrayFromFile("lp"+File.separator+"value"+problemId+".txt")[0];
		
		//double norm = MatrixUtils.createRealMatrix(A).operate(MatrixUtils.createRealVector(expectedSol)).subtract(MatrixUtils.createRealVector(b)).getNorm();
		//assertTrue(norm < 1.e-10);
		
		LPOptimizationRequest or = new LPOptimizationRequest();
		or.setC(c);
		or.setG(G);
		or.setH(h);
		or.setA(A);
		or.setB(b);
		or.setCheckKKTSolutionAccuracy(true);
		or.setToleranceKKT(1.E-7);
		or.setToleranceFeas(1.E-7);
		or.setTolerance(1.E-7);
		or.setDumpProblem(true);
		or.setAlpha(0.75);
		or.setInitialPoint(new double[]{0.9999998735888544,-999.0000001264111,1000.0,0.9999998735888544,0.0,-999.0000001264111,0.9999999661257591,0.9999998735888544,1000.0,0.0,0.9999998735888544,0.0,0.9999998735888544,0.9999998735888544,0.9999998735888544,0.0,0.0,0.9999998735888544,-1000.0,0.9999999198573067,9.253690467190285E-8,1000.0,-999.0000001264111,0.9999998735888544,-1000.0,-1000.0});
		
		//optimization
		//LPPrimalDualMethodOLD opt = new LPPrimalDualMethodOLD();
		LPPrimalDualMethod opt = new LPPrimalDualMethod();
		
		opt.setLPOptimizationRequest(or);
		int returnCode = opt.optimize();
		
		if(returnCode == OptimizationResponse.FAILED){
			fail();
		}
		
		LPOptimizationResponse response = opt.getLPOptimizationResponse();
		double[] sol = response.getSolution();
		RealVector cVector = new ArrayRealVector(c);
		RealVector solVector = new ArrayRealVector(sol);
		double value = cVector.dotProduct(solVector);
		log.debug("sol   : " + ArrayUtils.toString(sol));
		log.debug("value : "	+ value);
		
		//check constraints
		RealVector x = MatrixUtils.createRealVector(sol);
		RealMatrix GMatrix = MatrixUtils.createRealMatrix(G); 
		RealVector hvector = MatrixUtils.createRealVector(h);
		RealMatrix AMatrix = MatrixUtils.createRealMatrix(A); 
		RealVector bvector = MatrixUtils.createRealVector(b);
		RealVector Gxh = GMatrix.operate(x).subtract(hvector);
		for(int i=0; i<Gxh.getDimension(); i++){
			assertTrue(Gxh.getEntry(i)<=0);//not strictly because some constraint has been treated as a bound
		}
		RealVector Axb = AMatrix.operate(x).subtract(bvector);
		assertEquals(0., Axb.getNorm(), or.getToleranceFeas());
		
		//check value
		assertEquals(expectedvalue, value, or.getTolerance());
		
	}
	
	/**
	 * Problem in the form
	 * min(c.x) s.t.
	 * G.x < h
	 * A.x = b
	 * lb <= x <= ub
	 * 
	 * This is the same as testCGhAb3, but lb and ub are outside G.
	 * The presolved problem has a deterministic solution, that is, all the variables have a fixed value.
	 * Submitted 01/09/2013 by Chris Myers.
	 */
	public void testCGhAbLbUb2() throws Exception {
		log.debug("testCGhAbLbUb2");
		
		String problemId = "2";
		
		log.debug("problemId: " + problemId);
		double[] c = Utils.loadDoubleArrayFromFile("lp"+File.separator+"c"+problemId+".txt");
		double[][] G = Utils.loadDoubleMatrixFromFile("lp"+File.separator+"G"+problemId+".txt", " ".charAt(0));
		double[] h = Utils.loadDoubleArrayFromFile("lp"+File.separator+"h"+problemId+".txt");;
		double[][] A = Utils.loadDoubleMatrixFromFile("lp"+File.separator+"A"+problemId+".txt", " ".charAt(0));
		double[] b = Utils.loadDoubleArrayFromFile("lp"+File.separator+"b"+problemId+".txt");
		double[] lb = Utils.loadDoubleArrayFromFile("lp"+File.separator+"lb"+problemId+".txt");
		double[] ub = Utils.loadDoubleArrayFromFile("lp"+File.separator+"ub"+problemId+".txt");
		double[] expectedSol = Utils.loadDoubleArrayFromFile("lp"+File.separator+"sol"+problemId+".txt");
		double expectedvalue = Utils.loadDoubleArrayFromFile("lp"+File.separator+"value"+problemId+".txt")[0];
		double minLb = 0;
		double maxUb = 1.0E15;//it is do high because of the very high values of the elements of h
		
		LPOptimizationRequest or = new LPOptimizationRequest();
		or.setC(c);
		or.setG(G);
		or.setH(h);
		or.setA(A);
		or.setB(b);
		or.setLb(lb);
		or.setUb(ub);
		//or.setInitialPoint(new double[] {100000.00000000377, 2000000.0000000752, 100000.00000000095, 2000000.0000000189, 100000.00000000095, 2000000.0000000189, 100000.00000000095, 2000000.0000000189, 100000.00000000095, 2000000.0000000189});
		or.setCheckKKTSolutionAccuracy(true);
		//or.setToleranceKKT(1.e-5);
		//or.setToleranceFeas(5.E-5);
		//or.setTolerance(1.E-7);
		or.setDumpProblem(true);
		//or.setPresolvingDisabled(true);
		
		//optimization
		LPPrimalDualMethod opt = new LPPrimalDualMethod(minLb, maxUb);
		
		opt.setLPOptimizationRequest(or);
		int returnCode = opt.optimize();
		
		if(returnCode == OptimizationResponse.FAILED){
			fail();
		}
		
		LPOptimizationResponse response = opt.getLPOptimizationResponse();
		double[] sol = response.getSolution();
		RealVector cVector = new ArrayRealVector(c);
		RealVector solVector = new ArrayRealVector(sol);
		double value = cVector.dotProduct(solVector);
		log.debug("sol   : " + ArrayUtils.toString(sol));
		log.debug("value : "	+ value);
		
		//check constraints
		assertEquals(lb.length, sol.length);
		assertEquals(ub.length, sol.length);
		RealVector x = MatrixUtils.createRealVector(sol);
		RealMatrix GMatrix = MatrixUtils.createRealMatrix(G); 
		RealVector hvector = MatrixUtils.createRealVector(h);
		RealMatrix AMatrix = MatrixUtils.createRealMatrix(A); 
		RealVector bvector = MatrixUtils.createRealVector(b);
		for(int i=0; i<lb.length; i++){
			double di = Double.isNaN(lb[i])? -Double.MAX_VALUE : lb[i];
			assertTrue(di <= x.getEntry(i));
		}
		for(int i=0; i<ub.length; i++){
			double di = Double.isNaN(ub[i])? Double.MAX_VALUE : ub[i];
			assertTrue(di>= x.getEntry(i));
		}
		RealVector Gxh = GMatrix.operate(x).subtract(hvector);
		for(int i=0; i<Gxh.getDimension(); i++){
			assertTrue(Gxh.getEntry(i)<0);
		}
		RealVector Axb = AMatrix.operate(x).subtract(bvector);
		assertEquals(0., Axb.getNorm(), or.getToleranceFeas());
		
		assertEquals( expectedSol.length, sol.length);
		for(int i=0; i<sol.length; i++){
			assertEquals(expectedSol[0], sol[0], 1.e-7);
		}
		assertEquals(expectedvalue, value, 1.e-7);
	
	}
	
	/**
	 * Simple problem in the form
	 * min(c.x) s.t.
	 * G.x < h
	 * A.x = b
	 * 
	 * This is the same as testCGhAbLbUb2, but lb and ub are into G.
	 * The presolved problem has a deterministic solution, that is, all the variables have a fixed value.
	 */
	public void testCGhAb3() throws Exception {
		log.debug("testCGhAb3");
		
		String problemId = "3";
		
		log.debug("problemId: " + problemId);
		double[] c = Utils.loadDoubleArrayFromFile("lp"+File.separator+"c"+problemId+".txt");
		double[][] G = Utils.loadDoubleMatrixFromFile("lp"+File.separator+"G"+problemId+".txt", " ".charAt(0));
		double[] h = Utils.loadDoubleArrayFromFile("lp"+File.separator+"h"+problemId+".txt");;
		double[][] A = Utils.loadDoubleMatrixFromFile("lp"+File.separator+"A"+problemId+".txt", " ".charAt(0));
		double[] b = Utils.loadDoubleArrayFromFile("lp"+File.separator+"b"+problemId+".txt");
		double[] expectedSol = Utils.loadDoubleArrayFromFile("lp"+File.separator+"sol"+problemId+".txt");
		double expectedvalue = Utils.loadDoubleArrayFromFile("lp"+File.separator+"value"+problemId+".txt")[0];
		double minLb = 0;
		double maxUb = 1.0E15;//it is so high because of the very high values of the elements of h
		
		LPOptimizationRequest or = new LPOptimizationRequest();
		or.setC(c);
		or.setG(G);
		or.setH(h);
		or.setA(A);
		or.setB(b);
		or.setCheckKKTSolutionAccuracy(true);
		//or.setToleranceKKT(1.e-5);
//		or.setToleranceFeas(5.E-5);
//		or.setTolerance(1.E-7);
		or.setDumpProblem(true);
		//or.setPresolvingDisabled(true);
		
		//optimization
		LPPrimalDualMethod opt = new LPPrimalDualMethod(minLb, maxUb);
		
		opt.setLPOptimizationRequest(or);
		int returnCode = opt.optimize();
		
		if(returnCode == OptimizationResponse.FAILED){
			fail();
		}
		
		LPOptimizationResponse response = opt.getLPOptimizationResponse();
		double[] sol = response.getSolution();
		RealVector cVector = new ArrayRealVector(c);
		RealVector solVector = new ArrayRealVector(sol);
		double value = cVector.dotProduct(solVector);
		log.debug("sol   : " + ArrayUtils.toString(sol));
		log.debug("value : "	+ value);
		
		//check constraints
		RealVector x = MatrixUtils.createRealVector(sol);
		RealMatrix GMatrix = MatrixUtils.createRealMatrix(G); 
		RealVector hvector = MatrixUtils.createRealVector(h);
		RealMatrix AMatrix = MatrixUtils.createRealMatrix(A); 
		RealVector bvector = MatrixUtils.createRealVector(b);
		RealVector Gxh = GMatrix.operate(x).subtract(hvector);
		for(int i=0; i<Gxh.getDimension(); i++){
			assertTrue(Gxh.getEntry(i)<=0);
		}
		RealVector Axb = AMatrix.operate(x).subtract(bvector);
		assertEquals(0., Axb.getNorm(), or.getToleranceFeas());
		
		assertEquals( expectedSol.length, sol.length);
		for(int i=0; i<sol.length; i++){
			assertEquals(expectedSol[0], sol[0], or.getTolerance());
		}
		assertEquals(expectedvalue, value, or.getTolerance());
	
	}
	
	/**
	 * Problem in the form
	 * min(c.x) s.t.
	 * A.x = b
	 * lb <= x <= ub
	 * 
	 * This problem involves recursive column duplicate reductions.
	 * This is a good for testing with a small size problem.
	 */
	public void testCAbLbUb5() throws Exception {
		log.debug("testCAbLbUb5");
		
		String problemId = "5";
		
		log.debug("problemId: " + problemId);
		double[] c = Utils.loadDoubleArrayFromFile("lp"+File.separator+"c"+problemId+".txt");
		double[][] A = Utils.loadDoubleMatrixFromFile("lp"+File.separator+"A"+problemId+".txt", " ".charAt(0));
		double[] b = Utils.loadDoubleArrayFromFile("lp"+File.separator+"b"+problemId+".txt");
		double[] lb = Utils.loadDoubleArrayFromFile("lp"+File.separator+"lb"+problemId+".txt");
		double[] ub = Utils.loadDoubleArrayFromFile("lp"+File.separator+"ub"+problemId+".txt");
		double[] expectedSol = Utils.loadDoubleArrayFromFile("lp"+File.separator+"sol"+problemId+".txt");
		double expectedvalue = Utils.loadDoubleArrayFromFile("lp"+File.separator+"value"+problemId+".txt")[0];
		
		LPOptimizationRequest or = new LPOptimizationRequest();
		or.setC(c);
		or.setA(A);
		or.setB(b);
		or.setLb(lb);
		or.setUb(ub);
		or.setCheckKKTSolutionAccuracy(true);
//		or.setToleranceKKT(1.e-7);
//		or.setToleranceFeas(1.E-7);
//		or.setTolerance(1.E-7);
		or.setDumpProblem(true);
		
		//optimization
		LPPrimalDualMethod opt = new LPPrimalDualMethod();
		
		opt.setLPOptimizationRequest(or);
		int returnCode = opt.optimize();
		
		if(returnCode == OptimizationResponse.FAILED){
			fail();
		}
		
		LPOptimizationResponse response = opt.getLPOptimizationResponse();
		double[] sol = response.getSolution();
		RealVector cVector = new ArrayRealVector(c);
		RealVector solVector = new ArrayRealVector(sol);
		double value = cVector.dotProduct(solVector);
		log.debug("sol   : " + ArrayUtils.toString(sol));
		log.debug("value : "	+ value);
		
		//check constraints
		assertEquals(lb.length, sol.length);
		assertEquals(ub.length, sol.length);
		RealVector x = MatrixUtils.createRealVector(sol);
		RealMatrix AMatrix = MatrixUtils.createRealMatrix(A); 
		RealVector bvector = MatrixUtils.createRealVector(b);
		for(int i=0; i<lb.length; i++){
			double di = Double.isNaN(lb[i])? -Double.MAX_VALUE : lb[i];
			assertTrue(di <= x.getEntry(i));
		}
		for(int i=0; i<ub.length; i++){
			double di = Double.isNaN(ub[i])? Double.MAX_VALUE : ub[i];
			assertTrue(di >= x.getEntry(i));
		}
		RealVector Axb = AMatrix.operate(x).subtract(bvector);
		assertEquals(0., Axb.getNorm(), or.getToleranceFeas());
		
		//check value
		assertEquals(expectedvalue, value, or.getTolerance());
	}
	
	/**
	 * Problem in the form
	 * min(c.x) s.t.
	 * A.x = b
	 * lb <= x <= ub
	 * 
	 * This problem involves column duplicate reduction.
	 * This is a good for testing with a small size problem.
	 */
	public void testCAbLbUb6() throws Exception {
		log.debug("testCAbLbUb6");
		
		String problemId = "6";
		
		log.debug("problemId: " + problemId);
		double[] c = Utils.loadDoubleArrayFromFile("lp"+File.separator+"c"+problemId+".txt");
		double[][] A = Utils.loadDoubleMatrixFromFile("lp"+File.separator+"A"+problemId+".txt", " ".charAt(0));
		double[] b = Utils.loadDoubleArrayFromFile("lp"+File.separator+"b"+problemId+".txt");
		double[] lb = Utils.loadDoubleArrayFromFile("lp"+File.separator+"lb"+problemId+".txt");
		double[] ub = Utils.loadDoubleArrayFromFile("lp"+File.separator+"ub"+problemId+".txt");
		double[] expectedSol = Utils.loadDoubleArrayFromFile("lp"+File.separator+"sol"+problemId+".txt");
		double expectedvalue = Utils.loadDoubleArrayFromFile("lp"+File.separator+"value"+problemId+".txt")[0];
		
		LPOptimizationRequest or = new LPOptimizationRequest();
		or.setC(c);
		or.setA(A);
		or.setB(b);
		or.setLb(lb);
		or.setUb(ub);
		or.setCheckKKTSolutionAccuracy(true);
//		or.setToleranceKKT(1.e-7);
//		or.setToleranceFeas(1.E-7);
//		or.setTolerance(1.E-7);
		or.setDumpProblem(true);
		
		//optimization
		LPPrimalDualMethod opt = new LPPrimalDualMethod();
		
		opt.setLPOptimizationRequest(or);
		int returnCode = opt.optimize();
		
		if(returnCode == OptimizationResponse.FAILED){
			fail();
		}
		
		LPOptimizationResponse response = opt.getLPOptimizationResponse();
		double[] sol = response.getSolution();
		RealVector cVector = new ArrayRealVector(c);
		RealVector solVector = new ArrayRealVector(sol);
		double value = cVector.dotProduct(solVector);
		log.debug("sol   : " + ArrayUtils.toString(sol));
		log.debug("value : "	+ value);
		
		//check constraints
		assertEquals(lb.length, sol.length);
		assertEquals(ub.length, sol.length);
		RealVector x = MatrixUtils.createRealVector(sol);
		RealMatrix AMatrix = MatrixUtils.createRealMatrix(A); 
		RealVector bvector = MatrixUtils.createRealVector(b);
		for(int i=0; i<lb.length; i++){
			double di = Double.isNaN(lb[i])? -Double.MAX_VALUE : lb[i];
			assertTrue(di <= x.getEntry(i));
		}
		for(int i=0; i<ub.length; i++){
			double di = Double.isNaN(ub[i])? Double.MAX_VALUE : ub[i];
			assertTrue(di>= x.getEntry(i));
		}
		RealVector Axb = AMatrix.operate(x).subtract(bvector);
		assertEquals(0., Axb.getNorm(), or.getToleranceFeas());
		
		//check value
		assertEquals(expectedvalue, value, or.getTolerance());
	}
	
	/**
	 * Problem in the form
	 * min(c.x) s.t.
	 * G.x < h
	 * A.x = b
	 * lb <= x <= ub
	 * 
	 */
	public void testCGhAbLbUb7() throws Exception {
		log.debug("testCGhAbLbUb7");
		
		String problemId = "7";
		
		log.debug("problemId: " + problemId);
		double[] c = Utils.loadDoubleArrayFromFile("lp"+File.separator+"c"+problemId+".txt");
		double[][] G = Utils.loadDoubleMatrixFromFile("lp"+File.separator+"G"+problemId+".txt", " ".charAt(0));
		double[] h = Utils.loadDoubleArrayFromFile("lp"+File.separator+"h"+problemId+".txt");;
		double[][] A = Utils.loadDoubleMatrixFromFile("lp"+File.separator+"A"+problemId+".txt", " ".charAt(0));
		double[] b = Utils.loadDoubleArrayFromFile("lp"+File.separator+"b"+problemId+".txt");
		double[] lb = Utils.loadDoubleArrayFromFile("lp"+File.separator+"lb"+problemId+".txt");
		double[] ub = Utils.loadDoubleArrayFromFile("lp"+File.separator+"ub"+problemId+".txt");
		double[] expectedSol = Utils.loadDoubleArrayFromFile("lp"+File.separator+"sol"+problemId+".txt");
		double expectedvalue = Utils.loadDoubleArrayFromFile("lp"+File.separator+"value"+problemId+".txt")[0];
		
		//the unbounded bounds are saved on the files with NaN values, so substitute them with acceptable values
		lb = Utils.replaceValues(lb, Double.NaN, LPPrimalDualMethod.DEFAULT_MIN_LOWER_BOUND);
		ub = Utils.replaceValues(ub, Double.NaN, LPPrimalDualMethod.DEFAULT_MAX_UPPER_BOUND);
		
		LPOptimizationRequest or = new LPOptimizationRequest();
		or.setC(c);
		or.setG(G);
		or.setH(h);
		or.setA(A);
		or.setB(b);
		or.setLb(lb);
		or.setUb(ub);
		or.setCheckKKTSolutionAccuracy(true);
//		or.setToleranceKKT(1.e-7);
//		or.setToleranceFeas(1.E-7);
//		or.setTolerance(1.E-7);
		or.setDumpProblem(true);
		//or.setPresolvingDisabled(true);
		//or.setRescalingDisabled(true);
		
		//optimization
		LPPrimalDualMethod opt = new LPPrimalDualMethod();
		
		opt.setLPOptimizationRequest(or);
		int returnCode = opt.optimize();
		
		if(returnCode == OptimizationResponse.FAILED){
			fail();
		}
		
		LPOptimizationResponse response = opt.getLPOptimizationResponse();
		double[] sol = response.getSolution();
		RealVector cVector = new ArrayRealVector(c);
		RealVector solVector = new ArrayRealVector(sol);
		double value = cVector.dotProduct(solVector);
		log.debug("sol   : " + ArrayUtils.toString(sol));
		log.debug("value : "	+ value);
		
		//check constraints
		assertEquals(lb.length, sol.length);
		assertEquals(ub.length, sol.length);
		RealVector x = MatrixUtils.createRealVector(sol);
		RealMatrix GMatrix = MatrixUtils.createRealMatrix(G); 
		RealVector hvector = MatrixUtils.createRealVector(h);
		RealMatrix AMatrix = MatrixUtils.createRealMatrix(A); 
		RealVector bvector = MatrixUtils.createRealVector(b);
		for(int i=0; i<lb.length; i++){
			assertTrue(lb[i] <= x.getEntry(i));
		}
		for(int i=0; i<ub.length; i++){
			double di = Double.isNaN(lb[i])? -Double.MAX_VALUE : lb[i];
			assertTrue(di <= x.getEntry(i));
		}
		RealVector Gxh = GMatrix.operate(x).subtract(hvector);
		for(int i=0; i<Gxh.getDimension(); i++){
			double di = Double.isNaN(ub[i])? Double.MAX_VALUE : ub[i];
			assertTrue(di>= x.getEntry(i));
		}
		RealVector Axb = AMatrix.operate(x).subtract(bvector);
		assertEquals(0., Axb.getNorm(), or.getToleranceFeas());
		
		assertEquals( expectedSol.length, sol.length);
		for(int i=0; i<sol.length; i++){
			assertEquals(expectedSol[0], sol[0], or.getTolerance());
		}
		assertEquals(expectedvalue, value, or.getTolerance());
	
	}
	
	/**
	 * Problem in the form
	 * min(c.x) s.t.
	 * A.x = b
	 * lb <= x <= ub
	 */
	public void testCAbLbUb8() throws Exception {
		log.debug("testCAbLbUb8");
		
		String problemId = "8";
		
		log.debug("problemId: " + problemId);
		double[] c = Utils.loadDoubleArrayFromFile("lp"+File.separator+"c"+problemId+".txt");
		double[][] A = Utils.loadDoubleMatrixFromFile("lp"+File.separator+"A"+problemId+".txt", " ".charAt(0));
		double[] b = Utils.loadDoubleArrayFromFile("lp"+File.separator+"b"+problemId+".txt");
		double[] lb = Utils.loadDoubleArrayFromFile("lp"+File.separator+"lb"+problemId+".txt");
		double[] ub = Utils.loadDoubleArrayFromFile("lp"+File.separator+"ub"+problemId+".txt");
		double[] expectedSol = Utils.loadDoubleArrayFromFile("lp"+File.separator+"sol"+problemId+".txt");
		double expectedvalue = Utils.loadDoubleArrayFromFile("lp"+File.separator+"value"+problemId+".txt")[0];
		
		//the unbounded bounds are saved on the files with NaN values, so substitute them with acceptable values
		lb = Utils.replaceValues(lb, Double.NaN, LPPrimalDualMethod.DEFAULT_MIN_LOWER_BOUND);
		ub = Utils.replaceValues(ub, Double.NaN, LPPrimalDualMethod.DEFAULT_MAX_UPPER_BOUND);
		
		LPOptimizationRequest or = new LPOptimizationRequest();
		or.setC(c);
		or.setA(A);
		or.setB(b);
		or.setLb(lb);
		or.setUb(ub);
		or.setCheckKKTSolutionAccuracy(true);
//		or.setToleranceKKT(1.e-7);
//		or.setToleranceFeas(1.E-7);
//		or.setTolerance(1.E-7);
		or.setDumpProblem(true);
		or.setRescalingDisabled(true);
		
		//optimization
		LPPrimalDualMethod opt = new LPPrimalDualMethod();
		
		opt.setLPOptimizationRequest(or);
		int returnCode = opt.optimize();
		
		if(returnCode == OptimizationResponse.FAILED){
			fail();
		}
		
		LPOptimizationResponse response = opt.getLPOptimizationResponse();
		double[] sol = response.getSolution();
		RealVector cVector = new ArrayRealVector(c);
		RealVector solVector = new ArrayRealVector(sol);
		double value = cVector.dotProduct(solVector);
		log.debug("sol   : " + ArrayUtils.toString(sol));
		log.debug("value : "	+ value);
		
		//check constraints
		assertEquals(lb.length, sol.length);
		assertEquals(ub.length, sol.length);
		RealVector x = MatrixUtils.createRealVector(sol);
		RealMatrix AMatrix = MatrixUtils.createRealMatrix(A); 
		RealVector bvector = MatrixUtils.createRealVector(b);
		for(int i=0; i<lb.length; i++){
			double di = Double.isNaN(lb[i])? -Double.MAX_VALUE : lb[i];
			assertTrue(di <= x.getEntry(i));
		}
		for(int i=0; i<ub.length; i++){
			double di = Double.isNaN(ub[i])? Double.MAX_VALUE : ub[i];
			assertTrue(di>= x.getEntry(i));
		}
		RealVector Axb = AMatrix.operate(x).subtract(bvector);
		assertEquals(0., Axb.getNorm(), or.getToleranceFeas());
		
		//check value
		assertEquals(expectedvalue, value, or.getTolerance());	
	}
	
	/**
	 * Problem in the form
	 * min(c.x) s.t.
	 * G.x < h
	 * A.x = b
	 * lb <= x <= ub
	 * 
	 */
	public void testCGhAbLbUb10() throws Exception {
		log.debug("testCGhAbLbUb7");
		
		String problemId = "10";
		
		log.debug("problemId: " + problemId);
		double[] c = Utils.loadDoubleArrayFromFile("lp"+File.separator+"c"+problemId+".txt");
		double[][] G = Utils.loadDoubleMatrixFromFile("lp"+File.separator+"G"+problemId+".txt", " ".charAt(0));
		double[] h = Utils.loadDoubleArrayFromFile("lp"+File.separator+"h"+problemId+".txt");;
		double[][] A = Utils.loadDoubleMatrixFromFile("lp"+File.separator+"A"+problemId+".txt", " ".charAt(0));
		double[] b = Utils.loadDoubleArrayFromFile("lp"+File.separator+"b"+problemId+".txt");
		double[] lb = Utils.loadDoubleArrayFromFile("lp"+File.separator+"lb"+problemId+".txt");
		double[] ub = Utils.loadDoubleArrayFromFile("lp"+File.separator+"ub"+problemId+".txt");
		double[] expectedSol = Utils.loadDoubleArrayFromFile("lp"+File.separator+"sol"+problemId+".txt");
		double expectedvalue = Utils.loadDoubleArrayFromFile("lp"+File.separator+"value"+problemId+".txt")[0];
		
		//the unbounded bounds are saved on the files with NaN values, so substitute them with acceptable values
		lb = Utils.replaceValues(lb, Double.NaN, LPPrimalDualMethod.DEFAULT_MIN_LOWER_BOUND);
		ub = Utils.replaceValues(ub, Double.NaN, LPPrimalDualMethod.DEFAULT_MAX_UPPER_BOUND);
		
		LPOptimizationRequest or = new LPOptimizationRequest();
		or.setC(c);
		or.setG(G);
		or.setH(h);
		or.setA(A);
		or.setB(b);
		or.setLb(lb);
		or.setUb(ub);
		or.setCheckKKTSolutionAccuracy(true);
//		or.setToleranceKKT(1.e-7);
//		or.setToleranceFeas(1.E-7);
//		or.setTolerance(1.E-7);
		or.setDumpProblem(true);
		//or.setPresolvingDisabled(true);
		//or.setRescalingDisabled(true);
		
		//optimization
		LPPrimalDualMethod opt = new LPPrimalDualMethod();
		
		opt.setLPOptimizationRequest(or);
		int returnCode = opt.optimize();
		
		if(returnCode == OptimizationResponse.FAILED){
			fail();
		}
		
		LPOptimizationResponse response = opt.getLPOptimizationResponse();
		double[] sol = response.getSolution();
		RealVector cVector = new ArrayRealVector(c);
		RealVector solVector = new ArrayRealVector(sol);
		double value = cVector.dotProduct(solVector);
		log.debug("sol   : " + ArrayUtils.toString(sol));
		log.debug("value : "	+ value);
		
		//check constraints
		assertEquals(lb.length, sol.length);
		assertEquals(ub.length, sol.length);
		RealVector x = MatrixUtils.createRealVector(sol);
		RealMatrix GMatrix = MatrixUtils.createRealMatrix(G); 
		RealVector hvector = MatrixUtils.createRealVector(h);
		RealMatrix AMatrix = MatrixUtils.createRealMatrix(A); 
		RealVector bvector = MatrixUtils.createRealVector(b);
		for(int i=0; i<lb.length; i++){
			assertTrue(lb[i] <= x.getEntry(i));
		}
		for(int i=0; i<ub.length; i++){
			double di = Double.isNaN(lb[i])? -Double.MAX_VALUE : lb[i];
			assertTrue(di <= x.getEntry(i));
		}
		RealVector Gxh = GMatrix.operate(x).subtract(hvector);
		for(int i=0; i<Gxh.getDimension(); i++){
			double di = Double.isNaN(ub[i])? Double.MAX_VALUE : ub[i];
			assertTrue(di>= x.getEntry(i));
		}
		RealVector Axb = AMatrix.operate(x).subtract(bvector);
		assertEquals(0., Axb.getNorm(), or.getToleranceFeas());
		
		assertEquals( expectedSol.length, sol.length);
		for(int i=0; i<sol.length; i++){
			assertEquals(expectedSol[0], sol[0], or.getTolerance());
		}
		assertEquals(expectedvalue, value, or.getTolerance());
	
	}
	
}
