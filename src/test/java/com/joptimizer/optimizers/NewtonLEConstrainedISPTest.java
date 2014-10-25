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

import junit.framework.TestCase;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import com.joptimizer.functions.LinearMultivariateRealFunction;
import com.joptimizer.functions.PDQuadraticMultivariateRealFunction;

/**
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class NewtonLEConstrainedISPTest extends TestCase {
	private Log log = LogFactory.getLog(this.getClass().getName());

	public void testOptimize1() throws Exception {
		log.debug("testOptimize1");
		
		// START SNIPPET: NewtonLEConstrainedISP-1
		
		//commons-math client code
		RealMatrix Pmatrix = new Array2DRowRealMatrix(new double[][] { 
				{ 1.68, 0.34, 0.38 },
				{ 0.34, 3.09, -1.59 }, 
				{ 0.38, -1.59, 1.54 } });
		RealVector qVector = new ArrayRealVector(new double[] { 0.018, 0.025, 0.01 });

		// Objective function
		double theta = 0.01522;
		RealMatrix P = Pmatrix.scalarMultiply(theta);
		RealVector q = qVector.mapMultiply(-1);
		PDQuadraticMultivariateRealFunction objectiveFunction = new PDQuadraticMultivariateRealFunction(P.getData(), q.toArray(), 0);

		OptimizationRequest or = new OptimizationRequest();
		or.setF0(objectiveFunction);
		or.setInitialPoint(new double[] { 0.1, 0.1, 0.1 });//LE-infeasible starting point
		or.setA(new double[][] { { 1, 1, 1 } });
		or.setB(new double[] { 1 });

		// optimization
		NewtonLEConstrainedISP opt = new NewtonLEConstrainedISP();
		opt.setOptimizationRequest(or);
		int returnCode = opt.optimize();
		
		// END SNIPPET: NewtonLEConstrainedISP-1
		
		if(returnCode==OptimizationResponse.FAILED){
			fail();
		}
		
		OptimizationResponse response = opt.getOptimizationResponse();
		double[] sol = response.getSolution();
		log.debug("sol   : " + ArrayUtils.toString(sol));
		log.debug("value : " + objectiveFunction.value(sol));
		assertEquals(0.04632311555988555, sol[0], 0.000000000000001);
		assertEquals(0.5086308460954377,  sol[1], 0.000000000000001);
		assertEquals(0.44504603834467693, sol[2], 0.000000000000001);
	}
	
	/**
	 * Minimize x subject to 
	 * x+y=4, 
	 * x-y=2. 
	 * Should return (3,1).
	 * This problem is the same as LPPrimalDualMethodTest.testSimple4()
	 * and can be solved only with the use of a linear presolving phase:
	 * if passed directly to the solver, it will fail because JOptimizer
	 * does not want rank-deficient inequalities matrices like that of this problem.
	 */
	public void testOptimize2() throws Exception {
		log.debug("testOptimize2");
		double[] minimizeF = new double[] { 1.0, 0.0 };
		LinearMultivariateRealFunction objectiveFunction = new LinearMultivariateRealFunction(minimizeF, 0.0);

		// Equalities:
		double[][] equalityAMatrix = new double[][] { { 1.0, 1.0 }, { 1.0, -1.0 } };
		double[] equalityBVector = new double[] { 4.0, 2.0 };

		//optimization problem
		OptimizationRequest or = new OptimizationRequest();
		or.setF0(objectiveFunction);
		or.setA(equalityAMatrix);
		or.setB(equalityBVector);
		
		//optimization
		NewtonLEConstrainedISP opt = new NewtonLEConstrainedISP();
		opt.setOptimizationRequest(or);
		try{
			opt.optimize();
			fail();
		}catch(Exception e){
			//this problem cannot be passed directly to the solvers of JOptimizer
			//because they do not want rank-deficient inequalities matrices
			assertTrue(true);
		}
	}
	
	/**
	 * Minimize 0 subject to 
	 * x+y=4. 
	 * Should return any feasible solution.
	 */
	public void testOptimize3() throws Exception {
		log.debug("testOptimize3");
		double[] minimizeF = new double[] { 0.0, 0.0 };
		LinearMultivariateRealFunction objectiveFunction = new LinearMultivariateRealFunction(minimizeF, 0.0);

		// Equalities:
		double[][] equalityAMatrix = new double[][] { { 1.0, 1.0 } };
		double[] equalityBVector = new double[] { 4.0 };

		//optimization problem
		OptimizationRequest or = new OptimizationRequest();
		or.setF0(objectiveFunction);
		or.setA(equalityAMatrix);
		or.setB(equalityBVector);
		
		//optimization
		NewtonLEConstrainedISP opt = new NewtonLEConstrainedISP();
		opt.setOptimizationRequest(or);
		int returnCode = opt.optimize();
		
		if(returnCode==OptimizationResponse.FAILED){
			fail();
		}

		OptimizationResponse response = opt.getOptimizationResponse();
		double[] sol = response.getSolution();
		log.debug("sol: " + ArrayUtils.toString(sol));
		log.debug("value  : " + objectiveFunction.value(sol));
		assertEquals(4.0, sol[0] + sol[1], 1e-8);
	}
}
