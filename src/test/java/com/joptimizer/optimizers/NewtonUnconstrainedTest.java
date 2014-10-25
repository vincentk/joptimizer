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
import org.apache.commons.math3.linear.CholeskyDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import cern.jet.math.Mult;

import com.joptimizer.functions.PDQuadraticMultivariateRealFunction;
import com.joptimizer.util.Utils;

/**
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class NewtonUnconstrainedTest extends TestCase {
	private Algebra ALG = Algebra.DEFAULT;
	private DoubleFactory1D F1 = DoubleFactory1D.dense;
	private DoubleFactory2D F2 = DoubleFactory2D.dense;
	private Log log = LogFactory.getLog(this.getClass().getName());

	/**
	 * Quadratic objective.
	 */
	public void testOptimize() throws Exception {
		log.debug("testOptimize");
		// START SNIPPET: newtonUnconstrained-1
		
		RealMatrix PMatrix = new Array2DRowRealMatrix(new double[][] {
				{ 1.68, 0.34, 0.38 },
				{ 0.34, 3.09, -1.59 }, 
				{ 0.38, -1.59, 1.54 } });
		RealVector qVector = new ArrayRealVector(new double[] { 0.018, 0.025, 0.01 });

	    // Objective function.
		double theta = 0.01522;
		RealMatrix P = PMatrix.scalarMultiply(theta);
		RealVector q = qVector.mapMultiply(-1);
		PDQuadraticMultivariateRealFunction objectiveFunction = new PDQuadraticMultivariateRealFunction(P.getData(), q.toArray(), 0);
		
		OptimizationRequest or = new OptimizationRequest();
		or.setF0(objectiveFunction);
		or.setInitialPoint(new double[] {0.04, 0.50, 0.46});
		or.setTolerance(1.e-8);
		
	    //optimization
		NewtonUnconstrained opt = new NewtonUnconstrained();
		opt.setOptimizationRequest(or);
		int returnCode = opt.optimize(); 
		
		// END SNIPPET: newtonUnconstrained-1
		
		if(returnCode==OptimizationResponse.FAILED){
			fail();
		}
		
		OptimizationResponse response = opt.getOptimizationResponse();
		double[] sol = response.getSolution();
		log.debug("sol   : " + ArrayUtils.toString(sol));
		log.debug("value : "	+ objectiveFunction.value(sol));

		// we know the analytic solution of the problem
		// sol = -PInv * q
		CholeskyDecomposition cFact = new CholeskyDecomposition(P);
		RealVector benchSol = cFact.getSolver().solve(q).mapMultiply(-1);
		log.debug("benchSol   : " + ArrayUtils.toString(benchSol.toArray()));
		log.debug("benchValue : " + objectiveFunction.value(benchSol.toArray()));

		assertEquals(benchSol.getEntry(0), sol[0], 0.00000000000001);
		assertEquals(benchSol.getEntry(1), sol[1], 0.00000000000001);
		assertEquals(benchSol.getEntry(2), sol[2], 0.00000000000001);
	}

	/**
	 * Test with quite large positive definite symmetric matrix.
	 */
	public void testOptimize2() throws Exception {
		log.debug("testOptimize2");

		int dim = 40;
		
		// positive definite matrix
		Long seed = new Long(54321);
		DoubleMatrix2D mySymmPD = Utils.randomValuesPositiveMatrix(dim, dim, -0.01, 15.5, seed);
		DoubleMatrix1D CVector = Utils.randomValuesMatrix(1, dim, -0.01, 15.5, seed).viewRow(0);
		MySymmFunction objectiveFunction = new MySymmFunction(mySymmPD,	CVector);
		
		//optimization
		OptimizationRequest or = new OptimizationRequest();
		or.setF0(objectiveFunction);
		NewtonUnconstrained opt = new NewtonUnconstrained();
		opt.setOptimizationRequest(or);
		int returnCode = opt.optimize();
		
		if(returnCode==OptimizationResponse.FAILED){
			fail();
		}
		
		OptimizationResponse response = opt.getOptimizationResponse();
		double[] sol = response.getSolution();
		log.debug("sol   : " + ArrayUtils.toString(sol));
		log.debug("value : " + objectiveFunction.value(sol));
		
	  // we know the analytic solution of the problem: Qinv.sol = - C
		cern.colt.matrix.linalg.CholeskyDecomposition cFact = new cern.colt.matrix.linalg.CholeskyDecomposition(mySymmPD);
		DoubleMatrix1D benchSol = cFact.solve(F2.make(CVector.copy().assign(Mult.mult(-1)).toArray(), CVector.size())).viewColumn(0);
		log.debug("benchSol   : " + ArrayUtils.toString(benchSol.toArray()));
		log.debug("benchValue : "	+ objectiveFunction.value(benchSol.toArray()));

		for(int i=0; i<dim;i++){
			assertEquals(benchSol.get(i), sol[i], 0.000001);
		}
	}

	private class MySymmFunction extends PDQuadraticMultivariateRealFunction {

		public MySymmFunction(DoubleMatrix2D P, DoubleMatrix1D q) {
			super(P.toArray(), q.toArray(), 0);
		}

	}
}
