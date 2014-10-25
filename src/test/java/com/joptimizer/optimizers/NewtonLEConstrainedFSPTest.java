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

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import cern.jet.math.Mult;

import com.joptimizer.functions.ConvexMultivariateRealFunction;
import com.joptimizer.functions.PDQuadraticMultivariateRealFunction;

/**
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class NewtonLEConstrainedFSPTest extends TestCase {
	private Algebra ALG = Algebra.DEFAULT;
	private DoubleFactory1D F1 = DoubleFactory1D.dense;
	private DoubleFactory2D F2 = DoubleFactory2D.dense;
	private Log log = LogFactory.getLog(this.getClass().getName());

	public void testOptimize() throws Exception {
		log.debug("testOptimize");
		DoubleMatrix2D pMatrix = F2.make(new double[][] { 
				{ 1.68, 0.34, 0.38 },
				{ 0.34, 3.09, -1.59 }, 
				{ 0.38, -1.59, 1.54 } });
		DoubleMatrix1D qVector = F1.make(new double[] { 0.018, 0.025, 0.01 });

		// Objective function (Risk-Aversion).
		double theta = 0.01522;
		double[][] P = pMatrix.assign(Mult.mult(theta)).toArray();
		double[] q = qVector.assign(Mult.mult(-1)).toArray();
		PDQuadraticMultivariateRealFunction objectiveFunction = new PDQuadraticMultivariateRealFunction(P, q, 0);

		OptimizationRequest or = new OptimizationRequest();
		or.setF0(objectiveFunction);
		or.setInitialPoint(new double[] { 0.8, 0.1, 0.1 });
		or.setA(new double[][] { { 1, 1, 1 } });
		or.setB(new double[] { 1 });

		// optimization
		NewtonLEConstrainedFSP opt = new NewtonLEConstrainedFSP();
		opt.setOptimizationRequest(or);
		int returnCode = opt.optimize();
		
		if(returnCode==OptimizationResponse.FAILED){
			fail();
		}
		
		OptimizationResponse response = opt.getOptimizationResponse();
		double[] sol = response.getSolution();
		log.debug("sol   : " + ArrayUtils.toString(sol));
		log.debug("value : " + objectiveFunction.value(sol));
		assertEquals(0.04632311555988555, sol[0], 0.00000000000001);
		assertEquals(0.5086308460954377,  sol[1], 0.00000000000001);
		assertEquals(0.44504603834467693, sol[2], 0.00000000000001);
	}
	
	
	/**
	 * Minimize x - Log[-x^2 + 1], 
	 * dom f ={x | x^2<1}
	 * N.B.: this simulate a centering step of the barrier method 
	 * applied to the problem:
	 * Minimize x
	 * s.t. x^2<1
	 * when t=1.
	 */
	public void testOptimize2() throws Exception {
		log.debug("testOptimize2");
		
		// START SNIPPET: NewtonLEConstrainedFSP-1

		// Objective function
		ConvexMultivariateRealFunction objectiveFunction = new ConvexMultivariateRealFunction() {
			
			public double value(double[] X) {
				double x = X[0];
				return x - Math.log(1-x*x);
			}
			
			public double[] gradient(double[] X) {
				double x = X[0];
				return new double[]{1+2*x/(1-x*x)};
			}
			
			public double[][] hessian(double[] X) {
				double x = X[0];
				return new double[][]{{4*Math.pow(x, 2)/Math.pow(1-x*x, 2)+2/(1-x*x)}};
			}
			
			public int getDim() {
				return 1;
			}
		};

		OptimizationRequest or = new OptimizationRequest();
		or.setCheckKKTSolutionAccuracy(true);
		or.setF0(objectiveFunction);
		or.setInitialPoint(new double[] {0});//must be feasible
		
		// optimization
		NewtonLEConstrainedFSP opt = new NewtonLEConstrainedFSP();
		opt.setOptimizationRequest(or);
		int returnCode = opt.optimize();
		
		// END SNIPPET: NewtonLEConstrainedFSP-1
		
		if(returnCode==OptimizationResponse.FAILED){
			fail();
		}
		
		OptimizationResponse response = opt.getOptimizationResponse();
		double[] sol = response.getSolution();
		double value = objectiveFunction.value(sol);
		log.debug("sol   : " + ArrayUtils.toString(sol));
		log.debug("value : " + value);
		assertEquals(-0.41421356, sol[0], 0.0000001);//=1-Math.sqrt(2)
		assertEquals(-0.22598716, value , 0.0000001);
	}
	
	/**
	 * Minimize 100(2x+y) - Log[x] - Log[y], 
	 * s.t. x+y=1
	 * N.B.: this simulate a centering step of the barrier method 
	 * applied to the problem:
	 * Minimize 2x + y
	 * s.t. -x<0, 
	 *      -y<0
	 *      x+y=1
	 * when t=100; 
	 */
	public void testOptimize3() throws Exception {
		log.debug("testOptimize3");
		
			// Objective function (linear)
		ConvexMultivariateRealFunction objectiveFunction = new ConvexMultivariateRealFunction() {
			
			public double value(double[] X) {
				double x = X[0];
				double y = X[1];
				return 100 * (2*x + y) - Math.log(x)- Math.log(y);
			}
			
			public double[] gradient(double[] X) {
				double x = X[0];
				double y = X[1];
				return new double[]{200-1./x, 100-1./y};
			}
			
			public double[][] hessian(double[] X) {
				double x = X[0];
				double y = X[1];
				return new double[][]{{1./Math.pow(x,2), 0},{0,1./Math.pow(y,2)}};
			}
			
			public int getDim() {
				return 2;
			}
		};

		OptimizationRequest or = new OptimizationRequest();
		or.setF0(objectiveFunction);
		or.setInitialPoint(new double[] {0.0900980486377967, 0.9099019513622053});
		or.setA(new double[][] { { 1, 1} });
		or.setB(new double[] { 1 });
		
		// optimization
		NewtonLEConstrainedFSP opt = new NewtonLEConstrainedFSP(true);
		opt.setOptimizationRequest(or);
		int returnCode = opt.optimize();
		
		if(returnCode==OptimizationResponse.FAILED){
			fail();
		}
		
		OptimizationResponse response = opt.getOptimizationResponse();
		double[] sol = response.getSolution();
		double value = objectiveFunction.value(sol);
		log.debug("sol   : " + ArrayUtils.toString(sol));
		log.debug("value : " + value);
		assertEquals(0., sol[0], 0.01);
		assertEquals(1., sol[1], 0.01);
		assertEquals(1., sol[0]+sol[1],   0.000000000001);//check constraint
	}
}
