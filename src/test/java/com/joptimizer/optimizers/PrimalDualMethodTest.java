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

import java.util.Arrays;

import junit.framework.TestCase;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import cern.jet.math.Functions;
import cern.jet.math.Mult;

import com.joptimizer.functions.ConvexMultivariateRealFunction;
import com.joptimizer.functions.FunctionsUtils;
import com.joptimizer.functions.LinearMultivariateRealFunction;
import com.joptimizer.functions.PDQuadraticMultivariateRealFunction;
import com.joptimizer.functions.PSDQuadraticMultivariateRealFunction;
import com.joptimizer.functions.StrictlyConvexMultivariateRealFunction;
import com.joptimizer.util.Utils;

/**
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class PrimalDualMethodTest extends TestCase {

	private Algebra ALG = Algebra.DEFAULT;
	private DoubleFactory1D F1 = DoubleFactory1D.dense;
	private DoubleFactory2D F2 = DoubleFactory2D.dense;
	private Log log = LogFactory.getLog(this.getClass().getName());

	/**
	 * Quadratic objective with linear eq and ineq.
	 */
	public void testOptimize() throws Exception {
		log.debug("testOptimize");
		DoubleMatrix2D pMatrix = F2.make(new double[][] { 
				{ 1.68, 0.34, 0.38 },
				{ 0.34, 3.09, -1.59 }, 
				{ 0.38, -1.59, 1.54 } });
		DoubleMatrix1D qVector = F1.make(new double[] { 0.018, 0.025, 0.01 });

		// Objective function.
		double theta = 0.01522;
		DoubleMatrix2D P = pMatrix.assign(Mult.mult(theta));
		DoubleMatrix1D q = qVector.assign(Mult.mult(-1));
		PDQuadraticMultivariateRealFunction objectiveFunction = new PDQuadraticMultivariateRealFunction(P.toArray(), q.toArray(), 0);

	    ConvexMultivariateRealFunction[] inequalities = new ConvexMultivariateRealFunction[3];
		inequalities[0] = new LinearMultivariateRealFunction(new double[] {-1, 0, 0 }, 0);
		inequalities[1] = new LinearMultivariateRealFunction(new double[] { 0, -1, 0 }, 0);
		inequalities[2] = new LinearMultivariateRealFunction(new double[] {	0, 0, -1 }, 0);
	  
		OptimizationRequest or = new OptimizationRequest();
		or.setCheckKKTSolutionAccuracy(true);
		or.setF0(objectiveFunction);
		or.setInitialPoint(new double[] { 0.25, 0.25, 0.5 });
		// inequalities
		or.setFi(inequalities);
		// equalities
		or.setA(new double[][] { { 1, 1, 1 } });
		or.setB(new double[] { 1 });
		//tolerances
		or.setTolerance(1.E-11);
		or.setToleranceFeas(1.E-8);

		// optimization
		PrimalDualMethod opt = new PrimalDualMethod();
		opt.setOptimizationRequest(or);
		int returnCode = opt.optimize();
		if(returnCode == OptimizationResponse.FAILED){
			fail();
		}
		OptimizationResponse response = opt.getOptimizationResponse();
		double[] sol = response.getSolution();
		log.debug("sol   : " + ArrayUtils.toString(sol));
		log.debug("value : " + objectiveFunction.value(sol));
		assertEquals(0.04632311555988555, sol[0], 0.00000001);
		assertEquals(0.5086308460954377,  sol[1], 0.00000001);
		assertEquals(0.44504603834467693, sol[2], 0.00000001);
	}
	
	/**
	 * Quadratic objective with linear eq and ineq
	 * without initial point.
	 */
	public void testOptimize2() throws Exception {
		log.debug("testOptimize2");
		DoubleMatrix2D qq = F2.make(new double[][] { 
				{ 1.68, 0.34, 0.38 },
				{ 0.34, 3.09, -1.59 }, 
				{ 0.38, -1.59, 1.54 } });
		DoubleMatrix1D ll = F1.make(new double[] { 0.018, 0.025, 0.01 });

		// Objective function (Risk-Aversion).
		double theta = 0.01522;
		DoubleMatrix2D P = qq.assign(Mult.mult(theta));
		DoubleMatrix1D q = ll.assign(Mult.mult(-1));
		PDQuadraticMultivariateRealFunction objectiveFunction = new PDQuadraticMultivariateRealFunction(P.toArray(), q.toArray(), 0);

		ConvexMultivariateRealFunction[] inequalities = new ConvexMultivariateRealFunction[3];
		inequalities[0] = new LinearMultivariateRealFunction(new double[] {-1, 0, 0 }, 0);
		inequalities[1] = new LinearMultivariateRealFunction(new double[] { 0, -1, 0 }, 0);
		inequalities[2] = new LinearMultivariateRealFunction(new double[] {	0, 0, -1 }, 0);

		OptimizationRequest or = new OptimizationRequest();
		or.setCheckKKTSolutionAccuracy(true);
		or.setF0(objectiveFunction);
		// inequalities
		or.setFi(inequalities);
		// equalities
		or.setA(new double[][] { { 1, 1, 1 } });
		or.setB(new double[] { 1 });
		//tolerances
		or.setTolerance(1.E-12);

		// optimization
		PrimalDualMethod opt = new PrimalDualMethod();
		opt.setOptimizationRequest(or);
		int returnCode = opt.optimize();
		
		if(returnCode == OptimizationResponse.FAILED){
			fail();
		}
		
		OptimizationResponse response = opt.getOptimizationResponse();
		double[] sol = response.getSolution();
		log.debug("sol   : " + ArrayUtils.toString(sol));
		log.debug("value : " + objectiveFunction.value(sol));
		assertEquals(0.04632311555988555, sol[0], 0.0000001);
		assertEquals(0.5086308460954377,  sol[1], 0.0000001);
		assertEquals(0.44504603834467693, sol[2], 0.0000001);
	}

	/**
	 * Quadratic objective with linear eq and quadratic ineq.
	 */
	public void testOptimize3() throws Exception {
		log.debug("testOptimize3");
		DoubleMatrix2D qq = F2.make(new double[][] { 
				{ 1.68, 0.34, 0.38 },
				{ 0.34, 3.09, -1.59 }, 
				{ 0.38, -1.59, 1.54 } });
		DoubleMatrix1D ll = F1.make(new double[] { 0.018, 0.025, 0.01 });

		// Objective function (Risk-Aversion).
		double theta = 0.01522;
		DoubleMatrix2D PMatrix = qq.assign(Mult.mult(theta));
		DoubleMatrix1D Qvector = ll.assign(Mult.mult(-1));
		PDQuadraticMultivariateRealFunction objectiveFunction = new PDQuadraticMultivariateRealFunction(PMatrix.toArray(), Qvector.toArray(), 0);

		ConvexMultivariateRealFunction[] inequalities = new ConvexMultivariateRealFunction[4];
		inequalities[0] = new LinearMultivariateRealFunction(new double[] { -1, 0, 0 }, 0);
		inequalities[1] = new LinearMultivariateRealFunction(new double[] { 0, -1, 0 }, 0);
		inequalities[2] = new LinearMultivariateRealFunction(new double[] { 0, 0, -1 }, 0);
		inequalities[3] = FunctionsUtils.createCircle(3, 5);//not linear

		OptimizationRequest or = new OptimizationRequest();
		or.setCheckKKTSolutionAccuracy(true);
		or.setF0(objectiveFunction);
		or.setInitialPoint(new double[] { 0.2, 0.2, 0.6 });
		or.setInitialLagrangian(new double[]{0.5,0.5,0.5,0.5});
		// Inquality constraints
		or.setFi(inequalities);
		// Equality constraints
		or.setA(new double[][] { { 1, 1, 1 } });
		or.setB(new double[] { 1 });
		//tolerances
		or.setTolerance(1.E-10);

		// optimization
		PrimalDualMethod opt = new PrimalDualMethod();
		opt.setOptimizationRequest(or);
        int returnCode = opt.optimize();
		
		if(returnCode == OptimizationResponse.FAILED){
			fail();
		}
		
		OptimizationResponse response = opt.getOptimizationResponse();
		double[] sol = response.getSolution();
		log.debug("sol    : " + ArrayUtils.toString(sol));
		log.debug("value      : " + objectiveFunction.value(sol));
		assertEquals(0.04632311555988555, sol[0], 0.0000001);
		assertEquals(0.5086308460954377,  sol[1], 0.0000001);
		assertEquals(0.44504603834467693, sol[2], 0.0000001);
	}

	/**
	 * Linear objective with quadratic ineq.
	 */
	public void testOptimize4() throws Exception {
		log.debug("testOptimize4");

		// Objective function (linear)
		LinearMultivariateRealFunction objectiveFunction = new LinearMultivariateRealFunction(new double[] { 1, 1 }, 0);

		// Inquality constraints
		ConvexMultivariateRealFunction[] inequalities = new ConvexMultivariateRealFunction[1];
		inequalities[0] = FunctionsUtils.createCircle(2, 1);

		OptimizationRequest or = new OptimizationRequest();
		or.setCheckKKTSolutionAccuracy(true);
		or.setToleranceKKT(1.E-4);
		or.setF0(objectiveFunction);
		or.setInitialPoint(new double[] { 0, 0 });
		or.setFi(inequalities);

		// optimization
		PrimalDualMethod opt = new PrimalDualMethod();
		opt.setOptimizationRequest(or);
		int returnCode = opt.optimize();
		
		if(returnCode == OptimizationResponse.FAILED){
			fail();
		}
		
		OptimizationResponse response = opt.getOptimizationResponse();
		double[] sol = response.getSolution();
		double value = objectiveFunction.value(sol);
		log.debug("sol    : " + ArrayUtils.toString(sol));
		log.debug("value      : " + value);
		assertEquals(-Math.sqrt(2),   value,          0.00000001);
        assertEquals(-Math.sqrt(2)/2, sol[0], 0.00000001);
        assertEquals(-Math.sqrt(2)/2, sol[1], 0.00000001);
	}
	
	/**
	 * Linear objective with linear eq and ineq. 
	 */
	public void testOptimize5() throws Exception {
		log.debug("testOptimize5");
		// START SNIPPET: PrimalDualMethod-1

		// Objective function (linear)
		LinearMultivariateRealFunction objectiveFunction = new LinearMultivariateRealFunction(new double[] { 2, 1 }, 0);

		// Inquality constraints
		ConvexMultivariateRealFunction[] inequalities = new ConvexMultivariateRealFunction[2];
		inequalities[0] = new LinearMultivariateRealFunction(new double[] { -1,  0 }, 0);
		inequalities[1] = new LinearMultivariateRealFunction(new double[] {  0, -1 }, 0);

		OptimizationRequest or = new OptimizationRequest();
		or.setF0(objectiveFunction);
		or.setInitialPoint(new double[] { 0.9, 0.1 });
		or.setFi(inequalities);
		// Equality constraints
		or.setA(new double[][] { { 1, 1} });
		or.setB(new double[] { 1 });
		or.setTolerance(1.E-9);
		
		// optimization
		PrimalDualMethod opt = new PrimalDualMethod();
		opt.setOptimizationRequest(or);
		int returnCode = opt.optimize();
		
		// END SNIPPET: PrimalDualMethod-1
		
		if(returnCode == OptimizationResponse.FAILED){
			fail();
		}
		
		OptimizationResponse response = opt.getOptimizationResponse();
		double[] sol = response.getSolution();
		double value = objectiveFunction.value(sol);
		log.debug("sol   : " + ArrayUtils.toString(sol));
		log.debug("value : " + value);
		assertEquals(0.,   sol[0], 0.00000001);
		assertEquals(1.,   sol[1], 0.00000001);
		assertEquals(1.,   value,      0.00000001);
	}
	
	/**
	 * Linear objective with quadratic ineq 
	 * and without initial point.
	 * NOTE: changing c to 1 or 10 we get a KKT solution failed error:
	 * this is because rDual (that is proportional to the gradient of F0, that
	 * is proportional to c) does not decrease well during the iterations.
	 * @TODO: does this suggest to rescaling the objective function if it is linear?
	 */
	public void testOptimize6() throws Exception {
		log.debug("testOptimize6");
		// START SNIPPET: PrimalDualMethod-2

		// Objective function (linear)
		double c = 0.1;
		LinearMultivariateRealFunction objectiveFunction = new LinearMultivariateRealFunction(new double[] { c, c }, 0);

		// Inequality constraints
		ConvexMultivariateRealFunction[] inequalities = new ConvexMultivariateRealFunction[1];
		inequalities[0] = FunctionsUtils.createCircle(2, 1);

		OptimizationRequest or = new OptimizationRequest();
		or.setF0(objectiveFunction);
		or.setInitialLagrangian(new double[]{10});
		or.setFi(inequalities);
		or.setInteriorPointMethod(JOptimizer.PRIMAL_DUAL_METHOD);//this is also the default
		or.setToleranceFeas(5.E-6);
		//or.setCheckKKTSolutionAccuracy(true);
		//or.setCheckProgressConditions(true);

		// optimization
		JOptimizer opt = new JOptimizer();
		opt.setOptimizationRequest(or);
		int  returnCode = opt.optimize();
		
		// END SNIPPET: PrimalDualMethod-2
		
		if(returnCode==OptimizationResponse.FAILED){
			fail();
		}
		
		OptimizationResponse response = opt.getOptimizationResponse();
		double[] sol = response.getSolution();
		double value = objectiveFunction.value(sol);
		log.debug("sol   : " + ArrayUtils.toString(sol));
		log.debug("value : " + value);
		assertEquals(-Math.sqrt(2) * c,   value,  or.getTolerance());
		assertEquals(-Math.sqrt(2)/2, sol[0], 0.00001);
		assertEquals(-Math.sqrt(2)/2, sol[1], 0.00001);
	}

	/**
	 * Exponential objective with quadratic ineq. 
	 * f0 = exp[z^2], z=(x-1, y-2) 
	 * f1 = x^2+y^2<=3^2
	 */
	public void testOptimize7() throws Exception {
		log.debug("testOptimize7");
		StrictlyConvexMultivariateRealFunction objectiveFunction = new StrictlyConvexMultivariateRealFunction() {

			public double value(double[] X) {
				DoubleMatrix1D Z = F1.make(new double[] { X[0] - 1, X[1] - 2, });
				return Math.exp(Z.zDotProduct(Z));
			}

			public double[] gradient(double[] X) {
				DoubleMatrix1D Z = F1.make(new double[] { X[0] - 1, X[1] - 2, });
				return Z.assign(Mult.mult(2 * Math.exp(Z.zDotProduct(Z)))).toArray();
			}

			public double[][] hessian(double[] X) {
				DoubleMatrix1D Z = F1.make(new double[] { X[0] - 1, X[1] - 2, });
				double d = Math.exp(Z.zDotProduct(Z));
				DoubleMatrix2D ID = F2.identity(2);
				DoubleMatrix2D ret = ALG.multOuter(Z, Z, null).assign(ID, Functions.plus).assign(Mult.mult(2 * d));
				return ret.toArray();
			}

			public int getDim() {
				return 2;
			}
		};

		// Inquality constraints
		ConvexMultivariateRealFunction[] inequalities = new ConvexMultivariateRealFunction[1];
		inequalities[0] = FunctionsUtils.createCircle(2, 3);

		OptimizationRequest or = new OptimizationRequest();
		or.setF0(objectiveFunction);
		or.setInitialPoint(new double[] { 0.2, 0.2 });
		or.setFi(inequalities);

		// optimization
		PrimalDualMethod opt = new PrimalDualMethod();
		opt.setOptimizationRequest(or);
		int returnCode = opt.optimize();
		
		if(returnCode == OptimizationResponse.FAILED){
			fail();
		}
		
		OptimizationResponse response = opt.getOptimizationResponse();
		double[] sol = response.getSolution();
		double value = objectiveFunction.value(sol);
		log.debug("sol   : " + ArrayUtils.toString(sol));
		log.debug("value : " + value);
		assertEquals(1., sol[0], or.getTolerance());
		assertEquals(2., sol[1], or.getTolerance());
	}
	
	/**
	 * Min(s) s.t.
	 * x^2-y-s<0
	 * x+y=4
	 */
	public void testOptimize8() throws Exception {
		log.debug("testOptimize8");
		
		// Objective function (linear)
		LinearMultivariateRealFunction objectiveFunction = new LinearMultivariateRealFunction(new double[] { 0, 0, 1 }, 0);

		// Equalities:
		double[][] equalityAMatrix = new double[][] { { 1.0, 1.0, 0 } };
		double[] equalityBVector = new double[] { 4.0 };

		// inequalities
		ConvexMultivariateRealFunction[] inequalities = new ConvexMultivariateRealFunction[1];
		double[][] PMatrix = new double[][]{{2.0,0,0},{0,0,0},{0,0,0}}; 
		double[] qVector = new double[]{0,-1,-1};
		inequalities[0] = new PSDQuadraticMultivariateRealFunction(PMatrix, qVector, 0);

		OptimizationRequest or = new OptimizationRequest();
		or.setF0(objectiveFunction);
		or.setInitialPoint(new double[]{2,2,2000});
		or.setA(equalityAMatrix);
		or.setB(equalityBVector);
		or.setFi(inequalities);
		//or.setTolerance(1.E-7);//ok
		//or.setToleranceFeas(1.E-7);//ok
		or.setToleranceFeas(1E-6);//ko
		or.setTolerance(2E-6);//ko
		or.setInteriorPointMethod(JOptimizer.PRIMAL_DUAL_METHOD);//this is also the default

		// optimization
		JOptimizer opt = new JOptimizer();
		opt.setOptimizationRequest(or);
		int  returnCode = opt.optimize();
		
		if(returnCode==OptimizationResponse.FAILED){
			fail();
		}
		
		OptimizationResponse response = opt.getOptimizationResponse();
		double[] sol = response.getSolution();
		double value = objectiveFunction.value(sol);
		log.debug("sol   : " + ArrayUtils.toString(sol));
		log.debug("value : " + value);
		assertEquals(-0.5,   sol[0], 0.01);
		assertEquals( 4.5,   sol[1], 0.01);
		assertEquals(-4.25,  sol[2], 0.01);
	}
	
	/**
	 * Quadratic objective with linear eq and ineq.
	 */
	public void testOptimize10D() throws Exception {
		log.debug("testOptimize10D");

		int dim = 10;
		
		// Objective function
		DoubleMatrix2D P = Utils.randomValuesPositiveMatrix(dim, dim, -0.5, 0.5, 7654321L);
		DoubleMatrix1D q = Utils.randomValuesMatrix(1, dim, -0.5, 0.5, 7654321L).viewRow(0);

		PDQuadraticMultivariateRealFunction objectiveFunction = new PDQuadraticMultivariateRealFunction(P.toArray(), q.toArray(), 0);

		// equalities
		double[][] AEMatrix = new double[1][dim];
		Arrays.fill(AEMatrix[0], 1.);
		double[] BEVector = new double[] { 1 };

		// inequalities
		double[][] AIMatrix = new double[dim][dim];
		for (int i = 0; i < dim; i++) {
			AIMatrix[i][i] = -1;
		}
		ConvexMultivariateRealFunction[] inequalities = new ConvexMultivariateRealFunction[dim];
		for (int i = 0; i < dim; i++) {
			inequalities[i] = new LinearMultivariateRealFunction(AIMatrix[i], 0);
		}

		OptimizationRequest or = new OptimizationRequest();
		or.setF0(objectiveFunction);
		double[] ip = new double[dim];
		Arrays.fill(ip, 1. / dim);
		or.setInitialPoint(ip);
		or.setA(AEMatrix);
		or.setB(BEVector);
		or.setFi(inequalities);

		// optimization
		PrimalDualMethod opt = new PrimalDualMethod();
		opt.setOptimizationRequest(or);
		int returnCode = opt.optimize();
		
		if (returnCode == OptimizationResponse.FAILED) {
			fail();
		}
		
		OptimizationResponse response = opt.getOptimizationResponse();
		double[] sol = response.getSolution();
		double value = objectiveFunction.value(sol);
		log.debug("sol   : " + ArrayUtils.toString(sol));
		log.debug("value : " + value);
	}
	
}
