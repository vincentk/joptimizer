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

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;

import com.joptimizer.functions.ConvexMultivariateRealFunction;
import com.joptimizer.functions.FunctionsUtils;
import com.joptimizer.functions.LinearMultivariateRealFunction;
import com.joptimizer.util.ColtUtils;
import com.joptimizer.util.Utils;

/**
 * Basic Phase I Method (implemented as a Primal-Dual Method).
 * 
 * @see "S.Boyd and L.Vandenberghe, Convex Optimization, p. 579"
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class BasicPhaseIPDM {
	
	private PrimalDualMethod originalProblem;
	private int originalDim =-1;
	private int dim =-1;
	private Algebra ALG = Algebra.DEFAULT;
	private DoubleFactory1D F1 = DoubleFactory1D.dense;
	private DoubleFactory2D F2 = DoubleFactory2D.dense;
	private Log log = LogFactory.getLog(this.getClass().getName());

	public BasicPhaseIPDM(PrimalDualMethod originalProblem) {
		this.originalProblem = originalProblem;
		originalDim = originalProblem.getDim();
		this.dim = originalProblem.getDim()+1;//variable Y=(X, s)
	}
	
	public DoubleMatrix1D findFeasibleInitialPoint() throws Exception{
		log.debug("findFeasibleInitialPoint");
		
		OptimizationRequest or = new OptimizationRequest();
		
		//objective function: s
		DoubleMatrix1D C = F1.make(dim);
		C.set(dim-1, 1.);
		LinearMultivariateRealFunction objectiveFunction = new LinearMultivariateRealFunction(C.toArray(), 0);
		or.setF0(objectiveFunction);
		or.setToleranceFeas(originalProblem.getToleranceFeas());
		or.setTolerance(originalProblem.getTolerance());
		
		//@TODO: remove this
		//or.setToleranceKKT(originalProblem.getToleranceKKT());
		//or.setCheckKKTSolutionAccuracy(originalProblem.isCheckKKTSolutionAccuracy());
		
	  // Inequality constraints: fi(X)-s
		ConvexMultivariateRealFunction[] inequalities = new ConvexMultivariateRealFunction[originalProblem.getMieq()];
		for(int i=0; i<inequalities.length; i++){
			
			final ConvexMultivariateRealFunction originalFi = originalProblem.getFi()[i];
			
			ConvexMultivariateRealFunction fi = new ConvexMultivariateRealFunction() {
				
				public double value(double[] Y) {
					DoubleMatrix1D y = DoubleFactory1D.dense.make(Y);
					DoubleMatrix1D X = y.viewPart(0, originalDim);
					return originalFi.value(X.toArray()) - y.get(dim-1);
				}
				
				public double[] gradient(double[] Y) {
					DoubleMatrix1D y = DoubleFactory1D.dense.make(Y);
					DoubleMatrix1D X = y.viewPart(0, originalDim);
					DoubleMatrix1D origGrad = F1.make(originalFi.gradient(X.toArray()));
					DoubleMatrix1D ret = F1.make(1, -1);
					ret = F1.append(origGrad, ret);
					return ret.toArray();
				}
				
				public double[][] hessian(double[] Y) {
					DoubleMatrix1D y = DoubleFactory1D.dense.make(Y);
					DoubleMatrix1D X = y.viewPart(0, originalDim);
					double[][] originalFiHess = originalFi.hessian(X.toArray());
					if(originalFiHess==FunctionsUtils.ZEROES_2D_ARRAY_PLACEHOLDER){
						return FunctionsUtils.ZEROES_2D_ARRAY_PLACEHOLDER;
					}else{
						DoubleMatrix2D origHess = (originalFiHess!=FunctionsUtils.ZEROES_2D_ARRAY_PLACEHOLDER)? F2.make(originalFi.hessian(X.toArray())) : F2.make(X.size(), X.size());
						DoubleMatrix2D[][] parts = new DoubleMatrix2D[][]{{origHess, null},{null,F2.make(1, 1)}};
						return F2.compose(parts).toArray();
					}
				}
				
				public int getDim() {
					return dim;
				}
			};
			inequalities[i] = fi;
		}
		or.setFi(inequalities);
		
	  // Equality constraints: add a final zeroes column
		DoubleMatrix2D AEorig = originalProblem.getA();
		DoubleMatrix1D BEorig = originalProblem.getB();
		if(AEorig!=null){
			DoubleMatrix2D zeroCols = F2.make(AEorig.rows(), 1); 
			DoubleMatrix2D[][] parts = new DoubleMatrix2D[][]{{AEorig, zeroCols}};
			DoubleMatrix2D AE = F2.compose(parts);
			DoubleMatrix1D BE = BEorig;
			or.setA(AE);
			or.setB(BE);
		}
		
		//initial point
		DoubleMatrix1D X0 = originalProblem.getNotFeasibleInitialPoint();
		if(X0==null){
			if(AEorig!=null){
				X0 = findOneRoot(AEorig, BEorig);
			}else{
				X0 = F1.make(originalProblem.getDim(), 1./originalProblem.getDim());
			}
		}
		
		//check primal norm
		if (AEorig!=null) {
			//DoubleMatrix1D originalRPriX0 = AEorig.zMult(X0, BEorig.copy(), 1., -1., false);
			DoubleMatrix1D originalRPriX0 = ColtUtils.zMult(AEorig, X0, BEorig, -1);
			double norm = Math.sqrt(ALG.norm2(originalRPriX0));
			log.debug("norm: " + norm);
			if(norm > originalProblem.getToleranceFeas()){
				throw new Exception("The initial point for Basic Phase I Method must be equalities-feasible");
			}
		}
		
		DoubleMatrix1D originalFiX0 = originalProblem.getFi(X0);
		
		//lucky strike?
		int maxIneqIndex = Utils.getMaxIndex(originalFiX0);
		if(originalFiX0.get(maxIneqIndex) + originalProblem.getTolerance()<0){
			//the given notFeasible starting point is in fact already feasible
			return X0;
		}
		
		//DoubleMatrix1D initialPoint = F1.make(1, -Double.MAX_VALUE);
		DoubleMatrix1D initialPoint = F1.make(1, Math.sqrt(originalProblem.getToleranceFeas()));
		initialPoint = F1.append(X0, initialPoint);
		for(int i=0; i<originalFiX0.size(); i++){
			//initialPoint.set(dim-1, Math.max(initialPoint.get(dim-1), originalFiX0.get(i)+Math.sqrt(originalProblem.getToleranceFeas())));
			initialPoint.set(dim-1, Math.max(initialPoint.get(dim-1), originalFiX0.get(i)*Math.pow(originalProblem.getToleranceFeas(),-0.5)));
		}
		or.setInitialPoint(initialPoint.toArray());
		
	  //optimization
		PrimalDualMethod opt = new PhaseIPrimalDualMethod();
		opt.setOptimizationRequest(or);
		if(opt.optimize() == OptimizationResponse.FAILED){
			throw new Exception("Failed to find an initial feasible point");
		}
		OptimizationResponse response = opt.getOptimizationResponse();
		DoubleMatrix1D sol = F1.make(response.getSolution());
		DoubleMatrix1D ret = sol.viewPart(0, originalDim);
		DoubleMatrix1D ineq = originalProblem.getFi(ret);
		maxIneqIndex = Utils.getMaxIndex(ineq);
		if(log.isDebugEnabled()){
			log.debug("ineq        : "+ArrayUtils.toString(ineq.toArray()));
			log.debug("max ineq pos: "+maxIneqIndex);
			log.debug("max ineq val: "+ineq.get(maxIneqIndex));
		}
		//if(sol[dim-1]>0){
		if(ineq.get(maxIneqIndex)>=0){	
			throw new Exception("Infeasible problem");
		}

        return ret;
	}
	
	private class PhaseIPrimalDualMethod extends PrimalDualMethod{
		@Override
		protected boolean checkCustomExitConditions(DoubleMatrix1D Y){
			DoubleMatrix1D X = Y.viewPart(0, getDim()-1);
			DoubleMatrix1D ineqX = originalProblem.getFi(X);
			int ineqMaxIndex = Utils.getMaxIndex(ineqX);
			
			boolean isInternal = (ineqX.get(ineqMaxIndex) + getTolerance() <0) || Y.get(Y.size()-1)<0;
			log.info("isInternal  : " + isInternal);
			if(!isInternal){
				return false;
			}
			
			DoubleMatrix1D originalRPriX = F1.make(0);
			if(getA()!=null){
				//originalRPriX = originalProblem.getA().zMult(X, originalProblem.getB().copy(), 1., -1., false);
				originalRPriX = ColtUtils.zMult(originalProblem.getA(), X, originalProblem.getB(), -1);
			}
			boolean isPrimalFeas = Math.sqrt(ALG.norm2(originalRPriX)) < originalProblem.getToleranceFeas();
			log.info("isPrimalFeas: " + isPrimalFeas);
			
			log.info("checkCustomExitConditions: " + (isInternal && isPrimalFeas));
			return isInternal && isPrimalFeas;
		}
	}
	
	/**
	 * Just looking for one out of all the possible solutions.
	 * @see "Convex Optimization, C.5 p. 681".
	 */
	private DoubleMatrix1D findOneRoot(DoubleMatrix2D A, DoubleMatrix1D b) throws Exception{
		return originalProblem.findEqFeasiblePoint(A, b);
	}
}
