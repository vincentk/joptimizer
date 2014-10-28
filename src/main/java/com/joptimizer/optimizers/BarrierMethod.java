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

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.jet.math.Functions;
import cern.jet.math.Mult;

import com.joptimizer.functions.BarrierFunction;
import com.joptimizer.functions.ConvexMultivariateRealFunction;
import com.joptimizer.functions.FunctionsUtils;

/**
 * Barrier method.
 * 
 * @see "S.Boyd and L.Vandenberghe, Convex Optimization, p. 568"
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class BarrierMethod extends OptimizationRequestHandler {

	private BarrierFunction barrierFunction = null;
	private Log log = LogFactory.getLog(this.getClass().getName());
	
	public BarrierMethod(BarrierFunction barrierFunction) {
		this.barrierFunction = barrierFunction;
	}
	
	@Override
	public int optimize() throws Exception {
		log.info("optimize");
		long tStart = System.currentTimeMillis();
		OptimizationResponse response = new OptimizationResponse();

		// @TODO: check assumptions!!!
//		if(getA()!=null){
//			if(ALG.rank(getA())>=getA().rows()){
//				throw new IllegalArgumentException("A-rank must be less than A-rows");
//			}
//		}
		
		DoubleMatrix1D X0 = getInitialPoint();
		if(X0==null){
			DoubleMatrix1D X0NF = getNotFeasibleInitialPoint();
			if(X0NF!=null){
				double rPriX0NFNorm = Math.sqrt(ALG.norm2(rPri(X0NF)));
				if(rPriX0NFNorm <= getToleranceFeas() && !Double.isNaN(this.barrierFunction.value(X0NF.toArray()))){
					log.debug("the provided initial point is already feasible");
					X0 = X0NF;
				}
//				DoubleMatrix1D fiX0NF = getFi(X0NF);
//				int maxIndex = Utils.getMaxIndex(fiX0NF);
//				double maxValue = fiX0NF.get(maxIndex);
//				if (log.isDebugEnabled()) {
//					log.debug("X0NF  :  " + ArrayUtils.toString(X0NF.toArray()));
//					log.debug("fiX0NF:  " + ArrayUtils.toString(fiX0NF.toArray()));
//				}
//				if(maxValue<0){
//					//the provided not-feasible starting point is already feasible
//					log.debug("the provided initial point is already feasible");
//					X0 = X0NF;
//				}
			}
			if(X0 == null){
				BasicPhaseIBM bf1 = new BasicPhaseIBM(this);
				X0 = F1.make(bf1.findFeasibleInitialPoint());
			}
		}
		
		//check X0 feasibility
		double rPriX0Norm = Math.sqrt(ALG.norm2(rPri(X0)));
		if(Double.isNaN(this.barrierFunction.value(X0.toArray())) || rPriX0Norm > getToleranceFeas()){
			throw new Exception("initial point must be strictly feasible");
		}
//		DoubleMatrix1D fiX0 = getFi(X0);
//		if(fiX0!=null){
//			int maxIndex = Utils.getMaxIndex(fiX0);
//			double maxValue = fiX0.get(maxIndex);
//			if(maxValue >= 0){
//				log.debug("ineqX0      : " + ArrayUtils.toString(fiX0.toArray()));
//				log.debug("max ineq index: " + maxIndex);
//				log.debug("max ineq value: " + maxValue);
//				throw new Exception("initial point must be strictly feasible");
//			}
//		}
		
		DoubleMatrix1D V0 = (getA()!=null)? F1.make(getA().rows()) : F1.make(0);
		
		if(log.isDebugEnabled()){
			log.debug("X0: " + ArrayUtils.toString(X0.toArray()));
			log.debug("V0: " + ArrayUtils.toString(V0.toArray()));
		}

		DoubleMatrix1D X = X0;
		final int dim = X.size();
		double t = 1d;
		int outerIteration = 0;
		while (true) {
			outerIteration++;
			if(log.isDebugEnabled()){
				log.debug("outerIteration: " + outerIteration);
				log.debug("X=" + ArrayUtils.toString(X.toArray()));
				log.debug("f(X)=" + getF0(X));
			}
			
			//Stopping criterion: quit if gap < tolerance.
			double gap = this.barrierFunction.getDualityGap(t);
			log.debug("gap: "+gap);
			if(gap <= getTolerance()){
				break;
			}
			
			// custom exit condition
			if(checkCustomExitConditions(X)){
				response.setReturnCode(OptimizationResponse.SUCCESS);
				break;
			}
			
			//Centering step: compute x*(t) by minimizing tf0 + phi (the barrier function), subject to Ax = b, starting at x.
			final double tIter = t;
			log.debug("t: "+tIter);
			ConvexMultivariateRealFunction newObjectiveFunction = new ConvexMultivariateRealFunction() {
				
				@Override
				public double value(double[] X) {
					DoubleMatrix1D x = F1.make(X);
					double phi = barrierFunction.value(X);
					return tIter * getF0(x) + phi;
				}
				
				@Override
				public double[] gradient(double[] X) {
					DoubleMatrix1D x = F1.make(X);
					DoubleMatrix1D phiGrad = F1.make(barrierFunction.gradient(X));
					return getGradF0(x).assign(Mult.mult(tIter)).assign(phiGrad, Functions.plus).toArray();
				}
				
				@Override
				public double[][] hessian(double[] X) {
					DoubleMatrix1D x = F1.make(X);
					DoubleMatrix2D hessF0X = getHessF0(x);
					double[][] hessX = barrierFunction.hessian(X);
					if(hessX == FunctionsUtils.ZEROES_2D_ARRAY_PLACEHOLDER){
						return hessF0X.assign(Mult.mult(tIter)).toArray();
					}else{
						DoubleMatrix2D phiHess = F2.make(hessX);
						return hessF0X.assign(Mult.mult(tIter)).assign(phiHess, Functions.plus).toArray();
					}
				}

				@Override
				public int getDim() {
					return dim;
				}
			};
			
			//NB: cannot use the same request object for the inner step
			OptimizationRequest or = new OptimizationRequest();
			or.setA( (getA()!=null)? getA().toArray() : null );
			or.setAlpha(getAlpha());
			or.setB((getB()!=null)? getB().toArray() : null);
			or.setBeta(getBeta());
			or.setCheckKKTSolutionAccuracy(isCheckKKTSolutionAccuracy());
			or.setCheckProgressConditions(isCheckProgressConditions());
			or.setF0(newObjectiveFunction);
			or.setInitialPoint(X.toArray());
			or.setMaxIteration(getMaxIteration());
			or.setMu(getMu());
			or.setTolerance(getToleranceInnerStep());
			or.setToleranceKKT(getToleranceKKT());
			
			BarrierNewtonLEConstrainedFSP opt = new BarrierNewtonLEConstrainedFSP(true, this);
			opt.setOptimizationRequest(or);
			if(opt.optimize() == OptimizationResponse.FAILED){
				response.setReturnCode(OptimizationResponse.FAILED);
				break;
			}
			OptimizationResponse newtonResponse = opt.getOptimizationResponse();
			
			//Update. x := x*(t).
			X = F1.make(newtonResponse.getSolution());
			
//			//Stopping criterion: quit if gap < tolerance.
//			double gap = this.barrierFunction.getDualityGap(t);
//			log.debug("gap: "+gap);
//			if(gap <= getTolerance()){
//				break;
//			}
//			
//			// custom exit condition
//			if(checkCustomExitConditions(X)){
//				response.setReturnCode(OptimizationResponse.SUCCESS);
//				break;
//			}
			
			//Increase t: t := mu*t.
			t = getMu() * t;
			
			// iteration limit condition
			if (outerIteration == getMaxIteration()) {
				response.setReturnCode(OptimizationResponse.FAILED);
				log.error("Max iterations limit reached");
				throw new Exception("Max iterations limit reached");
			}
		}

		long tStop = System.currentTimeMillis();
		log.debug("time: " + (tStop - tStart));
		response.setSolution(X.toArray());
		setOptimizationResponse(response);
		return response.getReturnCode();
	}
	
	/**
	 * Use the barrier function instead.
	 */
	@Override
	protected DoubleMatrix1D getFi(DoubleMatrix1D X){
		throw new UnsupportedOperationException();
	}
	
	/**
	 * Use the barrier function instead.
	 */
	@Override
	protected DoubleMatrix2D getGradFi(DoubleMatrix1D X){
		throw new UnsupportedOperationException();
	}
	
	/**
	 * Use the barrier function instead.
	 */
	@Override
	protected DoubleMatrix2D[] getHessFi(DoubleMatrix1D X){
		throw new UnsupportedOperationException();
	}
	
	protected BarrierFunction getBarrierFunction(){
		return this.barrierFunction;
	}

	private class BarrierNewtonLEConstrainedFSP extends NewtonLEConstrainedFSP{
		BarrierMethod father = null;
		
		public BarrierNewtonLEConstrainedFSP(boolean activateChain, BarrierMethod father){
			super(activateChain);
			this.father = father;
		}
		
		@Override
		protected boolean checkCustomExitConditions(DoubleMatrix1D Y){
			boolean ret = father.checkCustomExitConditions(Y);
			log.debug("checkCustomExitConditions: " + ret);
			return ret;
		}
	}
	
//	private DoubleMatrix1D rPri(DoubleMatrix1D X) {
//		if(getA()==null){
//			return F1.make(0);
//		}
//		return getA().zMult(X, getB().copy(), 1., -1., false);
//	}
		
}
