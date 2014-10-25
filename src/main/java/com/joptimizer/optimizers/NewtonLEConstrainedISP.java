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

import com.joptimizer.functions.BarrierFunction;
import com.joptimizer.functions.LogarithmicBarrier;
import com.joptimizer.solvers.BasicKKTSolver;
import com.joptimizer.solvers.KKTSolver;
import com.joptimizer.util.ColtUtils;

/**
 * Linear equality constrained newton optimizer, with infeasible starting point.
 * 
 * @see "S.Boyd and L.Vandenberghe, Convex Optimization, p. 521"
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 * @TODO: switch linear search to a the feasible one, once primal feasibility
 *        has reached (see p. 535).
 */
public class NewtonLEConstrainedISP extends OptimizationRequestHandler {

	private KKTSolver kktSolver;
	private Log log = LogFactory.getLog(this.getClass().getName());
	
	public NewtonLEConstrainedISP(boolean activateChain){
		if(activateChain){
			this.successor = new PrimalDualMethod();
		}
	}
	
	public NewtonLEConstrainedISP(){
		this(false);
	}

	@Override
	public int optimize() throws Exception {
		log.debug("optimize");
		OptimizationResponse response = new OptimizationResponse();
		
	  //checking responsibility
		if (getFi() != null) {
			// forward to the chain
			return forwardOptimizationRequest();
		}
		
		long tStart = System.currentTimeMillis();
		DoubleMatrix1D X0 = getInitialPoint();
		if (X0 == null) {
			if(getA()!=null){
				X0 = findEqFeasiblePoint(getA(), getB());
				log.debug("Switch to the linear equality feasible starting point Newton algorithm ");
				NewtonLEConstrainedFSP opt = new NewtonLEConstrainedFSP();
				OptimizationRequest req = getOptimizationRequest();
				req.setInitialPoint(X0.toArray());
				opt.setOptimizationRequest(req);
				int retcode = opt.optimize(); 
				OptimizationResponse resp = opt.getOptimizationResponse();
				setOptimizationResponse(resp);
				return retcode;
			}else{
				X0 = F1.make(getDim());
			}
		}
		DoubleMatrix1D V0 = (getA()!=null)? F1.make(getA().rows()) : F1.make(0);
		if(log.isDebugEnabled()){
			log.debug("X0:  " + ArrayUtils.toString(X0.toArray()));
		}

		DoubleMatrix1D X = X0;
		DoubleMatrix1D V = V0;
		double F0X;
		DoubleMatrix1D gradX = null;
		DoubleMatrix2D hessX = null;
		DoubleMatrix1D rDualXV = null;
		DoubleMatrix1D rPriX = null;
		double previousF0X = Double.NaN;
		double previousRPriXNorm = Double.NaN;
		double previousRXVNorm = Double.NaN;
		int iteration = 0;
		while (true) {
			iteration++;
			F0X = getF0(X);
			if(log.isDebugEnabled()){
				log.debug("iteration " + iteration);
				log.debug("X=" + ArrayUtils.toString(X.toArray()));
				log.debug("V=" + ArrayUtils.toString(V.toArray()));
				log.debug("f(X)=" + F0X);
			}
			
//			if(!Double.isNaN(previousF0X)){
//				if (previousF0X < F0X) {
//					throw new Exception("critical minimization problem");
//				}
//			}
//			previousF0X = F0X;
			
			// custom exit condition
			if(checkCustomExitConditions(X)){
				response.setReturnCode(OptimizationResponse.SUCCESS);
				break;
			}
			
			gradX = getGradF0(X);
			hessX = getHessF0(X);
			rDualXV = rDual(X, V, gradX);
			rPriX = rPri(X);
			
			// exit condition
			double rPriXNorm = Math.sqrt(ALG.norm2(rPriX)); 
			double rDualXVNorm = Math.sqrt(ALG.norm2(rDualXV));
			log.debug("rPriXNorm : "+rPriXNorm);
			log.debug("rDualXVNorm: "+rDualXVNorm);
			double rXVNorm = Math.sqrt(Math.pow(rPriXNorm, 2)+ Math.pow(rDualXVNorm, 2));
			if (rPriXNorm <= getTolerance() && rXVNorm <= getTolerance()) {
				response.setReturnCode(OptimizationResponse.SUCCESS);
				break;
			}

			// Newton step and decrement
			if(this.kktSolver==null){
				//this.kktSolver = new BasicKKTSolver();
				this.kktSolver = new BasicKKTSolver();
			}
			if(isCheckKKTSolutionAccuracy()){
				kktSolver.setCheckKKTSolutionAccuracy(isCheckKKTSolutionAccuracy());
				kktSolver.setToleranceKKT(getToleranceKKT());
			}
			kktSolver.setHMatrix(hessX);
			kktSolver.setGVector(rDualXV);
			if(getA()!=null){
				kktSolver.setAMatrix(getA());
				kktSolver.setHVector(rPriX);
			}
			DoubleMatrix1D[] sol = kktSolver.solve();
			DoubleMatrix1D stepX = sol[0];
			DoubleMatrix1D stepV = (sol[1]!=null)? sol[1] : F1.make(0);
			if(log.isDebugEnabled()){
				log.debug("stepX: " + ArrayUtils.toString(stepX.toArray()));
				log.debug("stepV: " + ArrayUtils.toString(stepV.toArray()));
			}

//			// exit condition
//			double rPriXNorm = Math.sqrt(ALG.norm2(rPriX)); 
//			double rDualXVNorm = Math.sqrt(ALG.norm2(rDualXV));
//			log.debug("rPriXNorm : "+rPriXNorm);
//			log.debug("rDualXVNorm: "+rDualXVNorm);
//			double rXVNorm = Math.sqrt(Math.pow(rPriXNorm, 2)+ Math.pow(rDualXVNorm, 2));
//			if (rPriXNorm <= getTolerance() && rXVNorm <= getTolerance()) {
//				response.setReturnCode(OptimizationResponse.SUCCESS);
//				break;
//			}
			
			// iteration limit condition
			if (iteration == getMaxIteration()) {
				response.setReturnCode(OptimizationResponse.FAILED);
				log.error("Max iterations limit reached");
				throw new Exception("Max iterations limit reached");
			}
			
		    // progress conditions
			if(isCheckProgressConditions()){
				if (!Double.isNaN(previousRPriXNorm)	&& !Double.isNaN(previousRXVNorm)) {
					if ((previousRPriXNorm <= rPriXNorm && rPriXNorm >= getTolerance())|| 
						(previousRXVNorm <= rXVNorm && rXVNorm >= getTolerance())) {
						log.error("No progress achieved, exit iterations loop without desired accuracy");
						response.setReturnCode(OptimizationResponse.FAILED);
						throw new Exception("No progress achieved, exit iterations loop without desired accuracy");
					
					}
				} 
			}
			previousRPriXNorm = rPriXNorm;
			previousRXVNorm = rXVNorm;

			// backtracking line search
			double s = 1d;
			DoubleMatrix1D X1 = null;
			DoubleMatrix1D V1 = null;
			DoubleMatrix1D gradX1 = null;
			DoubleMatrix1D rDualX1V1 = null;
			DoubleMatrix1D rPriX1V1 = null;
			double previousNormRX1V1 = Double.NaN;
			while (true) {
        // @TODO: can we use 9.7.1?
			
				// X + s*stepX
				//X1 = X.copy().assign(stepX.copy().assign(Mult.mult(s)), Functions.plus);
				X1 = ColtUtils.add(X, stepX, s);
			  // V + s*stepV
				//V1 = V.copy().assign(stepV.copy().assign(Mult.mult(s)), Functions.plus);
				V1 = ColtUtils.add(V, stepV, s);
				if(isInDomainF0(X1)){
					gradX1 = getGradF0(X1);
					rDualX1V1 = rDual(X1, V1, gradX1);
					rPriX1V1 = rPri(X1);
					double normRX1V1 = Math.sqrt(ALG.norm2(rDualX1V1) + ALG.norm2(rPriX1V1));
					if (normRX1V1 <= (1 - getAlpha() * s) * rXVNorm) {
						break;
					}
					
					log.debug("normRX1V1: "+normRX1V1);
					if (!Double.isNaN(previousNormRX1V1)) {
						if (previousNormRX1V1 <= normRX1V1) {
							log.warn("No progress achieved in backtracking with norm");
							break;
						}
					}
					previousNormRX1V1 = normRX1V1;
				}
				
				s = getBeta() * s;
			}
			log.debug("s: " + s);

			// update
			X = X1;
			V = V1;
		}

		long tStop = System.currentTimeMillis();
		log.debug("time: " + (tStop - tStart));
		response.setSolution(X.toArray());
		setOptimizationResponse(response);
		return response.getReturnCode();
	}

	// rDual(x,v) := gradF(X)+[A]T*V (p 532)
	private DoubleMatrix1D rDual(DoubleMatrix1D X, DoubleMatrix1D V, DoubleMatrix1D gradX) {
		if(getA()==null){
			return gradX;
		}
		//return getAT().zMult(V, gradX.copy(), 1., 1, false);
		return ColtUtils.zMult(getAT(), V, gradX, 1);
	}

	// rPri(x,v) := Ax - b (p 532)
//	private DoubleMatrix1D rPri(DoubleMatrix1D X) {
//		if(getA()==null){
//			return F1.make(0);
//		}
//		return getA().zMult(X, getB().copy(), 1., -1., false);
//	}
	
	@Override
	protected int forwardOptimizationRequest() throws Exception {
		if (successor != null) {
			//this mean the chain was activated
			if(JOptimizer.PRIMAL_DUAL_METHOD.equals(getInteriorPointMethod())){
				this.successor = new PrimalDualMethod();
			}else if(JOptimizer.BARRIER_METHOD.equals(getInteriorPointMethod())){
				BarrierFunction bf = new LogarithmicBarrier(getFi(), getDim());
				this.successor = new BarrierMethod(bf);
			}
		}
		return super.forwardOptimizationRequest();
	}
	
	public void setKKTSolver(KKTSolver kktSolver) {
		this.kktSolver = kktSolver;
	}
}
