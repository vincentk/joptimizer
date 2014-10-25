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
import cern.colt.matrix.linalg.SeqBlas;
import cern.jet.math.Functions;
import cern.jet.math.Mult;

import com.joptimizer.functions.FunctionsUtils;
import com.joptimizer.solvers.BasicKKTSolver;
import com.joptimizer.solvers.KKTSolver;
import com.joptimizer.util.ColtUtils;
import com.joptimizer.util.Utils;

/**
 * Primal-dual interior-point method.
 * 
 * @see "S.Boyd and L.Vandenberghe, Convex Optimization, p. 609"
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class PrimalDualMethod extends OptimizationRequestHandler {

	private KKTSolver kktSolver;
	private Log log = LogFactory.getLog(this.getClass().getName());

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
				DoubleMatrix1D fiX0NF = getFi(X0NF);
				int maxIndex = Utils.getMaxIndex(fiX0NF);
				double maxValue = fiX0NF.get(maxIndex);
				if (log.isDebugEnabled()) {
					log.debug("rPriX0NFNorm :  " + rPriX0NFNorm);
					log.debug("X0NF         :  " + ArrayUtils.toString(X0NF.toArray()));
					log.debug("fiX0NF       :  " + ArrayUtils.toString(fiX0NF.toArray()));
				}
				if(maxValue<0 && rPriX0NFNorm<=getToleranceFeas()){
					//the provided not-feasible starting point is already feasible
					log.debug("the provided initial point is already feasible");
					X0 = X0NF;
				}
			}
			if(X0 == null){
				BasicPhaseIPDM bf1 = new BasicPhaseIPDM(this);
				X0 = bf1.findFeasibleInitialPoint();
			}
		}
		
		//check X0 feasibility
		DoubleMatrix1D fiX0 = getFi(X0);
		int maxIndex = Utils.getMaxIndex(fiX0);
		double maxValue = fiX0.get(maxIndex);
		double rPriX0Norm = Math.sqrt(ALG.norm2(rPri(X0)));
		if(maxValue >= 0 || rPriX0Norm > getToleranceFeas()){
			log.debug("rPriX0Norm  : " + rPriX0Norm);
			log.debug("ineqX0      : " + ArrayUtils.toString(fiX0.toArray()));
			log.debug("max ineq index: " + maxIndex);
			log.debug("max ineq value: " + maxValue);
			throw new Exception("initial point must be strictly feasible");
		}

		DoubleMatrix1D V0 = (getA()!=null)? F1.make(getA().rows()) : F1.make(0);
		
		DoubleMatrix1D L0 = getInitialLagrangian();
		if(L0!=null){
			for (int j = 0; j < L0.size(); j++) {
				// must be >0
				if(L0.get(j) <= 0){
					throw new IllegalArgumentException("initial lagrangian must be strictly > 0");
				}
			}
		}else{
			//L0 = F1.make(getFi().length, 1.);// must be >0
			L0 = F1.make(getMieq(), Math.min(1,(double)getDim()/getMieq()));// must be >0
		}
		if(log.isDebugEnabled()){
			log.debug("X0:  " + ArrayUtils.toString(X0.toArray()));
			log.debug("V0:  " + ArrayUtils.toString(V0.toArray()));
			log.debug("L0:  " + ArrayUtils.toString(L0.toArray()));
			log.debug("toleranceFeas:  " + getToleranceFeas());
			log.debug("tolerance    :  " + getTolerance());
		}

		DoubleMatrix1D X = X0;
		DoubleMatrix1D V = V0;
		DoubleMatrix1D L = L0;
		//double F0X;
		//DoubleMatrix1D gradF0X = null;
		//DoubleMatrix1D fiX = null;
		//DoubleMatrix2D GradFiX = null;
		//DoubleMatrix1D rPriX = null;
		//DoubleMatrix1D rCentXLt = null;
		//DoubleMatrix1D rDualXLV = null;
		//double rPriXNorm = Double.NaN;
		//double rCentXLtNorm = Double.NaN;
		//double rDualXLVNorm = Double.NaN;
		//double normRXLVt = Double.NaN;
		double previousF0X = Double.NaN;
		double previousRPriXNorm = Double.NaN;
		double previousRDualXLVNorm = Double.NaN;
		double previousSurrDG = Double.NaN;
		double t;
		int iteration = 0;
		while (true) {
			
			iteration++;
		    // iteration limit condition
			if (iteration == getMaxIteration()+1) {
				response.setReturnCode(OptimizationResponse.FAILED);
				log.error("Max iterations limit reached");
				throw new Exception("Max iterations limit reached");
			}
			
		    double F0X = getF0(X);
			if(log.isDebugEnabled()){
				log.debug("iteration: " + iteration);
				log.debug("X=" + ArrayUtils.toString(X.toArray()));
				log.debug("L=" + ArrayUtils.toString(L.toArray()));
				log.debug("V=" + ArrayUtils.toString(V.toArray()));
				log.debug("f0(X)=" + F0X);
			}
			
//			if(!Double.isNaN(previousF0X)){
//				if (previousF0X < F0X) {
//					throw new Exception("critical minimization problem");
//				}
//			}
//			previousF0X = F0X;
			
			// determine functions evaluations
			DoubleMatrix1D gradF0X = getGradF0(X);
			DoubleMatrix1D fiX     = getFi(X);
			DoubleMatrix2D GradFiX = getGradFi(X);
			DoubleMatrix2D[] HessFiX = getHessFi(X);
			
			// determine t
			double surrDG = getSurrogateDualityGap(fiX, L);
			t = getMu() * getMieq() / surrDG;
			log.debug("t:  " + t);
						
			// determine residuals
			DoubleMatrix1D rPriX    = rPri(X);
			DoubleMatrix1D rCentXLt = rCent(fiX, L, t);
			DoubleMatrix1D rDualXLV = rDual(GradFiX, gradF0X, L, V);
			double rPriXNorm    = Math.sqrt(ALG.norm2(rPriX));
			double rCentXLtNorm = Math.sqrt(ALG.norm2(rCentXLt));
			double rDualXLVNorm = Math.sqrt(ALG.norm2(rDualXLV));
			double normRXLVt    = Math.sqrt(Math.pow(rPriXNorm, 2) + Math.pow(rCentXLtNorm, 2) + Math.pow(rDualXLVNorm, 2));
			log.debug("rPri  norm: " + rPriXNorm);
			log.debug("rCent norm: " + rCentXLtNorm);
			log.debug("rDual norm: " + rDualXLVNorm);
			log.debug("surrDG    : " + surrDG);
			
			// custom exit condition
			if(checkCustomExitConditions(X)){
				response.setReturnCode(OptimizationResponse.SUCCESS);
				break;
			}
			
			// exit condition
			if (rPriXNorm <= getToleranceFeas() && rDualXLVNorm <= getToleranceFeas() && surrDG <= getTolerance()) {
				response.setReturnCode(OptimizationResponse.SUCCESS);
				break;
			}
			
		  // progress conditions
			if(isCheckProgressConditions()){
				if (!Double.isNaN(previousRPriXNorm)
					&& !Double.isNaN(previousRDualXLVNorm)
					&& !Double.isNaN(previousSurrDG)) {
					if (  (previousRPriXNorm <= rPriXNorm && rPriXNorm >= getToleranceFeas())
						|| (previousRDualXLVNorm <= rDualXLVNorm && rDualXLVNorm >= getToleranceFeas())) {
						log.error("No progress achieved, exit iterations loop without desired accuracy");
						response.setReturnCode(OptimizationResponse.FAILED);
						throw new Exception("No progress achieved, exit iterations loop without desired accuracy");
					}
				}
				previousRPriXNorm = rPriXNorm;
				previousRDualXLVNorm = rDualXLVNorm;
				previousSurrDG = surrDG;
			}

			// compute primal-dual search direction
			// a) prepare 11.55 system
			DoubleMatrix2D HessSum = getHessF0(X);
			for (int j = 0; j < getMieq(); j++) {
				if(HessFiX[j] != FunctionsUtils.ZEROES_MATRIX_PLACEHOLDER){
					//HessSum += L[i]*HessFiX[j]
					//HessSum.assign(HessFiX[j].copy().assign(Mult.mult(L.get(j))), Functions.plus);
					HessSum = ColtUtils.add(HessSum, HessFiX[j], L.get(j));
				}
				//log.debug("HessSum    : " + ArrayUtils.toString(HessSum.toArray()));
			}
			
//			DoubleMatrix2D GradSum = F2.make(getDim(), getDim());
//			for (int j = 0; j < getFi().length; j++) {
//				DoubleMatrix1D g = GradFiX.viewRow(j);
//				GradSum.assign(ALG.multOuter(g, g, null).assign(Mult.mult(-L.get(j) / fiX.get(j))), Functions.plus);
//				//log.debug("GradSum    : " + ArrayUtils.toString(GradSum.toArray()));
//			}
			DoubleMatrix2D GradSum = F2.make(getDim(), getDim());
			for (int j = 0; j < getMieq(); j++) {
				final double c = -L.getQuick(j) / fiX.getQuick(j);
				DoubleMatrix1D g = GradFiX.viewRow(j);
				SeqBlas.seqBlas.dger(c, g, g, GradSum);
				//log.debug("GradSum    : " + ArrayUtils.toString(GradSum.toArray()));
			}

			DoubleMatrix2D Hpd = HessSum.assign(GradSum, Functions.plus);
			//DoubleMatrix2D Hpd = getHessF0(X).assign(HessSum, Functions.plus).assign(GradSum, Functions.plus);

			DoubleMatrix1D gradSum = F1.make(getDim());
			for (int j = 0; j < getMieq(); j++) {
				//gradSum += GradFiX.viewRow(j)/(-t * fiX.get(j));
				//gradSum.assign(GradFiX.viewRow(j).copy().assign(Mult.div(-t * fiX.get(j))), Functions.plus);
				gradSum = ColtUtils.add(gradSum, GradFiX.viewRow(j), 1./(-t * fiX.get(j)));
				//log.debug("gradSum    : " + ArrayUtils.toString(gradSum.toArray()));
			}
			DoubleMatrix1D g = null;
			if(getAT()==null){
				//g = gradF0X.copy().assign(gradSum, Functions.plus);
				g = ColtUtils.add(gradF0X, gradSum);
			}else{
				//g = gradF0X.copy().assign(gradSum, Functions.plus).assign(ALG.mult(getAT(), V), Functions.plus);
				g = ColtUtils.add(ColtUtils.add(gradF0X, gradSum), ALG.mult(getAT(), V));
			}
			
			// b) solving 11.55 system
			if(this.kktSolver==null){
				//this.kktSolver = new BasicKKTSolver();
				this.kktSolver = new BasicKKTSolver(request.isRescalingDisabled());
			}
			//KKTSolver solver = new DiagonalKKTSolver();
			if(isCheckKKTSolutionAccuracy()){
				kktSolver.setCheckKKTSolutionAccuracy(true);
				kktSolver.setToleranceKKT(getToleranceKKT());
			}
			kktSolver.setHMatrix(Hpd);
			kktSolver.setGVector(g);
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

			// c) solving for L
			DoubleMatrix1D stepL = null;
//			DoubleMatrix2D diagFInv = F2.diagonal(fiX.copy().assign(Functions.inv));
//			DoubleMatrix2D diagL = F2.diagonal(L);
//			DoubleMatrix1D a1 = ColtUtils.diagonalMatrixMult(diagFInv, rCentXLt);
//			DoubleMatrix1D b1 = ColtUtils.diagonalMatrixMult(diagL, ALG.mult(GradFiX, stepX));
//			DoubleMatrix1D c1 = ColtUtils.diagonalMatrixMult(diagFInv, b1);
//			stepL = a1.assign(c1, Functions.minus);
			
			//DoubleMatrix1D diagFInvElements = fiX.copy().assign(Functions.inv);
			//DoubleMatrix1D diagLElements = L.copy();
			DoubleMatrix1D a2 = rCentXLt.copy().assign(fiX, Functions.div);
			DoubleMatrix1D b2 = ALG.mult(GradFiX, stepX).assign(L, Functions.mult);
			DoubleMatrix1D c2 = b2.assign(fiX, Functions.div);
			stepL = ColtUtils.add(a2, c2, -1);
			if(log.isDebugEnabled()){
				log.debug("stepL: " + ArrayUtils.toString(stepL.toArray()));
			}

			// line search and update
			// a) sMax computation 
			double sMax = Double.MAX_VALUE;
			for (int j = 0; j < getMieq(); j++) {
				if (stepL.get(j) < 0) {
					sMax = Math.min(-L.get(j) / stepL.get(j), sMax);
				}
			}
			sMax = Math.min(1, sMax);
			double s = 0.99 * sMax;
			// b) backtracking with f
			DoubleMatrix1D X1 = F1.make(X.size());
			DoubleMatrix1D L1 = F1.make(L.size());
			DoubleMatrix1D V1 = F1.make(V.size());
			DoubleMatrix1D fiX1 = null;
			DoubleMatrix1D gradF0X1 = null;
			DoubleMatrix2D GradFiX1 = null;
			DoubleMatrix1D rPriX1 = null;
			DoubleMatrix1D rCentX1L1t = null;
			DoubleMatrix1D rDualX1L1V1 = null;
			int cnt = 0;
			boolean areAllNegative = true;
			while (cnt < 500) {
				cnt++;
				// X1 = X + s*stepX
				X1 = stepX.copy().assign(Mult.mult(s)).assign(X, Functions.plus);
				DoubleMatrix1D ineqValueX1 = getFi(X1);
				areAllNegative = true;
				for (int j = 0; areAllNegative && j < getMieq(); j++) {
					areAllNegative = (Double.compare(ineqValueX1.get(j), 0.) < 0);
				}
				if (areAllNegative) {
					break;
				}
				s = getBeta() * s;
			}
			
			if(!areAllNegative){
				//exited from the feasible region
				throw new Exception("Optimization failed: impossible to remain within the faesible region");
			}
			
			log.debug("s: " + s);
			// c) backtracking with norm
			double previousNormRX1L1V1t = Double.NaN;
			cnt = 0;
			while (cnt < 500) {
				cnt++;
				X1 = ColtUtils.add(X, stepX, s);
				L1 = ColtUtils.add(L, stepL, s);
				V1 = ColtUtils.add(V, stepV, s);
				
				if (isInDomainF0(X1)) {
					fiX1 = getFi(X1);
					gradF0X1 = getGradF0(X1);
					GradFiX1 = getGradFi(X1);
					
					rPriX1 = rPri(X1);
					rCentX1L1t = rCent(fiX1, L1, t);
					rDualX1L1V1 = rDual(GradFiX1, gradF0X1, L1, V1);
					//log.debug("rPriX1     : "+ArrayUtils.toString(rPriX1.toArray()));
					//log.debug("rCentX1L1t : "+ArrayUtils.toString(rCentX1L1t.toArray()));
					//log.debug("rDualX1L1V1: "+ArrayUtils.toString(rDualX1L1V1.toArray()));
					double normRX1L1V1t = Math.sqrt(ALG.norm2(rPriX1)
							                          + ALG.norm2(rCentX1L1t)
							                          + ALG.norm2(rDualX1L1V1));
					//log.debug("normRX1L1V1t: "+normRX1L1V1t);
					if (normRX1L1V1t <= (1 - getAlpha() * s) * normRXLVt) {
						break;
					}
					
					if (!Double.isNaN(previousNormRX1L1V1t)) {
						if (previousNormRX1L1V1t <= normRX1L1V1t) {
							log.warn("No progress achieved in backtracking with norm");
							break;
						}
					}
					previousNormRX1L1V1t = normRX1L1V1t;
				}
				
				s = getBeta() * s;
				//log.debug("s: " + s);
			}

			// update
			X = X1;
			V = V1;
			L = L1;
			
//			fiX     = fiX1;
//			gradF0X = gradF0X1;
//			GradFiX  = GradFiX1;
//			
//			rPriX    = rPriX1;
//			rCentXLt = rCentX1L1t;
//			rDualXLV = rDualX1L1V1;
//			rPriXNorm    = Math.sqrt(ALG.norm2(rPriX));
//			rCentXLtNorm = Math.sqrt(ALG.norm2(rCentXLt));
//			rDualXLVNorm = Math.sqrt(ALG.norm2(rDualXLV));
//			normRXLVt = Math.sqrt(Math.pow(rPriXNorm, 2) + Math.pow(rCentXLtNorm, 2) + Math.pow(rDualXLVNorm, 2));
//			if(log.isDebugEnabled()){
//				log.debug("rPri  norm: " + rPriXNorm);
//				log.debug("rCent norm: " + rCentXLtNorm);
//				log.debug("rDual norm: " + rDualXLVNorm);
//				log.debug("surrDG    : " + surrDG);
//			}
		}

		long tStop = System.currentTimeMillis();
		log.debug("time: " + (tStop - tStart));
		log.debug("sol : " + ArrayUtils.toString(X.toArray()));
		log.debug("ret code: " + response.getReturnCode());
		response.setSolution(X.toArray());
		setOptimizationResponse(response);
		return response.getReturnCode();
	}

	/**
	 * Surrogate duality gap.
	 * 
	 * @see "Convex Optimization, 11.59"
	 */
	private double getSurrogateDualityGap(DoubleMatrix1D fiX, DoubleMatrix1D L) {
		return -ALG.mult(fiX, L);
	}

	/**
	 * @see "Convex Optimization, p. 610"
	 */
	private DoubleMatrix1D rDual(DoubleMatrix2D GradFiX, DoubleMatrix1D gradF0X, DoubleMatrix1D L, DoubleMatrix1D V) {
		if(getA()==null){
			//return GradFiX.zMult(L, gradF0X.copy(), 1., 1., true);
			return ColtUtils.zMultTranspose(GradFiX, L, gradF0X, 1.);
		}
		//return getA().zMult(V, GradFiX.zMult(L, gradF0X.copy(), 1., 1., true), 1., 1., true);
		return ColtUtils.zMultTranspose(getA(), V, ColtUtils.zMultTranspose(GradFiX, L, gradF0X, 1.), 1.);
	}

	/**
	 * @see "Convex Optimization, p. 610"
	 */
	private DoubleMatrix1D rCent(DoubleMatrix1D fiX, DoubleMatrix1D L, double t) {
		DoubleMatrix1D ret = F1.make(L.size());
		for(int i=0; i<ret.size(); i++){
			ret.setQuick(i, -L.getQuick(i)*fiX.getQuick(i) - 1. / t);
		}
		return ret;
	}
	
	public void setKKTSolver(KKTSolver kktSolver) {
		this.kktSolver = kktSolver;
	}

}
