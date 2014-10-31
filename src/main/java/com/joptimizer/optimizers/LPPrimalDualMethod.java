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
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import cern.jet.math.Functions;
import cern.jet.math.Mult;

import com.joptimizer.solvers.KKTSolver;
import com.joptimizer.solvers.UpperDiagonalHKKTSolver;
import com.joptimizer.util.ColtUtils;
import com.joptimizer.util.Utils;

/**
 * Primal-dual interior-point method for LP problems in the form (1):
 * 
 * <br>min(c) s.t.
 * <br>G.x < h
 * <br>A.x = b
 * <br>lb <= x <= ub
 * 
 * <br>If lower and/or upper bounds are not passed in, a default value is assigned on them, that is:
 * <br>-)if the vector lb is not passed in, all the lower bounds are assumed to be equal to the value of the field <i>minLBValue</i> 
 * <br>-)if the vector ub is not passed in, all the upper bounds are assumed to be equal to the value of the field <i>maxUBValue</i>
 * 
 * <br>The problem is first transformed in the standard form, then presolved and finally solved.
 * 
 * <br>Note 1: avoid to set minLBValue or maxUBValue to fake unbounded values.
 * 
 * @see "S.Boyd and L.Vandenberghe, Convex Optimization, p. 609"
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class LPPrimalDualMethod extends LPOptimizationRequestHandler {

	public static final double DEFAULT_MIN_LOWER_BOUND = -99999;
	public static final double DEFAULT_MAX_UPPER_BOUND = +99999;
//	public static final double DEFAULT_UNSPECIFIED_LOWER_BOUND = -9999;
//	public static final double DEFAULT_UNSPECIFIED_UPPER_BOUND = +9999;
	
//	private double unspecifiedLBValue = DEFAULT_UNSPECIFIED_LOWER_BOUND;
//	private double unspecifiedUBValue = DEFAULT_UNSPECIFIED_UPPER_BOUND;
	private double minLBValue = DEFAULT_MIN_LOWER_BOUND;
	private double maxUBValue = DEFAULT_MAX_UPPER_BOUND;
	/**
	 * Lower bounds with min value limited to the value of the field <i>minLBValue</i>.
	 */
	private DoubleMatrix1D limitedLb;
	
	/**
	 * Upper bounds with max value limited to the value of the field <i>maxUBValue</i>.
	 */
	private DoubleMatrix1D limitedUb;

	private KKTSolver kktSolver;
	private Log log = LogFactory.getLog(this.getClass().getName());
	
//	public LPPrimalDualMethod(){
//		this(DEFAULT_UNSPECIFIED_LOWER_BOUND, DEFAULT_UNSPECIFIED_UPPER_BOUND, 
//				DEFAULT_UNBOUNDED_LOWER_BOUND, DEFAULT_UNBOUNDED_UPPER_BOUND);
//	}
	
	public LPPrimalDualMethod(){
		this(DEFAULT_MIN_LOWER_BOUND, DEFAULT_MAX_UPPER_BOUND);
	}
	
//	public LPPrimalDualMethod(double unspecifiedLBValue, double unspecifiedUBValue, double unboundedLBValue, double unboundedUBValue){
//		this.unspecifiedLBValue = unspecifiedLBValue;
//		this.unspecifiedUBValue = unspecifiedUBValue;
//		this.unboundedLBValue = unboundedLBValue;
//		this.unboundedUBValue = unboundedUBValue;
//	}
	
	public LPPrimalDualMethod(double minLBValue, double maxUBValue){
		if(Double.isNaN(minLBValue) || Double.isInfinite(minLBValue) ){
			throw new IllegalArgumentException("The field minLBValue must not be set to Double.NaN or Double.NEGATIVE_INFINITY");
		}
		if(Double.isNaN(maxUBValue) || Double.isInfinite(maxUBValue) ){
			throw new IllegalArgumentException("The field maxUBValue must not be set to Double.NaN or Double.POSITIVE_INFINITY");
		}
		this.minLBValue = minLBValue;
		this.maxUBValue = maxUBValue;
	}
	
	/**
	 * Solves an LP in the form of:
	 * min(c) s.t.
	 * A.x = b
	 * G.x < h
	 * lb <= x <= ub
	 * 
	 */
	@Override
	public int optimize() throws Exception {
		log.info("optimize");
		
		LPOptimizationRequest lpRequest = getLPOptimizationRequest();
		if(log.isDebugEnabled() && lpRequest.isDumpProblem()){
			log.debug("LP problem: " + lpRequest.toString());
		}
		
		//standard form conversion
		LPStandardConverter lpConverter = new LPStandardConverter();//the slack variables will have default unboundedUBValue
		
		lpConverter.toStandardForm(getC(), getG(), getH(), getA(), getB(), getLb(), getUb());
		int nOfSlackVariables = lpConverter.getStandardS();
		log.debug("nOfSlackVariables: " + nOfSlackVariables);
		DoubleMatrix1D standardC = lpConverter.getStandardC();
		DoubleMatrix2D standardA = lpConverter.getStandardA();
		DoubleMatrix1D standardB = lpConverter.getStandardB();
		DoubleMatrix1D standardLb = lpConverter.getStandardLB();
		DoubleMatrix1D standardUb = lpConverter.getStandardUB();
		
		//solve the standard form problem
		LPOptimizationRequest standardLPRequest = lpRequest.cloneMe();
		standardLPRequest.setC(standardC);
		standardLPRequest.setA(standardA);
		standardLPRequest.setB(standardB);
		standardLPRequest.setLb(ColtUtils.replaceValues(standardLb, lpConverter.getUnboundedLBValue(), minLBValue));//substitute not-double numbers
		standardLPRequest.setUb(ColtUtils.replaceValues(standardUb, lpConverter.getUnboundedUBValue(), maxUBValue));//substitute not-double numbers
		if(getInitialPoint()!=null){
			standardLPRequest.setInitialPoint(lpConverter.getStandardComponents(getInitialPoint().toArray()));
		}
		if(getNotFeasibleInitialPoint()!=null){
			standardLPRequest.setNotFeasibleInitialPoint(lpConverter.getStandardComponents(getNotFeasibleInitialPoint().toArray()));
		}
		
		//optimization
		LPPrimalDualMethod opt = new LPPrimalDualMethod(minLBValue, maxUBValue);
		opt.setLPOptimizationRequest(standardLPRequest);
		if(opt.optimizeStandardLP(nOfSlackVariables) == OptimizationResponse.FAILED){
			return OptimizationResponse.FAILED;
		}
		
		//back to original form
		LPOptimizationResponse lpResponse = opt.getLPOptimizationResponse();
		double[] standardSolution = lpResponse.getSolution();
		double[] originalSol = lpConverter.postConvert(standardSolution);
		lpResponse.setSolution(originalSol);
		setLPOptimizationResponse(lpResponse);
		return lpResponse.getReturnCode();
	}
	
	/**
	 * Solves a standard form LP problem in the form of
	 * min(c) s.t.
	 * A.x = b
	 * lb <= x <= ub
	 */
	protected int optimizeStandardLP(int nOfSlackVariables) throws Exception {
		log.info("optimizeStandardLP");
		
		LPOptimizationRequest lpRequest = getLPOptimizationRequest();
		if(log.isDebugEnabled() && lpRequest.isDumpProblem()){
			log.debug("LP problem: " + lpRequest.toString());
		}
		
		LPOptimizationResponse lpResponse;
		if(lpRequest.isPresolvingDisabled()){
			//optimization
			LPPrimalDualMethod opt = new LPPrimalDualMethod(minLBValue, maxUBValue);
			opt.setLPOptimizationRequest(lpRequest);
			if(opt.optimizePresolvedStandardLP() == OptimizationResponse.FAILED){
				return OptimizationResponse.FAILED;
			}
			lpResponse = opt.getLPOptimizationResponse();
			setLPOptimizationResponse(lpResponse);
		}else{
			//presolving
			LPPresolver lpPresolver = new LPPresolver();
			lpPresolver.setAvoidScaling(lpRequest.isRescalingDisabled());
			lpPresolver.setAvoidFillIn(lpRequest.isAvoidPresolvingFillIn());
			lpPresolver.setAvoidIncreaseSparsity(lpRequest.isAvoidPresolvingIncreaseSparsity());
			lpPresolver.setNOfSlackVariables((short)nOfSlackVariables);
			lpPresolver.presolve(getC(), getA(), getB(), getLb(), getUb());
			int presolvedDim = lpPresolver.getPresolvedN();
			
			if(presolvedDim==0){
				//deterministic problem
				log.debug("presolvedDim : " + presolvedDim);
				log.debug("deterministic LP problem");
				lpResponse = new LPOptimizationResponse();
				lpResponse.setReturnCode(OptimizationResponse.SUCCESS);
				lpResponse.setSolution(new double[]{});
			}else{
				//solving the presolved problem
				DoubleMatrix1D presolvedC = lpPresolver.getPresolvedC();
				DoubleMatrix2D presolvedA = lpPresolver.getPresolvedA();
				DoubleMatrix1D presolvedB = lpPresolver.getPresolvedB();
				if(log.isDebugEnabled()){
					if(lpPresolver.getPresolvedYlb()!=null){
						log.debug("Ylb: " + ArrayUtils.toString(lpPresolver.getPresolvedYlb().toArray()));
						log.debug("Yub: " + ArrayUtils.toString(lpPresolver.getPresolvedYub().toArray()));
					}
					if(lpPresolver.getPresolvedZlb()!=null){
						log.debug("Zlb: " + ArrayUtils.toString(lpPresolver.getPresolvedZlb().toArray()));
						log.debug("Zub: " + ArrayUtils.toString(lpPresolver.getPresolvedZub().toArray()));
					}
				}
				
				//new LP problem (the presolved problem)
				LPOptimizationRequest presolvedLPRequest = lpRequest.cloneMe();
				presolvedLPRequest.setC(presolvedC);
				presolvedLPRequest.setA(presolvedA);
				presolvedLPRequest.setB(presolvedB);
				presolvedLPRequest.setLb(lpPresolver.getPresolvedLB());
				presolvedLPRequest.setUb(lpPresolver.getPresolvedUB());
				presolvedLPRequest.setYlb(lpPresolver.getPresolvedYlb());
				presolvedLPRequest.setYub(lpPresolver.getPresolvedYub());
				presolvedLPRequest.setZlb(lpPresolver.getPresolvedZlb());
				presolvedLPRequest.setZub(lpPresolver.getPresolvedZub());
				if(getInitialPoint()!=null){
					presolvedLPRequest.setInitialPoint(lpPresolver.presolve(getInitialPoint().toArray()));
				}
				if(getNotFeasibleInitialPoint()!=null){
					presolvedLPRequest.setNotFeasibleInitialPoint(lpPresolver.presolve(getNotFeasibleInitialPoint().toArray()));
				}
				
				//optimization
				//NB: because of rescaling during the presolving phase, minLB and maxUB could have been rescaled
				double rescaledMinLBValue = (Double.isNaN(lpPresolver.getMinRescaledLB()))? this.minLBValue : lpPresolver.getMinRescaledLB();
				double rescaledMaxUBValue = (Double.isNaN(lpPresolver.getMaxRescaledUB()))? this.maxUBValue : lpPresolver.getMaxRescaledUB();
				LPPrimalDualMethod opt = new LPPrimalDualMethod(rescaledMinLBValue, rescaledMaxUBValue);
				opt.setLPOptimizationRequest(presolvedLPRequest);
				if(opt.optimizePresolvedStandardLP() == OptimizationResponse.FAILED){
					return OptimizationResponse.FAILED;
				}
				lpResponse = opt.getLPOptimizationResponse();
			}
			
			//postsolving
			double[] postsolvedSolution = lpPresolver.postsolve(lpResponse.getSolution());
			lpResponse.setSolution(postsolvedSolution);
			setLPOptimizationResponse(lpResponse);			
		}
		
		return lpResponse.getReturnCode();
	}

	/**
	 * Solves a presolved standard form LP problem in the form of
	 * min(c) s.t.
	 * A.x = b
	 * lb <= x <= ub
	 */
	protected int optimizePresolvedStandardLP() throws Exception {
		log.info("optimizePresolvedStandardLP");
		
		long tStart = System.currentTimeMillis();
		
		LPOptimizationRequest lpRequest = getLPOptimizationRequest();
		if(log.isDebugEnabled() && lpRequest.isDumpProblem()){
			log.debug("LP problem: " + lpRequest.toString());
		}
		
		if(this.dim <= -1){
			if (getLb().size() != getUb().size()) {
				log.error("Lower and upper bounds must have the same dimension");
				throw new IllegalArgumentException("Lower and upper bounds must have the same dimension");
			}
			this.dim = getLb().size();
			double minDeltaBoundsValue = Double.MAX_VALUE;
			int minDeltaBoundsIndex = -1;
			for(int i=0; i<getDim(); i++){
				double deltai = getUb().getQuick(i) - getLb().getQuick(i);
				if(deltai < minDeltaBoundsValue){
					minDeltaBoundsValue = deltai;
					minDeltaBoundsIndex = i;
				}
			}
			log.info("min delta bounds index: " + minDeltaBoundsIndex);
			log.info("min delta bounds value: " + minDeltaBoundsValue);
		}
		//this.boundedLb = new boolean[getDim()];
		//this.boundedUb = new boolean[getDim()];
//		for(int i=0; i<getDim(); i++){
//			if(!isLbUnbounded(getLb().getQuick(i))){
//				boundedLb[i] = true;
//				nOfBoundedLb++;
//			}
//			if(!isUbUnbounded(getUb().getQuick(i))){
//				boundedUb[i] = true;
//				nOfBoundedUb++;
//			}
//		}
		this.meq = (this.meq>-1)? this.meq : ((getA()!=null)? getA().rows() : 0);
		//this.mieq = (this.mieq>-1)? this.mieq : (nOfBoundedLb+nOfBoundedUb);
		this.mieq = (this.mieq>-1)? this.mieq : (2 * getDim());
		
		if(log.isDebugEnabled()){
			log.debug("dim : " + getDim());
			log.debug("meq : " + getMeq());
			log.debug("mieq: " + getMieq());
		}
		
		LPOptimizationResponse lpResponse = new LPOptimizationResponse();
		
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
				BasicPhaseILPPDM bf1 = new BasicPhaseILPPDM(this);
				X0 = bf1.findFeasibleInitialPoint();
			}
		}
		
		//check X0 feasibility
		DoubleMatrix1D fiX0 = getFi(X0);
		int maxIndex = Utils.getMaxIndex(fiX0);
		double maxValue = fiX0.get(maxIndex);
		double rPriX0Norm = Math.sqrt(ALG.norm2(rPri(X0)));
		if(maxValue >= 0. || rPriX0Norm > getToleranceFeas()){//must be fi STRICTLY < 0
			log.warn("rPriX0Norm  : " + rPriX0Norm);
			log.warn("ineqX0      : " + ArrayUtils.toString(fiX0.toArray()));
			log.warn("max ineq index: " + maxIndex);
			log.warn("max ineq value: " + maxValue);
			//the point must be INTERNAL, fi are used as denominators
			throw new Exception("initial point must be strictly feasible");
		}

		DoubleMatrix1D V0 = F1.make(getMeq());
		if(getYlb()!=null && getYub()!=null){
			//NB: the Lagrangian multipliers for eq. constraints used in this interior point method (v)
			//are the opposite of the Lagrangian  multipliers for eq. constraints used in the presolver (y)
			//and so Ylb<=y<=Yub becomes -Yub<=v<=-Ylb  
			for(int i=0; i<getMeq(); i++){
				double v0i = 0;
				if(!isLbUnbounded(getYlb().getQuick(i))){
					if(!isUbUnbounded(getYub().getQuick(i))){
						v0i = -(getYub().getQuick(i)+getYlb().getQuick(i))/2;
					}else{
						v0i = -getYlb().getQuick(i);
					}
				}else{
					if(!isUbUnbounded(getYub().getQuick(i))){
						v0i = -getYub().getQuick(i);
					}else{
						v0i = 0;
					}
				}
				V0.setQuick(i, v0i);
			}
		}
		
		DoubleMatrix1D L0 = getInitialLagrangian();
		if(L0!=null){
			for (int j = 0; j < L0.size(); j++) {
				// must be >0
				if(L0.get(j) <= 0){
					throw new IllegalArgumentException("initial lagrangian must be strictly > 0");
				}
			}
		}else{
			L0 = F1.make(getMieq(), 1.);// must be >0 strictly
			if(getZlb()!=null && getZub()!=null){
				//Zlb<= L <=Zub, meaning that:
				//zlb[i] and zub[i] are the bounds on the Lagrangian of the constraint associated with lb[i]<x[i]<ub[i] 
				//note that zlb.size = zub.size = lb.size = ub.size (and = n of variables of the problem (= getDim())
				//and that L.size = nOfBoundedLb + nOfBoundedUb (and in general < 2*getDim())
				int cntLB = 0;
				int cntUB = 0;
				for(int i=0; i<getDim(); i++){
					double zlbi = (isLbUnbounded(getZlb().getQuick(i)))? 0 : getZlb().getQuick(i);//L must be > 0
					double zubi = (isUbUnbounded(getZub().getQuick(i)))? 1 : getZub().getQuick(i);
					L0.setQuick(cntLB, (zubi- zlbi)/2);
					cntLB++;
					L0.setQuick(getDim() + cntUB, (zubi- zlbi)/2);
					cntUB++;
				}
			}else{
				//inequalities comes from the pairs lower bounds-upper bounds
				//in the calculation of the H matrix fro the KKT system, each pairs gives terms of the form:
				//t = tl + tu
				//tl = -L[i] / fi[i] for the lower bound 
				//tu =  L[dim+i] / fi[dim+i] for the upper bound
				//we want t = 1, and hence
				//L[i] > -cc * fi[i]
				//L[dim+i] = (1 + L[i] / fi[i]) * fi[dim+i]
//				double cc = 10;
//				int nOfLB = getMieq()/2;
//				for (int i = 0; i < nOfLB; i++) {
//					L0.setQuick(i, -cc * fiX0.getQuick(i));
//					L0.setQuick(nOfLB + i, (1 - 10)	* fiX0.getQuick(nOfLB + i));
//					double sum = -L0.getQuick(i)/fiX0.getQuick(i)+L0.getQuick(nOfLB + i)/fiX0.getQuick(nOfLB + i);
//					log.debug("sum["+i+"]:  " + sum);
//				}
			}
		}
		if(log.isDebugEnabled()){
			log.debug("X0:  " + ArrayUtils.toString(X0.toArray()));
			log.debug("V0:  " + ArrayUtils.toString(V0.toArray()));
			log.debug("L0:  " + ArrayUtils.toString(L0.toArray()));
		}
		if(log.isInfoEnabled()){
			log.info("toleranceFeas:  " + getToleranceFeas());
			log.info("tolerance    :  " + getTolerance());
		}

		DoubleMatrix1D X = X0;
		DoubleMatrix1D V = V0;
		DoubleMatrix1D L = L0;
		double previousRPriXNorm = Double.NaN;
		double previousRDualXLVNorm = Double.NaN;
		double previousSurrDG = Double.NaN;
		double t;
		int iteration = 0;
		while (true) {
			
			iteration++;
		    // iteration limit condition
			if (iteration == getMaxIteration()+1) {
				lpResponse.setReturnCode(OptimizationResponse.FAILED);
				log.error("Max iterations limit reached");
				throw new Exception("Max iterations limit reached");
			}
			
			//XList.add(XList.size(), X);
			double F0X = getF0(X);
		  if(log.isInfoEnabled()){
				log.info("iteration: " + iteration);
				log.info("f0(X)=" + F0X);
			}
			if(log.isDebugEnabled()){
				log.debug("X=" + ArrayUtils.toString(X.toArray()));
				log.debug("L=" + ArrayUtils.toString(L.toArray()));
				log.debug("V=" + ArrayUtils.toString(V.toArray()));
			}
			
			
			// determine functions evaluations
			DoubleMatrix1D gradF0X = getGradF0(X);
			DoubleMatrix1D fiX     = getFi(X);
			log.debug("fiX=" + ArrayUtils.toString(fiX.toArray()));
			//DoubleMatrix2D GradFiX = getGradFi(X);
			//DoubleMatrix2D GradFiXOLD = getGradFiOLD(X);
			
			// determine t
			double surrDG = getSurrogateDualityGap(fiX, L);
			t = getMu() * getMieq() / surrDG;
			log.debug("t:  " + t);
						
			// determine residuals
			DoubleMatrix1D rPriX    = rPri(X);
			DoubleMatrix1D rCentXLt = rCent(fiX, L, t);
			DoubleMatrix1D rDualXLV = rDual(gradF0X, L, V);
			//DoubleMatrix1D rDualXLVOLD = rDualOLD(GradFiXOLD, gradF0X, L, V);
			//log.debug("delta: " + ALG.normInfinity(rDualXLVOLD.assign(rDualXLV, Functions.minus)));
			double rPriXNorm    = Math.sqrt(ALG.norm2(rPriX));
			double rCentXLtNorm = Math.sqrt(ALG.norm2(rCentXLt));
			double rDualXLVNorm = Math.sqrt(ALG.norm2(rDualXLV));
			double normRXLVt    = Math.sqrt(Math.pow(rPriXNorm, 2) + Math.pow(rCentXLtNorm, 2) + Math.pow(rDualXLVNorm, 2));
			//@TODO: set log.debug not log.info
			log.info("rPri  norm: " + rPriXNorm);
			log.info("rCent norm: " + rCentXLtNorm);
			log.info("rDual norm: " + rDualXLVNorm);
			log.info("surrDG    : " + surrDG);
			
			// custom exit condition
			if(checkCustomExitConditions(X)){
				lpResponse.setReturnCode(OptimizationResponse.SUCCESS);
				break;
			}
			
			// exit condition
			if (rPriXNorm <= getToleranceFeas() && rDualXLVNorm <= getToleranceFeas() && surrDG <= getTolerance()) {
				lpResponse.setReturnCode(OptimizationResponse.SUCCESS);
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
						lpResponse.setReturnCode(OptimizationResponse.FAILED);
						throw new Exception("No progress achieved, exit iterations loop without desired accuracy");
					}
				}
				previousRPriXNorm = rPriXNorm;
				previousRDualXLVNorm = rDualXLVNorm;
				previousSurrDG = surrDG;
			}

			// compute primal-dual search direction
			// a) prepare 11.55 system
			DoubleMatrix2D Hpd = GradLSum(L, fiX);
			//DoubleMatrix2D HpdOLD = GradLSumOLD(GradFiXOLD, L, fiX);
			//log.debug("delta: " + ALG.normInfinity(HpdOLD.assign(Hpd, Functions.minus)));
			DoubleMatrix1D gradSum = gradSum(t, fiX);
			DoubleMatrix1D g = null;
			//if(getAT()==null){
			if(getA()==null){
				g = ColtUtils.add(gradF0X, gradSum);
			}else{
				//g = ColtUtils.add(ColtUtils.add(gradF0X, gradSum), ALG.mult(getAT(), V));
				g = ColtUtils.add(ColtUtils.add(gradF0X, gradSum), ColtUtils.zMultTranspose(getA(), V, F1.make(getDim()), 0));
			}
			
			// b) solving 11.55 system
			if(this.kktSolver==null){
				this.kktSolver = new UpperDiagonalHKKTSolver(getDim(), lpRequest.isRescalingDisabled());
				//this.kktSolver = new DiagonalHKKTSolver(getDim(), lpRequest.isRescalingDisabled());
			}
			if(isCheckKKTSolutionAccuracy()){
				kktSolver.setCheckKKTSolutionAccuracy(true);
				kktSolver.setToleranceKKT(getToleranceKKT());
			}
			kktSolver.setHMatrix(Hpd);
			kktSolver.setGVector(g);
			if(getA()!=null){
				kktSolver.setAMatrix(getA());
				//kktSolver.setATMatrix(getAT());
				kktSolver.setHVector(rPriX);
//				if(rPriXNorm > getToleranceFeas()){
//					kktSolver.setHVector(rPriX);
//				}
			}
			DoubleMatrix1D[] sol = kktSolver.solve();
			DoubleMatrix1D stepX = sol[0];
			//double[] signa = new double[stepX.size()];
//			for(int p=0; p<stepX.size(); p++){
//				signa[p] =	Math.signum(stepX.getQuick(p)); 
//			}
			//SList.add(SList.size(), signa);
			DoubleMatrix1D stepV = (sol[1]!=null)? sol[1] : F1.make(0);
			if(log.isDebugEnabled()){
				log.debug("stepX: " + ArrayUtils.toString(stepX.toArray()));
				log.debug("stepV: " + ArrayUtils.toString(stepV.toArray()));
			}

			// c) solving for L
			DoubleMatrix1D stepL = F1.make(getMieq());
			DoubleMatrix1D gradFiStepX = gradFiStepX(stepX);
			for(int i=0; i<getMieq(); i++){
				stepL.setQuick(i, (- L.getQuick(i) * gradFiStepX.getQuick(i) + rCentXLt.getQuick(i)) / fiX.getQuick(i));
			}
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
			//DoubleMatrix2D GradFiX1 = null;
			//DoubleMatrix2D GradFiX1 = null;
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
					//GradFiX1 = getGradFi(X1);
					
					rPriX1 = rPri(X1);
					rCentX1L1t = rCent(fiX1, L1, t);
					rDualX1L1V1 = rDual(gradF0X1, L1, V1);
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
		}
		
//		if(lpRequest.isCheckOptimalDualityConditions()){
//			//check duality conditions:
//			if(!checkDualityConditions(X, L, V)){
//				log.error("duality conditions not satisfied");
//				lpResponse.setReturnCode(OptimizationResponse.FAILED);
//				throw new Exception("duality conditions not satisfied");
//			}
//		}
		
		if(lpRequest.isCheckOptimalLagrangianBounds()){
			//check equality constraints Lagrangian bounds
//			if(!checkEqConstraintsLagrangianBounds(V)){
//				log.error("equality constraints Lagrangian multipliers bounds not satisfied");
//				lpResponse.setReturnCode(OptimizationResponse.FAILED);
//				throw new Exception("equality constraints Lagrangian multipliers bounds not satisfied");
//			}
			
			//check inequality constraints Lagrangian bounds
//			if(!checkIneqConstraintsLagrangianBounds(X, L)){
//				log.error("inequality constraints Lagrangian multipliers bounds not satisfied");
//				lpResponse.setReturnCode(OptimizationResponse.FAILED);
//				throw new Exception("inequality constraints Lagrangian multipliers bounds not satisfied");
//			}
		}

		long tStop = System.currentTimeMillis();
		log.debug("time: " + (tStop - tStart));
		log.debug("sol : " + ArrayUtils.toString(X.toArray()));
		log.debug("ret code: " + lpResponse.getReturnCode());
//		log.debug("XList : " + ArrayUtils.toString(XList));
//		for(int s=0; s<SList.size(); s++){
//			log.debug("SList : " + ArrayUtils.toString(SList.get(s)));
//		}
		lpResponse.setSolution(X.toArray());
		setLPOptimizationResponse(lpResponse);
		return lpResponse.getReturnCode();
	}

//	protected DoubleMatrix1D gradSumOLD(double t, DoubleMatrix1D fiX, DoubleMatrix2D GradFiX) {
//		DoubleMatrix1D gradSum = F1.make(getDim());
//		for (int j = 0; j < getMieq(); j++) {
//			//gradSum += GradFiX.viewRow(j)/(-t * fiX.get(j));
//			gradSum = ColtUtils.add(gradSum, GradFiX.viewRow(j), 1./(-t * fiX.get(j)));
//			//log.debug("gradSum    : " + ArrayUtils.toString(gradSum.toArray()));
//		}
//		return gradSum;
//	}
	
	/**
	 * Calculates the second term of the first row of (11.55) "Convex Optimization".
	 * @see "Convex Optimization, 11.55"
	 */
	protected DoubleMatrix1D gradSum(double t, DoubleMatrix1D fiX) {
		DoubleMatrix1D gradSum = F1.make(getDim());
		for(int i=0; i<dim; i++){
			double d = 0;
			d +=  1. / (t * fiX.getQuick(i));
			d += -1. / (t * fiX.getQuick(getDim() + i));
			gradSum.setQuick(i, d);
		}
		return gradSum;
	}

//	protected DoubleMatrix2D GradLSumOLD(DoubleMatrix2D GradFiX, DoubleMatrix1D L, DoubleMatrix1D fiX) {
//		DoubleMatrix2D GradSum = F2.make(getDim(), getDim());
//		for (int j = 0; j < getMieq(); j++) {
//			final double c = -L.getQuick(j) / fiX.getQuick(j);
//			DoubleMatrix1D g = GradFiX.viewRow(j);
//			SeqBlas.seqBlas.dger(c, g, g, GradSum);
//			//log.debug("GradSum    : " + ArrayUtils.toString(GradSum.toArray()));
//		}
//		return GradSum;
//	}
	
	/**
	 * Return the H matrix (that is diagonal).
	 * This is the third addendum of (11.56) of "Convex Optimization".
	 * @see "Convex Optimization, 11.56"
	 */
	protected DoubleMatrix2D GradLSum(DoubleMatrix1D L, DoubleMatrix1D fiX) {
		//DoubleMatrix2D GradLSum = F2.make(1, getDim());
		SparseDoubleMatrix2D GradLSum = new SparseDoubleMatrix2D(getDim(), getDim(), getDim(), 0.001, 0.01); 
		for(int i=0; i<getDim(); i++){
			double d = 0;
			d -= L.getQuick(i) / fiX.getQuick(i);
			d -= L.getQuick(getDim() + i) / fiX.getQuick(getDim() + i);
			//GradLSum.setQuick(0, i, d);
			GradLSum.setQuick(i, i, d);
		}
		
		return GradLSum;
	}
	
	/**
	 * Computes the term Grad[fi].stepX
	 */
	protected DoubleMatrix1D gradFiStepX(DoubleMatrix1D stepX){
		
		DoubleMatrix1D ret = F1.make(getMieq());
		for(int i=0; i<getDim(); i++){
			ret.setQuick(         i, - stepX.getQuick(i));
			ret.setQuick(getDim()+i,   stepX.getQuick(i));
		}
		
		return ret;
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
//	private DoubleMatrix1D rDualOLD(DoubleMatrix2D GradFiX, DoubleMatrix1D gradF0X, DoubleMatrix1D L, DoubleMatrix1D V) {
//		DoubleMatrix1D m1 = ColtUtils.zMultTranspose(GradFiX, L, gradF0X, 1.);
//		if(getMeq()==0){
//			return m1;
//		}
//		return ColtUtils.zMultTranspose(getA(), V, m1, 1.);
//	}
	
	/**
	 * @see "Convex Optimization, p. 610"
	 */
	protected DoubleMatrix1D rDual(DoubleMatrix1D gradF0X, DoubleMatrix1D L, DoubleMatrix1D V) {
		//m1 = GradFiX[T].L + gradF0X
		DoubleMatrix1D m1 = F1.make(getDim());
		for(int i=0; i<getDim(); i++){
			double m = 0;
			m += - L.getQuick(i);
			m += L.getQuick(getDim() + i);
			m1.setQuick(i, m + gradF0X.get(i));
		}
		if(getMeq()==0){
			return m1;
		}
		return ColtUtils.zMultTranspose(getA(), V, m1, 1.);
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
	
	/**
	 * Objective function value at X.
	 * This is C.X
	 */
	@Override
	protected double getF0(DoubleMatrix1D X) {
		return getC().zDotProduct(X);
	}

	/**
	 * Objective function gradient at X.
	 * This is C itself.
	 */
	@Override
	protected DoubleMatrix1D getGradF0(DoubleMatrix1D X) {
		return getC();
	}
	
	/**
	 * Objective function hessian at X.
	 */
	@Override
	protected DoubleMatrix2D getHessF0(DoubleMatrix1D X) {
		throw new RuntimeException("Hessians are null for LP");
	}
	
	/**
	 * Inequality functions values at X.
	 * This is (-x+lb) for all bounded lb and (x-ub) for all bounded ub. 
	 */
	@Override
	protected DoubleMatrix1D getFi(DoubleMatrix1D X){
		double[] ret = new double[getMieq()];
		for(int i=0; i<getDim(); i++){
			ret[i] = -X.getQuick(i) + getLb().getQuick(i);
			ret[getDim() + i] = X.getQuick(i) - getUb().getQuick(i) ;
		}
		return F1.make(ret);
	}
	
	/**
	 * Inequality functions gradients values at X.
	 * This is -1 for all bounded lb and 1 for all bounded ub.
	 */
//	protected DoubleMatrix2D getGradFiOLD(DoubleMatrix1D X) {
//		double[][] ret = new double[getMieq()][getDim()];
//		int cntLb = 0;
//		int cntUb = 0;
//		for(int i=0; i<getDim(); i++){
//			if(boundedLb[i]){
//				ret[cntLb][i] = -1;
//				cntLb++;
//			}
//			if(boundedUb[i]){
//				ret[nOfBoundedLb + cntUb][i] = 1;
//				cntUb++;
//			}
//		}
//		return F2.make(ret);
//	}
	
	/**
	 * Inequality functions gradients values at X.
	 * This is -1 for all bounded lb and 1 for all bounded ub, and it is returned in a 2-rows compressed format.
	 * @return a 2xdim matrix, 1 row for the lower bounds and 1 row for the upper bounds gradients
	 */
	@Override
	protected DoubleMatrix2D getGradFi(DoubleMatrix1D X) {
//		double[][] ret = new double[2][getDim()];
//		double[] ret0 = new double[getDim()]; 
//		double[] ret1 = new double[getDim()];
//		for(int i=0; i<getDim(); i++){
//			if(boundedLb[i]){
//				ret0[i] = -1;
//			}
//			if(boundedUb[i]){
//				ret1[i] = 1;
//			}
//		}
//		ret[0] = ret0;
//		ret[1] = ret1;
//		return F2.make(ret);
		throw new RuntimeException("GradFi are not used for LP");
	}
	
	/**
	 * Inequality functions hessians values at X.
	 */
	@Override
	protected DoubleMatrix2D[] getHessFi(DoubleMatrix1D X){
		throw new RuntimeException("Hessians are null for LP");
	}
	
	/**
	 * rPri := Ax - b
	 */
	@Override
	protected DoubleMatrix1D rPri(DoubleMatrix1D X) {
		if(getMeq()==0){
			return F1.make(0);
		}
		return ColtUtils.zMult(getA(), X, getB(), -1);
	}
	
	/**
	 * Objective function domain.
	 */
	@Override
	protected boolean isInDomainF0(DoubleMatrix1D X) {
		return true;
	}
	
	/**
	 * Return the lower bounds for the problem. If the original lower bounds are null, the they are set to 
	 * the value of <i>minLBValue</i>. Otherwise, any lower bound is limited to the value of <i>minLBValue</i> 
	 */
	@Override
	protected DoubleMatrix1D getLb() {
		if (this.limitedLb == null) {
			if (super.getLb() == null) {
				this.limitedLb = F1.make(getC().size(), minLBValue);
			} else {
				this.limitedLb = F1.make(super.getLb().size());
				for (int i = 0; i < super.getLb().size(); i++) {
					double lbi = super.getLb().getQuick(i);
					if (lbi < minLBValue) {
						log.warn("the " + i +"-th lower bound was limited form "+lbi+" to the value of minLBValue: " + minLBValue);
						limitedLb.setQuick(i, minLBValue);
					} else {
						limitedLb.setQuick(i, lbi);
					}
				}
			}
		}
		return limitedLb;
	}
	
	/**
	 * Return the upper bounds for the problem. If the original upper bounds are null, the they are set to 
	 * the value of <i>maxUBValue</i>. Otherwise, any upper bound is limited to the value of <i>maxUBValue</i> 
	 */
	@Override
	protected DoubleMatrix1D getUb() {
		if (this.limitedUb == null) {
			if (super.getUb() == null) {
				this.limitedUb = F1.make(getC().size(), maxUBValue);
			} else {
				this.limitedUb = F1.make(super.getUb().size());
				for (int i = 0; i < super.getUb().size(); i++) {
					double ubi = super.getUb().getQuick(i);
					if (maxUBValue < ubi) {
						log.warn("the " + i +"-th upper bound was limited form "+ubi+" to the value of maxUBValue: " + maxUBValue);
						limitedUb.setQuick(i, maxUBValue);
					} else {
						limitedUb.setQuick(i, ubi);
					}
				}
			}
		}
		return limitedUb;
	}
	
	protected boolean isLbUnbounded(Double lb){
		//return Double.compare(unboundedLBValue, lb)==0;
		return Double.isNaN(lb);
	}
	
	protected boolean isUbUnbounded(Double ub){
		//return Double.compare(unboundedUBValue, ub)==0;
		return Double.isNaN(ub);
	}
	
//	/**
//	 * Check the duality conditions
//	 * GradF0(x) +	Sum[L[i] * GradFi(x] + A[T].V = 0
//	 * L[i] * fi(x) = 0,
//	 * 
//	 * NB1: that these are rDual and rCent
//	 * NB2: use this only for debugging purpose 
//	 * 
//	 * @return true if satisfied
//	 */
//	protected boolean checkDualityConditions(DoubleMatrix1D X, DoubleMatrix1D L, DoubleMatrix1D V){
//		//check GradF0(x) +	Sum[L[i] * GradFi(x] + A[T].V = 0
//		DoubleMatrix1D res1 = rDual(getGradF0(X), L, V);
//		//log.debug("duality res 1: " + ArrayUtils.toString(res1.toArray()));
//		double norm1 = ALG.normInfinity(res1);
//		log.debug("duality res norm 1: " + norm1);
//		
//		//check L[i] * fi(x) = 0
//		DoubleMatrix1D res2 = rCent(getFi(X), L, Double.POSITIVE_INFINITY);
//		//log.debug("duality res 2: " + ArrayUtils.toString(res2.toArray()));
//		double norm2 = ALG.normInfinity(res2);
//		log.debug("duality res norm 2: " + norm2);
//		
//		boolean ret = norm1 < getToleranceFeas() && norm2 < getToleranceFeas(); 
//		log.debug("checkDualityConditions: " + ret);
//		return ret;
//	}
	
//	/**
//	 * Check if the bound conditions on the optimal equality constraints Lagrangian coefficients are respected.
//	 * NB1: use this only for debugging purpose 
//	 * NB2: the bounds are checked within a tolerance, because the optimum is not exact, but within a tolerance
//	 * 
//	 * @param V the equality constraints Lagrangian multipliers relative to the optimal solution
//	 * @return true if satisfied
//	 */
//	protected boolean checkEqConstraintsLagrangianBounds(DoubleMatrix1D V){
//		boolean ret = true;
//		if(testPresolver!=null && getYlb()!=null && getYub()!=null){
//			//NB: the Lagrangian multipliers for eq. constraints used in this interior point method (v)
//			//are the opposite of the Lagrangian  multipliers for eq. constraints used in the presolver (y)
//			//and so Ylb<=y<=Yub becomes -Yub<=v<=-Ylb
//			
//			for(int i=0; i<getMeq() && ret; i++){
//				double vi = V.getQuick(i);
//				if(!testPresolver.isLBUnbounded(getYlb().getQuick(i))){
//					if(!testPresolver.isUBUnbounded(getYub().getQuick(i))){
//						ret = (vi + getToleranceFeas() >= -getYub().getQuick(i)) && (vi <= getToleranceFeas() - getYlb().getQuick(i));
//					}else{
//						ret = (vi <= getToleranceFeas() - getYlb().getQuick(i));
//					}
//				}else{
//					if(!testPresolver.isUBUnbounded(getYub().getQuick(i))){
//						ret = (vi + getToleranceFeas() >= -getYub().getQuick(i));
//					}else{
//						ret = true;
//					}
//				}
//			}
//		}
//		log.debug("checkEqConstraintsLagrangianBounds: " + ret);
//		return ret;
//	}
	
//	/**
//	 * Check if the bound conditions on the optimal inequality constraints Lagrangian coefficients are respected.
//	 * NB1: use this only for debugging purpose 
//	 * 
//	 * @param X the optimal solution
//	 * @param L the inequality constraints Lagrangian multipliers relative to the optimal solution
//	 * @return true if satisfied
//	 */
//	protected boolean checkIneqConstraintsLagrangianBounds(DoubleMatrix1D X, DoubleMatrix1D L){
//		boolean ret = true;
//		if(testPresolver!=null && getZlb()!=null && getZub()!=null){
//			//Zlb and Zub are bounds on the Lagrangian given by the presolver:
//			//Zlb<= L <=Zub, meaning that:
//			//zlb[i] and zub[i] are the bounds on the Lagrangian of the constraint associated with lb[i]<x[i]<ub[i] 
//			//note that zlb.size = zub.size = lb.size = ub.size (and = n of variables of the problem (= getDim())
//			//and that L.size = nOfBoundedLb + nOfBoundedUb (and in general < 2*n = 2*getDim())
//			//See Table 1 of Andersen & Andersen for the meaning of the values of Zlb and Zub
//			int cntLB = 0;
//			int cntUB = 0;
//			for(int i=0; i<getDim() && ret; i++){
//				
//				if(getZlb().getQuick(i)>0){
//					//if we already know that zlb > 0, then the constraint must be active
//					throw new RuntimeException("just to see");
//				}
//				
//				double xi = X.getQuick(i);
//				double zlbi = (testPresolver.isLBUnbounded(getZlb().getQuick(i)))? -Double.MAX_VALUE : getZlb().getQuick(i);//@TODO: this is wrong, because zlbi can be negative, but L is always positive 
//				double zubi = (testPresolver.isUBUnbounded(getZub().getQuick(i)))?  Double.MAX_VALUE : getZub().getQuick(i);
//				//there is a inequality constraint (and its Lagrangian) relative to this lower bound
//				double lcnt = L.getQuick(cntLB);
//				ret = (zlbi <= lcnt) && (lcnt <= zubi);
//				double delta = Math.abs(xi - getLb().get(i)); 
//				if(delta > getTolerance()){
//					//if the optimal xi is not on its lower bound, we must have L[i] = 0
//					//note that this is already checked by rCent norm condition
//					ret &= (lcnt*delta < getTolerance());
//				}
//				cntLB++;
//				if(ret){
//					//there is a inequality constraint (and its Lagrangian) relative to this upper bound
//					lcnt = L.getQuick(getDim() + cntUB);
//					ret = (zlbi <= lcnt) && (lcnt <= zubi);
//					delta = Math.abs(getUb().get(i) - xi); 
//					if(delta > getTolerance()){
//						//if the optimal xi is not on its upper bound, we must have L[i] = 0
//						//note that this is already checked by rCent norm condition
//						ret &= (lcnt*delta < getTolerance());
//					}
//					cntUB++;
//				}
//			}
//		}
//		log.debug("checkIneqConstraintsLagrangianBounds: " + ret);
//		return ret;
//	}
}
