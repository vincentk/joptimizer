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
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import cern.jet.math.Functions;

import com.joptimizer.solvers.UpperDiagonalHKKTSolver;
import com.joptimizer.util.ColtUtils;
import com.joptimizer.util.Utils;

/**
 * Basic Phase I Method form LP problems (implemented as a Primal-Dual Method).
 * 
 * @see "S.Boyd and L.Vandenberghe, Convex Optimization, p. 579"
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class BasicPhaseILPPDM {
	
	private LPPrimalDualMethod originalProblem;
	private int originalDim =-1;
	private int dim =-1;
	private Algebra ALG = Algebra.DEFAULT;
	private DoubleFactory1D F1 = DoubleFactory1D.dense;
//	private DoubleFactory2D F2 = DoubleFactory2D.dense;
	private Log log = LogFactory.getLog(this.getClass().getName());

	public BasicPhaseILPPDM(LPPrimalDualMethod originalProblem) {
		this.originalProblem = originalProblem;
		this.originalDim = originalProblem.getDim();
		this.dim = originalProblem.getDim()+1;//variable Y=(X, s)
	}
	
	public DoubleMatrix1D findFeasibleInitialPoint() throws Exception{
		log.debug("findFeasibleInitialPoint");
		
		LPOptimizationRequest or = new LPOptimizationRequest();
		
		//objective function: s
		DoubleMatrix1D c = F1.make(dim);
		c.set(dim-1, 1.);
		or.setC(c);
//		double[] lb = new double[dim]; 
//		double[] ub = new double[dim];
////		Arrays.fill(lb, LPPrimalDualMethod.DEFAULT_MIN_LOWER_BOUND);
////		Arrays.fill(ub, LPPrimalDualMethod.DEFAULT_MAX_UPPER_BOUND);
//		for(int i=0; i<originalProblem.getDim(); i++){
//			lb[i] = originalProblem.getLb().getQuick(i) + LPPrimalDualMethod.DEFAULT_MIN_LOWER_BOUND; 
//			ub[i] = originalProblem.getUb().getQuick(i) + LPPrimalDualMethod.DEFAULT_MAX_UPPER_BOUND;
//		}
//		or.setLb(lb);
//		or.setUb(ub);
		or.setToleranceFeas(originalProblem.getToleranceFeas());
		or.setTolerance(originalProblem.getTolerance());
		
		//@TODO: remove this
		//or.setToleranceKKT(originalProblem.getToleranceKKT());
		//or.setCheckKKTSolutionAccuracy(originalProblem.isCheckKKTSolutionAccuracy());
		
	  // Equality constraints: add a final zeroes column
		DoubleMatrix2D AEorig = originalProblem.getA();
		DoubleMatrix1D BEorig = originalProblem.getB();
		if(AEorig!=null){ 
			DoubleFactory2D F2 = (AEorig instanceof SparseDoubleMatrix2D)? DoubleFactory2D.sparse : DoubleFactory2D.dense;
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
		double s0 = initialPoint.getQuick(dim-1);
		for(int i=0; i<originalFiX0.size(); i++){
			//initialPoint.set(dim-1, Math.max(initialPoint.get(dim-1), originalFiX0.get(i)+Math.sqrt(originalProblem.getToleranceFeas())));
			s0 = Math.max(s0, originalFiX0.get(i) * 1.1);
		}
		initialPoint.setQuick(dim-1, s0);
		or.setInitialPoint(initialPoint.toArray());
//		double minLBValue =  Double.MAX_VALUE;
//		double maxUBValue = -Double.MAX_VALUE;
//		double[] lb = new double[dim - 1];//the new variable s is not bounded
//		double[] ub = new double[dim - 1];//the new variable s is not bounded
//		for(int i=0; i<originalProblem.getDim(); i++){
//			double lbi = originalProblem.getLb().getQuick(i) - 2*s0;
//			lb[i] = lbi;//bounds are less tight
//			minLBValue = Math.min(minLBValue, lbi);
//			double ubi = originalProblem.getUb().getQuick(i) + 2*s0;
//			maxUBValue = Math.max(maxUBValue, ubi);
//			ub[i] = ubi;//bounds are less tight
//		}
//		lb[dim-1] = LPPrimalDualMethod.DEFAULT_MIN_LOWER_BOUND;
//		ub[dim-1] = 2*s0;
		//or.setLb(lb);
		//or.setUb(ub);
		
		//optimization
		//LPPrimalDualMethod opt = new PhaseILPPrimalDualMethod(Math.min(minLBValue, lb[dim-1]), Math.max(maxUBValue, ub[dim-1]));
		LPPrimalDualMethod opt = new PhaseILPPrimalDualMethod(2*s0);//the bounds of this problem are not used
		//opt.setKKTSolver(new SparseKKTSolver());
		opt.setKKTSolver(new UpperDiagonalHKKTSolver(originalDim));
		opt.setOptimizationRequest(or);
		if(opt.optimizePresolvedStandardLP() == OptimizationResponse.FAILED){
			throw new Exception("Failed to find an initial feasible point");
		}
		LPOptimizationResponse response = opt.getLPOptimizationResponse();
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
	
	private class PhaseILPPrimalDualMethod extends LPPrimalDualMethod{
	
		private double maxSValue;
		
		/**
		 * We need this constructor because this class is used to solve a problem
		 * that is not strictly standard (the introduction of the variable s)
		 * transforms the original bounds in classical inequalities constraints.
		 * This is a little forcing but we can go on anyway. 
		 */
		PhaseILPPrimalDualMethod(double maxSValue){
			//super(minLBValue, maxUBValue);
			this.maxSValue = maxSValue;
			this.dim = originalProblem.getDim()+1;
			this.meq = originalProblem.getMeq();
			this.mieq = originalProblem.getMieq() + 1;//the variable s has an upper bound
		}
		
		/**
		 * Inequality functions values at X.
		 * This is (-x+lb-s) for all original bounded lb 
		 * and (x-ub-s) for all original bounded ub
		 * ans (s - maxSValue)
		 */
		@Override
		protected DoubleMatrix1D getFi(DoubleMatrix1D Xs){
			double[] ret = new double[originalProblem.getMieq() + 1];
			double s = Xs.getQuick(getDim() -1);
			//loop on the original bounded bounds
			for(int i=0; i<originalProblem.getDim(); i++){
				ret[i] = -Xs.getQuick(i) -s + originalProblem.getLb().getQuick(i);//-x -s +lb < 0
				ret[originalProblem.getDim() + i] = Xs.getQuick(i) -s - originalProblem.getUb().getQuick(i) ;//x -s -ub < 0
			}
			ret[originalProblem.getMieq()] = s - this.maxSValue;
			       
			return F1.make(ret);
		}
		
		/**
		 * Inequality functions gradients values at X.
		 * This is -1,...,-1 for all original bounded lb and 1,...,-1 for all original bounded ub.
		 */
//		@Override
//		protected DoubleMatrix2D getGradFiOLD(DoubleMatrix1D Xs) {
//			double[][] ret = new double[originalProblem.getMieq()][getDim()];
//			int cntLb = 0;
//			int cntUb = 0;
//			//loop on the original bounded bounds
//			for(int i=0; i<originalProblem.getDim(); i++){
//				if(originalProblem.boundedLb[i]){
//					//grad(-x -s)
//					ret[cntLb][i] = -1;
//					ret[cntLb][getDim()-1] = -1;
//					cntLb++;
//				}
//				if(originalProblem.boundedUb[i]){
//					//grad(-x -s)
//					ret[originalProblem.nOfBoundedLb + cntUb][i] = 1;
//					ret[originalProblem.nOfBoundedLb + cntUb][getDim()-1] = -1;
//					cntUb++;
//				}
//			}
//			return F2.make(ret);
//		}
		
		/**
		 * Inequality functions gradients values at X.
		 * This is -1,...,-1 for all original bounded lb 
		 * and 1,...,-1 for all original bounded ub
		 * and 1 for grad(s - maxSValue)
		 * The number of inequalities is the same as the original problem + 1 
		 * because the new variable s upper bounded.
		 * Returns the result in a compressed 2-rows matrix.
		 * @return a 3xdim matrix, 1 row for the lower bounds and 1 row for the upper bounds gradients
		 */
//		@Override
//		protected DoubleMatrix2D getGradFi(DoubleMatrix1D X) {
//			double[][] ret = new double[2][getDim()];
//			double[] ret0 = new double[getDim()]; 
//			double[] ret1 = new double[getDim()];
//			double[] ret2 = new double[getDim()];
//			//loop on the original bounded bounds
//			for(int i=0; i<originalProblem.getDim(); i++){
//				//grad(-x -s)
//				ret0[i] = -1;
//				ret0[getDim()-1] = -1;
//				//grad(-x -s)
//				ret1[i] = 1;
//				ret1[getDim()-1] = -1;
//			}
//			ret2[getDim()-1] = 1;
//			
//			ret[0] = ret0;
//			ret[1] = ret1;
//			ret[2] = ret2;
//			return F2.make(ret);
//		}
		
		/**
		 * {@inheritDoc}
		 */
		@Override
		protected DoubleMatrix1D rDual(DoubleMatrix1D gradF0X, DoubleMatrix1D L, DoubleMatrix1D V) {
			//m1 = GradFiX[T].L + gradF0X
			
		  //take into account the inequalities given by the original bounds
			//part 1
			DoubleMatrix1D m1 = F1.make(getDim());
			for(int i=0; i<originalProblem.getDim(); i++){
				double m = 0;
				m += - L.getQuick(i);
				m +=   L.getQuick(originalProblem.getDim() + i);
				m1.setQuick(i, m);
			}
			
		  //part 2
			//take into account the last column of -1 terms
			double d = 0d;
			for(int i=0; i<originalProblem.getMieq(); i++){
				d += L.getQuick(i);
			}
			m1.setQuick(getDim()-1, m1.getQuick(getDim()-1) - d);
			
			//take into account gradF0X
			m1 = m1.assign(gradF0X, Functions.plus);
			
			//take into account the upper bound on s: s < maxSValue
			double m = L.getQuick(L.size() - 1);
			m1.setQuick(getDim() - 1, m1.getQuick(getDim() - 1) + m);
			
			if(getMeq()==0){
				return m1;
			}
			return ColtUtils.zMultTranspose(getA(), V, m1, 1.);
		}
		
		/**
		 * {@inheritDoc}
		 */
		@Override
		protected DoubleMatrix1D gradSum(double t, DoubleMatrix1D fiX) {
			DoubleMatrix1D gradSum = F1.make(getDim());
			double ddim = 0;
			for (int i = 0; i < originalProblem.getDim(); i++) {
				double di = 0;
				double fiXL = fiX.getQuick(i);
				di += 1. / (t * fiXL);
				ddim += 1. / (t * fiXL);
				double fiXU = fiX.getQuick(originalProblem.getDim() + i);
				di += -1. / (t * fiXU);
				ddim += 1. / (t * fiXU);
				gradSum.setQuick(i, di);
			}
			double fsU = fiX.getQuick(fiX.size() - 1);
			ddim += -1. / (t * fsU);//given by s < maxSValue 
			gradSum.setQuick(getDim() - 1, ddim);

			return gradSum;
		}
		
		/**
		 * This is the third addendum of (11.56).
		 * Returns only the subdiagonal elements in a sparse matrix.
		 * Remember that the inequality functions are f[i] - s, 
		 * (i.e. (-x+lb-s) for all original bounded lb and (x-ub-s) for all original bounded ub)
		 * plus the upper bound for s
		 */
		@Override
		protected DoubleMatrix2D GradLSum(DoubleMatrix1D L, DoubleMatrix1D fiX) {
			SparseDoubleMatrix2D GradLSum = new SparseDoubleMatrix2D(getDim(), getDim());
			double ddimdim = 0;
			//counts for the original bound terms
			for(int i=0; i<originalProblem.getDim(); i++){
				double dii = 0;
				double didim = 0;
				double fiXL = fiX.getQuick(i);
				double LI = L.getQuick(i);
				double LIFiXL = - LI / fiXL;
				dii     += LIFiXL;
				didim   += LIFiXL;
				ddimdim += LIFiXL;
				double LDimI = L.getQuick(originalProblem.getDim() + i);
				double fiXU = fiX.getQuick(originalProblem.getDim() + i);
				double LDimIFiXU = - LDimI / fiXU;
				dii     += LDimIFiXU;
				didim   -= LDimIFiXU;
				ddimdim += LDimIFiXU;
				GradLSum.setQuick(         i,          i, dii);
//				if(Math.abs(didim) < 1.E-11){
//					log.warn("didim: " + didim);
//				}
				GradLSum.setQuick(getDim()-1,          i, didim);
				//GradLSum.setQuick(         i, getDim()-1, didim);
			}
			
			//counts for the new upper bound on s
			double LDimS = L.getQuick(L.size() - 1);
			double fsU = fiX.getQuick(fiX.size() - 1);
			double LDimSFsU = - LDimS / fsU;
			ddimdim += LDimSFsU;
			
			GradLSum.setQuick(getDim()-1, getDim()-1, ddimdim);
			
			return GradLSum;
		}
		
		/**
		 * {@inheritDoc}
		 */
		@Override
		protected DoubleMatrix1D gradFiStepX(DoubleMatrix1D stepX){
			
			DoubleMatrix1D ret = F1.make(getMieq());
			for(int i=0; i<originalProblem.getDim(); i++){
				ret.setQuick(                         i, - stepX.getQuick(i) - stepX.getQuick(getDim()-1));
				ret.setQuick(originalProblem.getDim()+i,   stepX.getQuick(i) - stepX.getQuick(getDim()-1));
			}
			//counts for the new upper bound on s
			ret.setQuick(getMieq()-1, stepX.getQuick(getDim()-1));
			
			return ret;
		}
		
		@Override
		protected boolean checkCustomExitConditions(DoubleMatrix1D Xs){
			DoubleMatrix1D X = Xs.viewPart(0, getDim()-1);
			DoubleMatrix1D ineqX = originalProblem.getFi(X);
			int ineqMaxIndex = Utils.getMaxIndex(ineqX);
			//log.debug("ineqMaxIndex: " + ineqMaxIndex);
			//log.debug("ineqMaxValue: " + ineqX.get(ineqMaxIndex));
			
			boolean isInternal = (ineqX.get(ineqMaxIndex) + getTolerance() <0) || Xs.get(Xs.size()-1)<0;
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
		
		/**
		 * Return the lower bounds for the problem. 
		 * It is always null because only the bounds of the original problem are used. 
		 */
		@Override
		protected DoubleMatrix1D getLb() {
			return null;
		}
		
		/**
		 * Return the lower bounds for the problem. 
		 * It is always null because only the bounds of the original problem are used. 
		 */
		@Override
		protected DoubleMatrix1D getUb() {
			return null;
		}
		
		/**
		 * {@inheritDoc}
		 */
		protected boolean checkDualityConditions(DoubleMatrix1D X, DoubleMatrix1D L, DoubleMatrix1D V){
			//return true because the Phase I solution does not have to give a convergence (a real minimum of its objective function)
			return true;
		}
		
		/**
		 * {@inheritDoc}
		 */
		protected boolean checkEqConstraintsLagrangianBounds(DoubleMatrix1D V){
			//return true because the Phase I solution does not have to be a convergence (a real minimum of its objective function)
			return true;
		}
		
		/**
		 * {@inheritDoc}
		 */
		protected boolean checkIneqConstraintsLagrangianBounds(DoubleMatrix1D X, DoubleMatrix1D L){
			//return true because the Phase I solution does not have to be a convergence (a real minimum of its objective function)
			return true;
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
