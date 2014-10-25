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

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

import com.joptimizer.functions.ConvexMultivariateRealFunction;

/**
 * Linear optimization problem.
 * The general form is:
 * 
 * min(c) s.t.
 * <br>G.x < h
 * <br>A.x = b
 * <br>lb <= x <= ub
 * 
 * Lower and upper bounds can be stated for a more user friendly usage.
 * 
 * @see "S.Boyd and L.Vandenberghe, Convex Optimization"
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class LPOptimizationRequest extends OptimizationRequest{

	/**
	 * Linear objective function.
	 */
	private DoubleMatrix1D c;
	
	/**
	 * Linear inequalities constraints matrix.
	 */
	private DoubleMatrix2D G;
	
	/**
	 * Linear inequalities constraints coefficients.
	 */
	private DoubleMatrix1D h;
	
	/**
	 * Lower bounds.
	 */
	private DoubleMatrix1D lb;
	
	/**
	 * Upper bounds.
	 */
	private DoubleMatrix1D ub;
	
	/**
	 * Lagrangian lower bounds for linear constraints (A rows).
	 */
	private DoubleMatrix1D ylb;
	
	/**
	 * Lagrangian upper bounds for linear constraints (A rows).
	 */
	private DoubleMatrix1D yub;
	
	/**
	 * Lagrangian lower bounds for lb constraints.
	 */
	private DoubleMatrix1D zlb;
	
	/**
	 * Lagrangian upper bounds for ub constraints.
	 */
	private DoubleMatrix1D zub;
	
	/**
	 * Should LP presolving be disabled?
	 */
	private boolean presolvingDisabled = false;
	
	/**
	 * If true, no method for making normal equations sparser will be applied during the presolving phase.
	 * @see Jacek Gondzio "Presolve analysis of linear programs prior to applying an interior point method", 3
	 */
	private boolean avoidPresolvingIncreaseSparsity = false;

	/**
	 * If true, no methods that cause fill-in in the original matrices will be called during the presolving phase.
	 */
	private boolean avoidPresolvingFillIn = false;
	
//	/**
//	 * Perform duality condition check on the optimal solution.
//	 * @TODO: move to parent class
//	 */
//	private boolean checkOptimalDualityConditions = false;
	
	/**
	 * Check if the bound conditions on the optimal equality constraints Lagrangian coefficients are respected.
	 */
	private boolean checkOptimalLagrangianBounds = false;
	
	/**
	 * Dump the problem to the log file?
	 */
	private boolean dumpProblem = false;
	
	public DoubleMatrix1D getC() {
		return c;
	}

	public void setC(double[] c) {
		if(c!=null){
			setC(DoubleFactory1D.dense.make(c));
		}
	}
	
	public void setC(DoubleMatrix1D c) {
		this.c = c;
	}
	
	public DoubleMatrix2D getG() {
		return G;
	}

	public void setG(double[][] G) {
		if(G!=null){
			setG(DoubleFactory2D.dense.make(G));
		}
	}
	
	public void setG(DoubleMatrix2D G) {
		this.G = G;
	}

	public DoubleMatrix1D getH() {
		return h;
	}

	public void setH(double[] h) {
		if(h!=null){
			setH(DoubleFactory1D.dense.make(h));
		}
	}
	
	public void setH(DoubleMatrix1D h) {
		this.h = h;
	}
	
	public DoubleMatrix1D getLb() {
		return this.lb;
	}
	
	public void setLb(double[] lb) {
		if(lb!=null){
			setLb(DoubleFactory1D.dense.make(lb));
		}
	}
	
	public void setLb(DoubleMatrix1D lb) {
		for (int i = 0; i < lb.size(); i++) {
			double lbi = lb.getQuick(i);
			if (Double.isNaN(lbi) || Double.isInfinite(lbi)) {
				throw new IllegalArgumentException("The lower bounds can not be set to Double.NaN or Double.INFINITY");
			}
		}
		this.lb = lb;
	}
	
	public DoubleMatrix1D getUb() {
		return this.ub;
	}
	
	public void setUb(double[] ub) {
		if(ub!=null){
			setUb(DoubleFactory1D.dense.make(ub));
		}
	}
	
	public void setUb(DoubleMatrix1D ub) {
		for (int i = 0; i < ub.size(); i++) {
			double ubi = ub.getQuick(i);
			if (Double.isNaN(ubi) || Double.isInfinite(ubi)) {
				throw new IllegalArgumentException("The upper bounds can not be set to Double.NaN or Double.INFINITY");
			}
		}
		this.ub = ub;
	}
	
	public DoubleMatrix1D getYlb() {
		return this.ylb;
	}
	
	public void setYlb(double[] ylb) {
		if(ylb!=null){
			setYlb(DoubleFactory1D.dense.make(ylb));
		}
	}
	
	public void setYlb(DoubleMatrix1D ylb) {
		this.ylb = ylb;
	}
	
	public DoubleMatrix1D getYub() {
		return this.yub;
	}
	
	public void setYub(double[] yub) {
		if(yub!=null){
			setYub(DoubleFactory1D.dense.make(yub));
		}
	}
	
	public void setYub(DoubleMatrix1D yub) {
		this.yub = yub;
	}
	
	public DoubleMatrix1D getZlb() {
		return this.zlb;
	}
	
	public void setZlb(double[] zlb) {
		if(zlb!=null){
			setZlb(DoubleFactory1D.dense.make(zlb));
		}
	}
	
	public void setZlb(DoubleMatrix1D zlb) {
		this.zlb = zlb;
	}
	
	public DoubleMatrix1D getZub() {
		return this.zub;
	}
	
	public void setZub(double[] zub) {
		if(zub!=null){
			setZub(DoubleFactory1D.dense.make(zub));
		}
	}
	
	public void setZub(DoubleMatrix1D zub) {
		this.zub = zub;
	}
	
	public boolean isAvoidPresolvingIncreaseSparsity() {
		return avoidPresolvingIncreaseSparsity;
	}

	public void setAvoidPresolvingIncreaseSparsity(
			boolean avoidPresolvingIncreaseSparsity) {
		this.avoidPresolvingIncreaseSparsity = avoidPresolvingIncreaseSparsity;
	}

	public boolean isAvoidPresolvingFillIn() {
		return avoidPresolvingFillIn;
	}

	public void setAvoidPresolvingFillIn(boolean avoidPresolvingFillIn) {
		this.avoidPresolvingFillIn = avoidPresolvingFillIn;
	}
	
	public boolean isPresolvingDisabled() {
		return this.presolvingDisabled;
	}
	
	public void setPresolvingDisabled(boolean presolvingDisabled) {
		this.presolvingDisabled = presolvingDisabled;
	}
	
//	public boolean isCheckOptimalDualityConditions() {
//		return this.checkOptimalDualityConditions;
//	}
//	
//	public void setCheckOptimalDualityConditions(boolean checkOptimalDualityConditions) {
//		this.checkOptimalDualityConditions = checkOptimalDualityConditions;
//	}
	
	public boolean isCheckOptimalLagrangianBounds() {
		return this.checkOptimalLagrangianBounds;
	}
	
	public void setCheckOptimalLagrangianBounds(boolean checkOptimalLagrangianBounds) {
		this.checkOptimalLagrangianBounds = checkOptimalLagrangianBounds;
	}
	
	public boolean isDumpProblem() {
		return this.dumpProblem;
	}
	
	public void setDumpProblem(boolean dumpProblem) {
		this.dumpProblem = dumpProblem;
	}

	@Override
	public void setF0(ConvexMultivariateRealFunction f0) {
		throw new UnsupportedOperationException("Use the matrix formulation for this linear problem");
	}
	
	@Override
	public void setFi(ConvexMultivariateRealFunction[] fi) {
		throw new UnsupportedOperationException("Use the matrix formulation for this linear problem");
	}
	
	@Override
	public String toString(){
		try{
			StringBuffer sb = new StringBuffer();
			sb.append(this.getClass().getName() + ": ");
			sb.append("\nmin(c) s.t.");
			if(getG() != null && getG().rows()>0){
				sb.append("\nG.x < h");
			}
			if(getA()!=null && getA().rows()>0){
				sb.append("\nA.x = b");
			}
			if(getLb()!=null && getUb()!=null){
				sb.append("\nlb <= x <= ub");
			}else if(getLb()!=null){
				sb.append("\nlb <= x");
			}else if(getUb()!=null){
				sb.append("\nx <= ub");
			}
			sb.append("\nc: " + ArrayUtils.toString(c.toArray()));
			if(G != null){
				sb.append("\nG: " + ArrayUtils.toString(G.toArray()));
				sb.append("\nh: " + ArrayUtils.toString(h.toArray()));
			}
			if(getA()!=null){
				sb.append("\nA: " + ArrayUtils.toString(getA().toArray()));
				sb.append("\nb: " + ArrayUtils.toString(getB().toArray()));
			}
			if(getLb()!=null){
				sb.append("\nlb: " + ArrayUtils.toString(getLb().toArray()));
			}
			if(getUb()!=null){
				sb.append("\nub: " + ArrayUtils.toString(getUb().toArray()));
			}
			if(getYlb()!=null){
				sb.append("\nylb: " + ArrayUtils.toString(getYlb().toArray()));
			}
			if(getYub()!=null){
				sb.append("\nyub: " + ArrayUtils.toString(getYub().toArray()));
			}
			if(getZlb()!=null){
				sb.append("\nzlb: " + ArrayUtils.toString(getZlb().toArray()));
			}
			if(getZub()!=null){
				sb.append("\nzub: " + ArrayUtils.toString(getZub().toArray()));
			}
			
			return sb.toString();
		}catch(Exception e){
			
			return "";
		}
	}
	
	public LPOptimizationRequest cloneMe(){
		LPOptimizationRequest clonedLPRequest = new LPOptimizationRequest();
		clonedLPRequest.setToleranceFeas(getToleranceFeas());
		clonedLPRequest.setDumpProblem(isDumpProblem());
		clonedLPRequest.setPresolvingDisabled(isPresolvingDisabled());
		clonedLPRequest.setRescalingDisabled(isRescalingDisabled());
		clonedLPRequest.setAvoidPresolvingFillIn(isAvoidPresolvingFillIn());
		clonedLPRequest.setAvoidPresolvingIncreaseSparsity(isAvoidPresolvingIncreaseSparsity());
		//clonedLPRequest.setCheckOptimalDualityConditions(isCheckOptimalDualityConditions());
		clonedLPRequest.setCheckOptimalLagrangianBounds(isCheckOptimalLagrangianBounds());
		clonedLPRequest.setAlpha(getAlpha());
		clonedLPRequest.setBeta(getBeta());
		clonedLPRequest.setCheckKKTSolutionAccuracy(isCheckKKTSolutionAccuracy());
		clonedLPRequest.setToleranceKKT(getToleranceKKT());
		clonedLPRequest.setCheckProgressConditions(isCheckProgressConditions());
		clonedLPRequest.setMaxIteration(getMaxIteration());
		clonedLPRequest.setMu(getMu());
		clonedLPRequest.setTolerance(getTolerance());
		
		return clonedLPRequest;
	}
}
