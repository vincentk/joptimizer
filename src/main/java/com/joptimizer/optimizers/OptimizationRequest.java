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

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

import com.joptimizer.functions.ConvexMultivariateRealFunction;


/**
 * Optimization problem.
 * Setting the field's values you define an optimization problem.
 * 
 * @see "S.Boyd and L.Vandenberghe, Convex Optimization"
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class OptimizationRequest {

	/**
	 * Maximum number of iteration in the search algorithm.
	 * Not mandatory, default is provided.
	 */
	private int maxIteration = JOptimizer.DEFAULT_MAX_ITERATION;
	
	/**
	 * Tolerance for the minimum value.
	 * Not mandatory, default is provided.
	 * NOTE: as a golden rule, do not ask for more accuracy than you really need.
	 * @see "Convex Optimization, p. 11.7.3"
	 */
	private double tolerance = JOptimizer.DEFAULT_TOLERANCE;
	
	/**
	 * Tolerance for the constraints satisfaction.
	 * Not mandatory, default is provided.
	 * NOTE: as a golden rule, do not ask for more accuracy than you really need.
	 * @see "Convex Optimization, p. 11.7.3"
	 */
	private double toleranceFeas = JOptimizer.DEFAULT_FEASIBILITY_TOLERANCE;
	
	/**
	 * Tolerance for inner iterations in the barrier-method.
	 * NB: it makes sense only for barrier method.
	 * Not mandatory, default is provided.
	 * NOTE: as a golden rule, do not ask for more accuracy than you really need.
	 * @see "Convex Optimization, p. 11.7.3"
	 */
	private double toleranceInnerStep = JOptimizer.DEFAULT_TOLERANCE_INNER_STEP;
	
	/**
	 * Calibration parameter for line search.
	 * Not mandatory, default is provided.
	 * @see "Convex Optimization, p. 11.7.3"
	 */
	private double alpha = JOptimizer.DEFAULT_ALPHA;
	
	/**
	 * Calibration parameter for line search.
	 * Not mandatory, default is provided.
	 * @see "Convex Optimization, p. 11.7.3"
	 */
	private double beta = JOptimizer.DEFAULT_BETA;
	
	/**
	 * Calibration parameter for line search.
	 * Not mandatory, default is provided.
	 * @see "Convex Optimization, p. 11.7.3"
	 */
	private double mu = JOptimizer.DEFAULT_MU;
	
	/**
	 * Activate progress condition check during iterations.
     * If true, a progress in the relevant algorithm norms is required during iterations,
     * otherwise the iteration will be exited with a warning (and solution
     * must be manually checked against the desired accuracy).
	 * Not mandatory, default is provided.
	 * @see "Convex Optimization, p. 11.7.3"
	 */
	private boolean checkProgressConditions = false;
	
	/**
	 * Check the accuracy of the solution of KKT system during iterations.
     * If true, every inversion of the system must have an accuracy that satisfy
     * the given toleranceKKT.
	 * Not mandatory, default is provided.
	 * @see "Convex Optimization, p. 11.7.3"
	 */
	private boolean checkKKTSolutionAccuracy = false;
	
	/**
	 * Acceptable tolerance for KKT system resolution.
	 * Not mandatory, default is provided.
	 */
	private double toleranceKKT = JOptimizer.DEFAULT_KKT_TOLERANCE;
	
	/**
	 * Should matrix rescaling be disabled?
	 * Rescaling is involved in LP presolving and in the solution of the KKT systems associated with the problem.
	 * It is an heuristic process, in some situations it could be useful to turn off this feature.
	 */
	private boolean rescalingDisabled = false;
	
	/**
	 * The objective function to minimize.
	 * Mandatory.
	 */
	private ConvexMultivariateRealFunction f0;
	
	/**
	 * Feasible starting point for the minimum search.
	 * It must be feasible.
	 * Not mandatory.
	 */
	private DoubleMatrix1D initialPoint = null;
	
	/**
	 * Not-feasible starting point for the minimum search.
	 * It does not have to be feasible. This provide the possibility to give the algorithm
	 * a starting point even if it does not satisfies inequality constraints. The algorithm
	 * will search a feasible point starting from here.
	 * Not mandatory.
	 */
	private DoubleMatrix1D notFeasibleInitialPoint = null;
	
	/**
	 * Starting point for the Lagrangian multipliers.
	 * Must have the same dimension of the inequalities constraints array.
	 * Not mandatory, but very useful in some case.
	 */
	private DoubleMatrix1D initialLagrangian = null;
	
	/**
	 * Equalities constraints matrix.
	 * Must be rank(A) < dimension of the variable.
	 * Not mandatory.
	 * @see "Convex Optimization, 11.1"
	 */
	private DoubleMatrix2D A = null;
	
	/**
	 * Equalities constraints vector.
	 * Not mandatory.
	 * @see "Convex Optimization, 11.1"
	 */
	private DoubleMatrix1D b = null;
	
	/**
	 * Inequalities constraints array.
	 * Not mandatory.
	 * @see "Convex Optimization, 11.1"
	 */
	private ConvexMultivariateRealFunction[] fi;
	
	/**
	 * The chosen interior-point method.
	 * Must be barrier-method or primal-dual method. 
	 */
	private String interiorPointMethod = JOptimizer.PRIMAL_DUAL_METHOD;

	public int getMaxIteration() {
		return maxIteration;
	}

	public void setMaxIteration(int maxIteration) {
		this.maxIteration = maxIteration;
	}
	
	double getTolerance() {
		return tolerance;
	}

	public void setTolerance(double tolerance) {
		this.tolerance = tolerance;
	}

	public double getToleranceFeas() {
		return toleranceFeas;
	}

	public void setToleranceFeas(double toleranceFeas) {
		this.toleranceFeas = toleranceFeas;
	}

	public double getToleranceInnerStep() {
		return toleranceInnerStep;
	}

	public void setToleranceInnerStep(double toleranceInnerStep) {
		this.toleranceInnerStep = toleranceInnerStep;
	}

	public double getAlpha() {
		return alpha;
	}

	public void setAlpha(double alpha) {
		this.alpha = alpha;
	}

	public double getBeta() {
		return beta;
	}

	public void setBeta(double beta) {
		this.beta = beta;
	}

	public double getMu() {
		return mu;
	}

	public void setMu(double mu) {
		this.mu = mu;
	}

	public boolean isCheckProgressConditions() {
		return checkProgressConditions;
	}

	public void setCheckProgressConditions(boolean checkProgressConditions) {
		this.checkProgressConditions = checkProgressConditions;
	}

	public boolean isCheckKKTSolutionAccuracy() {
		return checkKKTSolutionAccuracy;
	}

	public void setCheckKKTSolutionAccuracy(boolean checkKKTSolutionAccuracy) {
		this.checkKKTSolutionAccuracy = checkKKTSolutionAccuracy;
	}

	public double getToleranceKKT() {
		return toleranceKKT;
	}

	public void setToleranceKKT(double toleranceKKT) {
		this.toleranceKKT = toleranceKKT;
	}

	public ConvexMultivariateRealFunction getF0() {
		return f0;
	}

	public void setF0(ConvexMultivariateRealFunction f0) {
		this.f0 = f0;
	}

	public DoubleMatrix1D getInitialPoint() {
		return initialPoint;
	}

	public void setInitialPoint(double[] initialPoint) {
		this.initialPoint = DoubleFactory1D.dense.make(initialPoint);
	}

	public DoubleMatrix1D getNotFeasibleInitialPoint() {
		return notFeasibleInitialPoint;
	}

	public void setNotFeasibleInitialPoint(double[] notFeasibleInitialPoint) {
		this.notFeasibleInitialPoint = DoubleFactory1D.dense.make(notFeasibleInitialPoint);
	}

	public DoubleMatrix1D getInitialLagrangian() {
		return initialLagrangian;
	}

	public void setInitialLagrangian(double[] initialLagrangian) {
		this.initialLagrangian = DoubleFactory1D.dense.make(initialLagrangian);
	}

	public DoubleMatrix2D getA() {
		return A;
	}

	public void setA(double[][] a) {
		if(a!=null){
			A = DoubleFactory2D.dense.make(a);
		}
	}
	
	public void setA(DoubleMatrix2D A) {
		this.A = A;
	}

	public DoubleMatrix1D getB() {
		return b;
	}

	public void setB(double[] b) {
		if(b!=null){
			this.b = DoubleFactory1D.dense.make(b);
		}
	}
	
	public void setB(DoubleMatrix1D b) {
		this.b = b;
	}

	public ConvexMultivariateRealFunction[] getFi() {
		return fi;
	}

	public void setFi(ConvexMultivariateRealFunction[] fi) {
		this.fi = fi;
	}

	public String getInteriorPointMethod() {
		return interiorPointMethod;
	}

	public void setInteriorPointMethod(String interiorPointMethod) {
		this.interiorPointMethod = interiorPointMethod;
	}
	
	public void setRescalingDisabled(boolean rescalingDisabled) {
		this.rescalingDisabled = rescalingDisabled;
	}

	public boolean isRescalingDisabled() {
		return rescalingDisabled;
	}
}
