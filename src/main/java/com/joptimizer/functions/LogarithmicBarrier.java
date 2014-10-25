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
package com.joptimizer.functions;

import com.joptimizer.util.Utils;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import cern.jet.math.Functions;
import cern.jet.math.Mult;

/**
 * Default barrier function for the barrier method algorithm.
 * <br>If f_i(x) are the inequalities of the problem, theh we have:
 * <br><i>&Phi;</i> = - Sum_i[log(-f_i(x))]
 * @see "S.Boyd and L.Vandenberghe, Convex Optimization, 11.2.1"
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class LogarithmicBarrier implements BarrierFunction {

	final Algebra ALG = Algebra.DEFAULT;
	final DoubleFactory1D F1 = DoubleFactory1D.dense;
	final DoubleFactory2D F2 = DoubleFactory2D.dense;
	private ConvexMultivariateRealFunction[] fi = null;
	private int dim = -1;

	/**
	 * Create the logarithmic barrier function.
	 * @see "S.Boyd and L.Vandenberghe, Convex Optimization, 11.2.1"
	 */
	public LogarithmicBarrier(ConvexMultivariateRealFunction[] fi, int dim) {
		this.fi = fi;
		this.dim = dim;
	}

	public double value(double[] X) {
		double psi = 0;
		for(int j=0; j<fi.length; j++){
			double ineqValuejX = fi[j].value(X);
			if(ineqValuejX>=0){
				return Double.NaN;
			}
			psi -= Math.log(-ineqValuejX);
		}
		return psi;
	}

	public double[] gradient(double[] X) {
		DoubleMatrix1D gradFiSum = F1.make(getDim());
		for(int j=0; j<fi.length; j++){
			double ineqValuejX = fi[j].value(X);
			DoubleMatrix1D ineqGradjX = F1.make(fi[j].gradient(X));
			gradFiSum.assign(ineqGradjX.assign(Mult.mult(-1./ineqValuejX)), Functions.plus);
		}
		return gradFiSum.toArray();
	}
	
	public double[][] hessian(double[] X) {
		DoubleMatrix2D HessSum = F2.make(new double[getDim()][getDim()]);
		DoubleMatrix2D GradSum = F2.make(new double[getDim()][getDim()]);
		for (int j = 0; j < fi.length; j++) {
			double ineqValuejX = fi[j].value(X);
			double[][] fijHessianX = fi[j].hessian(X);
			DoubleMatrix2D ineqHessjX = (fijHessianX!=FunctionsUtils.ZEROES_2D_ARRAY_PLACEHOLDER)? F2.make(fijHessianX) : FunctionsUtils.ZEROES_MATRIX_PLACEHOLDER;
			DoubleMatrix1D ineqGradjX = F1.make(fi[j].gradient(X));
			if(ineqHessjX!=FunctionsUtils.ZEROES_MATRIX_PLACEHOLDER){
				HessSum.assign(ineqHessjX.assign(Mult.mult(-1./ineqValuejX)), Functions.plus);
			}
			GradSum.assign(ALG.multOuter(ineqGradjX, ineqGradjX, null).assign(Mult.mult(1. / Math.pow(ineqValuejX, 2))), Functions.plus);
		}
		return HessSum.assign(GradSum, Functions.plus).toArray();
	}
	
	public int getDim() {
		return this.dim;
	}
	
	public double getDualityGap(double t) {
		return ((double)fi.length) / t;
	}
	
	/**
	 * Create the barrier function for the Phase I.
	 * It is a LogarithmicBarrier for the constraints: 
	 * <br>fi(X)-s, i=1,...,n
	 */
	public BarrierFunction createPhase1BarrierFunction(){
		
		final int dimPh1 = dim +1;
		ConvexMultivariateRealFunction[] inequalitiesPh1 = new ConvexMultivariateRealFunction[this.fi.length];
		for(int i=0; i<inequalitiesPh1.length; i++){
			
			final ConvexMultivariateRealFunction originalFi = this.fi[i];
			
			ConvexMultivariateRealFunction fi = new ConvexMultivariateRealFunction() {
				
				public double value(double[] Y) {
					DoubleMatrix1D y = DoubleFactory1D.dense.make(Y);
					DoubleMatrix1D X = y.viewPart(0, dim);
					return originalFi.value(X.toArray()) - y.get(dimPh1-1);
				}
				
				public double[] gradient(double[] Y) {
					DoubleMatrix1D y = DoubleFactory1D.dense.make(Y);
					DoubleMatrix1D X = y.viewPart(0, dim);
					DoubleMatrix1D origGrad = F1.make(originalFi.gradient(X.toArray()));
					DoubleMatrix1D ret = F1.make(1, -1);
					ret = F1.append(origGrad, ret);
					return ret.toArray();
				}
				
				public double[][] hessian(double[] Y) {
					DoubleMatrix1D y = DoubleFactory1D.dense.make(Y);
					DoubleMatrix1D X = y.viewPart(0, dim);
					DoubleMatrix2D origHess;
					double[][] origFiHessX = originalFi.hessian(X.toArray());
					if(origFiHessX == FunctionsUtils.ZEROES_2D_ARRAY_PLACEHOLDER){
						return FunctionsUtils.ZEROES_2D_ARRAY_PLACEHOLDER;
					}else{
						origHess = F2.make(origFiHessX);
						DoubleMatrix2D[][] parts = new DoubleMatrix2D[][]{{origHess, null},{null,F2.make(1, 1)}};
						return F2.compose(parts).toArray();
					}
				}
				
				public int getDim() {
					return dimPh1;
				}
			};
			inequalitiesPh1[i] = fi;
		}
		
		BarrierFunction bfPh1 = new LogarithmicBarrier(inequalitiesPh1, dimPh1);
		return bfPh1;
	}
	
	/**
	 * Calculates the initial value for the s parameter in Phase I.
	 * Return s = max(fi(x))
	 * @see "S.Boyd and L.Vandenberghe, Convex Optimization, 11.6.2"
	 */
	public double calculatePhase1InitialFeasiblePoint(double[] originalNotFeasiblePoint, double tolerance){
		//DoubleMatrix1D X0NF = F1.make(originalNotFeasiblePoint);
		DoubleMatrix1D fiX0NF = F1.make(fi.length);
		for(int i=0; i<fi.length; i++){
			fiX0NF.set(i, this.fi[i].value(originalNotFeasiblePoint));
		}
		
		//lucky strike?
		int maxIneqIndex = Utils.getMaxIndex(fiX0NF);
		if(fiX0NF.get(maxIneqIndex) < 0){
			//the given notFeasible starting point is in fact already feasible
			return -1;
		}
		
		double s = Math.pow(tolerance,-0.5);
		for(int i=0; i<fiX0NF.size(); i++){
			s = Math.max(s, fiX0NF.get(i)*Math.pow(tolerance,-0.5));
		}
		
		return s;
	}
}
