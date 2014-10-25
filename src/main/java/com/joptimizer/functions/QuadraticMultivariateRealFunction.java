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

import com.joptimizer.util.ColtUtils;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import cern.colt.matrix.linalg.Property;

/**
 * Represent a function in the form of
 * <br>f(x) := 1/2 x.P.x + q.x + r
 * <br>where x, q &#8712; R<sup>n</sup>, P is a symmetric nXn matrix and r &#8712; R.
 * 
 * <br>NOTE1: remember the two following propositions hold:
 * <ol>
 * 	<li>A function f(x) is a quadratic form if and only if it can be written as 
 * f(x) = x.P.x 
 * for a symmetric matrix P (f can even be written as x.P1.x with P1 not symmetric, for example 
 * <br>f = x^2 + 2 x y + y^2 we can written with P={{1, 1}, {1, 1}} symmetric or 
 * with P1={{1, -1}, {3, 1}} not symmetric, but here we are interested in 
 * symmetric matrices for they convexity properties).</li>
 * 	<li>Let f(x) = x.P.x be a quadratic form with associated symmetric matrix P, then we have:
 * 		<ul>
 * 			<li>f is convex <=> P is positive semidefinite</li>
 * 			<li>f is concave <=> P is negative semidefinite</li>
 * 			<li>f is strictly convex <=> P is positive definite</li>
 * 			<li>f is strictly concave <=> P is negative definite</li>
 * 		</ul>
 *  </li>
 * </ol>
 * 
 * NOTE2: precisely speaking, this class should have been named "PolynomialOfDegreeTwo", because
 * by definition a quadratic form in the variables x1,x2,...,xn is a polynomial function where all terms 
 * in the functional expression have order two. A general polynomial function f(x) of degree two can be written
 * as the sum of a quadratic form Q = x.P.x and a linear form L = q.x  (plus a constant term r):
 * <br>f(x) = Q + L + r
 * <br>Because L is convex, f is convex if so is Q.
 *  
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 * @see "Eivind Eriksen, Quadratic Forms and Convexity"
 */
public class QuadraticMultivariateRealFunction implements TwiceDifferentiableMultivariateRealFunction {

	/**
	 * Dimension of the function argument.
	 */
	protected int dim = -1;

	/**
	 * Quadratic factor.
	 */
	protected DoubleMatrix2D P = null;

	/**
	 * Linear factor.
	 */
	protected DoubleMatrix1D q = null;

	/**
	 * Constant factor.
	 */
	protected double r = 0;
	
	private Algebra ALG = Algebra.DEFAULT;
	
	public QuadraticMultivariateRealFunction(double[][] PMatrix, double[] qVector, double r, boolean checkSymmetry){
		this.P = (PMatrix!=null)? DoubleFactory2D.dense.make(PMatrix) : null;
		this.q = (qVector!=null)? DoubleFactory1D.dense.make(qVector) : null;
		this.r = r;
		
		if(P==null && q==null){
			throw new IllegalArgumentException("Impossible to create the function");
		}
		if (P != null && !Property.DEFAULT.isSquare(P)) {
			throw new IllegalArgumentException("P is not square");
		}
		if (P != null && checkSymmetry && !Property.DEFAULT.isSymmetric(P)) {
			throw new IllegalArgumentException("P is not symmetric");
		}
		
		this.dim = (P != null)? P.columns() : q.size();
		if (this.dim < 0) {
			throw new IllegalArgumentException("Impossible to create the function");
		}
  }

	public QuadraticMultivariateRealFunction(double[][] PMatrix, double[] qVector, double r) {
		this(PMatrix, qVector, r, false);
	}

	public final double value(double[] X) {
		DoubleMatrix1D x = DoubleFactory1D.dense.make(X);
		double ret = r;
		if (P != null) {
			ret += 0.5 * ALG.mult(x, ALG.mult(P, x));
		}
		if (q != null) {
			ret += ALG.mult(q, x);
		}
		return ret;
	}

	public final double[] gradient(double[] X) {
		DoubleMatrix1D x = DoubleFactory1D.dense.make(X);
		DoubleMatrix1D ret = null;
		if(P!=null){
			if (q != null) {
				//P.x + q
				//ret = P.zMult(x, q.copy(), 1, 1, false);
				ret = ColtUtils.zMult(P, x, q, 1);
			} else {
				ret = ALG.mult(P, x);
			}
		}else{
			ret = q;
		}
		return ret.toArray();
		
	}

	public final double[][] hessian(double[] X) {
		DoubleMatrix2D ret = null;
		if(P!=null){
			ret = P;
		}else{
			//ret = DoubleFactory2D.dense.make(dim, dim);
			return FunctionsUtils.ZEROES_2D_ARRAY_PLACEHOLDER;
		}
		return ret.toArray();
	}

	public DoubleMatrix2D getP() {
		return P;
	}

	public DoubleMatrix1D getQ() {
		return q;
	}

	public double getR() {
		return r;
	}

	public int getDim() {
		return this.dim;
	}

}
