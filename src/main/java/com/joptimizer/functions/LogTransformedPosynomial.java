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

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import cern.jet.math.Functions;
import cern.jet.math.Mult;

/**
 * This function represents the logarithm of a posynomial after a change of variables x->y=log(x).
 * <br>It represents a posynomial:
 * 
 * <br>(1) f(x) = Sum[k=1, K](c[k]*x[1]^a[1,k]*x[2]^a[2,k]*****x[n]^a[n,k]), c[k]>0
 * 
 * <br>in the form of:
 * 
 * <br>(2) g(y) = log(Sum[k=1, K](e^(aT[k]*y+b[k])))
 * 
 * <br>where a[k]=(a[1,k],..,a[n,k]) and b[k]=log(c[k])
 * 
 * <br>It is useful in geometric programming.
 * 
 * @see "S.Boyd and L.Vandenberghe, Convex Optimization, 4.5.3"
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class LogTransformedPosynomial implements ConvexMultivariateRealFunction {
	
	private DoubleMatrix2D A = null;
	private DoubleMatrix2D AT = null;
	private DoubleMatrix1D b = null;
	private int dim = -1;
	private Algebra ALG = Algebra.DEFAULT;
	
	/**
	 * A representation of a posynomial (1) in the form (2).
	 * @param akArray the matrix (a[1,k],..,a[n,k]) in expression (2)
	 * @param bkArray the vector b[k] in expression (2)
	 */
	public LogTransformedPosynomial(double[][] akArray, double[] bkArray){
		this.A = DoubleFactory2D.dense.make(akArray);
		this.b = DoubleFactory1D.dense.make(bkArray);
		if(A.rows() != b.size()){
			throw new IllegalArgumentException("Impossible to create the function");
		}
		this.AT = ALG.transpose(A);
		this.dim = A.columns();
	}

	public double value(double[] X) {
		DoubleMatrix1D x = DoubleFactory1D.dense.make(X);
		DoubleMatrix1D g = ALG.mult(A, x).assign(b, Functions.plus).assign(Functions.exp);
		return Math.log(g.zSum());
	}

	public double[] gradient(double[] X) {
		DoubleMatrix1D x = DoubleFactory1D.dense.make(X);
		DoubleMatrix1D g = ALG.mult(A, x).assign(b, Functions.plus).assign(Functions.exp);
		double den = g.zSum();
		double[] R = new double[dim];
		for(int i=0; i<dim; i++){
			double d = 0d;
			for(int k=0; k<A.rows(); k++){
				d += g.get(k) * A.get(k, i);
			}
			R[i] = d/den;
		}
	    return R;
	}

	public double[][] hessian(double[] X) {
		DoubleMatrix1D x = DoubleFactory1D.dense.make(X);
		DoubleMatrix1D g = ALG.mult(A, x).assign(b, Functions.plus).assign(Functions.exp);
		double den = g.zSum();
		DoubleMatrix1D r = DoubleFactory1D.dense.make(dim);
		for(int i=0; i<dim; i++){
			double d = 0d;
			for(int k=0; k<A.rows(); k++){
				d += g.get(k) * A.get(k, i);
			}
			r.set(i, d);
		}
		
		DoubleMatrix2D ret = DoubleFactory2D.dense.make(dim, dim);
		ret.assign(ALG.multOuter(r, r, null).assign(Mult.mult(-1d/Math.pow(den, 2))), Functions.plus);
		for(int k=0; k<A.rows(); k++){
			ret.assign(ALG.multOuter(A.viewRow(k), A.viewRow(k), null).assign(Mult.mult(g.get(k))).assign(Mult.mult(1d/den)), Functions.plus);
		}
		return ret.toArray();
	}

	public int getDim() {
		return this.dim;
	}

}
