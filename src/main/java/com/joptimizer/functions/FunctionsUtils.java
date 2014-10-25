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

import java.util.Arrays;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.jet.math.Functions;
import cern.jet.math.Mult;

/**
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class FunctionsUtils {

	/**
	 * Placeholder for 2D-arrays of zeroes.
	 */
	public static final double[][] ZEROES_2D_ARRAY_PLACEHOLDER = new double[0][0];
	
	/**
	 * Placeholder for a matrix of zeroes.
	 */
	public static final DoubleMatrix2D ZEROES_MATRIX_PLACEHOLDER = DoubleFactory2D.dense.make(0, 0);
	private static DoubleFactory1D F1 = DoubleFactory1D.dense;
	private static DoubleFactory2D F2 = DoubleFactory2D.dense;
	
	public static ConvexMultivariateRealFunction createCircle(final int dim,
			final double radius) {
		double[] center = new double[dim];
		return createCircle(dim, radius, center);
	}

	public static ConvexMultivariateRealFunction createCircle(final int dim,
			final double radius, final double[] center) {

		final DoubleMatrix1D C = F1.make(center);
		return new ConvexMultivariateRealFunction() {

			/**
			 * Sum[ (x[i]-center[i])^2 ] - radius^2.
			 */
			public double value(double[] X) {
				DoubleMatrix1D x = DoubleFactory1D.dense.make(X);
				DoubleMatrix1D D = x.assign(C, Functions.minus);
				double d = D.zDotProduct(D) - Math.pow(radius, 2);
				return d;
			}

			public double[] gradient(double[] X) {
				DoubleMatrix1D x = DoubleFactory1D.dense.make(X);
				DoubleMatrix1D D = x.assign(C, Functions.minus);
				return D.assign(Mult.mult(2)).toArray();
			}

			public double[][] hessian(double[] X) {
				double[] d = new double[dim];
				Arrays.fill(d, 2);
				return F2.diagonal(F1.make(d)).toArray();
			}

			public int getDim() {
				return dim;
			}
		};
	}
}
