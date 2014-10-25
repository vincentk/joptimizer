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


/**
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public interface  TwiceDifferentiableMultivariateRealFunction {

	/**
	 * Evaluation of the function at point X.
	 */
	public double value(double[] X);
	
	/**
	 * Function gradient at point X.
	 */
	public double[] gradient(double[] X);
	
	/**
	 * Function hessian at point X.
	 */
	public double[][] hessian(double[] X);
	
	/**
	 * Dimension of the function argument.
	 */
	public int getDim();
}
