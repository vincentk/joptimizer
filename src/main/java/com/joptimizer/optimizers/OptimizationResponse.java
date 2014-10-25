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

/**
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class OptimizationResponse {

	public static final int SUCCESS = 0;
	public static final int WARN = 1;
	public static final int FAILED = 2;

	/**
	 * The optimization retun code. In the case of WARN, you are given a result
	 * by the optimizer but you must manually check if this is appropriate for
	 * you (i.e. you have to manually check if constraints are satisfied within
	 * an acceptable tolerance). It can happen, for example, when the algorithm
	 * exceeds the available number of iterations.
	 */
	private int returnCode;

	private double[] solution;
	
	private double f0;

	public void setReturnCode(int returnCode) {
		this.returnCode = returnCode;
	}

	public int getReturnCode() {
		return returnCode;
	}

	public void setSolution(double[] solution) {
		this.solution = solution;
	}

	public double[] getSolution() {
		return solution;
	}
}
