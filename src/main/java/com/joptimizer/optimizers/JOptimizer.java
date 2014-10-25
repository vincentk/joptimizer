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
 * Convex Optimizer.
 * 
 * The algorithm selection is implemented as a Chain of Responsibility pattern,
 * and this class is the client of the chain.
 * 
 * @see "S.Boyd and L.Vandenberghe, Convex Optimization"
 * @author <a href="mailto:alberto.trivellato@gmail.com">alberto trivellato</a>
 */
public class JOptimizer {

	public static final int DEFAULT_MAX_ITERATION = 500;
	public static final double DEFAULT_FEASIBILITY_TOLERANCE = 1.E-6;
	public static final double DEFAULT_TOLERANCE = 1.E-5;
	public static final double DEFAULT_TOLERANCE_INNER_STEP = 1.E-5;
	public static final double DEFAULT_KKT_TOLERANCE = 1.E-9;
	public static final double DEFAULT_ALPHA = 0.055;
	public static final double DEFAULT_BETA = 0.55;
	public static final double DEFAULT_MU = 10;
	public static final String BARRIER_METHOD = "BARRIER_METHOD";
	public static final String PRIMAL_DUAL_METHOD = "PRIMAL_DUAL_METHOD";
	public static final String DEFAULT_INTERIOR_POINT_METHOD = PRIMAL_DUAL_METHOD;
	
	private OptimizationRequest request = null;
	private OptimizationResponse response = null;
	
	public int optimize() throws Exception {
		//start with the first step in the chain.
		OptimizationRequestHandler handler = new NewtonUnconstrained(true);
		handler.setOptimizationRequest(request);
		int retCode = handler.optimize();
		this.response = handler.getOptimizationResponse();
		return retCode;
	}
	
	public void setOptimizationRequest(OptimizationRequest or) {
		this.request = or;
	}

	public OptimizationResponse getOptimizationResponse() {
		return response;
	}
	
}
