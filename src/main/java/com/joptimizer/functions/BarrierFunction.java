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
 * Interface for the barrier function used by a given barrier optimization method.
 * @see "S.Boyd and L.Vandenberghe, Convex Optimization, 11.2"
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public interface BarrierFunction extends TwiceDifferentiableMultivariateRealFunction{

	/**
	* Calculates the duality gap for a barrier method build with this barrier function. 
	*/
	public double getDualityGap(double t);
	
	/**
	 * Create the barrier function for the basic Phase I method.
	 * @see "S.Boyd and L.Vandenberghe, Convex Optimization, 11.4.1"
	 */
	public BarrierFunction createPhase1BarrierFunction();
	
	/**
	 * Calculates the initial value for the additional variable s in basic Phase I method.
	 * @see "S.Boyd and L.Vandenberghe, Convex Optimization, 11.4.1"
	 */
	public double calculatePhase1InitialFeasiblePoint(double[] originalNotFeasiblePoint, double tolerance);
}
