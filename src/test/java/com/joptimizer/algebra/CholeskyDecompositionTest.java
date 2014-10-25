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
package com.joptimizer.algebra;

import junit.framework.TestCase;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.CholeskyDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import com.joptimizer.util.Utils;

/**
 * tests Commons-Math CholeskyDecomposition
 * 
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class CholeskyDecompositionTest extends TestCase {

	private Log log = LogFactory.getLog(this.getClass().getName());

	/**
	 * good decomposition.
	 */
	public void testDecomposition1() throws Exception {
		log.debug("testDecomposition1");
		RealMatrix P1 = new Array2DRowRealMatrix(new double[][] {
				{8.08073550734687,1.59028724315583},
				{1.59028724315583,0.3250861184011492}});
		CholeskyDecomposition cFact1 = new CholeskyDecomposition(P1);
		log.debug("L: " + cFact1.getL());
		log.debug("LT: " + cFact1.getLT());
		// check L.LT-Q=0
		RealMatrix P1Inv = cFact1.getL().multiply(cFact1.getLT()); 
		double norm1 = P1Inv.subtract(P1).getNorm();
		log.debug("norm1: " + norm1);
		assertTrue(norm1 < 1.E-12);
	}

	/**
	 * poor decomposition.
	 * rescaling can help in doing it better
	 */
	public void testDecomposition2() throws Exception {
		log.debug("testDecomposition2");
		RealMatrix P1 = new Array2DRowRealMatrix(new double[][] {
				{ 8.185301256666552E9, 1.5977225251367908E9 },
				{ 1.5977225251367908E9, 3.118660129093004E8 } });
		CholeskyDecomposition cFact1 = new CholeskyDecomposition(P1);
		log.debug("L: " + cFact1.getL());
		log.debug("LT: " + cFact1.getLT());
		// check L.LT-Q=0
		double norm1 = cFact1.getL().multiply(cFact1.getLT()).subtract(P1).getNorm();
		log.debug("norm1: " + norm1);
		assertTrue(norm1 < 1.E-5);
		
		//poor precision, try to make it better
	
		//geometric eigenvalues mean
		DescriptiveStatistics ds = new DescriptiveStatistics(new double[]{8.5E9, 0.00572});
		RealMatrix P2 = P1.scalarMultiply(1./ds.getGeometricMean());
		CholeskyDecomposition cFact2 = new CholeskyDecomposition(P2);
		log.debug("L: " + cFact2.getL());
		log.debug("LT: " + cFact2.getLT());
		// check L.LT-Q=0
		double norm2 = cFact2.getL().multiply(cFact2.getLT()).subtract(P2).getNorm();
		log.debug("norm2: " + norm2);
		assertTrue(norm2 < Utils.getDoubleMachineEpsilon());	
	}
}
