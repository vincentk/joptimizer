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

import com.joptimizer.util.Utils;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.RCDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;

/**
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class ComparisonTest extends TestCase {

	private Algebra ALG = Algebra.DEFAULT;
	private DoubleFactory1D F1 = DoubleFactory1D.dense;
	private DoubleFactory2D F2 = DoubleFactory2D.dense;
	private Log log = LogFactory.getLog(this.getClass().getName());

	public void testDummy() throws Exception{
		log.debug("testDummy");
	}

	public void testSparseMatrix1() throws Exception{
		log.debug("testSparseMatrix1");
		int rows = 750;
		int cols = 750;
		int dim = rows*cols;
		DoubleMatrix2D sMatrix = Utils.randomValuesSparseMatrix(rows, cols, -5, 5, 0.50, 12345L);
		//log.debug("sMatrix: " + Utils.toString(sMatrix.toArray()));
		log.debug("cardinality: " + sMatrix.cardinality());
		int nz = dim - sMatrix.cardinality();
		log.debug("sparsity index: " + 100*new Double(nz)/dim +" %");
		
		//try sparse multiplication
		long t0 = System.currentTimeMillis();
		Algebra.DEFAULT.mult(sMatrix, sMatrix);
		log.debug("sparse time: " + (System.currentTimeMillis()-t0));
		
		//try RC sparse multiplication
		DoubleMatrix2D rcMatrix  = new RCDoubleMatrix2D(sMatrix.toArray()); 
		long t1 = System.currentTimeMillis();
//		// Linear algebraic y = A * x
//		DoubleMatrix2D Y = new RCDoubleMatrix2D(rows, cols);
//		rcMatrix.forEachNonZero(
//		   new cern.colt.function.IntIntDoubleFunction() {
//		      public double apply(int row, int column, double value) {
//		         T.setQuick(row, Y.getQuick(row) + value * x.getQuick(column));
//		         return value;
//		      }
//		   }
//		);
		Algebra.DEFAULT.mult(rcMatrix, rcMatrix);
		log.debug("rc time: " + (System.currentTimeMillis()-t1));
		
		//try dense multiplication
		DoubleMatrix2D dMatrix = DoubleFactory2D.dense.make(sMatrix.toArray());
		long t2 = System.currentTimeMillis();
		Algebra.DEFAULT.mult(dMatrix, dMatrix);
		log.debug("dense time: " + (System.currentTimeMillis()-t2));
	}
}
