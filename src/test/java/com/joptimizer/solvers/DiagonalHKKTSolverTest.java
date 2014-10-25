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
package com.joptimizer.solvers;

import java.io.File;

import junit.framework.TestCase;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import cern.colt.matrix.linalg.Property;
import cern.jet.math.Functions;

import com.joptimizer.util.Utils;

/**
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class DiagonalHKKTSolverTest extends TestCase {

	private Algebra ALG = Algebra.DEFAULT;
	private DoubleFactory1D F1 = DoubleFactory1D.dense;
	private DoubleFactory2D F2 = DoubleFactory2D.sparse;
	private Property P = Property.TWELVE;
	private Log log = LogFactory.getLog(this.getClass().getName());

	public void testDummy(){
		assertTrue(true);
	}
	
	/**
	 * this is the KKT system relative to the first iteration of the minimization phase
	 * in the primal dual method for the afiro netlib problem.
	 * @TODO: solve this test
	 */
	public void xxxtestSolve() throws Exception {
		log.debug("testSolve");
		double[][] H = Utils.loadDoubleMatrixFromFile("lp" + File.separator	+ "H13.csv");
		double[][] A = Utils.loadDoubleMatrixFromFile("lp" + File.separator	+ "A13.csv");
		double[] g = Utils.loadDoubleArrayFromFile("lp" + File.separator	+ "g13.csv");
		double[] h = Utils.loadDoubleArrayFromFile("lp" + File.separator	+ "hh13.csv");
		DoubleMatrix2D HMatrix = F2.make(H);
		DoubleMatrix2D AMatrix = F2.make(A);
		DoubleMatrix1D gVector = F1.make(g);
		DoubleMatrix1D hVector = F1.make(h);

		DiagonalHKKTSolver solver = new DiagonalHKKTSolver();
		solver.setHMatrix(HMatrix);
		solver.setAMatrix(AMatrix);
		solver.setGVector(gVector);
		solver.setHVector(hVector);
		DoubleMatrix1D[] sol = solver.solve();
		DoubleMatrix1D v = sol[0];
		DoubleMatrix1D w = sol[1];
		log.debug("v: " + ArrayUtils.toString(v.toArray()));
		log.debug("w: " + ArrayUtils.toString(w.toArray()));

		DoubleMatrix1D a = ALG.mult(HMatrix, v).assign(ALG.mult(ALG.transpose(AMatrix), w), Functions.plus).assign(gVector, Functions.plus);
		DoubleMatrix1D b = ALG.mult(AMatrix, v).assign(hVector, Functions.plus);
		log.debug("a: " + ArrayUtils.toString(a.toArray()));
		log.debug("b: " + ArrayUtils.toString(b.toArray()));
		for (int i = 0; i < a.size(); i++) {
			assertEquals(0, a.get(i), 1.E-14);
		}
		for (int i = 0; i < b.size(); i++) {
			assertEquals(0, b.get(i), 1.E-14);
		}
	}
}
