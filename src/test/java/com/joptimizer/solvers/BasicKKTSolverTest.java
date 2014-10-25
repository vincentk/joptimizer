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

/**
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class BasicKKTSolverTest extends TestCase {

	private Algebra ALG = Algebra.DEFAULT;
	private DoubleFactory1D F1 = DoubleFactory1D.dense;
	private DoubleFactory2D F2 = DoubleFactory2D.dense;
	private Property P = Property.TWELVE;
	private Log log = LogFactory.getLog(this.getClass().getName());

	public void testSolveSimple() throws Exception {
		log.debug("testSolveSimple");
		double[][] HMatrix = new double[][] { { 3 } };
		double[][] AMatrix = new double[][] { { 2 } };
		DoubleMatrix2D H = F2.make(HMatrix);
		DoubleMatrix2D A = F2.make(AMatrix);
		DoubleMatrix2D AT = ALG.transpose(A.copy());
		DoubleMatrix1D g = F1.make(1, -3);
		DoubleMatrix1D h = F1.make(1, 0);

		KKTSolver solver = new BasicKKTSolver();
		solver.setHMatrix(H);
		solver.setAMatrix(A);
		solver.setGVector(g);
		solver.setHVector(h);
		DoubleMatrix1D[] sol = solver.solve();
		DoubleMatrix1D v = sol[0];
		DoubleMatrix1D w = sol[1];
		log.debug("v: " + ArrayUtils.toString(v.toArray()));
		log.debug("w: " + ArrayUtils.toString(w.toArray()));

		DoubleMatrix1D a = ALG.mult(H, v).assign(ALG.mult(AT, w), Functions.plus).assign(g, Functions.plus);
		DoubleMatrix1D b = ALG.mult(A, v).assign(h, Functions.plus);
		log.debug("a: " + ArrayUtils.toString(a.toArray()));
		log.debug("b: " + ArrayUtils.toString(b.toArray()));
		for (int i = 0; i < a.size(); i++) {
			assertEquals(0, a.get(i), 1.E-14);
		}
		for (int i = 0; i < b.size(); i++) {
			assertEquals(0, b.get(i), 1.E-14);
		}
	}

	public void testSolve2() throws Exception {
		log.debug("testSolve2");
		double[][] HMatrix = new double[][] { 
				{ 1.68, 0.34, 0.38 },
				{ 0.34, 3.09, -1.59 }, 
				{ 0.38, -1.59, 1.54 } };
		double[][] AMatrix = new double[][] { { 1, 2, 3 } };
		DoubleMatrix2D H = F2.make(HMatrix);
		DoubleMatrix2D A = F2.make(AMatrix);
		DoubleMatrix2D AT = ALG.transpose(A.copy());
		DoubleMatrix1D g = F1.make(new double[] { 2, 5, 1 });
		DoubleMatrix1D h = F1.make(new double[] { 1 });

		KKTSolver solver = new BasicKKTSolver();
		solver.setHMatrix(H);
		solver.setAMatrix(A);
		solver.setGVector(g);
		solver.setHVector(h);
		DoubleMatrix1D[] sol = solver.solve();
		DoubleMatrix1D v = sol[0];
		DoubleMatrix1D w = sol[1];
		log.debug("v: " + ArrayUtils.toString(v.toArray()));
		log.debug("w: " + ArrayUtils.toString(w.toArray()));

		DoubleMatrix1D a = ALG.mult(H, v).assign(ALG.mult(AT, w), Functions.plus).assign(g, Functions.plus);
		DoubleMatrix1D b = ALG.mult(A, v).assign(h, Functions.plus);
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
