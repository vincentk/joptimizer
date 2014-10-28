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

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import cern.colt.function.IntIntDoubleFunction;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.jet.math.Functions;
import cern.jet.math.Mult;

import com.joptimizer.algebra.CholeskyFactorization;
import com.joptimizer.algebra.MatrixRescaler;
import com.joptimizer.algebra.Matrix1NornRescaler;
import com.joptimizer.util.ColtUtils;

/**
 * Solves the KKT system
 * 
 * H.v + [A]T.w = -g, <br>
 * A.v = -h
 * 
 * with singular H. The KKT matrix is nonsingular if and only if H + ATQA > 0
 * for some Q > 0, 0, in which case, H + ATQA > 0 for all Q > 0. This class uses
 * the diagonal matrix Q = s.Id with scalar s > 0 to try finding the solution.
 * NOTE: matrix A can not be null for this solver
 * 
 * @see "S.Boyd and L.Vandenberghe, Convex Optimization, p. 547"
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class AugmentedKKTSolver extends KKTSolver {

	private double s = 1.e-6;

	private Log log = LogFactory.getLog(this.getClass().getName());

	/**
	 * Returns the two vectors v and w.
	 */
	@Override
	public DoubleMatrix1D[] solve() throws Exception {

		if (A == null) {
			throw new IllegalStateException("Matrix A cannot be null");
		}

		DoubleMatrix1D v = null;// dim equals cols of A
		DoubleMatrix1D w = null;// dim equals rank of A

		if (log.isDebugEnabled()) {
			log.debug("H: " + ArrayUtils.toString(H.toArray()));
			log.debug("g: " + ArrayUtils.toString(g.toArray()));
			log.debug("A: " + ArrayUtils.toString(A.toArray()));
			if (h != null) {
				log.debug("h: " + ArrayUtils.toString(h.toArray()));
			}
		}

		// augmentation
		final DoubleMatrix2D HAugm = ColtUtils.subdiagonalMultiply(AT, A);// H + ATQA
		HAugm.forEachNonZero(new IntIntDoubleFunction() {
			@Override
			public double apply(int i, int j, double HAugmij) {
				return s * HAugmij;
			}
		});
		H.forEachNonZero(new IntIntDoubleFunction() {
			@Override
			public double apply(int i, int j, double Hij) {
				if (i + 1 > j) {
					// the subdiagonal elements
					HAugm.setQuick(i, j, Hij + HAugm.getQuick(i, j));
				}
				return Hij;
			}
		});

		DoubleMatrix1D gAugm = null;// g + ATQh
		if (h != null) {
			DoubleMatrix1D ATQh = ALG.mult(AT, ColtUtils.diagonalMatrixMult(F1.make(A.rows(), 1), h));
			DoubleMatrix1D gATQh = ColtUtils.add(g, ATQh, defaultScalar);
			gAugm = gATQh;
		}else{
			gAugm = g.copy();
		}

		// solving the augmented system
		CholeskyFactorization HFact = new CholeskyFactorization(HAugm, (MatrixRescaler) new Matrix1NornRescaler());
		try {
			HFact.factorize();
		} catch (Exception e) {
			// The KKT matrix is nonsingular if and only if H + ATQA > 0 for some Q > 0
			log.error("singular KKT system");
			throw new Exception("singular KKT system");
		}

		// Solving KKT system via elimination
		DoubleMatrix1D HInvg = HFact.solve(gAugm);
		DoubleMatrix2D HInvAT = HFact.solve(AT);
		DoubleMatrix2D MenoSLower = ColtUtils.subdiagonalMultiply(A, HInvAT);
		DoubleMatrix1D AHInvg = ALG.mult(A, HInvg);

		CholeskyFactorization MSFact = new CholeskyFactorization(MenoSLower, (MatrixRescaler) new Matrix1NornRescaler());
		MSFact.factorize();
		if (h == null) {
			w = MSFact.solve(ColtUtils.scalarMult(AHInvg, -1));
		} else {
			w = MSFact.solve(ColtUtils.add(h, AHInvg, -1));
		}

		v = HInvg.assign(ALG.mult(HInvAT, w), Functions.plus).assign(Mult.mult(-1));

		// solution checking
		if (this.checkKKTSolutionAccuracy && !this.checkKKTSolutionAccuracy(v, w)) {
			log.error("KKT solution failed");
			throw new Exception("KKT solution failed");
		}

		DoubleMatrix1D[] ret = new DoubleMatrix1D[2];
		ret[0] = v;
		ret[1] = w;
		return ret;
	}
	
	public void setS(double s) {
		this.s = s;
	}
}
