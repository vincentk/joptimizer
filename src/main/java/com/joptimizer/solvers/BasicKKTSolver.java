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

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.jet.math.Functions;
import cern.jet.math.Mult;

import com.joptimizer.algebra.CholeskyFactorization;
import com.joptimizer.algebra.Matrix1NornRescaler;
import com.joptimizer.util.ColtUtils;

/**
 * H.v + [A]T.w = -g, <br>
 * A.v = -h
 * 
 * @see "S.Boyd and L.Vandenberghe, Convex Optimization, p. 542"
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class BasicKKTSolver extends KKTSolver {

	private boolean avoidScaling = false;
	private Log log = LogFactory.getLog(this.getClass().getName());

	public BasicKKTSolver(){
		this(false);
	}

	public BasicKKTSolver(boolean avoidScaling) {
		this.avoidScaling = avoidScaling;
	}

	/**
	 * Returns the two vectors v and w.
	 */
	@Override
	public DoubleMatrix1D[] solve() throws Exception {

		DoubleMatrix1D v = null;// dim equals cols of A
		DoubleMatrix1D w = null;// dim equals rank of A

		if (log.isDebugEnabled()) {
			log.debug("H: " + ArrayUtils.toString(H.toArray()));
			log.debug("g: " + ArrayUtils.toString(g.toArray()));
			if (A != null) {
				log.debug("A: " + ArrayUtils.toString(A.toArray()));
			}
			if (h != null) {
				log.debug("h: " + ArrayUtils.toString(h.toArray()));
			}
		}
		CholeskyFactorization HFact = new CholeskyFactorization(H, new Matrix1NornRescaler());
		boolean isHReducible = true;
		try{
			HFact.factorize();
		}catch(Exception e){
			isHReducible = false;
		}

		if (isHReducible) {
			// Solving KKT system via elimination
			DoubleMatrix1D HInvg = HFact.solve(g);
			if (A != null) {
				DoubleMatrix2D HInvAT = HFact.solve(AT);
				DoubleMatrix2D MenoSLower = ColtUtils.subdiagonalMultiply(A, HInvAT);
				DoubleMatrix1D AHInvg = ALG.mult(A, HInvg);
				
				CholeskyFactorization MSFact = new CholeskyFactorization(MenoSLower, new Matrix1NornRescaler());
				MSFact.factorize();
				if(h == null){
					w = MSFact.solve(ColtUtils.scalarMult(AHInvg, -1));
				}else{
					//w = MSFact.solve(h.copy().assign(AHInvg, Functions.minus));
					w = MSFact.solve(ColtUtils.add(h, AHInvg, -1));
				}
				
				v = HInvg.assign(ALG.mult(HInvAT, w), Functions.plus).assign(Mult.mult(-1));
			} else {
				w = null;
				v = HInvg.assign(Mult.mult(-1));
			}
		} else {
			// H is singular
			// Solving the full KKT system
			if(A!=null){
//				KKTSolver kktSolver = new BasicKKTSolver();
//				kktSolver.setCheckKKTSolutionAccuracy(false);
				DoubleMatrix1D[] fullSol =  this.solveAugmentedKKT();
				v = fullSol[0];
				w = fullSol[1];
			}else{
				//@TODO: try with rescaled H
				throw new Exception("KKT solution failed");
			}
		}

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
}