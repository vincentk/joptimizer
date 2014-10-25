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

import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import cern.jet.math.Mult;

import com.joptimizer.algebra.QRSparseFactorization;
import com.joptimizer.algebra.Matrix1NornRescaler;
import com.joptimizer.util.ColtUtils;

/**
 * Solves the KKT system
 * 
 * H.v + [A]T.w = -g, <br>
 * A.v = -h
 * 
 * as a whole. Note that we can not use the Cholesky factorization
 * for inverting the full KKT matrix, because it is symmetric but not
 * positive in general. 
 * 
 * @see "S.Boyd and L.Vandenberghe, Convex Optimization, p. 542"
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class FullKKTSolver extends KKTSolver {

	private Log log = LogFactory.getLog(this.getClass().getName());

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
		
		//compose the full KKT matrix
		DoubleMatrix2D KKT = null;
		DoubleMatrix1D b = null;
		final DoubleMatrix2D HFull = ColtUtils.fillSubdiagonalSymmetricMatrix((SparseDoubleMatrix2D)this.H);
		
		if (this.A != null) {
			if(h!=null){
				//H.v + [A]T.w = -g
				//A.v = -h
				DoubleMatrix2D[][] parts = {
						{ HFull, this.AT },
						{ this.A, null } };
				if(HFull instanceof SparseDoubleMatrix2D && A instanceof SparseDoubleMatrix2D){
					KKT = DoubleFactory2D.sparse.compose(parts);
				}else{
					KKT = DoubleFactory2D.dense.compose(parts);
				}
				
				b = F1.append(g, h).assign(Mult.mult(-1));
			}else{
				//H.v + [A]T.w = -g
				DoubleMatrix2D[][] parts = {{ HFull, this.AT }};
				if(HFull instanceof SparseDoubleMatrix2D && A instanceof SparseDoubleMatrix2D){
					KKT = DoubleFactory2D.sparse.compose(parts);
				}else{
					KKT = DoubleFactory2D.dense.compose(parts);
				}
				b = ColtUtils.scalarMult(g, -1);
			}
		}else{
			KKT = HFull;
			b = ColtUtils.scalarMult(g, -1);
		}
		
		//factorization
		//LDLTPermutedFactorization KKTFact = new LDLTPermutedFactorization(KKT, new RequestFilter());
		QRSparseFactorization KKTFact = new QRSparseFactorization((SparseDoubleMatrix2D)KKT, new Matrix1NornRescaler());
		try{
			KKTFact.factorize();
		}catch(Exception e){
			log.error("singular KKT system");
			throw new Exception("singular KKT system");
		}
		
		DoubleMatrix1D x = KKTFact.solve(b);
		v = x.viewPart(0, H.rows());
		w = x.viewPart(H.rows()-1, x.size()-H.rows());

		// solution checking
		if (this.checkKKTSolutionAccuracy && !this.checkKKTSolutionAccuracy(v, w)) {
			log.error("KKT solution failed");
			throw new Exception("KKT solution failed");
		}

		DoubleMatrix1D[] ret = new DoubleMatrix1D[2];
		ret[0] = v;// dim equals cols of A
		ret[1] = w;
		return ret;
	}
}
