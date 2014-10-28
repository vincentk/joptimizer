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
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import cern.jet.math.Functions;
import cern.jet.math.Mult;

import com.joptimizer.algebra.CholeskyFactorization;
import com.joptimizer.algebra.MatrixRescaler;
import com.joptimizer.algebra.Matrix1NornRescaler;
import com.joptimizer.util.ColtUtils;
import com.joptimizer.util.Utils;

/**
 * Solves
 * 
 * H.v + [A]T.w = -g, <br>
 * A.v = -h
 * 
 * for diagonal H.
 * H is expected to be diagonal.
 * Only the subdiagonal elements are relevant.
 * 
 * @see "S.Boyd and L.Vandenberghe, Convex Optimization, p. 542"
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 * @TODO: do not use this until AHAT calculation is incorrect.
 */
public class DiagonalHKKTSolver extends KKTSolver {

	private boolean avoidScaling = false;
	private Log log = LogFactory.getLog(this.getClass().getName());
	
	public DiagonalHKKTSolver(){
		this(false);
	}

	public DiagonalHKKTSolver(boolean avoidScaling) {
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
			log.debug("H: " + ArrayUtils.toString(ColtUtils.fillSubdiagonalSymmetricMatrix(this.H).toArray()));
			log.debug("g: " + ArrayUtils.toString(g.toArray()));
			if (A!=null) {
				log.debug("A: " + ArrayUtils.toString(A.toArray()));
				log.debug("h: " + ArrayUtils.toString(h.toArray()));
			}
		}
  	
    // Solving KKT system via elimination
		DoubleMatrix1D HInvg = F1.make(H.rows());
		for(int i=0; i<H.rows(); i++){
			double d = H.getQuick(i, i);
			if (d < Utils.getDoubleMachineEpsilon()) {
				throw new Exception("not positive definite matrix");
			}
			HInvg.setQuick(i, g.getQuick(i) / d);
		}
		
		if (A != null) {
			DoubleMatrix2D HInvAT = calculateHAT(H, A);
			DoubleMatrix2D MenoSLower = calculateSubdiagonalAHAT(this.A, this.H);
			log.debug("MenoS: " + ArrayUtils.toString(ColtUtils.fillSubdiagonalSymmetricMatrix(MenoSLower).toArray()));
			DoubleMatrix1D AHInvg = ALG.mult(A, HInvg);
			
			MatrixRescaler rescaler = (this.avoidScaling)? null : new Matrix1NornRescaler();
			CholeskyFactorization MSFact = new CholeskyFactorization(MenoSLower, rescaler);
			//CholeskySparseFactorization MSFact = new CholeskySparseFactorization((SparseDoubleMatrix2D)MenoSLower, new RequestFilter());
			try{
				MSFact.factorize();
				if(h == null){
					w = MSFact.solve(ColtUtils.scalarMult(AHInvg, -1));
				}else{
					DoubleMatrix1D hmAHInvg = ColtUtils.add(h, AHInvg, -1);
					//log.debug("hmAHInvg: " + ArrayUtils.toString(hmAHInvg.toArray()));
					w = MSFact.solve(hmAHInvg);
				}
				
				v = HInvg.assign(ALG.mult(HInvAT, w), Functions.plus).assign(Mult.mult(-1));
			}catch(Exception e){
				log.warn("warn: "+ e.getMessage());
				log.debug("MenoS: " + ArrayUtils.toString(ColtUtils.fillSubdiagonalSymmetricMatrix(MenoSLower).toArray()));
				//is it a numeric issue? try solving the full KKT system
				try{
					//NOTE: it would be more appropriate to try solving the full KKT, but if the decomposition 
					//of the Shur complement of H (>0) in KKT fails it is certainty for a numerical issue and
					//the augmented KKT seems to be more able to recover from this situation
					//DoubleMatrix1D[] fullSol =  this.solveFullKKT();
					DoubleMatrix1D[] fullSol =  this.solveAugmentedKKT();
					v = fullSol[0];
					w = fullSol[1];
				}catch(Exception ex){
					log.error(ex.getMessage());
					throw new Exception("KKT solution failed");
				}
			}
			
		} else {
			//A==null
			w = null;
			v = HInvg.assign(Mult.mult(-1));
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
	
	private DoubleMatrix2D calculateHAT(final DoubleMatrix2D HH,	final DoubleMatrix2D AA) {
		final DoubleMatrix2D ret = DoubleFactory2D.sparse.make(AA.columns(), AA.rows());
		AA.forEachNonZero(new IntIntDoubleFunction() {
			@Override
			public double apply(int i, int j, double aij) {
				ret.setQuick(j, i, aij * HH.getQuick(j, j));
				return aij;
			}
		});
		return ret;
	}
	
	/**
	 * @FIXME: fix this method (wrong return)
	 * try for example with H and A of the first iteration of the afiro netlib problem 
	 */
	private DoubleMatrix2D calculateSubdiagonalAHAT(final DoubleMatrix2D AA, final DoubleMatrix2D HH) {
		final DoubleMatrix2D ret = DoubleFactory2D.sparse.make(AA.rows(), AA.rows());
		final int[] rowHolder = new int[] { -1 };
		final int[] colHolder = new int[] { -1 };
		final double[] valueHolder = new double[] { Double.NaN };
		final IntIntDoubleFunction myFunc = new IntIntDoubleFunction() {
			@Override
			public double apply(int r, int c, double AAColrc) {
				if (c < rowHolder[0] + 1) {
					//log.debug("sum " + AAColrc + "*" + HH.getQuick(colHolder[0], colHolder[0]) + " to AHAT(" + r + ","	+ rowHolder[0] + ")");
					ret.setQuick(r, rowHolder[0], ret.getQuick(r, rowHolder[0]) + valueHolder[0] * AAColrc	* HH.getQuick(colHolder[0], colHolder[0]));
				}
				return AAColrc;
			}
		};
		AA.forEachNonZero(new IntIntDoubleFunction() {
			@Override
			public double apply(final int i, final int j, final double aij) {
				rowHolder[0] = i;
				colHolder[0] = j;
				valueHolder[0] = aij;
				//log.debug("a(" + i + "," + j + "): " + aij);
				DoubleMatrix2D AACol = AA.viewPart(0, j, AA.rows(), 1);
				//log.debug("ACol(" + j + "): " + ArrayUtils.toString(AACol.toArray()));
//				ACol.forEachNonZero(new IntIntDoubleFunction() {
//					public double apply(int r, int c, double AColrc) {
//						if (c < r + 1) {
//							logger.debug("sum " + AColrc + "*" + H.getQuick(j, j) + " to AHAT(" + r + "," + i + ")");
//							ret.setQuick(r, i, ret.getQuick(r, i) + aij * AColrc * H.getQuick(j, j));
//						}
//						return AColrc;
//					}
//				});
				AACol.forEachNonZero(myFunc);
				return aij;
			}
		});
		return ret;
	}

	/**
	 * Check the solution of the system
	 * 
	 * 	KKT.x = b
	 * 
	 * against the scaled residual
	 * 
	 * 	beta < gamma, 
	 * 
	 * where gamma is a parameter chosen by the user and beta is
	 * the scaled residual,
	 * 
	 * 	beta = ||KKT.x-b||_oo/( ||KKT||_oo . ||x||_oo + ||b||_oo ), 
	 * with ||x||_oo = max(||x[i]||)
	 */
	@Override
	protected boolean checkKKTSolutionAccuracy(DoubleMatrix1D v, DoubleMatrix1D w) {
		log.debug("checkKKTSolutionAccuracy");
		DoubleMatrix2D KKT = null;
		DoubleMatrix1D x = null;
		DoubleMatrix1D b = null;
		
		if (this.A != null) {
			if(h!=null){
				//H.v + [A]T.w = -g
				//A.v = -h
				DoubleMatrix2D[][] parts = { 
						{ H, this.AT },
						{ this.A, null } };
				if(A instanceof SparseDoubleMatrix2D){
					KKT = DoubleFactory2D.sparse.compose(parts);
				}else{
					KKT = DoubleFactory2D.dense.compose(parts);
				}
				
				x = F1.append(v, w);
				b = F1.append(g, h).assign(Mult.mult(-1));
			}else{
				//H.v + [A]T.w = -g
				DoubleMatrix2D[][] parts = {{ H, this.AT }};
				if(A instanceof SparseDoubleMatrix2D){
					KKT = DoubleFactory2D.sparse.compose(parts);
				}else{
					KKT = DoubleFactory2D.dense.compose(parts);
				}
				x = F1.append(v, w);
				//b = g.copy().assign(Mult.mult(-1));
				b = ColtUtils.scalarMult(g, -1);
			}
		}else{
			//H.v = -g
			KKT = H;
			x = v;
			//b = g.copy().assign(Mult.mult(-1));
			b = ColtUtils.scalarMult(g, -1);
		}
		
		//checking residual
		double scaledResidual = Utils.calculateScaledResidual(KKT, x, b);
		log.info("KKT inversion scaled residual: " + scaledResidual);
		return scaledResidual < toleranceKKT;
	}
}
