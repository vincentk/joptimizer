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
import cern.jet.math.Functions;
import cern.jet.math.Mult;

import com.joptimizer.algebra.CholeskyFactorization;
import com.joptimizer.algebra.CholeskyUpperDiagonalFactorization;
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
 * for upper diagonal H.
 * H is expected to be diagonal in its upper left corner of dimension diagonalLength.
 * Only the subdiagonal elements are relevant.
 * 
 * @see "S.Boyd and L.Vandenberghe, Convex Optimization, p. 542"
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class UpperDiagonalHKKTSolver extends KKTSolver {

	private boolean avoidScaling = false;
	private int diagonalLength;
	private Log log = LogFactory.getLog(this.getClass().getName());
	
	public UpperDiagonalHKKTSolver(int diagonalLength){
		this(diagonalLength, false);
	}

	public UpperDiagonalHKKTSolver(int diagonalLength, boolean avoidScaling) {
		this.diagonalLength = diagonalLength;
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
			log.debug("H: " + ArrayUtils.toString(ColtUtils.fillSubdiagonalSymmetricMatrix((SparseDoubleMatrix2D)this.H).toArray()));
			log.debug("g: " + ArrayUtils.toString(g.toArray()));
			if (A!=null) {
				log.debug("A: " + ArrayUtils.toString(A.toArray()));
				log.debug("h: " + ArrayUtils.toString(h.toArray()));
			}
		}
    	MatrixRescaler rescaler = (avoidScaling)? null : new Matrix1NornRescaler();
		CholeskyUpperDiagonalFactorization HFact = new CholeskyUpperDiagonalFactorization((SparseDoubleMatrix2D)H, diagonalLength, rescaler);
		boolean isHReducible = true;
		try{
			HFact.factorize();
		}catch(Exception e){
			log.warn("warn", e);
			isHReducible = false;
		}

		if (isHReducible) {
			// Solving KKT system via elimination
			DoubleMatrix1D HInvg;
			HInvg = HFact.solve(g);
			
			if (A != null) {
				DoubleMatrix2D HInvAT;
				HInvAT = HFact.solve(AT);
				
//				double scaledResidualX = Utils.calculateScaledResidual(ColtUtils.fillSubdiagonalSymmetricMatrix(H), HInvAT, AT);
//				log.info("scaledResidualX: " + scaledResidualX);
				
				DoubleMatrix2D MenoSLower = ColtUtils.subdiagonalMultiply(A, HInvAT);
				log.debug("MenoS: " + ArrayUtils.toString(ColtUtils.fillSubdiagonalSymmetricMatrix(MenoSLower).toArray()));
				DoubleMatrix1D AHInvg = ALG.mult(A, HInvg);
				
				CholeskyFactorization MSFact = new CholeskyFactorization(MenoSLower, new Matrix1NornRescaler());
				//CholeskySparseFactorization MSFact = new CholeskySparseFactorization((SparseDoubleMatrix2D)MenoSLower, new RequestFilter());
				try{
					MSFact.factorize();
					if(h == null){
						w = MSFact.solve(ColtUtils.scalarMult(AHInvg, -1));
					}else{
						//w = MSFact.solve(h.copy().assign(AHInvg, Functions.minus));
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
		} else {
			// H not isReducible, try solving the augmented KKT system
			DoubleMatrix1D[] fullSol = this.solveAugmentedKKT();
			v = fullSol[0];
			w = fullSol[1];
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
	 * 
	 * @TODO: avoid overriding, leave only the parent 
	 */
	@Override
	protected boolean checkKKTSolutionAccuracy(DoubleMatrix1D v, DoubleMatrix1D w) {
		log.debug("checkKKTSolutionAccuracy");
		DoubleMatrix2D KKT = null;
		DoubleMatrix1D x = null;
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
				
				x = F1.append(v, w);
				b = F1.append(g, h).assign(Mult.mult(-1));
			}else{
				//H.v + [A]T.w = -g
				DoubleMatrix2D[][] parts = {{ HFull, this.AT }};
				if(HFull instanceof SparseDoubleMatrix2D && A instanceof SparseDoubleMatrix2D){
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
			KKT = HFull;
			x = v;
			//b = g.copy().assign(Mult.mult(-1));
			b = ColtUtils.scalarMult(g, -1);
		}
		
		//checking residual
		double scaledResidual = Utils.calculateScaledResidual(KKT, x, b);
		log.info("KKT inversion scaled residual: " + scaledResidual);
		return scaledResidual < toleranceKKT;
	}

	public void setDiagonalLength(int diagonalLength) {
		this.diagonalLength = diagonalLength;
	}

	public int getDiagonalLength() {
		return diagonalLength;
	}
}
