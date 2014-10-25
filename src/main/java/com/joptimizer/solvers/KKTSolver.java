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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import cern.jet.math.Mult;

import com.joptimizer.util.ColtUtils;
import com.joptimizer.util.Utils;

/**
 * Solves the KKT system:
 * 
 * H.v + [A]T.w = -g, <br>
 * A.v = -h, <br>
 * 
 * (H is square and symmetric)
 * 
 * @see "S.Boyd and L.Vandenberghe, Convex Optimization, p. 542"
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public abstract class KKTSolver {

	protected DoubleMatrix2D H;
	protected DoubleMatrix2D A;
	protected DoubleMatrix2D AT;
	protected DoubleMatrix1D g;
	protected DoubleMatrix1D h;
	protected double toleranceKKT = Utils.getDoubleMachineEpsilon();
	protected boolean checkKKTSolutionAccuracy;
	protected Algebra ALG = Algebra.DEFAULT;
	protected DoubleFactory2D F2 = DoubleFactory2D.dense;
	protected DoubleFactory1D F1 = DoubleFactory1D.dense;
	protected double defaultScalar = 1.e-6;
	private Log log = LogFactory.getLog(this.getClass().getName());

	/**
	 * Returns two vectors v and w solutions of the KKT system.
	 */
	public abstract DoubleMatrix1D[] solve() throws Exception;

	public void setHMatrix(DoubleMatrix2D HMatrix) {
		this.H = HMatrix;
	}

	public void setAMatrix(DoubleMatrix2D AMatrix) {
		this.A = AMatrix;
		this.AT = ALG.transpose(A);
	}

	public void setGVector(DoubleMatrix1D gVector) {
		this.g = gVector;
	}

	public void setHVector(DoubleMatrix1D hVector) {
		this.h = hVector;
	}

	/**
	 * Acceptable tolerance for system resolution.
	 */
	public void setToleranceKKT(double tolerance) {
		this.toleranceKKT = tolerance;
	}

	public void setCheckKKTSolutionAccuracy(boolean b) {
		this.checkKKTSolutionAccuracy = b;
	}
	
	protected DoubleMatrix1D[] solveAugmentedKKT() throws Exception{
		log.info("solveAugmentedKKT");
		if(A==null){
			throw new Exception("KKT solution failed");
		}
		KKTSolver kktSolver = new AugmentedKKTSolver();
		kktSolver.setCheckKKTSolutionAccuracy(false);//if the caller has true, then it will make the check, otherwise no check at all
		kktSolver.setHMatrix(H);
		kktSolver.setAMatrix(A);
		kktSolver.setGVector(g);
		kktSolver.setHVector(h);
		return kktSolver.solve();
	}
	
	protected DoubleMatrix1D[] solveFullKKT() throws Exception{
		log.info("solveFullKKT");
		KKTSolver kktSolver = new FullKKTSolver();
		kktSolver.setCheckKKTSolutionAccuracy(false);//if the caller has true, then it will make the check, otherwise no check at all
		kktSolver.setHMatrix(H);
		kktSolver.setAMatrix(A);
		kktSolver.setGVector(g);
		kktSolver.setHVector(h);
		return kktSolver.solve();
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
	protected boolean checkKKTSolutionAccuracy(DoubleMatrix1D v, DoubleMatrix1D w) {
		DoubleMatrix2D KKT = null;
		DoubleMatrix1D x = null;
		DoubleMatrix1D b = null;
		
		if (this.A != null) {
			if(this.AT==null){
				this.AT = ALG.transpose(A);
			}
			if(h!=null){
				//H.v + [A]T.w = -g
				//A.v = -h
				DoubleMatrix2D[][] parts = { 
						{ this.H, this.AT },
						{ this.A, null } };
				if(H instanceof SparseDoubleMatrix2D && A instanceof SparseDoubleMatrix2D){
					KKT = DoubleFactory2D.sparse.compose(parts);
				}else{
					KKT = DoubleFactory2D.dense.compose(parts);
				}
				x = F1.append(v, w);
				b = F1.append(g, h).assign(Mult.mult(-1));
			}else{
				//H.v + [A]T.w = -g
				DoubleMatrix2D[][] parts = {{ this.H, this.AT }};
				if(H instanceof SparseDoubleMatrix2D && A instanceof SparseDoubleMatrix2D){
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
			KKT = this.H;
			x = v;
			//b = g.copy().assign(Mult.mult(-1));
			b = ColtUtils.scalarMult(g, -1);
		}
		
		//checking residual
		double scaledResidual = Utils.calculateScaledResidual(KKT, x, b);
		log.debug("KKT inversion scaled residual: " + scaledResidual);
		return scaledResidual < toleranceKKT;
	}
	
	/**
	 * Create a full data matrix starting form a symmetric matrix filled only in its subdiagonal elements.
	 */
	protected DoubleMatrix2D createFullDataMatrix(DoubleMatrix2D SubDiagonalSymmMatrix){
		int c = SubDiagonalSymmMatrix.columns();
		DoubleMatrix2D ret = F2.make(c, c);
		for(int i=0; i<c; i++){
			for(int j=0; j<=i; j++){
				ret.setQuick(i, j, SubDiagonalSymmMatrix.getQuick(i, j));
				ret.setQuick(j, i, SubDiagonalSymmMatrix.getQuick(i, j));
			}
		}
		return ret;
	}
}
