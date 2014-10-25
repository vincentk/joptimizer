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

import java.io.File;

import junit.framework.TestCase;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularValueDecomposition;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;

import com.joptimizer.util.ColtUtils;
import com.joptimizer.util.TestUtils;
import com.joptimizer.util.Utils;

/**
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class CholeskyFactorizationTest extends TestCase {
	protected Algebra ALG = Algebra.DEFAULT;
	protected DoubleFactory2D F2 = DoubleFactory2D.dense;
	protected DoubleFactory1D F1 = DoubleFactory1D.dense;
	private Log log = LogFactory.getLog(this.getClass().getName());

	public void testInvert1() throws Exception {
		log.debug("testInvert1");
		double[][] QData = new double[][] { 
				{ 1, .12, .13, .14, .15 },
				{ .12, 2, .23, .24, .25 }, 
				{ .13, .23, 3, 0, 0 },
				{ .14, .24, 0, 4, 0 }, 
				{ .15, .25, 0, 0, 5 } };
		RealMatrix Q = MatrixUtils.createRealMatrix(QData);

		CholeskyFactorization myc = new CholeskyFactorization(DoubleFactory2D.dense.make(QData));
		myc.factorize();
		RealMatrix L = new Array2DRowRealMatrix(myc.getL().toArray());
		RealMatrix LT = new Array2DRowRealMatrix(myc.getLT().toArray());
		log.debug("L: " + ArrayUtils.toString(L.getData()));
		log.debug("LT: " + ArrayUtils.toString(LT.getData()));
		log.debug("L.LT: " + ArrayUtils.toString(L.multiply(LT).getData()));
		log.debug("LT.L: " + ArrayUtils.toString(LT.multiply(L).getData()));
		
		// check Q = L.LT
		double norm = L.multiply(LT).subtract(Q).getNorm();
		log.debug("norm: " + norm);
		assertTrue(norm < 1.E-15);
		
		RealMatrix LInv = new SingularValueDecomposition(L).getSolver().getInverse();
		log.debug("LInv: " + ArrayUtils.toString(LInv.getData()));
		RealMatrix LInvT = LInv.transpose();
		log.debug("LInvT: " + ArrayUtils.toString(LInvT.getData()));
		RealMatrix LTInv = new SingularValueDecomposition(LT).getSolver().getInverse();
		log.debug("LTInv: " + ArrayUtils.toString(LTInv.getData()));
		RealMatrix LTInvT = LTInv.transpose();
		log.debug("LTInvT: " + ArrayUtils.toString(LTInvT.getData()));
		log.debug("LInv.LInvT: " + ArrayUtils.toString(LInv.multiply(LInvT).getData()));
		log.debug("LTInv.LTInvT: " + ArrayUtils.toString(LTInv.multiply(LTInvT).getData()));
		
		RealMatrix Id = MatrixUtils.createRealIdentityMatrix(Q.getRowDimension());
		//check Q.(LTInv * LInv) = 1
		norm = Q.multiply(LTInv.multiply(LInv)).subtract(Id).getNorm();
		log.debug("norm: " + norm);
		assertTrue(norm < 5.E-15);
		
		// check Q.QInv = 1
		RealMatrix QInv = MatrixUtils.createRealMatrix(myc.getInverse().toArray());
		norm = Q.multiply(QInv).subtract(Id).getNorm();
		log.debug("norm: " + norm);
		assertTrue(norm < 1.E-15);
	}
	
	/**
	 * The same as before, but with rescaling.
	 */
	public void testInvert2() throws Exception {
		log.debug("testInvert2");
		double[][] QData = new double[][] { 
				{ 1, .12, .13, .14, .15 },
				{ .12, 2, .23, .24, .25 }, 
				{ .13, .23, 3, 0, 0 },
				{ .14, .24, 0, 4, 0 }, 
				{ .15, .25, 0, 0, 5 } };
		RealMatrix Q = MatrixUtils.createRealMatrix(QData);

		CholeskyFactorization myc = new CholeskyFactorization(DoubleFactory2D.dense.make(QData), new Matrix1NornRescaler());
		myc.factorize();
		RealMatrix L = new Array2DRowRealMatrix(myc.getL().toArray());
		RealMatrix LT = new Array2DRowRealMatrix(myc.getLT().toArray());
		log.debug("L: " + ArrayUtils.toString(L.getData()));
		log.debug("LT: " + ArrayUtils.toString(LT.getData()));
		log.debug("L.LT: " + ArrayUtils.toString(L.multiply(LT).getData()));
		log.debug("LT.L: " + ArrayUtils.toString(LT.multiply(L).getData()));
		
		// check Q = L.LT
		double norm = L.multiply(LT).subtract(Q).getNorm();
		log.debug("norm: " + norm);
		assertTrue(norm < 1.E-15);
		
		RealMatrix LInv = new SingularValueDecomposition(L).getSolver().getInverse();
		log.debug("LInv: " + ArrayUtils.toString(LInv.getData()));
		RealMatrix LInvT = LInv.transpose();
		log.debug("LInvT: " + ArrayUtils.toString(LInvT.getData()));
		RealMatrix LTInv = new SingularValueDecomposition(LT).getSolver().getInverse();
		log.debug("LTInv: " + ArrayUtils.toString(LTInv.getData()));
		RealMatrix LTInvT = LTInv.transpose();
		log.debug("LTInvT: " + ArrayUtils.toString(LTInvT.getData()));
		log.debug("LInv.LInvT: " + ArrayUtils.toString(LInv.multiply(LInvT).getData()));
		log.debug("LTInv.LTInvT: " + ArrayUtils.toString(LTInv.multiply(LTInvT).getData()));
		
		RealMatrix Id = MatrixUtils.createRealIdentityMatrix(Q.getRowDimension());
		//check Q.(LTInv * LInv) = 1
		norm = Q.multiply(LTInv.multiply(LInv)).subtract(Id).getNorm();
		log.debug("norm: " + norm);
		assertTrue(norm < 5.E-15);
		
		// check Q.QInv = 1
		RealMatrix QInv = MatrixUtils.createRealMatrix(myc.getInverse().toArray());
		norm = Q.multiply(QInv).subtract(Id).getNorm();
		log.debug("norm: " + norm);
		assertTrue(norm < 1.E-15);
	}
	
	/**
	 * This test shows that the correct check of the inversion accuracy must be done with
	 * the scaled residual, not with the simple norm ||A.x-b||
	 */
	public void testScaledResidual() throws Exception {
		log.debug("testScaledResidual");
		double[][] G = TestUtils.loadDoubleMatrixFromFile("factorization" + File.separator	+ "matrix1.csv");
		RealMatrix Q = MatrixUtils.createRealMatrix(G);
		int dim = Q.getRowDimension();
	
		RealVector b = new ArrayRealVector(new double[]{1,2,3,4,5,6,7,8,9,10});
		
		CholeskyFactorization cs = new CholeskyFactorization(DoubleFactory2D.dense.make(Q.getData()));
		cs.factorize();
		RealVector x = new ArrayRealVector(cs.solve(DoubleFactory1D.dense.make(b.toArray())).toArray());
		
		//scaledResidual = ||Ax-b||_oo/( ||A||_oo . ||x||_oo + ||b||_oo )
		// with ||x||_oo = max(x[i])
		double scaledResidual = Utils.calculateScaledResidual(Q.getData(), x.toArray(), b.toArray());
		log.debug("scaledResidual: " + scaledResidual);
		assertTrue(scaledResidual < Utils.getDoubleMachineEpsilon());
		
		//b - Q.x
		//checking the simple norm, this will fail
		double n1 = b.subtract(Q.operate(x)).getNorm();
		log.debug("||b - Q.x||: " + n1);
		//assertTrue(n1 < 1.E-8);
	}
	
	/**
	 * Not positive matrix, must fail
	 */
	public void testInvertNotPositive() throws Exception {
		log.debug("testInvertNotPositive");
		double[][] G = TestUtils.loadDoubleMatrixFromFile("factorization" + File.separator	+ "matrix4.csv");
		DoubleMatrix2D QMatrix = DoubleFactory2D.sparse.make(G);
		log.debug("QMatrix: " + ArrayUtils.toString(QMatrix.toArray()));//10x10 symm positive
		log.debug("cardinality: " + QMatrix.cardinality());
		int rows = QMatrix.rows();
		int cols = QMatrix.columns();
		int dim = rows*cols;
		int nz = dim - QMatrix.cardinality();
		log.debug("sparsity index: " + 100*new Double(nz)/dim +" %");
		
		try{
			CholeskyFactorization cs = new CholeskyFactorization(DoubleFactory2D.dense.make(G));
			cs.factorize();
		}catch(Exception e){
			assertTrue(true);//ok, the matrix is not positive
			return;
		}
		
		//if here, not good
		fail();
		
	}
	
	/**
	 * The matrix6 has a regular Cholesky factorization (as given by Mathematica) 
	 * This test shows how rescaling a matrix can help its factorization.
	 */
	public void testScale6() throws Exception {
		log.debug("testScale6");
		DoubleFactory2D F2 = DoubleFactory2D.sparse;
		DoubleFactory1D F1 = DoubleFactory1D.sparse;
		Algebra ALG = Algebra.DEFAULT;
		
		String matrixId = "6";
		double[][] A = TestUtils.loadDoubleMatrixFromFile("factorization" + File.separator + "matrix" + matrixId + ".txt", " ".charAt(0));
		SparseDoubleMatrix2D AMatrix = (SparseDoubleMatrix2D) F2.make(A);
		int dim = AMatrix.rows();
		
		CholeskyFactorization cs;
		try{
			cs = new CholeskyFactorization(AMatrix);
			cs.factorize();
		}catch(Exception e){
			log.debug("numeric problem, try to rescale the matrix");
			MatrixRescaler rescaler = new Matrix1NornRescaler();
			DoubleMatrix1D Uv = rescaler.getMatrixScalingFactorsSymm(AMatrix);
			DoubleMatrix2D U = F2.diagonal(Uv);
			
			assertTrue(rescaler.checkScaling(ColtUtils.fillSubdiagonalSymmetricMatrix(AMatrix), Uv, Uv));
			
			DoubleMatrix2D AScaled = ColtUtils.diagonalMatrixMult(Uv, AMatrix, Uv);
			cs = new CholeskyFactorization(AScaled);
			cs.factorize();
			
			//NOTE: with scaling, we must solve U.A.U.z = U.b, after that we have x = U.z
			
			//solve Q.x = b
			DoubleMatrix1D b = Utils.randomValuesVector(dim, -1, 1, 12345L);
			DoubleMatrix1D x = cs.solve(ALG.mult(U, b));
			double scaledResidualx = Utils.calculateScaledResidual(AMatrix, ALG.mult(U, x), b);
			log.debug("scaledResidualx: " + scaledResidualx);
			assertTrue(scaledResidualx < Utils.getDoubleMachineEpsilon());
			
			//solve Q.X = B
			DoubleMatrix2D B = Utils.randomValuesMatrix(dim, 5, -1, 1, 12345L);
			DoubleMatrix2D X = cs.solve(ALG.mult(U, B));
			double scaledResidualX = Utils.calculateScaledResidual(AMatrix, ALG.mult(U, X), B);
			log.debug("scaledResidualX: " + scaledResidualX);
			assertTrue(scaledResidualX < Utils.getDoubleMachineEpsilon());
		}
	}
}
