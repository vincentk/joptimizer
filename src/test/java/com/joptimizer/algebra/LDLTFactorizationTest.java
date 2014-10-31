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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;

import com.joptimizer.util.ColtUtils;
import com.joptimizer.util.TestUtils;
import com.joptimizer.util.Utils;

/**
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class LDLTFactorizationTest extends TestCase {
	private Log log = LogFactory.getLog(this.getClass().getName());

	public void testFactorize1() throws Exception {
		log.debug("testFactorize1");
		double[][] QData = new double[][] { 
				{ 1, .12, .13, .14, .15 },
				{ .12, 2, .23, .24, .25 }, 
				{ .13, .23, 3, 0, 0 },
				{ .14, .24, 0, 4, 0 }, 
				{ .15, .25, 0, 0, 5 } };
		RealMatrix Q = MatrixUtils.createRealMatrix(QData);

		LDLTFactorization myc = new LDLTFactorization(new DenseDoubleMatrix2D(QData));
		myc.factorize();
		RealMatrix L = new Array2DRowRealMatrix(myc.getL().toArray());
		RealMatrix D = new Array2DRowRealMatrix(myc.getD().toArray());
		RealMatrix LT = new Array2DRowRealMatrix(myc.getLT().toArray());
		log.debug("L: " + L);
		log.debug("D: " + D);
		log.debug("LT: " + LT);
		log.debug("L.D.LT: " + L.multiply(D.multiply(LT)));

		// check Q = L.D.LT
		double norm = L.multiply(D).multiply(LT).subtract(Q).getNorm();
		log.debug("norm: " + norm);
		assertTrue(norm < Utils.getDoubleMachineEpsilon());
	}

	public void testFactorizeNotPositive() throws Exception {
		log.debug("testFactorizeNotPositive");
		double[][] QData = new double[][] { 
				{ 1, 0 }, 
				{ 0, -1 } };
		RealMatrix Q = MatrixUtils.createRealMatrix(QData);

		LDLTFactorization myc = new LDLTFactorization(new DenseDoubleMatrix2D(QData));
		myc.factorize();
		RealMatrix L = new Array2DRowRealMatrix(myc.getL().toArray());
		RealMatrix D = new Array2DRowRealMatrix(myc.getD().toArray());
		RealMatrix LT = new Array2DRowRealMatrix(myc.getLT().toArray());
		log.debug("L: " + L);
		log.debug("D: " + D);
		log.debug("LT: " + LT);
		log.debug("L.D.LT: " + L.multiply(D.multiply(LT)));
		
		// check Q = L.D.LT
		double norm = L.multiply(D).multiply(LT).subtract(Q).getNorm();
		log.debug("norm: " + norm);
		assertTrue(norm < Utils.getDoubleMachineEpsilon());
	}

	public void testFactorizeSingular() throws Exception {
		log.debug("testFactorizeSingular");
		double[][] QData = new double[][] { 
				{ 1, 0, 1 }, 
				{ 0, -1, 0 },
				{ 1, 0, 1 } };

		try{
			LDLTFactorization myc = new LDLTFactorization(new DenseDoubleMatrix2D(QData));
			myc.factorize();
			
			fail();//the factorization must detect the singularity
		}catch(Exception e){
			assertTrue(true);///OK
		}		
	}
	
	/**
	 * The matrix7 has a regular Cholesky factorization (as given by Mathematica) 
	 * so JOptimizer must be able to factorize it
	 */
	public void testScale6() throws Exception {
		log.debug("testScale6");
		DoubleFactory2D F2 = DoubleFactory2D.sparse;
		Algebra ALG = Algebra.DEFAULT;
		
		String matrixId = "6";
		double[][] A = TestUtils.loadDoubleMatrixFromFile("factorization" + File.separator + "matrix" + matrixId + ".txt", " ".charAt(0));
		SparseDoubleMatrix2D AMatrix = (SparseDoubleMatrix2D) F2.make(A);
		int dim = AMatrix.rows();
		
		LDLTFactorization myc;
		try{
			myc = new LDLTFactorization(new DenseDoubleMatrix2D(A));
			myc.factorize();
		}catch(Exception e){
			log.debug("numeric problem, try to rescale the matrix");
			MatrixRescaler rescaler = new Matrix1NornRescaler();
			DoubleMatrix1D Uv = rescaler.getMatrixScalingFactorsSymm(AMatrix);
			DoubleMatrix2D U = F2.diagonal(Uv);
			
			assertTrue(rescaler.checkScaling(ColtUtils.fillSubdiagonalSymmetricMatrix(AMatrix), Uv, Uv));
			
			DoubleMatrix2D AScaled = ColtUtils.diagonalMatrixMult(Uv, AMatrix, Uv);
			myc = new LDLTFactorization(AScaled);
			myc.factorize();
			
			//NOTE: with scaling, we must solve U.A.U.z = U.b, after that we have x = U.z
			
			//solve Q.x = b
			DoubleMatrix1D b = Utils.randomValuesVector(dim, -1, 1, 12345L);
			DoubleMatrix1D x = myc.solve(ALG.mult(U, b));
			double scaledResidualx = Utils.calculateScaledResidual(AMatrix, ALG.mult(U, x), b);
			log.debug("scaledResidualx: " + scaledResidualx);
			assertTrue(scaledResidualx < 1.e-15);
		}
	}
}
