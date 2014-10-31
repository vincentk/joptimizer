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
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import cern.colt.function.IntIntDoubleFunction;
import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;

import com.joptimizer.util.Utils;

/**
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class CholeskyComparisonTest extends TestCase {
	private Log log = LogFactory.getLog(this.getClass().getName());

	public void testDummy() throws Exception{
		log.debug("testDummy");
	}
	
	/**
	 * Test with a generic sparse matrix.
	 */
	public void testCompareFactorizations1() throws Exception{
		log.debug("testCompareFactorizations1");
		
		int iterations = 4;
		int m = 1000;
		int dim = m*m;
		
		DoubleMatrix2D sMatrix = Utils.randomValuesSparseMatrix(m, m, -10, 10, 0.97, 12345L);
		//log.debug("sMatrix: " + Utils.toString(sMatrix.toArray()));
		DoubleMatrix2D QMatrix = Algebra.DEFAULT.mult(sMatrix, Algebra.DEFAULT.transpose(sMatrix));//positive and symmetric
		double[][] QMatrixData = QMatrix.toArray();
		//log.debug("QMatrix: " + Utils.toString(QMatrix.toArray()));
		RealMatrix A  = MatrixUtils.createRealMatrix(QMatrix.toArray());
		log.debug("cardinality: " + QMatrix.cardinality());
		int nz = dim - QMatrix.cardinality();
		log.debug("sparsity index: " + 100*new Double(nz)/dim +" %");
		
		//try Cholesky 1
		long t1F = System.currentTimeMillis();
		CholeskyFactorization myc1 = null;
		for(int i=0; i<iterations; i++){
			//factorization
			myc1 = new CholeskyFactorization(QMatrix);
			myc1.factorize();
		}
		log.debug("Cholesky standard factorization time: " + (System.currentTimeMillis()-t1F));
		RealMatrix L1 = MatrixUtils.createRealMatrix(myc1.getL().toArray());
		double norm1 = A.subtract(L1.multiply(L1.transpose())).getNorm();
		log.debug("norm1                               : " + norm1);
		assertEquals(0., norm1, 1.E-12 * Math.sqrt(dim));
		long t1I = System.currentTimeMillis();
		for(int i=0; i<iterations; i++){
			//inversion
			myc1.getInverse();
		}
		log.debug("Cholesky standard inversion time    : " + (System.currentTimeMillis()-t1I));
		RealMatrix AInv1 = MatrixUtils.createRealMatrix(myc1.getInverse().toArray());
		
		//try Cholesky 2
		long t2F = System.currentTimeMillis();
		CholeskyRCTFactorization myc2 = null;
		for(int i=0; i<iterations; i++){
		  //factorization
			myc2 = new CholeskyRCTFactorization(QMatrix);
			myc2.factorize();
		}
		log.debug("Cholesky RCT factorization time     : " + (System.currentTimeMillis()-t2F));
		RealMatrix L2 = MatrixUtils.createRealMatrix(myc2.getL().toArray());
		double norm2 = A.subtract(L2.multiply(L2.transpose())).getNorm();
		log.debug("norm2                               : " + norm2);
		assertEquals(0., norm2, 1.E-12 * Math.sqrt(dim));
		assertEquals(0., L1.subtract(L2).getNorm(), 1.E-10 * Math.sqrt(dim));
		long t2I = System.currentTimeMillis();
		for(int i=0; i<iterations; i++){
			//inversion
			myc2.getInverse();
		}
		log.debug("Cholesky RCT inversion time         : " + (System.currentTimeMillis()-t2I));
		RealMatrix AInv2 = MatrixUtils.createRealMatrix(myc2.getInverse().toArray());
		assertEquals(0., AInv1.subtract(AInv2).getNorm(), 1.E-10 * Math.sqrt(dim));
		
		//try Cholesky 3
		long t3F = System.currentTimeMillis();
		CholeskyRCFactorization myc3 = null;
		for(int i=0; i<iterations; i++){
			//factorization
			myc3 = new CholeskyRCFactorization(QMatrix);
			myc3.factorize();
		}
		log.debug("Cholesky RC factorization time      : " + (System.currentTimeMillis()-t3F));
		RealMatrix L3 = MatrixUtils.createRealMatrix(myc3.getL().toArray());
		double norm3 = A.subtract(L3.multiply(L3.transpose())).getNorm();
		log.debug("norm3                               : " + norm3);
		assertEquals(0., norm3, 1.E-12 * Math.sqrt(dim));
		assertEquals(0., L1.subtract(L3).getNorm(), 1.E-10 * Math.sqrt(dim));
		long t3I = System.currentTimeMillis();
		for(int i=0; i<iterations; i++){
			//inversion
			myc3.getInverse();
		}
		log.debug("Cholesky RC inversion time          : " + (System.currentTimeMillis()-t3I));
		RealMatrix AInv3 = MatrixUtils.createRealMatrix(myc3.getInverse().toArray());
		assertEquals(0., AInv1.subtract(AInv3).getNorm(), 1.E-10 * Math.sqrt(dim));
		
		//try Cholesky 4
		long t4 = System.currentTimeMillis();
		LDLTFactorization myc4 = null;
		for(int i=0; i<iterations; i++){
			myc4 = new LDLTFactorization(new DenseDoubleMatrix2D(QMatrixData));
			myc4.factorize();
		}
		log.debug("Cholesky LDLT factorization time    : " + (System.currentTimeMillis()-t4));
		RealMatrix L4 = MatrixUtils.createRealMatrix(myc4.getL().toArray());
		RealMatrix D4 = MatrixUtils.createRealMatrix(myc4.getD().toArray());
		double norm4 = A.subtract(L4.multiply(D4.multiply(L4.transpose()))).getNorm();
		log.debug("norm4                               : " + norm4);
		assertEquals(0., norm4, 1.E-12 * Math.sqrt(dim));
		
		//try Cholesky 5
		long t5 = System.currentTimeMillis();
		CholeskySparseFactorization myc5 = null;
		for(int i=0; i<iterations; i++){
			myc5 = new CholeskySparseFactorization(new SparseDoubleMatrix2D(QMatrix.toArray()));
			myc5.factorize();
		}
		log.debug("Cholesky sparse factorization time  : " + (System.currentTimeMillis()-t5));
		RealMatrix L5 = MatrixUtils.createRealMatrix(myc5.getL().toArray());
		double norm5 = A.subtract(L5.multiply(L5.transpose())).getNorm();
		log.debug("norm5                               : " + norm5);
		assertEquals(0., norm5, 1.E-12 * Math.sqrt(dim));
		assertEquals(0., L1.subtract(L5).getNorm(), 1.E-10 * Math.sqrt(dim));
	}
	
	/**
	 * Test with a diagonal matrix.
	 */
	public void testCompareFactorizations2() throws Exception{
		log.debug("testCompareFactorizations2");
		
		int iterations = 4;
		int m = 1000;
		int dim = m*m;
		
		DoubleMatrix2D QMatrix = DoubleFactory2D.sparse.diagonal(DoubleFactory1D.sparse.make(m, 1.));
		//log.debug("QMatrix: " + Utils.toString(QMatrix.toArray()));
		RealMatrix A  = MatrixUtils.createRealMatrix(QMatrix.toArray());
		log.debug("cardinality: " + QMatrix.cardinality());
		int nz = dim - QMatrix.cardinality();
		log.debug("sparsity index: " + 100*new Double(nz)/dim +" %");
		
		//try Cholesky 1
		long t1F = System.currentTimeMillis();
		CholeskyFactorization myc1 = null;
		for(int i=0; i<iterations; i++){
			//factorization
			myc1 = new CholeskyFactorization(QMatrix);
			myc1.factorize();
		}
		log.debug("Cholesky standard factorization time: " + (System.currentTimeMillis()-t1F));
		RealMatrix L1 = MatrixUtils.createRealMatrix(myc1.getL().toArray());
		double norm1 = A.subtract(L1.multiply(L1.transpose())).getNorm();
		log.debug("norm1                               : " + norm1);
		assertEquals(0., norm1, 1.E-12 * Math.sqrt(dim));
		long t1I = System.currentTimeMillis();
		for(int i=0; i<iterations; i++){
			//inversion
			myc1.getInverse();
		}
		log.debug("Cholesky standard inversion time    : " + (System.currentTimeMillis()-t1I));
		
		//try Cholesky 5
		long t5 = System.currentTimeMillis();
		CholeskySparseFactorization myc5 = null;
		for(int i=0; i<iterations; i++){
			myc5 = new CholeskySparseFactorization(new SparseDoubleMatrix2D(QMatrix.toArray()));
			myc5.factorize();
		}
		log.debug("Cholesky sparse factorization time  : " + (System.currentTimeMillis()-t5));
		RealMatrix L5 = MatrixUtils.createRealMatrix(myc5.getL().toArray());
		double norm5 = A.subtract(L5.multiply(L5.transpose())).getNorm();
		log.debug("norm5                               : " + norm5);
		assertEquals(0., norm5, 1.E-12 * Math.sqrt(dim));
		assertEquals(0., L1.subtract(L5).getNorm(), 1.E-10 * Math.sqrt(dim));
	}
	
	/**
	 * This test shows poor precision with a high dimension matrix.
	 * The matrix is upper left diagonal with diagonalLength = dim -1.
	 * Is the closed form decomposition better than the classical one?
	 * The answer seems to be: it is not said...
	 */
	public void testFromFileSer1() throws Exception {
		log.debug("testFromFileSer1");
		String matrixId = "sparseDoubleMatrix2D_1";
		SparseDoubleMatrix2D AMatrix = (SparseDoubleMatrix2D)Utils.deserializeObject("factorization" + File.separator + matrixId + ".ser");
		int dim = AMatrix.rows();
		log.debug("dim: " + dim);
		
		CholeskyUpperDiagonalFactorization cs = new CholeskyUpperDiagonalFactorization(AMatrix, dim -1, new Matrix1NornRescaler());
		cs.factorize();
		
		//solve A.x = b
		DoubleMatrix1D b = Utils.randomValuesVector(dim, -1, 1, 12345L);
		DoubleMatrix1D x = cs.solve(b);
		double scaledResidualx1 = Utils.calculateScaledResidual(AMatrix, x, b);
		log.debug("scaledResidualx1: " + scaledResidualx1);
		assertTrue(scaledResidualx1 < 5.e-7);
		
		//solve A.y = B
		DoubleMatrix2D B = Utils.randomValuesMatrix(dim, 5, -1, 1, 12345L);
		DoubleMatrix2D X = cs.solve(B);
		double scaledResidualX1 = Utils.calculateScaledResidual(AMatrix, X, B);
		log.debug("scaledResidualX1: " + scaledResidualX1);
		assertTrue(scaledResidualX1 < 5.e-6);
		
		//try to normalize with respect to A[dim-1][dim-1] in order to have the closed form decomposition
		final double s = AMatrix.getQuick(dim-1, dim-1);
		log.debug("A[dim-1][dim-1]: " + AMatrix.getQuick(dim-1, dim-1));
		AMatrix.forEachNonZero(new IntIntDoubleFunction() {
			@Override
			public double apply(int i, int j, double aij) {
				return aij / s;
			}
		});
		
		cs = new CholeskyUpperDiagonalFactorization(AMatrix, dim -1, new Matrix1NornRescaler());
		cs.factorize();
		
		//solve A.x = b
		x = cs.solve(b);
		double scaledResidualx2 = Utils.calculateScaledResidual(AMatrix, x, b);
		log.debug("scaledResidualx2: " + scaledResidualx2);
		assertTrue(scaledResidualx2 < 5.e-7);
		
		//solve A.y = B
		X = cs.solve(B);
		double scaledResidualX2 = Utils.calculateScaledResidual(AMatrix, X, B);
		log.debug("scaledResidualX2: " + scaledResidualX2);
		assertTrue(scaledResidualX2 < 5.e-6);
	
		//we have a very small benefit with the closed form decomposition
		assertTrue(scaledResidualx2 < scaledResidualx1);
		//assertTrue(scaledResidualX2 < scaledResidualX1);//this fails
	}
}
