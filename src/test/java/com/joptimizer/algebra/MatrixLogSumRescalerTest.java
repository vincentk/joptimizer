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

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import cern.colt.matrix.linalg.SingularValueDecomposition;

import com.joptimizer.util.ColtUtils;
import com.joptimizer.util.Utils;

public class MatrixLogSumRescalerTest extends TestCase {

	private Log log = LogFactory.getLog(this.getClass().getName());
	
	public void testSimpleScalingNoSymm() throws Exception {
		log.debug("testSimpleScalingNoSymm");
		DoubleFactory2D F2 = DoubleFactory2D.sparse;
		DoubleFactory1D F1 = DoubleFactory1D.sparse;
		Algebra ALG = Algebra.DEFAULT;
		final double[][] A = new double[][]{
				{1, 0, 0},
				{0, 0, 2},
				{2, 3, 0},
				{0, 0, 4}
		};
		DoubleMatrix2D AMatrix = F2.make(A);
		MatrixRescaler rescaler = new MatrixLogSumRescaler();
		DoubleMatrix1D[] UV = rescaler.getMatrixScalingFactors(AMatrix);
		DoubleMatrix2D AScaled = ColtUtils.diagonalMatrixMult(UV[0], AMatrix, UV[1]);
		double cn0 = new SingularValueDecomposition(AMatrix).cond();
		double cn1 = new SingularValueDecomposition(AScaled).cond();
		double norm0 = ALG.normInfinity(AMatrix);
		double norm1 = ALG.normInfinity(AScaled);
		log.debug("U : " + ArrayUtils.toString(UV[0].toArray()));
		log.debug("V : " + ArrayUtils.toString(UV[1].toArray()));
		log.debug("AScaled : " + ArrayUtils.toString(AScaled.toArray()));
		log.debug("cn0: " + cn0);
		log.debug("cn1: " + cn1);
		log.debug("norm0: " + norm0);
		log.debug("norm1: " + norm1);
		assertTrue(rescaler.checkScaling(AMatrix, UV[0], UV[1]));
		assertFalse(cn1 > cn0);//not guaranteed by the rescaling
		assertFalse(norm1 > norm0);//not guaranteed by the rescaling
	}
	
	public void testSimpleScalingSymm() throws Exception {
		log.debug("testSimpleScalingSymm");
		DoubleFactory2D F2 = DoubleFactory2D.sparse;
		Algebra ALG = Algebra.DEFAULT;
		double[][] A = new double[][] {
				{1., 0.5e7, 0}, 
				{0.5e7, 2., 0}, 
				{0., 0., 3.e-9}};
		DoubleMatrix2D AMatrix = F2.make(A);
		MatrixRescaler rescaler = new MatrixLogSumRescaler();
		DoubleMatrix1D U = rescaler.getMatrixScalingFactorsSymm(AMatrix);
		DoubleMatrix2D AScaled = ColtUtils.diagonalMatrixMult(U, AMatrix, U);
		double cn0 = new SingularValueDecomposition(AMatrix).cond();
		double cn1 = new SingularValueDecomposition(AScaled).cond();
		double norm0 = ALG.normInfinity(AMatrix);
		double norm1 = ALG.normInfinity(AScaled);
		log.debug("U : " + ArrayUtils.toString(U.toArray()));
		log.debug("AScaled : " + ArrayUtils.toString(AScaled.toArray()));
		log.debug("cn0: " + cn0);
		log.debug("cn1: " + cn1);
		log.debug("norm0: " + norm0);
		log.debug("norm1: " + norm1);
		assertTrue(rescaler.checkScaling(AMatrix, U, U));
		assertFalse(cn1 > cn0);
		assertFalse(norm1 > norm0);
	}
	
	/**
	 * Test of the matrix in Gajulapalli example 2.1.
	 * It is a Pathological Square Matrix.
	 * @see Gajulapalli, Lasdon "Scaling Sparse Matrices for Optimization Algorithms"
	 */
	public void testPathologicalScalingNoSymm() throws Exception {
		log.debug("testPathologicalScalingNoSymm");
		DoubleFactory2D F2 = DoubleFactory2D.sparse;
		DoubleFactory1D F1 = DoubleFactory1D.sparse;
		Algebra ALG = Algebra.DEFAULT;
		double[][] A = new double[][] { 
				{ 1.e0, 1.e10, 1.e20 }, 
				{ 1.e10, 1.e30, 1.e50 }, 
				{ 1.e20, 1.e40, 1.e80 } };
		DoubleMatrix2D AMatrix = F2.make(A);
		MatrixRescaler rescaler = new MatrixLogSumRescaler();
		DoubleMatrix1D[] UV = rescaler.getMatrixScalingFactors(AMatrix);
		DoubleMatrix2D AScaled = ColtUtils.diagonalMatrixMult(UV[0], AMatrix, UV[1]);
		double cn0 = new SingularValueDecomposition(AMatrix).cond();
		double cn1 = new SingularValueDecomposition(AScaled).cond();
		double norm0 = ALG.normInfinity(AMatrix);
		double norm1 = ALG.normInfinity(AScaled);
		log.debug("U : " + ArrayUtils.toString(UV[0].toArray()));
		log.debug("V : " + ArrayUtils.toString(UV[1].toArray()));
		log.debug("AScaled : " + ArrayUtils.toString(AScaled.toArray()));
		log.debug("cn0: " + cn0);
		log.debug("cn1: " + cn1);
		log.debug("norm0: " + norm0);
		log.debug("norm1: " + norm1);
		assertTrue(rescaler.checkScaling(AMatrix, UV[0], UV[1]));
		assertFalse(cn1 > cn0);//not guaranteed by the rescaling
		assertFalse(norm1 > norm0);//not guaranteed by the rescaling
	}
	
	/**
	 * Test of the matrix in Gajulapalli example 3.1.
	 * It is a Pathological Square Matrix.
	 * @see Gajulapalli, Lasdon "Scaling Sparse Matrices for Optimization Algorithms"
	 */
	public void testPathologicalScalingSymm() throws Exception {
		log.debug("testPathologicalScalingSymm");
		DoubleFactory2D F2 = DoubleFactory2D.sparse;
		DoubleFactory1D F1 = DoubleFactory1D.sparse;
		Algebra ALG = Algebra.DEFAULT;
		double[][] A = new double[][] { 
				{ 1.e0,  1.e20, 1.e10, 1.e0  }, 
				{ 1.e20, 1.e20, 1.e0,  1.e40 }, 
				{ 1.e10, 1.e0,  1.e40, 1.e50 },
				{ 1.e0 , 1.e40, 1.e50, 1.e0 }};
		DoubleMatrix2D AMatrix = F2.make(A);
		MatrixRescaler rescaler = new MatrixLogSumRescaler();
		DoubleMatrix1D[] UV = rescaler.getMatrixScalingFactors(AMatrix);
		DoubleMatrix2D AScaled = ColtUtils.diagonalMatrixMult(UV[0], AMatrix, UV[1]);
		double cn0 = new SingularValueDecomposition(AMatrix).cond();
		double cn1 = new SingularValueDecomposition(AScaled).cond();
		double norm0 = ALG.normInfinity(AMatrix);
		double norm1 = ALG.normInfinity(AScaled);
		log.debug("U : " + ArrayUtils.toString(UV[0].toArray()));
		log.debug("V : " + ArrayUtils.toString(UV[1].toArray()));
		log.debug("AScaled : " + ArrayUtils.toString(AScaled.toArray()));
		log.debug("cn0: " + cn0);
		log.debug("cn1: " + cn1);
		log.debug("norm0: " + norm0);
		log.debug("norm1: " + norm1);
		assertTrue(rescaler.checkScaling(AMatrix, UV[0], UV[1]));
		assertFalse(cn1 > cn0);//not guaranteed by the rescaling
		assertFalse(norm1 > norm0);//not guaranteed by the rescaling
	}
	
	/**
	 * Test the matrix norm before and after scaling.
	 * Note that scaling is not guaranteed to give a better condition number.
	 * The test shows some issue with matrix norm, in that this type of scaling
	 * in not effective in the norm with this matrix.
	 */
	public void testMatrixNormScaling7() throws Exception {
		log.debug("testMatrixNormScaling7");
		DoubleFactory1D F1 = DoubleFactory1D.sparse;
		DoubleFactory2D F2 = DoubleFactory2D.sparse;
		Algebra ALG = Algebra.DEFAULT;
		
		String matrixId = "7";
		double[][] A = Utils.loadDoubleMatrixFromFile("factorization" + File.separator + "matrix" + matrixId + ".txt", " ".charAt(0));
		final DoubleMatrix2D AMatrix = F2.make(A);
		
		MatrixRescaler rescaler = new MatrixLogSumRescaler();
		final DoubleMatrix1D U = rescaler.getMatrixScalingFactorsSymm(AMatrix);
		final DoubleMatrix2D AScaled = ColtUtils.diagonalMatrixMult(U, AMatrix, U);
		//log.debug("AScaled : " + ArrayUtils.toString(AScaled.toArray()));
		
		double norm0 = ALG.normInfinity(AMatrix);
		double norm1 = ALG.normInfinity(AScaled);
		log.debug("U : " + ArrayUtils.toString(U.toArray()));
		log.debug("norm0: " + norm0);
		log.debug("norm1: " + norm1);
		
		assertTrue(rescaler.checkScaling(AMatrix, U, U));//note: this must be guaranteed
		log.debug("better matrix norm: " + (norm1 > norm0));
		//assertFalse(norm1 > norm0);//note: this is not guaranteed		
	}
	
	/**
	 * Test the rescaling of a is diagonal with some element < 1.e^16.
	 */
	public void testGetConditionNumberDiagonal() throws Exception {
		log.debug("testGetConditionNumberDiagonal");
		DoubleFactory2D F2 = DoubleFactory2D.sparse;
		DoubleFactory1D F1 = DoubleFactory1D.sparse;
		Algebra ALG = Algebra.DEFAULT;
		
		double[] A = new double[] {1.E-17,168148.06378241107,5.333317404302006E-11,9.724301428859958E-11,4.343924031677448E-10,53042.618161481514,1.2550281021203455E-12,55714.086057404944,16564.267914476874,1.6265469281243343E-12,7.228925943265697E-11,19486.564364392565,315531.47099006834,236523.83171379057,202769.6735227342,2.4925304834427544E-13,2.7996276724404553E-13,2.069135405949759E-12,2530058.817281487,4.663208124742273E-15,2.5926311225234777E-12,2454865.060218241,7.564594931528804E-14,2.944935006524965E-13,7.938509176903875E-13,2546775.969599124,4.36659839706981E-15,3.772728220251383E-9,985020.987902404,971715.0611851265,1941150.6250316042,3.3787344131154E-10,2.8903135775881254E-11,1263.9864262585922,873899.9914494107,153097.08545910483,3.738245318154646E-11,1267390.1117847422,6.50494734416794E-10,3.588511203703992E-11,1231.6604599987518,3.772810869560189E-9,85338.92515278656,3.7382488244903144E-11,437165.36165859725,9.954549425029816E-11,1.8376434881340742E-9,86069.90894488744,1.2087907925307217E11,1.1990761432334067E11,1.163424797835085E11,1.1205515861349094E11,1.2004378300642543E11,8.219259112337953E8,1.1244633984805448E-11,1.1373907469271675E-12,1.9743774924311214E-12,6.301661187526759E-16,6.249382377266375E-16,8.298198098742164E-16,6.447686765999485E-16,1.742229837554675E-16,1.663041351618635E-16};
		DoubleMatrix1D b = Utils.randomValuesVector(A.length, -1, 1, 12345L);
		
		SparseDoubleMatrix2D AMatrix = (SparseDoubleMatrix2D)F2.diagonal(F1.make(A));
		MatrixRescaler rescaler = new MatrixLogSumRescaler();
		DoubleMatrix1D U = rescaler.getMatrixScalingFactorsSymm(AMatrix);
		DoubleMatrix2D AScaled = ColtUtils.diagonalMatrixMult(U, AMatrix, U);
		//log.debug("AScaled: " + ArrayUtils.toString(AScaled.toArray()));
		
		double cn_original = new SingularValueDecomposition(AMatrix).cond();
		double[] cn_2_original = ColtUtils.getConditionNumberRange(new Array2DRowRealMatrix(AMatrix.toArray()), 2);
		double[] cn_00_original = ColtUtils.getConditionNumberRange(new Array2DRowRealMatrix(AMatrix.toArray()), Integer.MAX_VALUE);
		double cn_scaled = new SingularValueDecomposition(AScaled).cond();
		double[] cn_2_scaled = ColtUtils.getConditionNumberRange(new Array2DRowRealMatrix(AScaled.toArray()), Integer.MAX_VALUE);
		double[] cn_00_scaled = ColtUtils.getConditionNumberRange(new Array2DRowRealMatrix(AScaled.toArray()), Integer.MAX_VALUE);
		log.debug("cn_original   : " + ArrayUtils.toString(cn_original));
		log.debug("cn_2_original : " + ArrayUtils.toString(cn_2_original));
		log.debug("cn_00_original: " + ArrayUtils.toString(cn_00_original));
		log.debug("cn_scaled     : " + ArrayUtils.toString(cn_scaled));
		log.debug("cn_2_scaled   : " + ArrayUtils.toString(cn_2_scaled));
		log.debug("cn_00_scaled  : " + ArrayUtils.toString(cn_00_scaled));
		
		assertTrue(rescaler.checkScaling(AMatrix, U, U));//NB: this MUST BE guaranteed by the scaling algorithm
		log.debug("better matrix norm: " + (cn_scaled < cn_original));
		assertTrue(cn_scaled < cn_original);//NB: this IS NOT guaranteed by the scaling algorithm
	}
	
	/**
	 * Test the condition number before and after scaling.
	 * Note that scaling is not guaranteed to give a better condition number.
	 * The test shows some issue with condition number, in that this type of scaling
	 * in not effective in the condition number with this matrix.
	 */
	public void testGetConditionNumberFromFile7() throws Exception {
		log.debug("testGetConditionNumberFromFile7");
		DoubleFactory2D F2 = DoubleFactory2D.sparse;
		DoubleFactory1D F1 = DoubleFactory1D.sparse;
		Algebra ALG = Algebra.DEFAULT;
		
		String matrixId = "7";
		double[][] A = Utils.loadDoubleMatrixFromFile("factorization" + File.separator + "matrix" + matrixId + ".txt", " ".charAt(0));
		SparseDoubleMatrix2D AMatrix = (SparseDoubleMatrix2D)F2.make(A);

		MatrixRescaler rescaler = new MatrixLogSumRescaler();
		DoubleMatrix1D Uv = rescaler.getMatrixScalingFactorsSymm(AMatrix);
		DoubleMatrix2D AScaled = ColtUtils.diagonalMatrixMult(Uv, AMatrix, Uv);
		//log.debug("AScaled: " + ArrayUtils.toString(AScaled.toArray()));
		
		double cn_original = new SingularValueDecomposition(AMatrix).cond();
		double[] cn_2_original = ColtUtils.getConditionNumberRange(new Array2DRowRealMatrix(AMatrix.toArray()), 2);
		double[] cn_00_original = ColtUtils.getConditionNumberRange(new Array2DRowRealMatrix(AMatrix.toArray()), Integer.MAX_VALUE);
		double cn_scaled = new SingularValueDecomposition(AScaled).cond();
		double[] cn_2_scaled = ColtUtils.getConditionNumberRange(new Array2DRowRealMatrix(AScaled.toArray()), Integer.MAX_VALUE);
		double[] cn_00_scaled = ColtUtils.getConditionNumberRange(new Array2DRowRealMatrix(AScaled.toArray()), Integer.MAX_VALUE);
		log.debug("cn_original   : " + ArrayUtils.toString(cn_original));
		log.debug("cn_2_original : " + ArrayUtils.toString(cn_2_original));
		log.debug("cn_00_original: " + ArrayUtils.toString(cn_00_original));
		log.debug("cn_scaled     : " + ArrayUtils.toString(cn_scaled));
		log.debug("cn_2_scaled   : " + ArrayUtils.toString(cn_2_scaled));
		log.debug("cn_00_scaled  : " + ArrayUtils.toString(cn_00_scaled));

		assertTrue(rescaler.checkScaling(AMatrix, Uv, Uv));//NB: this MUST BE guaranteed by the scaling algorithm
		log.debug("better matrix norm: " + (cn_scaled < cn_original));
	  //assertTrue(cn_scaled < cn_original);//NB: this IS NOT guaranteed by the scaling algorithm
	}
	
	/**
	 * Test the condition number before and after scaling.
	 * Note that scaling is not guaranteed to give a better condition number.
	 * The test shows some issue with condition number, in that this type of scaling
	 * in not effective in the condition number with this matrix.
	 */
	public void testGetConditionNumberFromFile13() throws Exception {
		log.debug("testGetConditionNumberFromFile13");
		DoubleFactory2D F2 = DoubleFactory2D.sparse;
		DoubleFactory1D F1 = DoubleFactory1D.sparse;
		Algebra ALG = Algebra.DEFAULT;
		
		String matrixId = "13";
		double[][] A = Utils.loadDoubleMatrixFromFile("factorization" + File.separator + "matrix" + matrixId + ".csv");
		SparseDoubleMatrix2D AMatrix = (SparseDoubleMatrix2D)F2.make(A);

		MatrixRescaler rescaler = new MatrixLogSumRescaler();
		DoubleMatrix1D Uv = rescaler.getMatrixScalingFactorsSymm(AMatrix);
		DoubleMatrix2D AScaled = ColtUtils.diagonalMatrixMult(Uv, AMatrix, Uv);		
		//log.debug("AScaled: " + ArrayUtils.toString(AScaled.toArray()));
		
		double cn_original = new SingularValueDecomposition(AMatrix).cond();
		double[] cn_2_original = ColtUtils.getConditionNumberRange(new Array2DRowRealMatrix(AMatrix.toArray()), 2);
		double[] cn_00_original = ColtUtils.getConditionNumberRange(new Array2DRowRealMatrix(AMatrix.toArray()), Integer.MAX_VALUE);
		double cn_scaled = new SingularValueDecomposition(AScaled).cond();
		double[] cn_2_scaled = ColtUtils.getConditionNumberRange(new Array2DRowRealMatrix(AScaled.toArray()), Integer.MAX_VALUE);
		double[] cn_00_scaled = ColtUtils.getConditionNumberRange(new Array2DRowRealMatrix(AScaled.toArray()), Integer.MAX_VALUE);
		log.debug("cn_original   : " + ArrayUtils.toString(cn_original));
		log.debug("cn_2_original : " + ArrayUtils.toString(cn_2_original));
		log.debug("cn_00_original: " + ArrayUtils.toString(cn_00_original));
		log.debug("cn_scaled     : " + ArrayUtils.toString(cn_scaled));
		log.debug("cn_2_scaled   : " + ArrayUtils.toString(cn_2_scaled));
		log.debug("cn_00_scaled  : " + ArrayUtils.toString(cn_00_scaled));
		
		assertTrue(rescaler.checkScaling(AMatrix, Uv, Uv));//NB: this MUST BE guaranteed by the scaling algorithm
		log.debug("better matrix norm: " + (cn_scaled < cn_original));
		//assertTrue(cn_scaled < cn_original);//NB: this IS NOT guaranteed by the scaling algorithm
	}
}
