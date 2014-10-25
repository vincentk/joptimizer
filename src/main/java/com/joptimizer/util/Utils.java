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
package com.joptimizer.util;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.URI;
import java.util.Random;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import cern.jet.math.Functions;

/**
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class Utils {
	
	private static Double RELATIVE_MACHINE_PRECISION = Double.NaN;
	public static Log log = LogFactory.getLog(Utils.class);
	
	public static File getClasspathResourceAsFile(String resourceName) throws Exception{
    return new File(new URI(Thread.currentThread().getContextClassLoader().getResource(resourceName).toString()));
  }
	
	public static DoubleMatrix1D randomValuesVector(int dim, double min, double max) {
		return randomValuesVector(dim, min, max, null);
	}

	public static DoubleMatrix1D randomValuesVector(int dim, double min, double max, Long seed) {
		Random random = (seed != null) ? new Random(seed) : new Random();

		double[] v = new double[dim];
		for (int i = 0; i < dim; i++) {
			v[i] = min + random.nextDouble() * (max - min);
		}
		return DoubleFactory1D.dense.make(v);
	}
	
	public static DoubleMatrix2D randomValuesMatrix(int rows, int cols, double min, double max) {
		return randomValuesMatrix(rows, cols, min, max, null);
	}

	public static DoubleMatrix2D randomValuesMatrix(int rows, int cols, double min, double max, Long seed) {
		Random random = (seed != null) ? new Random(seed) : new Random();

		double[][] matrix = new double[rows][cols];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				matrix[i][j] = min + random.nextDouble() * (max - min);
			}
		}
		return DoubleFactory2D.dense.make(matrix);
	}
	
	public static DoubleMatrix2D randomValuesSparseMatrix(int rows, int cols, double min, double max, double sparsityIndex, Long seed) {
		Random random = (seed != null) ? new Random(seed) : new Random();
		double minThreshold = min + sparsityIndex * (max-min);
		
		double[][] matrix = new double[rows][cols];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				double d = min + random.nextDouble() * (max - min);
				if(d > minThreshold){
					matrix[i][j] = d;
				}
			}
		}
		return DoubleFactory2D.sparse.make(matrix);
	}
	
	/**
	 * @TODO: check this!!!
	 * @see "http://mathworld.wolfram.com/PositiveDefiniteMatrix.html"
	 */
	public static DoubleMatrix2D randomValuesPositiveMatrix(int rows, int cols, double min, double max, Long seed) {
		DoubleMatrix2D Q = Utils.randomValuesMatrix(rows, cols, min, max, seed);
		DoubleMatrix2D P = Algebra.DEFAULT.mult(Q, Algebra.DEFAULT.transpose(Q.copy()));
    return Algebra.DEFAULT.mult(P, P);
	}
	
	/**
	 * Calculate the scaled residual 
	 * <br> ||Ax-b||_oo/( ||A||_oo . ||x||_oo + ||b||_oo ), with
	 * <br> ||x||_oo = max(||x[i]||)
	 */
	public static double calculateScaledResidual(DoubleMatrix2D A, DoubleMatrix2D X, DoubleMatrix2D B){
		double residual = -Double.MAX_VALUE;
		double niX = Algebra.DEFAULT.normInfinity(X);
		double niB = Algebra.DEFAULT.normInfinity(B);
		if(Double.compare(niX, 0.)==0 && Double.compare(niB, 0.)==0){
			return 0;
		}else{
			double num = Algebra.DEFAULT.normInfinity(Algebra.DEFAULT.mult(A, X).assign(B,	Functions.minus));
			double den = Algebra.DEFAULT.normInfinity(A) * niX + niB;
			residual =  num / den;
			//log.debug("scaled residual: " + residual);
			
			return residual;
		}
	}
	
	/**
	 * Calculate the scaled residual 
	 * <br> ||Ax-b||_oo/( ||A||_oo . ||x||_oo + ||b||_oo ), with
	 * <br> ||x||_oo = max(||x[i]||)
	 */
	public static double calculateScaledResidual(double[][] A, double[][] X, double[][] B){
		DoubleMatrix2D AMatrix = DoubleFactory2D.dense.make(A); 
		DoubleMatrix2D XMatrix = DoubleFactory2D.dense.make(X);
		DoubleMatrix2D BMatrix = DoubleFactory2D.dense.make(B);
		return calculateScaledResidual(AMatrix, XMatrix, BMatrix);
	}
	
	/**
	 * Calculate the scaled residual
	 * <br> ||Ax-b||_oo/( ||A||_oo . ||x||_oo + ||b||_oo ), with
	 * <br> ||x||_oo = max(||x[i]||)
	 */
	public static double calculateScaledResidual(DoubleMatrix2D A, DoubleMatrix1D x, DoubleMatrix1D b){
		double residual = -Double.MAX_VALUE;
		double nix = Algebra.DEFAULT.normInfinity(x);
		double nib = Algebra.DEFAULT.normInfinity(b);
		if(Double.compare(nix, 0.)==0 && Double.compare(nib, 0.)==0){
			return 0;
		}else{
			double num = Algebra.DEFAULT.normInfinity(ColtUtils.zMult(A, x, b, -1));
			double den = Algebra.DEFAULT.normInfinity(A) * nix + nib;
			residual =  num / den;
			//log.debug("scaled residual: " + residual);
			
			return residual;
		}
	}
	
	/**
	 * Calculate the scaled residual 
	 * <br> ||Ax-b||_oo/( ||A||_oo . ||x||_oo + ||b||_oo ), with
	 * <br> ||x||_oo = max(||x[i]||)
	 */
	public static double calculateScaledResidual(double[][] A, double[] x, double[] b){
		DoubleMatrix2D AMatrix = DoubleFactory2D.dense.make(A); 
		DoubleMatrix1D xVector = DoubleFactory1D.dense.make(x);
		DoubleMatrix1D bVector = DoubleFactory1D.dense.make(b);
		return calculateScaledResidual(AMatrix, xVector, bVector);
	}
	
	/**
	 * Residual conditions check after resolution of A.x=b.
	 * 
	 * eps := The relative machine precision
	 * N   := matrix dimension
	 * 
     * Checking the residual of the solution. 
     * Inversion pass if scaled residuals are less than 10:
	 * ||Ax-b||_oo/( (||A||_oo . ||x||_oo + ||b||_oo) . N . eps ) < 10.
	 * 
	 * @param A not-null matrix
	 * @param x not-null vector
	 * @param b not-null vector
	 */
//	public static boolean checkScaledResiduals(DoubleMatrix2D A, DoubleMatrix1D x, DoubleMatrix1D b, Algebra ALG) {
//	  //The relative machine precision
//		double eps = RELATIVE_MACHINE_PRECISION;
//		int N = A.rows();//matrix dimension
//		double residual = -Double.MAX_VALUE;
//		if(Double.compare(ALG.normInfinity(x), 0.)==0 && Double.compare(ALG.normInfinity(b), 0.)==0){
//			return true;
//		}else{
//			residual = ALG.normInfinity(ALG.mult(A, x).assign(b,	Functions.minus)) / 
//	          ((ALG.normInfinity(A)*ALG.normInfinity(x) + ALG.normInfinity(b)) * N * eps);
//			log.debug("scaled residual: " + residual);
//			return residual < 10;
//		}
//	}
	
	/**
	 * The smallest positive (epsilon) such that 1.0 + epsilon != 1.0.
	 * @see http://en.wikipedia.org/wiki/Machine_epsilon#Approximation_using_Java
	 */
	public static final double getDoubleMachineEpsilon() {
		
		if(!Double.isNaN(RELATIVE_MACHINE_PRECISION)){
			return RELATIVE_MACHINE_PRECISION;
		}
		
		synchronized(RELATIVE_MACHINE_PRECISION){
			
			if(!Double.isNaN(RELATIVE_MACHINE_PRECISION)){
				return RELATIVE_MACHINE_PRECISION;
			}
			
			double eps = 1.;
			do {
				eps /= 2.;
			} while ((double) (1. + (eps / 2.)) != 1.);
			
			log.debug("Calculated double machine epsilon: " + eps);
			RELATIVE_MACHINE_PRECISION = eps;
		}
		
		return RELATIVE_MACHINE_PRECISION;
	}
	
	/**
	   * Get the index of the maximum entry.
	   */
	  public static int getMaxIndex(DoubleMatrix1D v){
	  	int maxIndex = -1;
	  	double maxValue = -Double.MAX_VALUE;
	  	for(int i=0; i<v.size(); i++){
	  		if(v.getQuick(i)>maxValue){
	  			maxIndex = i;
	  			maxValue = v.getQuick(i); 
	  		}
	  	}
	  	return maxIndex; 
	  } 
	  
	  
	  
	  /**
	   * Get the index of the minimum entry.
	   */
	  public static int getMinIndex(DoubleMatrix1D v){
	  	int minIndex = -1;
	  	double minValue = Double.MAX_VALUE;
	  	for(int i=0; i<v.size(); i++){
	  		if(v.getQuick(i)<minValue){
	  			minIndex = i;
	  			minValue = v.getQuick(i); 
	  		}
	  	}
	  	return minIndex; 
	  }
	  
	public static final double[][] createConstantDiagonalMatrix(int dim, double c) {
		double[][] matrix = new double[dim][dim];
		for (int i = 0; i < dim; i++) {
			matrix[i][i] = c;
		}
		return matrix;
	}
	
	/**
	 * @deprecated avoid calculating matrix inverse, better use the solve methods
	 */
	@Deprecated
	public static final double[][] upperTriangularMatrixInverse(double[][] L) throws Exception {
		
		// Solve L*X = Id
		int dim = L.length;
		double[][] x = Utils.createConstantDiagonalMatrix(dim, 1.);
		for (int j = 0; j < dim; j++) {
			final double[] LJ = L[j];
			final double LJJ = LJ[j];
			final double[] xJ = x[j];
			for (int k = 0; k < dim; ++k) {
				xJ[k] /= LJJ;
			}
			for (int i = j + 1; i < dim; i++) {
				final double[] xI = x[i];
				final double LJI = LJ[i];
				for (int k = 0; k < dim; ++k) {
					xI[k] -= xJ[k] * LJI;
				}
			}
		}
        
		return new Array2DRowRealMatrix(x).transpose().getData();
	}
	
	/**
	 * @deprecated avoid calculating matrix inverse, better use the solve methods
	 */
	@Deprecated
	public static final double[][] lowerTriangularMatrixInverse(double[][] L) throws Exception {
		double[][] LT = new Array2DRowRealMatrix(L).transpose().getData();
		double[][] x = upperTriangularMatrixInverse(LT);
		return new Array2DRowRealMatrix(x).transpose().getData();
	}
	
	/**
	 * Brute-force determinant calculation.
	 */
	public static final double calculateDeterminant(double[][] ai, int dim) {
		double det = 0;
		if (dim == 1) {
			det = ai[0][0];
		} else if (dim == 2) {
			det = ai[0][0] * ai[1][1] - ai[0][1] * ai[1][0];
		} else {
			double ai1[][] = new double[dim - 1][dim - 1];
			for (int k = 0; k < dim; k++) {
				for (int i1 = 1; i1 < dim; i1++) {
					int j = 0;
					for (int j1 = 0; j1 < dim; j1++) {
						if (j1 != k) {
							ai1[i1 - 1][j] = ai[i1][j1];
							j++;
						}
					}
				}
				if (k % 2 == 0) {
					det += ai[0][k] * calculateDeterminant(ai1, dim - 1);
				} else {
					det -= ai[0][k] * calculateDeterminant(ai1, dim - 1);
				}
			}
		}
		return det;
	}

//	public static final String toString(double[] array) {
//		if(array ==  null || array.length == 0){
//			return "{}";
//		}
//		StringBuffer sb = new StringBuffer();
//		sb.append("{");
//		for(int i=0; i<array.length-1; i++){
//			sb.append(array[i]);
//			sb.append(",");
//		}
//		sb.append(array[array.length-1]);
//		sb.append("}");
//        return sb.toString();
//    }
	
//	public static final String toString(double[][] array) {
//		if(array ==  null || array.length == 0){
//			return "{}";
//		}
//		StringBuffer sb = new StringBuffer();
//		sb.append("{");
//		for(int i=0; i<array.length-1; i++){
//			sb.append(toString(array[i]));
//			sb.append(",");
//		}
//		sb.append(toString(array[array.length-1]));
//		sb.append("}");
//        return sb.toString();
//    }
	
	public static final int[] getFullRankSubmatrixRowIndices(RealMatrix M) {
		int row = M.getRowDimension();
		int col = M.getColumnDimension();
		
		SingularValueDecomposition dFact1 = new SingularValueDecomposition(M);
		int  r = dFact1.getRank();
		int[] ret = new int[r];
		
		if(r<row){
			//we have to find a submatrix of M with row dimension = rank
			RealMatrix fullM = MatrixUtils.createRealMatrix(1, col);
			fullM.setRowVector(0, M.getRowVector(0));
			ret[0] = 0;
			int iRank = 1;
			for(int i=1; i<row; i++){
				RealMatrix tmp = MatrixUtils.createRealMatrix(fullM.getRowDimension()+1, col);
				tmp.setSubMatrix(fullM.getData(), 0, 0);
				tmp.setRowVector(fullM.getRowDimension(), M.getRowVector(i));
				SingularValueDecomposition dFact_i = new SingularValueDecomposition(tmp);
				int ri = dFact_i.getRank();
				if(ri>iRank){
					fullM = tmp;
					ret[iRank] = i;
					iRank = ri;
					if(iRank==r){
						break;//target reached!
					}
				}
			}
		}else{
			for(int i=0; i<r; i++){
				ret[i] = i;
			}
		}
		
		return ret;
	}
	
	/**
	 * Extract the sign (the leftmost bit), exponent (the 11 following bits) 
	 * and mantissa (the 52 rightmost bits) from a double.
	 * @see http://www.particle.kth.se/~lindsey/JavaCourse/Book/Part1/Tech/Chapter02/floatingPt.html
	 */
	public static final long[] getExpAndMantissa(double myDouble) {
		long lbits = Double.doubleToLongBits(myDouble);
		long lsign = lbits >>> 63;// 0(+) or 1(-)
		long lexp = (lbits >>> 52 & ((1 << 11) - 1)) - ((1 << 10) - 1);
		long lmantissa = lbits & ((1L << 52) - 1);
		long[] ret = new long[] { lsign, lexp, lmantissa };
		log.debug("double  : " + myDouble);
		log.debug("sign    : " + lsign);
		log.debug("exp     : " + lexp);
		log.debug("mantissa: " + lmantissa);
		log.debug("reverse : " + Double.longBitsToDouble((lsign << 63)	| (lexp + ((1 << 10) - 1)) << 52 | lmantissa));
		log.debug("log(d)  : " + Math.log1p(myDouble));
		return ret;
	}
	
	/**
	 * Return a new array with all the occurences of oldValue replaced by newValue.
	 */
	public static final double[] replaceValues(double[] v, double oldValue,	double newValue) {
		double[] ret = new double[v.length];
		for (int i = 0; i < v.length; i++) {
			double vi = v[i];
			if (Double.compare(oldValue, vi) != 0) {
				// no substitution
				ret[i] = vi;
			} else {
				ret[i] = newValue;
			}
		}
		return ret;
	}
	
	public static final double round(double d, double precision){
		return Math.round(d * precision) / precision;
	}
	
	public static final void serializeObject(Object obj, String filename) throws Exception{
		FileOutputStream fout = new FileOutputStream(filename, true);
	  ObjectOutputStream oos = new ObjectOutputStream(fout);
	  oos.writeObject(obj);
	  oos.close();
	}
	
	public static final Object deserializeObject(String classpathFileName) throws Exception{
		InputStream streamIn = Thread.currentThread().getContextClassLoader().getResourceAsStream(classpathFileName);
		ObjectInputStream objectinputstream = new ObjectInputStream(streamIn);
    return objectinputstream.readObject();
	}
	
}
