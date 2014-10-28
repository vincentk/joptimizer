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
package com.joptimizer.optimizers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import cern.colt.function.IntIntDoubleFunction;
import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;

import com.joptimizer.algebra.Matrix1NornRescaler;
import com.joptimizer.algebra.MatrixRescaler;
import com.joptimizer.util.ColtUtils;
import com.joptimizer.util.Utils;



/**
 * Presolver for a linear problem in the form of:
 * 
 * min(c)  s.t.
 * A.x = b
 * lb <= x <= ub
 * 
 * <br>Note 1: unboundedLBValue is the distinctive value of an unbounded lower bound. It must be one of the values:
 *  <ol>
 *   <li>Double.NaN (the default)</li>
 *   <li>Double.NEGATIVE_INFINITY</li>
 *  </ol>
 * Note 2: unboundedUBValue is the distinctive value of an unbounded upper bound. It must be one of the values:
 *  <ol>
 *   <li>Double.NaN (the default)</li>
 *   <li>Double.POSITIVE_INFINITY</li>
 *  </ol>
 * Note 3: if lb is null, each variable lower bound will be assigned the value of <i>unboundedLBValue</i>
 * <br>Note 4: if ub is null, each variable upper bound will be assigned the value of <i>unboundedUBValue</i>
 * 
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 * @see E.D. Andersen, K.D. Andersen, "Presolving in linear programming"
 * @see Jacek Gondzio "Presolve analysis of linear programs prior to applying an interior point method"
 * @see Xin Huang, "Preprocessing and Postprocessing in Linear Optimization"
 */
public class LPPresolver {

	public static final double DEFAULT_UNBOUNDED_LOWER_BOUND = Double.NaN;
	public static final double DEFAULT_UNBOUNDED_UPPER_BOUND = Double.NaN;
//	public static final double DEFAULT_UNSPECIFIED_LOWER_BOUND = DEFAULT_UNBOUNDED_LOWER_BOUND;
//	public static final double DEFAULT_UNSPECIFIED_UPPER_BOUND = DEFAULT_UNBOUNDED_UPPER_BOUND;
	
	private Algebra ALG = Algebra.DEFAULT;
	private DoubleFactory1D F1 = null;
	private DoubleFactory2D F2 = null; 
	
	private double eps = Utils.getDoubleMachineEpsilon();
	private boolean useSparsity = true;
	/**
	 * If true, no method for making normal equations sparser will be applied.
	 * @see Jacek Gondzio "Presolve analysis of linear programs prior to applying an interior point method", 3
	 */
	private boolean avoidIncreaseSparsity = false;
	/**
	 * If true, no methods that cause fill-in in the original matrices will be called.
	 */
	private boolean avoidFillIn = false;
	/**
	 * If true, no scaling on constraints matrices will be applied.
	 */
	private boolean avoidScaling = false;
//	private double unspecifiedLBValue = DEFAULT_UNSPECIFIED_LOWER_BOUND;
//	private double unspecifiedUBValue = DEFAULT_UNSPECIFIED_UPPER_BOUND;
	private double unboundedLBValue = DEFAULT_UNBOUNDED_LOWER_BOUND;
	private double unboundedUBValue = DEFAULT_UNBOUNDED_UPPER_BOUND;
	private int originalN;//number of variables
	private int originalMeq;//number of variables
	/**
	 * If the problem is in standard form, this is the number of slack variables (expected
	 * to be the first variables of the problem).
	 */
	private short nOfSlackVariables = -1;
	
	//after presolving fields
	private	int presolvedN = -1;
	private	int presolvedMeq = -1;
	private boolean[] indipendentVariables;
	private short[] presolvedX;
	private short[] presolvedPositions;
	private	DoubleMatrix1D presolvedC = null;
	private	DoubleMatrix2D presolvedA = null;
	private	DoubleMatrix1D presolvedB = null;
	private	DoubleMatrix1D presolvedLB = null;
	private	DoubleMatrix1D presolvedUB = null;
	private	DoubleMatrix1D presolvedYlb = null;
	private	DoubleMatrix1D presolvedYub = null;
	private	DoubleMatrix1D presolvedZlb = null;
	private	DoubleMatrix1D presolvedZub = null;

	/**
	 * Row by row non-zeroes entries of A.
	 */
	private short[][] vRowPositions;
	/**
	 * Column by column non-zeroes entries of A.
	 */
	private short[][] vColPositions;
	
	private double[] g;//min constraints values (g[i] <= A[i,j].x)
	private double[] h;//max constraints values (h[i] >= A[i,j].x)
	private int[][] vRowLengthMap;
	private int[][] vColLengthMap;
	
	private boolean someReductionDone = true;
	private DoubleMatrix1D T = null;//used for rescaling
	private DoubleMatrix1D R = null;//used for rescaling
	private double minRescaledLB = Double.NaN;//used for rescaling
	private double maxRescaledUB = Double.NaN;//used for rescaling
	private List<PresolvingStackElement> presolvingStack = new ArrayList<LPPresolver.PresolvingStackElement>();
	private Log log = LogFactory.getLog(this.getClass().getName());
	private double[] expectedSolution;//for testing purpose
	private double expectedTolerance = Double.NaN;//for testing purpose
	
	public LPPresolver(){
		this(DEFAULT_UNBOUNDED_LOWER_BOUND, DEFAULT_UNBOUNDED_UPPER_BOUND);
	}
	
	public LPPresolver(double unboundedLBValue, double unboundedUBValue){
		if(!Double.isNaN(unboundedLBValue) && !Double.isInfinite(unboundedLBValue) ){
			throw new IllegalArgumentException("The field unboundedLBValue must be set to Double.NaN or Double.NEGATIVE_INFINITY");
		}
		if(!Double.isNaN(unboundedUBValue) && !Double.isInfinite(unboundedUBValue) ){
			throw new IllegalArgumentException("The field unboundedUBValue must be set to Double.NaN or Double.POSITIVE_INFINITY");
		}
//		this.unspecifiedLBValue = unspecifiedLBValue;
//		this.unspecifiedUBValue = unspecifiedUBValue;
		this.unboundedLBValue = unboundedLBValue;
		this.unboundedUBValue = unboundedUBValue;
	}
	
	public boolean isUseSparsity() {
		return useSparsity;
	}

	public void setUseSparsity(boolean useSparsity) {
		this.useSparsity = useSparsity;
	}
	
	public boolean isAvoidIncreaseSparsity() {
		return avoidIncreaseSparsity;
	}

	public void setAvoidIncreaseSparsity(boolean avoidIncreaseSparsity) {
		this.avoidIncreaseSparsity = avoidIncreaseSparsity;
	}
	
	public void setAvoidFillIn(boolean avoidFillIn) {
		this.avoidFillIn = avoidFillIn;
	}

	public boolean isAvoidFillIn() {
		return avoidFillIn;
	}
	
	public void setAvoidScaling(boolean avoidScaling) {
		this.avoidScaling = avoidScaling;
	}

	public boolean isAvoidScaling() {
		return avoidScaling;
	}
	
	public void presolve(double[] originalC, 
			double[][] originalA, double[] originalB, 
			double[] originalLB, double[] originalUB){
		
		this.F1 = (useSparsity)? DoubleFactory1D.sparse : DoubleFactory1D.dense;
		this.F2 = (useSparsity)? DoubleFactory2D.sparse : DoubleFactory2D.dense;
		
		DoubleMatrix1D cVector = F1.make(originalC);
		DoubleMatrix2D AMatrix = null;
		DoubleMatrix1D bVector = null;
		if(originalA!=null){
			AMatrix = F2.make(originalA);
			bVector = F1.make(originalB);
		}
		
		if(originalLB != null && originalUB !=null){
			if(originalLB.length != originalUB.length){
				throw new IllegalArgumentException("lower and upper bounds have different lenght");
			}
		}
		DoubleMatrix1D lbVector = (originalLB!=null)? F1.make(originalLB) : null;
		DoubleMatrix1D ubVector = (originalUB!=null)? F1.make(originalUB) : null;

		presolve(cVector, AMatrix, bVector, lbVector, ubVector);
	}
	
	public void presolve(DoubleMatrix1D originalC, 
			DoubleMatrix2D originalA, DoubleMatrix1D originalB, 
			DoubleMatrix1D originalLB, DoubleMatrix1D originalUB){
		long t0 = System.currentTimeMillis();
		
		this.F1 = (useSparsity)? DoubleFactory1D.sparse : DoubleFactory1D.dense;
		this.F2 = (useSparsity)? DoubleFactory2D.sparse : DoubleFactory2D.dense;
		
		//working entities definition
		DoubleMatrix1D c;
		final DoubleMatrix2D A;
		DoubleMatrix1D b;
		DoubleMatrix1D lb;//primal variables lower bounds
		DoubleMatrix1D ub;//primal variables upper bounds
		DoubleMatrix1D ylb;//lagrangian lower bounds for linear constraints (A rows)
		DoubleMatrix1D yub;//lagrangian upper bounds for linear constraints (A rows)
		DoubleMatrix1D zlb;//lagrangian lower bounds for lb constraints
		DoubleMatrix1D zub;//lagrangian upper bounds for ub constraints
		
		c = originalC.copy();
		
		this.originalN = originalA.columns();
		this.originalMeq = originalA.rows();
		this.indipendentVariables = new boolean[originalN];
		Arrays.fill(indipendentVariables, true);
		this.g = new double[originalN];
		this.h = new double[originalN];
		
		if(originalLB==null){
			originalLB = F1.make(originalN, unboundedLBValue);
		}
		if(originalUB==null){
			originalUB = F1.make(originalN, unboundedUBValue);
		}
		for(int i=0; i<originalN; i++){
			if(originalUB.getQuick(i) < originalLB.getQuick(i)){
				log.debug("infeasible problem");
				throw new RuntimeException("infeasible problem");
			}
		}
		lb = originalLB.copy();//this will change during the process
		ub = originalUB.copy();//this will change during the process
		A = (useSparsity)? new SparseDoubleMatrix2D(originalA.rows(), originalA.columns()) : F2.make(originalA.rows(), originalA.columns());//this will change during the process
		b = originalB.copy();//this will change during the process
		vRowLengthMap = new int[originalN+1][];//the first position is for 0-length rows
		vColLengthMap = new int[originalMeq+1][];//the first position is for 0-length columns
		ylb = F1.make(originalMeq, unboundedLBValue);
		yub = F1.make(originalMeq, unboundedUBValue);
		zlb = F1.make(originalN, unboundedLBValue);
		zub = F1.make(originalN, unboundedUBValue);
		int entries = originalN * originalMeq;
		int cardinality = 0;
		short[] vColCounter = new short[originalN];//counter of non-zero values in each column
		vRowPositions = new short[originalMeq][0];
		if(originalA instanceof SparseDoubleMatrix2D){
			SparseDoubleMatrix2D Q = (SparseDoubleMatrix2D)originalA;
			final int[] cardinalityHolder = new int[]{cardinality};
			final short[] vColCounterPH = new short[originalN];
			
			//view Q column by column
			for (int column = 0; column < originalN; column++) {
				//log.debug("column:" + c);
				final int [] currentColumnIndexHolder = new int[]{column};
				DoubleMatrix2D P = Q.viewPart(0, column, originalMeq, 1);
				P.forEachNonZero(new IntIntDoubleFunction() {
					@Override
					public double apply(int i, int j, double qij) {
						//log.debug("i:"+i+",j:"+currentColumnIndexHolder[0]+", qij="+qij);
						cardinalityHolder[0] = cardinalityHolder[0] + 1;
						if(vRowPositions[i]==null){
							vRowPositions[i] =  new short[]{};
						}
						vRowPositions[i] = ArrayUtils.add(vRowPositions[i], vRowPositions[i].length, (short) currentColumnIndexHolder[0]);
						vColCounterPH[currentColumnIndexHolder[0]] = (short)(vColCounterPH[currentColumnIndexHolder[0]] + 1);
						A.setQuick(i, currentColumnIndexHolder[0], qij);
						return qij;
					}
				});
			}
			
			cardinality = cardinalityHolder[0];
			vColCounter = vColCounterPH;
			
      //check empty row
      for(int i=0; i<originalMeq; i++){
      	short[] vRowPositionsI = vRowPositions[i];
      	if(vRowPositionsI.length < 1){
					if(!isZero(originalB.getQuick(i))){	
						log.debug("infeasible problem");
						throw new RuntimeException("infeasible problem");
					}
				}
				if(this.vRowLengthMap[vRowPositionsI.length]==null){
					vRowLengthMap[vRowPositionsI.length] = new int[]{i};
				}else{
					vRowLengthMap[vRowPositionsI.length] = addToSortedArray(vRowLengthMap[vRowPositionsI.length], i);
				}
	    }	        
		}else{
			for(short i=0; i<originalMeq; i++){
				short[] vRowPositionsI = new short[]{};
				for(short j=0; j<originalN; j++){
					double originalAIJ = originalA.getQuick(i, j);
					if(!isZero(originalAIJ)){	
						cardinality ++;
						vRowPositionsI = ArrayUtils.add(vRowPositionsI, vRowPositionsI.length, (short) j);
						vColCounter[j] ++;
						A.setQuick(i, j, originalAIJ);
					}
				}
				//check empty row
				if(vRowPositionsI.length < 1){
					if(!isZero(originalB.getQuick(i))){	
						log.debug("infeasible problem");
						throw new RuntimeException("infeasible problem");
					}
				}
				vRowPositions[i] = vRowPositionsI;
				if(this.vRowLengthMap[vRowPositionsI.length]==null){
					vRowLengthMap[vRowPositionsI.length] = new int[]{i};
				}else{
					vRowLengthMap[vRowPositionsI.length] = addToSortedArray(vRowLengthMap[vRowPositionsI.length], i);
				}
			}
		}
		
	  //check empty columns
		for(int j=0; j<vColCounter.length; j++){
			if(vColCounter[j] == 0){
				//empty column
				if(originalC.getQuick(j)>0){
					if(isLBUnbounded(lb.getQuick(j))){
						log.debug("unbounded problem");
						throw new RuntimeException("unbounded problem");
					}else{
						//variable j fixed at its lower bound
						log.debug("found empty column: " + j);
						ub.setQuick(j, lb.getQuick(j));
					}
				}else if(originalC.getQuick(j)<0){
					if(isUBUnbounded(ub.getQuick(j))){
						log.debug("unbounded problem");
						throw new RuntimeException("unbounded problem");
					}else{
						//variable j fixed at its upper bound
						log.debug("found empty column: " + j);
						lb.setQuick(j, ub.getQuick(j));
					}
				}
			}
		}
		
		//fill not-zero columns holders
		this.vColPositions = new short[originalN][];
		for(int j = 0; j<originalN; j++){
			int length = vColCounter[j];
			this.vColPositions[j] = new short[length];
			if(this.vColLengthMap[length]==null){
				vColLengthMap[length] = new int[]{j};
			}else{
				vColLengthMap[length] = addToSortedArray(vColLengthMap[length], j);
			}
		}
		for(int i=0; i<vRowPositions.length; i++){
			short[] vRowPositionsI = vRowPositions[i]; 
			for(int j=0; j<vRowPositionsI.length; j++){
				short col = vRowPositionsI[j];
				this.vColPositions[col][vColPositions[col].length - vColCounter[col]] = (short)i;
				vColCounter[vRowPositionsI[j]]--;
			}
		}
		
		vColCounter = null;
		log.debug("sparsity index: " + 100*((double)(entries-cardinality)) / ((double)entries) +" % ("+cardinality+"nz/"+entries+"tot)");
		
		//pre-presolving check
		checkProgress(c, A, b, lb, ub, ylb, yub, zlb, zub);
		
		//remove all fixed variables
		removeFixedVariables(c, A, b, lb, ub, ylb, yub, zlb, zub);
		
		//repeat
		int iteration = 0;
		while(someReductionDone){
			iteration++;
			log.debug("iteration: " + iteration);
//			log.debug("c: "+ArrayUtils.toString(c));
//			log.debug("A: "+ArrayUtils.toString(A));
//			log.debug("b: "+ArrayUtils.toString(b));
//			log.debug("lb: "+ArrayUtils.toString(lb));
//			log.debug("ub: "+ArrayUtils.toString(ub));
			someReductionDone = false;//reset
			checkProgress(c, A, b, lb, ub, ylb, yub, zlb, zub);
			//Check rows
			//  Remove fixed variables
					removeFixedVariables(c, A, b, lb, ub, ylb, yub, zlb, zub);
				  checkProgress(c, A, b, lb, ub, ylb, yub, zlb, zub);
			//	Remove all row singletons
					removeSingletonRows(c, A, b, lb, ub, ylb, yub, zlb, zub);
					checkProgress(c, A, b, lb, ub, ylb, yub, zlb, zub);
			//	Remove all forcing constraints
					removeForcingConstraints(c, A, b, lb, ub, ylb, yub, zlb, zub);
					checkProgress(c, A, b, lb, ub, ylb, yub, zlb, zub);
			//tight the bounds
					if(iteration < 5){
						//the higher the iteration, the less useful it is
						compareBounds(c, A, b, lb, ub, ylb, yub, zlb, zub);
						checkProgress(c, A, b, lb, ub, ylb, yub, zlb, zub);
					}
			//Dominated constraints
			//	Remove all dominated constraints
					removeDominatedConstraints(c, A, b, lb, ub, ylb, yub, zlb, zub);
					checkProgress(c, A, b, lb, ub, ylb, yub, zlb, zub);
			//Check columns
			//	Remove all free, implied free column singletons and
			//	all column singletons in combination with a doubleton equation
					checkColumnSingletons(c, A, b, lb, ub, ylb, yub, zlb, zub);
					checkProgress(c, A, b, lb, ub, ylb, yub, zlb, zub);
			//Dominated columns
					removeDominatedColumns(c, A, b, lb, ub, ylb, yub, zlb, zub);
					checkProgress(c, A, b, lb, ub, ylb, yub, zlb, zub);
					if(!avoidIncreaseSparsity){
					//Duplicate rows
						removeDuplicateRow(c, A, b, lb, ub, ylb, yub, zlb, zub);
						checkProgress(c, A, b, lb, ub, ylb, yub, zlb, zub);
				//Duplicate columns
						removeDuplicateColumn(c, A, b, lb, ub, ylb, yub, zlb, zub);
						checkProgress(c, A, b, lb, ub, ylb, yub, zlb, zub);
					}
			//Remove row doubleton
					if(!avoidFillIn){
						removeDoubletonRow(c, A, b, lb, ub, ylb, yub, zlb, zub);
						checkProgress(c, A, b, lb, ub, ylb, yub, zlb, zub);
					}
			
			//Remove empty rows from the indexed list
			//removeEmptyRows();
		}
		removeAllEmptyRowsAndColumns(c, A, b, lb, ub, ylb, yub, zlb, zub);
		
		presolvedN = 0;
		presolvedX = new short[originalN];//longer than it needs
		Arrays.fill(presolvedX, (short)-1);
		presolvedPositions = new short[originalN];
		Arrays.fill(presolvedPositions, (short)-1);
		for(int j=0; j<indipendentVariables.length; j++){
			if(indipendentVariables[j]){
				presolvedX[presolvedN] = (short)j;
				presolvedPositions[j] = (short)presolvedN; 
				presolvedN++;
			}
		}
		if(log.isDebugEnabled()){
			log.debug("final n    : " + presolvedN);
			log.debug("presolvedX           : " + ArrayUtils.toString(presolvedX));
			log.debug("presolvedPositions   : " + ArrayUtils.toString(presolvedPositions));
			log.debug("indipendentVariables : " + ArrayUtils.toString(indipendentVariables));
		}
		
		presolvedMeq = 0;
		for(int i=0; i<vRowPositions.length; i++){
			if(vRowPositions[i].length > 0){
				presolvedMeq++;
			}
		}
		log.debug("final meq: " + ArrayUtils.toString(presolvedMeq));
		
		if(presolvedMeq>0){
			presolvedA = (useSparsity)? new SparseDoubleMatrix2D(presolvedMeq, presolvedN) : F2.make(presolvedMeq, presolvedN);
			presolvedB = F1.make(presolvedMeq);
			presolvedYlb = F1.make(presolvedMeq);
			presolvedYub = F1.make(presolvedMeq);
		}
		if(presolvedN>0){
			presolvedC = F1.make(presolvedN);
			presolvedLB = F1.make(presolvedN);
			presolvedUB = F1.make(presolvedN);
			presolvedZlb = F1.make(presolvedN);
			presolvedZub = F1.make(presolvedN);
		}
		short cntR = 0;
		for(int i=0; presolvedA!=null && i<vRowPositions.length; i++){
			short[] vRowPositionsI = vRowPositions[i];
			if(vRowPositionsI.length>0){
				for(int j=0; j<vRowPositionsI.length; j++){
					short jnz = vRowPositionsI[j];
					int col = presolvedPositions[jnz];
					presolvedA.setQuick(cntR, col, A.getQuick(i, jnz));
					presolvedB.setQuick(cntR, b.getQuick(i));
				}
				cntR++;
			}
		}
		cntR = 0;
		for(int i=0; i<vRowPositions.length; i++){
			if(vRowPositions[i].length>0){
				presolvedYlb.setQuick(cntR, ylb.getQuick(i));
				presolvedYub.setQuick(cntR, yub.getQuick(i));
				cntR++;
			}
		}
		for(int j=0; j<presolvedN; j++){
			int col = presolvedX[j];
			presolvedC.setQuick(j, c.getQuick(col));
			presolvedLB.setQuick(j, lb.getQuick(col));
			presolvedUB.setQuick(j, ub.getQuick(col));
			presolvedZlb.setQuick(j, zlb.getQuick(col));
			presolvedZub.setQuick(j, zub.getQuick(col));
		}
		
		if(!avoidScaling){
			scaling();
		}
		
		if(log.isDebugEnabled()){
			log.debug("presolvingStack : " + presolvingStack);
			if(this.R != null){
				log.debug("presolving R : " + ArrayUtils.toString(R.toArray()));
				log.debug("presolving T : " + ArrayUtils.toString(T.toArray()));
			}
		}
		log.info("end presolving ("+(System.currentTimeMillis()-t0)+" ms)");
	}
	
	/**
	 * From the full x, gives back its presolved elements. 
	 */
	public double[] presolve(double[] x){
		if(x.length != originalN){
			throw new IllegalArgumentException("wrong array dimension: " + x.length);
		}
		double[] presolvedX = Arrays.copyOf(x, x.length);
		for(int i=0; i<presolvingStack.size(); i++){
			presolvingStack.get(i).preSolve(presolvedX);
		}
		double[] ret = new double[presolvedN];
		int cntPosition = 0;
		for(int i=0; i<presolvedX.length; i++){
			if(indipendentVariables[i]){
				ret[cntPosition] = presolvedX[i];
				cntPosition++;
			}
		}
		if(this.T != null){
			//rescaling has been done: x = T.x1
			for(int i=0; i<ret.length; i++){
				ret[i] = ret[i] / T.getQuick(i);
			}
		}
		return ret;
	}
	
	public double[] postsolve(double[] x){
		
		double[] postsolvedX = new double[originalN];
		
		if(this.T != null){
			//rescaling has been done: x = T.x1
			for(int i=0; i<x.length; i++){
				x[i] = T.getQuick(i) * x[i];
			}
		}
		
		for(int i=0; i<x.length; i++){
			postsolvedX[presolvedX[i]] = x[i];
		}
		for(int i=presolvingStack.size()-1; i>-1; i--){
			presolvingStack.get(i).postSolve(postsolvedX);
		}
		return postsolvedX;
	}

	private void removeFixedVariables(DoubleMatrix1D c,
			DoubleMatrix2D A, DoubleMatrix1D b, 
			DoubleMatrix1D lb, DoubleMatrix1D ub, 
			DoubleMatrix1D ylb, DoubleMatrix1D yub,
			DoubleMatrix1D zlb, DoubleMatrix1D zub) {
		for(short j=0; j<indipendentVariables.length; j++){
			if(indipendentVariables[j]){		
				//this is an active variable
				if(!isLBUnbounded(lb.getQuick(j)) && isZero(lb.getQuick(j)-ub.getQuick(j))){	
					//variable x is fixed
					double v = lb.getQuick(j);
					log.debug("found fixed variables: x[" + j + "]="+v);
					addToPresolvingStack(new LinearDependency(j, null, null, v));
					
					//substitution into objective function @TODO
				
					//substitution
					for(short k=0; k< this.vRowPositions.length; k++){
						short[] vRowPositionsK = vRowPositions[k];
						for(short i=0; i<vRowPositionsK.length; i++){
							if(vRowPositionsK[i] == j){
								if(vRowPositionsK.length == 1){
									//this row contains only xj
									if(!isZero(v - b.getQuick(k) / A.getQuick(k, j))){	
										log.debug("infeasible problem");
										throw new RuntimeException("infeasible problem");
									}
								}
								b.setQuick(k, b.getQuick(k) - A.getQuick(k, j) * v);
								vRowPositions[k] = removeElementFromSortedArray(vRowPositionsK, j);
								changeRowsLengthPosition(k, vRowPositions[k].length+1, vRowPositions[k].length);
								A.setQuick(k, j, 0);
								break;
							}
							if(vRowPositionsK[i] > j){
								break;//the array is sorted
							}
						}
					}
					changeColumnsLengthPosition(j, vColPositions[j].length, 0);
					vColPositions[j] = new short[]{};
					this.someReductionDone = true;
				}
			}
		}
	}

	private void removeSingletonRows(DoubleMatrix1D c,
			DoubleMatrix2D A, DoubleMatrix1D b, 
			DoubleMatrix1D lb, DoubleMatrix1D ub, 
			DoubleMatrix1D ylb, DoubleMatrix1D yub,
			DoubleMatrix1D zlb, DoubleMatrix1D zub){
		short i = 0;
		while(i < vRowPositions.length){
			short[] vRowPositionsI = vRowPositions[i];
			if(vRowPositionsI.length == 1){
				//singleton found: A(i,j).x(j) = b(i)
				short j = vRowPositionsI[0];
				double AIJ = A.getQuick(i, j);
				if(isZero(AIJ)){
					log.debug("A["+i+"]["+j+"]: expected non-zero but was " + AIJ);
					throw new IllegalStateException("A["+i+"]["+j+"]: expected non-zero but was " + AIJ);
				}
				double xj = b.getQuick(i) / AIJ;
				log.debug("found singleton at row "+i+": x[" + j +"]=" + xj);
				addToPresolvingStack(new LinearDependency(j, null, null, xj));
				
				//substitution into the other equations
				for(short k=0; k<this.vRowPositions.length; k++){
					if(k != i){
						short[] vRowPositionsK = vRowPositions[k];
						for(short nz=0; nz<vRowPositionsK.length; nz++){
							if(vRowPositionsK[nz] == j){
								//this row contains xj at position nz
								if(vRowPositionsK.length == 1){
									if(!isZero(xj - b.getQuick(k) / A.getQuick(k, j))){
										log.debug("infeasible problem");
										throw new RuntimeException("infeasible problem");
									}
								}
								b.setQuick(k, b.getQuick(k) - A.getQuick(k, j) * xj);
								A.setQuick(k, j, 0.);
								vRowPositions[k] = ArrayUtils.remove(vRowPositionsK, nz);
								changeRowsLengthPosition(k, vRowPositions[k].length+1, vRowPositions[k].length);
								break;
							}else if(vRowPositionsK[nz] > j){
								break;
							}
						}
					}
				}
				A.setQuick(i, j, 0.);
				b.setQuick(i, 0.);
				changeColumnsLengthPosition(j, vColPositions[j].length, 0);
				vColPositions[j] = new short[]{};
				vRowPositions[i] = ArrayUtils.remove(vRowPositionsI, 0);//this row has only this nz-entry 
				changeRowsLengthPosition(i, vRowPositions[i].length+1, vRowPositions[i].length);
				this.someReductionDone = true;
				i = -1;//restart
			}
			i++;
		}
	}
	
	private void removeForcingConstraints(DoubleMatrix1D c,
			DoubleMatrix2D A, DoubleMatrix1D b, 
			DoubleMatrix1D lb, DoubleMatrix1D ub, 
			DoubleMatrix1D ylb, DoubleMatrix1D yub,
			DoubleMatrix1D zlb, DoubleMatrix1D zub) {
		for(short i=0; i<vRowPositions.length; i++){
			short[] vRowPositionsI = vRowPositions[i];
			if(vRowPositionsI.length > 0){
				g[i] = 0.;
				h[i] = 0.;
				boolean allPositive = true;
				boolean allNegative = true;
				boolean allLbPositive = true;
				boolean allUbFinite = true;
				for(short nz=0; nz<vRowPositionsI.length; nz++){
					short j = vRowPositionsI[nz];
					double AIJ = A.getQuick(i, j);
					if(isZero(AIJ)){
						log.debug("A["+i+"]["+j+"]: expected non-zero but was " + AIJ);
						throw new IllegalStateException("A["+i+"]["+j+"]: expected non-zero but was " + AIJ);
					}else if (AIJ > 0) {
						// j in P
						g[i] += AIJ * lb.getQuick(j);
						h[i] += AIJ * ub.getQuick(j);
						allNegative = false;
					} else {
						// j in M
						g[i] += AIJ * ub.getQuick(j);
						h[i] += AIJ * lb.getQuick(j);
						allPositive = false;
					}
					allLbPositive = allLbPositive && lb.getQuick(j)>=0;
					allUbFinite = allUbFinite && !isUBUnbounded(ub.getQuick(j));
				}
				
				if(h[i] < b.getQuick(i) || b.getQuick(i) < g[i]){
					log.debug("infeasible problem");
					throw new RuntimeException("infeasible problem");
				}
				
				//logger.debug(g[i]+","+b[i]);
				//logger.debug(h[i]+","+b[i]);
				short[] forcedVariablesI = new short[]{};
				if(isZero(g[i] - b.getQuick(i))){
					//forcing constraint
					log.debug("found forcing constraint at row: " + i);
					//the only feasible value of xj is l[j] (u[j]) if A(i,j) > 0 (A(i,j) < 0). 
					//Therefore, we can fix all variables in the ith constraint.
					for(short nz=0; nz<vRowPositionsI.length; nz++){
						short j = vRowPositionsI[nz];
						double aij = A.getQuick(i, j);
						if (aij > 0) {
							ub.setQuick(j, lb.getQuick(j));
						} else {
							lb.setQuick(j, ub.getQuick(j));
						}
						log.debug("x[" + j +"]=" + lb.getQuick(j));
						forcedVariablesI = ArrayUtils.add(forcedVariablesI, j);
						addToPresolvingStack(new LinearDependency(j, null, null, lb.getQuick(j)));
					}
				}else if(isZero(h[i]-b.getQuick(i))){
				  //forcing constraint
					log.debug("found forcing constraint at row: " + i);
					//the only feasible value of xj is u[j] (l[j]) if A(i,j) > 0 (A(i,j) < 0). 
					//Therefore, we can fix all variables in the ith constraint.
					for(short nz=0; nz<vRowPositionsI.length; nz++){
						short j = vRowPositionsI[nz];
						double aij = A.getQuick(i, j);
						if (aij > 0) {
							lb.setQuick(j, ub.getQuick(j));
						} else {
							ub.setQuick(j, lb.getQuick(j));
						}
						log.debug("x[" + j +"]=" + lb.getQuick(j));
						forcedVariablesI = ArrayUtils.add(forcedVariablesI, j);
						addToPresolvingStack(new LinearDependency(j, null, null, lb.getQuick(j)));
					}
				}
				if(forcedVariablesI.length > 0){
					//there are forced variables to substitute
					for(short fv =0; fv < forcedVariablesI.length; fv++){
						short j = forcedVariablesI[fv];
						double xj = lb.getQuick(j);
						//substitution into the other equations
						for(short k=0; k<this.vRowPositions.length; k++){
							if(k != i){
								short[] vRowPositionsK = vRowPositions[k];
								for(short nz=0; nz<vRowPositionsK.length; nz++){
									if(vRowPositionsK[nz] == j){
										//this row contains x[j]
										if(vRowPositionsK.length == 1){
											if(!isZero(xj - b.getQuick(k) / A.getQuick(k, j))){
												log.debug("infeasible problem");
												throw new RuntimeException("infeasible problem");
											}
										}
										b.setQuick(k, b.getQuick(k) - A.getQuick(k, j) * xj);
										A.setQuick(k, j, 0.);
										vRowPositions[k] = ArrayUtils.remove(vRowPositionsK, nz);
										changeRowsLengthPosition(k, vRowPositions[k].length+1, vRowPositions[k].length);
										changeColumnsLengthPosition(j, vColPositions[j].length, vColPositions[j].length-1);
										vColPositions[j] = ArrayUtils.remove(vColPositions[j], 0);//ordered row loop
										break;
									}else if(vRowPositionsK[nz] > j){
										break;
									}
								}
							}
						}
						A.setQuick(i, j, 0.);
						b.setQuick(i, 0.);
						vRowPositions[i] = removeElementFromSortedArray(vRowPositions[i], j);//this row has only this nz-entry 
						changeRowsLengthPosition(i, vRowPositions[i].length+1, vRowPositions[i].length);
						if(vColPositions[j].length != 1 && vColPositions[j][0] != j){
							log.debug("Expected empty column "+j+" but was not empty");
							throw new IllegalStateException("Expected empty column "+j+" but was not empty");
						}
						changeColumnsLengthPosition(j, vColPositions[j].length, 0);
						vColPositions[j] = new short[]{};
						this.someReductionDone = true;
					}
					//cancel the row
					if(vRowPositions[i].length > 0){
						log.debug("Expected empty row "+i+" but was not empty");
						throw new IllegalStateException("Expected empty row "+i+" but was not empty");
					}
					vRowPositions[i] = new short[]{};
					b.setQuick(i, 0);
					continue;
				}
				
				//check if we can tight upper bounds. leveraging the fact that, typically, 
				//the problem has lb => 0 for most variables.
				//if coefficients are all positive or all negative the bounds can be limited
				boolean aa = g[i] >= 0 && b.getQuick(i) >= 0;
				boolean bb = h[i] <= 0 && b.getQuick(i) <= 0;
				boolean t1 = aa && allPositive;
				boolean t2 = bb && allNegative;// same as above with a sign change
				if (t1 || t2) {
					boolean sameSignReductionDone = false;
					for (short nz = 0; nz < vRowPositionsI.length; nz++) {
						short nzj = vRowPositionsI[nz];
						double d = b.getQuick(i) / A.getQuick(i, nzj);
						// we can limit upper bound to d
						if (isUBUnbounded(ub.getQuick(nzj)) || ub.getQuick(nzj) > d) {
							ub.setQuick(nzj, d);
							sameSignReductionDone = true;
						}
					}
					if (sameSignReductionDone) {
						this.someReductionDone = true;
						log.debug("all same signs lb reduction at row " + i);
					}
				}
			}
		}
	}
	
	private void compareBounds(DoubleMatrix1D c,
			DoubleMatrix2D A, DoubleMatrix1D b, 
			DoubleMatrix1D lb, DoubleMatrix1D ub, 
			DoubleMatrix1D ylb, DoubleMatrix1D yub,
			DoubleMatrix1D zlb, DoubleMatrix1D zub) {
		int treshold = (avoidFillIn)? 2 : 3;//if avoidFillIn=true, doubleton are not managed 
		for(short i=0; i<vRowPositions.length; i++){
			short[] vRowPositionsI = vRowPositions[i];
			if(vRowPositionsI.length >= treshold){
				
				//define SP(x) = Sum[Aij * xj], Aij>=0
				//define SM(x) = Sum[-Aij * xj], Aij<0
				//we have SP(x) = b[i] + SM(x)
				
				boolean allLbPositive = true;
				int cntSP = 0;
				int cntSM = 0;
				boolean allSPLbFinite = true;
				boolean allSPUbFinite = true;
				boolean allSMLbFinite = true;
				boolean allSMUbFinite = true;
				for(short nz=0; nz<vRowPositionsI.length; nz++){
					short j = vRowPositionsI[nz];
					allLbPositive = allLbPositive && lb.getQuick(j)>=0;
					double aij = A.getQuick(i, j);
					if(aij >= 0){
						cntSP++;
						allSPLbFinite = allSPLbFinite && !isLBUnbounded(lb.getQuick(j));
						allSPUbFinite = allSPUbFinite && !isUBUnbounded(ub.getQuick(j));
					}else{
						cntSM++;
						allSMLbFinite = allSMLbFinite && !isLBUnbounded(lb.getQuick(j));
						allSMUbFinite = allSMUbFinite && !isUBUnbounded(ub.getQuick(j));
					}
					if(!(allLbPositive || allSPLbFinite || allSPUbFinite || allSMLbFinite || allSMUbFinite )){
						break;
					}
				}
				
				if(allLbPositive){
					if(allSPUbFinite){
						log.debug("all lb positive and ub of variables with positive coeff finite at row " + i);
						//we have SM < SP(ub) - b[i]
						//so ub[j] < (SP(ub) - b[i]) / -Aij, for each j in P
						int cntAijPositive = 0;
						int cntAijNegative = 0;
						double spub = 0;
						for (short nz = 0; nz < vRowPositionsI.length; nz++) {
							short j = vRowPositionsI[nz];
							double aij = A.getQuick(i, j);
							if(aij >= 0){
								cntAijPositive++;
								spub += aij * ub.getQuick(j);
							}
						}
						for (short nz = 0; nz < vRowPositionsI.length; nz++) {
							short j = vRowPositionsI[nz];
							double aij = A.getQuick(i, j);
							if(aij < 0){
								cntAijNegative++;
								if(isUBUnbounded(ub.getQuick(j)) || ub.getQuick(j) > -(spub - b.getQuick(i)) / aij){
									log.debug("old ub: " + ub.getQuick(j));
									log.debug("new ub: " + -(spub - b.getQuick(i)) / aij);
									ub.setQuick(j, -(spub - b.getQuick(i)) / aij);
									this.someReductionDone = true;
								}
							}
						}
					}
					if(allSMUbFinite){
						log.debug("all lb positive and ub of variables with negative coeff finite at row " + i);
						//we have SP < b[i] + SM(ub)
						//so ub[j] < (b[i] + SM(ub)) / Aij, for each j in SP
						double smub = 0;
						for (short nz = 0; nz < vRowPositionsI.length; nz++) {
							short j = vRowPositionsI[nz];
							double aij = A.getQuick(i, j);
							if(aij <= 0){
								smub -= aij * ub.getQuick(j);
							}
						}
						for (short nz = 0; nz < vRowPositionsI.length; nz++) {
							short j = vRowPositionsI[nz];
							double aij = A.getQuick(i, j);
							if(aij > 0){
								if(isUBUnbounded(ub.getQuick(j)) || ub.getQuick(j) > (b.getQuick(i) + smub) / aij){
									log.debug("old ub: " + ub.getQuick(j));
									log.debug("new ub: " + (b.getQuick(i) + smub) / aij);
									ub.setQuick(j, (b.getQuick(i) + smub) / aij);
									this.someReductionDone = true;
								}
							}
						}
					}
					if(cntSM == 1 && allSPLbFinite){
						log.debug("1 negative coeff and finite lb of positice coeff variables at row " + i);
						//we have SM > -b[i] + SP(lb)
						//so lb[m] > (-b[i] + SM(lb)) / Aim
						double splb = 0;
						int m = -1;
						for (short nz = 0; nz < vRowPositionsI.length; nz++) {
							short j = vRowPositionsI[nz];
							double aij = A.getQuick(i, j);
							if(aij >= 0){
								splb += aij * lb.getQuick(j);
							}else{
								m = j;
							}
						}
						double aim = -A.getQuick(i, m);
						if(isLBUnbounded(lb.getQuick(m)) || lb.getQuick(m) < (-b.getQuick(i) + splb) / aim){
							log.debug("old lb: " + lb.getQuick(m));
							log.debug("new lb: " + (-b.getQuick(i) + splb) / aim);
							lb.setQuick(m, (-b.getQuick(i) + splb) / aim);
							this.someReductionDone = true;
						}
					}
					if(cntSP == 1 && allSMLbFinite){
						log.debug("1 positive coeff and finite lb of negative coeff variables at row " + i);
						//we have SP > b[i] + SM(lb)
						//so lb[p] > (b[i] + SM(lb)) / Aip
						double smlb = 0;
						int p = -1;
						for (short nz = 0; nz < vRowPositionsI.length; nz++) {
							short j = vRowPositionsI[nz];
							double aij = A.getQuick(i, j);
							if(aij <= 0){
								smlb -= aij * lb.getQuick(j);
							}else{
								p = j;
							}
						}
						double aip = A.getQuick(i, p);
						if(isLBUnbounded(lb.getQuick(p)) || lb.getQuick(p) < (b.getQuick(i) + smlb) / aip){
							log.debug("old lb: " + lb.getQuick(p));
							log.debug("new lb: " + (b.getQuick(i) + smlb) / aip);
							lb.setQuick(p, (b.getQuick(i) + smlb) / aip);
							this.someReductionDone = true;
						}
					}
				}
			}
		}
	}
	
	private void removeDominatedConstraints(DoubleMatrix1D c,
			DoubleMatrix2D A, DoubleMatrix1D b, 
			DoubleMatrix1D lb, DoubleMatrix1D ub, 
			DoubleMatrix1D ylb, DoubleMatrix1D yub,
			DoubleMatrix1D zlb, DoubleMatrix1D zub) {
	}
	
	/**
	 * Manages:
	 * -)free column singletons
	 * -)doubleton equations combined with a column singleton
	 * -)implied free column singletons
	 */
	private void checkColumnSingletons(DoubleMatrix1D c,
			DoubleMatrix2D A, DoubleMatrix1D b, 
			DoubleMatrix1D lb, DoubleMatrix1D ub, 
			DoubleMatrix1D ylb, DoubleMatrix1D yub,
			DoubleMatrix1D zlb, DoubleMatrix1D zub) {
		for(short col=0; col<this.vColPositions.length; col++){
			if(vColPositions[col].length == 1){
				short row = vColPositions[col][0];
				log.debug("found column singleton at row " + row + ", col " + col);
				short[] vRowPositionsRow = vRowPositions[row];
				if(vRowPositionsRow.length < 2){
					continue;//this is a fixed variable
				}

				double ARcol = A.getQuick(row, col);
				double cCol = c.getQuick(col);
				boolean isCColNz = !isZero(cCol);
				double lbCol = lb.getQuick(col);
				double ubCol = ub.getQuick(col);
				boolean isLBUnbounded = isLBUnbounded(lbCol); 
				boolean isUBUnbounded = isUBUnbounded(ubCol);
				
				if(isLBUnbounded || isUBUnbounded){
					//bound on one of the optimal Lagrange multipliers.
					if(isLBUnbounded){
						if(isUBUnbounded){
							//table 2, row 1
							zlb.setQuick(col, 0);
							zub.setQuick(col, 0);
							ylb.setQuick(row, cCol / ARcol);
							yub.setQuick(row, cCol / ARcol);
						}else{
							if(ARcol > 0){
								//table 2, row 4
								zub.setQuick(col, 0);
								ylb.setQuick(row, cCol / ARcol);
							}else{
								//table 2, row 5
								zub.setQuick(col, 0);
								yub.setQuick(row, cCol / ARcol);
							}
						}
					}else{
						if(isUBUnbounded){
							if(ARcol > 0){
								//table 2, row 2
								zlb.setQuick(col, 0);
								yub.setQuick(row, cCol / ARcol);
							}else{
								//table 2, row 3
								zlb.setQuick(col, 0);
								ylb.setQuick(row, cCol / ARcol);
							}
						}
					}
					
					if(isLBUnbounded && isUBUnbounded){
						//free column singleton: one constraint and	one variable 
						//is removed from the problem without generating any fill-ins in A, 
						//although	the objective function is modified
						log.debug("free column singleton");
						//substitution into the objective function
						short[] xi = new short[vRowPositionsRow.length-1];
						double[] mi = new double[vRowPositionsRow.length - 1];
						int cntXi = 0;
						for (int j = 0; j < vRowPositionsRow.length; j++) {
							short nzJ = vRowPositionsRow[j];
							if (nzJ != col){
								xi[cntXi] = nzJ;
								mi[cntXi] = -A.getQuick(row, nzJ) / ARcol;
								cntXi++;
								if (isCColNz){
									c.setQuick(nzJ, c.getQuick(nzJ) - cCol * A.getQuick(row, nzJ) / ARcol);
								}
							}
						}
						//see Andersen & Andersen, eq (10) [that is incorrect!]
						addToPresolvingStack(new LinearDependency(col, xi, mi, b.getQuick(row)));
						for(short j = 0; j < vRowPositionsRow.length; j++){
							short column = vRowPositionsRow[j];//the nz column index
							if(column!=col && vColPositions[column].length == 1){
								//this is also a column singleton, we do not want an empty final column
								//so we fix the value of the variable
								//@TODO: fix this for unbounded bounds
								if(c.getQuick(column)<0){
									lb.setQuick(column, ub.getQuick(column));
								}else if(c.getQuick(column)>0){
									ub.setQuick(column, lb.getQuick(column));
								}else{
									ub.setQuick(column, lb.getQuick(column));
								}
								log.debug("found fixed variables: x[" + column + "]="+lb.getQuick(column));
								addToPresolvingStack(new LinearDependency(column, null, null, lb.getQuick(column)));
								pruneFixedVariable(column, c, A, b, lb, ub, ylb, yub, zlb, zub);
							}
							changeColumnsLengthPosition(column, vColPositions[column].length, vColPositions[column].length-1);
							vColPositions[column] = removeElementFromSortedArray(vColPositions[column], row);
							A.setQuick(row, column, 0.);
						}
						changeRowsLengthPosition(row, vRowPositions[row].length, 0);
						vRowPositions[row] = new short[]{};
						if(vColPositions[col].length > 0){
							log.debug("Expected empty column "+col+" but was not empty");
							throw new IllegalStateException("Expected empty column "+col+" but was not empty");
						}
						vColPositions[col] = new short[]{};
						ylb.setQuick(row, cCol / ARcol);//ok, but jet stated above
						yub.setQuick(row, cCol / ARcol);//ok, but jet stated above
						b.setQuick(row, 0);
						lb.setQuick(col, this.unboundedLBValue);
						ub.setQuick(col, this.unboundedUBValue);
						c.setQuick(col, 0);
						this.someReductionDone = true;
						continue;
					}
				}
				
				double impliedL;
				double impliedU;
				if(ARcol>0){
					impliedL = (b.getQuick(row) - h[row])/ARcol + ubCol;
					impliedU = (b.getQuick(row) - g[row])/ARcol + lbCol;
				}else{
					impliedL = (b.getQuick(row) - g[row])/ARcol + ubCol;
					impliedU = (b.getQuick(row) - h[row])/ARcol + lbCol;
				}
				boolean ifl = impliedL > lbCol;//do not use =, it will cause a loop
				boolean ifu = impliedU < ubCol;//do not use =, it will cause a loop
				if(ifl){
				  lb.setQuick(col, impliedL);//tighten the bounds
				  lbCol = impliedL;
				  this.someReductionDone = true;
				}
				if(ifu){
				  ub.setQuick(col, impliedU);//tighten the bounds
				  ubCol = impliedU;
				  this.someReductionDone = true;
				}
				boolean isImpliedFree = (ifl && ifu) || (isZero(impliedL - lbCol) && isZero(impliedU - ubCol));
								
				if(vRowPositionsRow.length == 2 || isImpliedFree){
					//substitution
					short y = -1;
					double q=0., m=0.;
					short[] xi = new short[vRowPositionsRow.length-1];
					double[] mi = new double[vRowPositionsRow.length-1];
					StringBuffer sb = new StringBuffer("x["+col+"]=");
					q = b.getQuick(row)/ARcol;
					sb.append(q);
					int cntXi = 0;
					for(int j=0; j<vRowPositionsRow.length; j++){
						short nzJ = vRowPositionsRow[j];  
						if(nzJ != col){
							double ARnzJ = A.getQuick(row, nzJ);
							m = -ARnzJ/ARcol;
							xi[cntXi] = nzJ;
							mi[cntXi] = m;
							cntXi++;
							sb.append(" + " + m + "*x["+nzJ+"]");
							if(isCColNz){
								//the objective function is modified
								double cc = c.getQuick(col) * ARnzJ / ARcol;
								c.setQuick(nzJ, c.getQuick(nzJ) - cc);
							}
							y = nzJ;
						}
					}
					addToPresolvingStack(new LinearDependency(col, xi, mi, q));
					
					if(vRowPositionsRow.length == 2){
					//NOTE: the row and the column are removed
						log.debug("doubleton equation combined with a column singleton: " + sb.toString());
						//x = m*y + q, x column singleton
						//addToDoubletonMap(col, y, m, q);
						//the bounds on the variable y are modified so	that the feasible region is unchanged even if the bounds on x are removed
						//y = x/m - q/m
						double lbY = lb.getQuick(y);
						double ubY = ub.getQuick(y);
						boolean isLBYUnbounded = isLBUnbounded(lbY);
						boolean isUBYUnbounded = isLBUnbounded(ubY);
						if(m>0){
							if(!isLBUnbounded){
								double l = lbCol/m - q/m;
								lb.setQuick(y, (isLBYUnbounded)? l : Math.max(lbY, l));
							}
							if(!isUBUnbounded){
								double u = ubCol/m - q/m;
								ub.setQuick(y, (isUBYUnbounded)? u : Math.min(ubY, u));
							}
						}else{
							if(!isUBUnbounded){
								double u = ubCol/m - q/m;
								lb.setQuick(y, (isLBYUnbounded)? u : Math.max(lbY, u));
							}
							if(!isLBUnbounded){
								double l = lbCol/m - q/m;
								ub.setQuick(y, (isUBYUnbounded)? l : Math.min(ubY, l));
							}
						}
						if(vColPositions[y].length == 1){
							//this is also a column singleton, we do not want an empty final column
							//so we fix the value of the variable
							if(c.getQuick(y)<0){
								if(isUBUnbounded(ub.getQuick(y))){
									throw new RuntimeException("unbounded problem");
								}
								lb.setQuick(y, ub.getQuick(y));
							}else if(c.getQuick(y)>0){
								if(isLBUnbounded(lb.getQuick(y))){
									throw new RuntimeException("unbounded problem");
								}
								ub.setQuick(y, lb.getQuick(y));
							}else{
								//any value is good
								if(isLBUnbounded(lb.getQuick(y)) && isUBUnbounded(ub.getQuick(y))){
									throw new RuntimeException("unbounded problem");
								}else if(!isLBUnbounded(lb.getQuick(y)) && !isUBUnbounded(ub.getQuick(y))){
									double d = (ub.getQuick(y)-lb.getQuick(y))/2;
									lb.setQuick(y, d);
									ub.setQuick(y, d);
								}else if(!isLBUnbounded(lb.getQuick(y))){
									ub.setQuick(y, lb.getQuick(y));
								}else{
									lb.setQuick(y, ub.getQuick(y));
								}
							}
						}
						//remove the bounds on col
						lb.setQuick(col, this.unboundedLBValue);
						ub.setQuick(col, this.unboundedUBValue);
						//remove the variable
						A.setQuick(row, col, 0.);
						A.setQuick(row, y, 0.);
						b.setQuick(row, 0.);
						changeColumnsLengthPosition(col, vColPositions[col].length, 0);
						vColPositions[col] = new short[]{};
						vRowPositionsRow = removeElementFromSortedArray(vRowPositionsRow, col);//just to have vRowPositionsRow[0] 
						changeRowsLengthPosition(row, vRowPositionsRow.length+1, vRowPositionsRow.length);
						changeColumnsLengthPosition(vRowPositionsRow[0], vColPositions[vRowPositionsRow[0]].length, vColPositions[vRowPositionsRow[0]].length-1);
						vColPositions[vRowPositionsRow[0]] = removeElementFromSortedArray(vColPositions[vRowPositionsRow[0]], row);
						vRowPositions[row] = new short[]{};
						this.someReductionDone = true;
						continue;
					} else {
						//NOTE: one constraint and	one variable is removed from the problem 
						//without generating any fill-ins in A, 
						//although the objective function is modified
						log.debug("implied free column singletons: " + sb.toString());
						ylb.setQuick(row, c.getQuick(col) / A.getQuick(row, col));//ok, but already stated above 
						yub.setQuick(row, c.getQuick(col) / A.getQuick(row, col));//ok, but already stated above
						for(short cc=0; cc<vRowPositions[row].length; cc++){
							short column = vRowPositions[row][cc];
							if(column == col){
								continue;
							}
							if(vColPositions[column].length == 1){
								//this is also a column singleton, we do not want an empty final column
								//so we fix the value of the variable
								if(c.getQuick(column)<0){
									//no problem of unbounded bound, this is an implied free column
									if(isUBUnbounded(ub.getQuick(column))){
										throw new RuntimeException("unbounded problem");
									}
									lb.setQuick(column, ub.getQuick(column));
								}else if(c.getQuick(column)>0){
									//no problem of unbounded bound, this is an implied free column
									if(isLBUnbounded(lb.getQuick(column))){
										throw new RuntimeException("unbounded problem");
									}
									ub.setQuick(column, lb.getQuick(column));
								}else{
									//no problem of unbounded bound, this is an implied free column
									if(isLBUnbounded(lb.getQuick(column)) || isUBUnbounded(ub.getQuick(column))){
										throw new RuntimeException("unbounded problem");
									}
									double d = (ub.getQuick(y) - lb.getQuick(y)) / 2;
									lb.setQuick(y, d);
									ub.setQuick(y, d);
								}
								log.debug("found fixed variables: x[" + column + "]="+lb.getQuick(column));
								addToPresolvingStack(new LinearDependency(column, null, null, lb.getQuick(column)));
								pruneFixedVariable(column, c, A, b, lb, ub, ylb, yub, zlb, zub);
							}
							changeColumnsLengthPosition(column, vColPositions[column].length, vColPositions[column].length-1);
							vColPositions[column] = removeElementFromSortedArray(vColPositions[column], row);
							A.setQuick(row, column, 0.);
						}
						A.setQuick(row, col, 0.);
						b.setQuick(row, 0);
						lb.setQuick(col, this.unboundedLBValue);
						ub.setQuick(col, this.unboundedUBValue);
						c.setQuick(col, 0);
						changeColumnsLengthPosition(col, vColPositions[col].length, 0);
						vColPositions[col] = new short[]{};
						changeRowsLengthPosition(row, vRowPositions[row].length, 0);
						vRowPositions[row] = new short[]{};
						this.someReductionDone = true;
						//checkProgress(c, A, b, lb, ub, ylb, yub, zlb, zub);
						continue;
					}
				}
			}		
		}
	}
	
	/**
	 * NOTE: this presolving technique needs no corresponding postsolving.
	 */
	private void removeDominatedColumns(DoubleMatrix1D c,
			DoubleMatrix2D A, DoubleMatrix1D b, 
			DoubleMatrix1D lb, DoubleMatrix1D ub, 
			DoubleMatrix1D ylb, DoubleMatrix1D yub,
			DoubleMatrix1D zlb, DoubleMatrix1D zub) {
		for(short col=0; col<this.vColPositions.length; col++){
			short[] vColPositionsCol = vColPositions[col];
			if(vColPositionsCol==null || vColPositionsCol.length==0){
				continue;
			}
			double e = 0.;
			double d = 0.;
			//it will be e <= d
			for(int i=0; i<vColPositionsCol.length; i++){
				short row = vColPositionsCol[i]; 
				double AIJ = A.getQuick(row, col);
				if(AIJ > 0){
					e += AIJ * ylb.getQuick(row); 
					d += AIJ * yub.getQuick(row);
				}else{
					e += AIJ * yub.getQuick(row); 
					d += AIJ * ylb.getQuick(row);
				}
			}
			
			double cmd = c.getQuick(col) - d;
			double cme = c.getQuick(col) - e;
			//logger.debug("cmd: " +cmd);
			//logger.debug("cme: " +cme);
			boolean isCmdPositive = cmd > 0 && !isZero(cmd);//strictly > 0
			boolean isCmeNegative = cme < 0 && !isZero(cme);//strictly < 0
			boolean isLBColUnbounded = isLBUnbounded(lb.getQuick(col));
			boolean isUBColUnbounded = isUBUnbounded(ub.getQuick(col));
			
			if(isCmdPositive || isCmeNegative){
			  //dominated column
				log.debug("found dominated column: " + col);
				if(isCmdPositive){
					zlb.setQuick(col, 0);
					if(isLBColUnbounded){
						log.debug("unbounded problem");
						throw new RuntimeException("unbounded problem");
					}
					ub.setQuick(col, lb.getQuick(col));
					log.debug("x[" + col + "]="+lb.getQuick(col));
					addToPresolvingStack(new LinearDependency(col, null, null, lb.getQuick(col)));
					pruneFixedVariable(col, c, A, b, lb, ub, ylb, yub, zlb, zub);
				}else if(isCmeNegative){
					zub.setQuick(col, 0);
					if(isUBColUnbounded){
						log.debug("unbounded problem");
						throw new RuntimeException("unbounded problem");
					}
					lb.setQuick(col, ub.getQuick(col));
					log.debug("x[" + col + "]="+ub.getQuick(col));
					addToPresolvingStack(new LinearDependency(col, null, null, ub.getQuick(col)));
					pruneFixedVariable(col, c, A, b, lb, ub, ylb, yub, zlb, zub);
				}
				continue;
			}
			
			//here we have cmd<=0 and cme>=0 (can even be unbounded)
			
			if(vColPositionsCol.length>1){//the column singletons are used to generate the bounds d and e and therefore they cannot be dropped with this test
				if(!isLBColUnbounded && isZero(cmd)){	
					//weakly dominated column, see A. & A. (27), (28)
					log.debug("found weakly dominated column: " + col);
					ub.setQuick(col, lb.getQuick(col));
					log.debug("x[" + col + "]="+lb.getQuick(col));
					addToPresolvingStack(new LinearDependency(col, null, null, lb.getQuick(col)));
					pruneFixedVariable(col, c, A, b, lb, ub, ylb, yub, zlb, zub);
					continue;
				}
				if(!isUBColUnbounded && isZero(cme)){	
					//weakly dominated column, see A. & A. (27), (28)
					log.debug("found weakly dominated column: " + col);
					lb.setQuick(col, ub.getQuick(col));
					log.debug("x[" + col + "]="+ub.getQuick(col));
					addToPresolvingStack(new LinearDependency(col, null, null, ub.getQuick(col)));
					pruneFixedVariable(col, c, A, b, lb, ub, ylb, yub, zlb, zub);
					continue;
				}
			}
			
			if(!isLBColUnbounded && isUBColUnbounded){
				//new bounds on the optimal Lagrange multipliers y
				for(int i=0; i<vColPositionsCol.length; i++){
					short row = vColPositionsCol[i]; 
					double AIJ = A.getQuick(row, col);
					if(AIJ > 0){
						if(!isUBUnbounded(cme/AIJ + ylb.getQuick(row))){
							log.debug("set new bounds on the optimal Lagrange multipliers: " + row);
							yub.setQuick(row, Math.min(yub.getQuick(row), cme/AIJ + ylb.getQuick(row)));
						}
					}else{
						if(!isLBUnbounded(cme/AIJ + yub.getQuick(row))){
							log.debug("set new bounds on the optimal Lagrange multipliers: " + row);
							ylb.setQuick(row, Math.max(ylb.getQuick(row), cme/AIJ + yub.getQuick(row)));
						}
					}
				}
			}
			if(isLBColUnbounded && !isUBColUnbounded){
			  //new bounds on the optimal Lagrange multipliers y
				for(int i=0; i<vColPositionsCol.length; i++){
					short row = vColPositionsCol[i]; 
					double AIJ = A.getQuick(row, col);
					if(AIJ > 0){
						if(!isLBUnbounded(cmd/AIJ + yub.getQuick(row))){
							log.debug("set new bounds on the optimal Lagrange multipliers: " + row);
							ylb.setQuick(row, Math.max(ylb.getQuick(row), cmd/AIJ + yub.getQuick(row)));
						}
					}else{
						if(!isUBUnbounded(cmd/AIJ + ylb.getQuick(row))){
							log.debug("set new bounds on the optimal Lagrange multipliers: " + row);
							yub.setQuick(row, Math.min(yub.getQuick(row), cmd/AIJ + ylb.getQuick(row)));
						}
					}
				}
			}
		}
	}
	
	/**
	 * NB: for the rows of A that contain the slack variables, there cannot be the same sparsity pattern
	 * (A is diagonal in its right-upper part)
	 */
	private void removeDuplicateRow(DoubleMatrix1D c,
			DoubleMatrix2D A, DoubleMatrix1D b, 
			DoubleMatrix1D lb, DoubleMatrix1D ub, 
			DoubleMatrix1D ylb, DoubleMatrix1D yub,
			DoubleMatrix1D zlb, DoubleMatrix1D zub) {
		//the position 0 is for empty rows, 1 is for row singleton and 2 for row doubleton
		int startingLength = 3;
		for(int i=startingLength; i<vRowLengthMap.length; i++){
			int[] vRowLengthMapI = vRowLengthMap[i];
			if(vRowLengthMapI == null || vRowLengthMapI.length < 1){
				//no rows has this number of nz
				continue;
			}
			
			boolean stop = false;
			for(int j=0; !stop && j<vRowLengthMapI.length; j++){
				short prow = (short)vRowLengthMapI[j];//the row of A that has this number of nz
				if(vRowPositions[prow].length==0 || prow < nOfSlackVariables){
					//the upper left part of A is diagonal if there are the slack variables:
					//there is no sparsity superset possible 
					continue;
				}
				short[] vRowPositionsProw = vRowPositions[prow];
				if(vRowPositionsProw.length != i){
					log.debug("Row "+prow+" has an unexpected number of nz: expected " + i + " but is " + vRowPositionsProw.length);
					throw new IllegalStateException();
				}
				for(int si=i; !stop && si<vRowLengthMap.length; si++){
					//look into rows with superset sparsity pattern
					int[] vRowLengthMapSI = vRowLengthMap[si];
					if(vRowLengthMapSI == null || vRowLengthMapSI.length < 1){
						continue;
					}
					for(int sj=0; sj<vRowLengthMapSI.length; sj++){
						if(si==i && sj <= j){
							continue;//look forward, not behind
						}
						short srow = (short)vRowLengthMapSI[sj];
						if(vRowPositions[srow].length==0){
							continue;
						}
						short[] vRowPositionsSrow = vRowPositions[srow];
						//same sparsity pattern?
						if(isSubsetSparsityPattern(vRowPositionsProw, vRowPositionsSrow)){
							log.debug("found superset sparsity pattern: row " + prow + " contained in row " + srow);
							
							//look for the higher number of coefficients that can be deleted
							Map<Double, List<Integer>> coeffRatiosMap = new HashMap<Double, List<Integer>>();
							for(short k=0; k<vRowPositionsProw.length; k++){
								short col = vRowPositionsProw[k];
								double APRL = A.getQuick(prow, col); 
								double ASRL = A.getQuick(srow, col);
								double ratio = -ASRL/APRL;
								//put the ratio and the column index in the map
								boolean added = false;
								for(Double keyRatio : coeffRatiosMap.keySet()){
									if(isZero(ratio - keyRatio)){
										coeffRatiosMap.get(keyRatio).add((int)col);
										added = true;
										break;
									}
								}
								if(!added){
									List<Integer> newList = new ArrayList<Integer>();
									newList.add((int)col);
									coeffRatiosMap.put(ratio, newList);
								}
							}
							//take the ratio(s) with the higher number of column indexes
							int maxNumberOfColumn = -1;
							List<Integer> candidatedColumns = null;
							for(Double keyRatio : coeffRatiosMap.keySet()){
								int size = coeffRatiosMap.get(keyRatio).size(); 
								if(size > maxNumberOfColumn){
									maxNumberOfColumn = size;
									candidatedColumns = coeffRatiosMap.get(keyRatio);
								}else if(size == maxNumberOfColumn){
									candidatedColumns.addAll(coeffRatiosMap.get(keyRatio));
								}
							}

							//look for the position with less column fill in
							short lessFilledColumn = -1;//cannot be greater
							int lessFilledColumnLength = this.originalMeq + 1;//cannot be greater
							for(short k=0; k<candidatedColumns.size(); k++){
								short col = candidatedColumns.get(k).shortValue();
								if(vColPositions[col].length>1 && vColPositions[col].length<lessFilledColumnLength ){
									lessFilledColumn = col;
									lessFilledColumnLength = vColPositions[col].length;
								}
							}
							log.debug("less filled column (" + lessFilledColumn +"): length=" + lessFilledColumnLength);
							double APRL = A.getQuick(prow, lessFilledColumn); 
							double ASRL = A.getQuick(srow, lessFilledColumn);
							double alpha = -ASRL/APRL;
							
							b.setQuick(srow, b.getQuick(srow) + alpha*b.getQuick(prow));
							//substitute A[prow] with A[prow] * alpha*A[row] for every nz entry of A[row]
							for(short t=0; t<vRowPositionsProw.length; t++){
								short cc = vRowPositionsProw[t];
								double nv = 0.;
								if(cc!=lessFilledColumn){
									nv = A.getQuick(srow, cc) + alpha*A.getQuick(prow, cc);
								}
								A.setQuick(srow, cc, nv);
								if(isZero(nv)){
									vRowPositions[srow] = removeElementFromSortedArray(vRowPositions[srow], cc);
									changeColumnsLengthPosition(cc, vColPositions[cc].length, vColPositions[cc].length-1);
									vColPositions[cc] = removeElementFromSortedArray(vColPositions[cc], srow);
									changeRowsLengthPosition(srow, vRowPositions[srow].length+1, vRowPositions[srow].length);
									A.setQuick(srow, cc, 0.);
								}
							}
							this.someReductionDone = true;
							stop = true;
							i=startingLength-1;//restart, ++ comes from the for loop
							break;
						}
					}
				}
			}
		}
	}
	
	private void removeDuplicateColumn(DoubleMatrix1D c,
			DoubleMatrix2D A, DoubleMatrix1D b, 
			DoubleMatrix1D lb, DoubleMatrix1D ub, 
			DoubleMatrix1D ylb, DoubleMatrix1D yub,
			DoubleMatrix1D zlb, DoubleMatrix1D zub) {
		//logger.debug("actual A: " + ArrayUtils.toString(A));
		//the position 0 is for empty columns, 1 is for column singleton
		int startingLength = 2;
		for(int i=startingLength; i<vColLengthMap.length; i++){
			int[] vColLengthMapI = vColLengthMap[i];
			if(vColLengthMapI == null || vColLengthMapI.length < 1){
				//no column has this number of nz
				continue;
			}
			
			boolean stop = false;
			for(int j=0; !stop && j<vColLengthMapI.length; j++){
				short pcol = (short)vColLengthMapI[j];//the column of A that has this number of nz
				short[] vColPositionsPcol = vColPositions[pcol];
				if(vColPositionsPcol.length != i){
					log.debug("Column "+pcol+" has an unexpected number of nz: expected " + i + " but is " + vColPositionsPcol.length);
					throw new IllegalStateException();
				}
				if(pcol < nOfSlackVariables){
					//the upper left part of A is diagonal if there are the slack variables:
					//the sparsity pattern can not be the same 
					continue;
				}
				//look into the next columns with the same sparsity pattern
				for(int sj=j+1; !stop && sj<vColLengthMapI.length; sj++){
					short scol = (short)vColLengthMapI[sj];
					short[] vColPositionsScol = vColPositions[scol];
					if(vColPositionsScol.length != i){
						log.debug("Column "+scol+" has an unexpected number of nz: expected " + i + " but is " + vColPositionsScol.length);
						throw new IllegalStateException();
					}
					if(isSameSparsityPattern(vColPositionsPcol, vColPositionsScol)){
						//found the same sparsity pattern
						//check if pcol = alfa * srow
						boolean isDuplicated = true;
						double v = A.getQuick(vColPositionsPcol[0], pcol) / A.getQuick(vColPositionsScol[0], scol);
						for(int k=1; k<i; k++){//"i" is the number of nz of pcol and scol
							isDuplicated = isZero(v - A.getQuick(vColPositionsPcol[k], pcol) / A.getQuick(vColPositionsScol[k], scol)); 
							if(!isDuplicated){
								break;
							}					
						}
						if(!isDuplicated){
							continue;
						}else{
							//here we have pcol = alfa *  scol
							//NB: for table 3 and 4, j=pcol and k=scol
							log.debug("found duplicated columns: col " + pcol + " and col " + scol + ": v=" + v);
							double cAlfaC = c.getQuick(pcol) - v*c.getQuick(scol); 
							boolean isLBPUnbounded = isLBUnbounded(lb.getQuick(pcol));
							boolean isUBPUnbounded = isUBUnbounded(ub.getQuick(pcol));
							if(!isZero(cAlfaC)){
								//Fixing a duplicate column
								//see table 3 of A. & A.
								if(isUBUnbounded(ub.getQuick(scol)) && zlb.getQuick(scol)>=0){
									if(v>=0 && cAlfaC>0){
										zlb.setQuick(pcol, 0);//table 3, row 1 (zj > 0)
										if(!isLBPUnbounded){//check table 1, row 1
											if(isUBPUnbounded){
												ub.setQuick(pcol, lb.getQuick(pcol));
												log.debug("found fixed variables: x[" + pcol + "]="+lb.getQuick(pcol));
												//presolvingStack.add(presolvingStack.size(), new LinearDependency(pcol, null, null, lb[pcol]));
												addToPresolvingStack(new LinearDependency(pcol, null, null, lb.getQuick(pcol)));
												pruneFixedVariable(pcol, c, A, b, lb, ub, ylb, yub, zlb, zub);
											}
										}else{//check table 1, row 3
											log.debug("unbounded problem");
											throw new RuntimeException("unbounded problem");
										}
										this.someReductionDone = true;
									}else if(v<=0 && cAlfaC<0){
										zub.setQuick(pcol, 0);//table 3, row 2 (zj < 0)
										if(isLBPUnbounded){//check table 1, row 2
											if(!isUBPUnbounded){
												lb.setQuick(pcol, ub.getQuick(pcol));
												log.debug("found fixed variables: x[" + pcol + "]="+ub.getQuick(pcol));
												//presolvingStack.add(presolvingStack.size(), new LinearDependency(pcol, null, null, ub[pcol]));
												addToPresolvingStack(new LinearDependency(pcol, null, null, ub.getQuick(pcol)));
												pruneFixedVariable(pcol, c, A, b, lb, ub, ylb, yub, zlb, zub);
											}
										}else{//check table 1, row 4
											log.debug("unbounded problem");
											throw new RuntimeException("unbounded problem");
										}
										this.someReductionDone = true;
									}
								}
								if(isLBUnbounded(lb.getQuick(scol)) && zlb.getQuick(scol)<=0){
									if(v>=0 && cAlfaC<0){
										zlb.setQuick(pcol, 0);//table 3, row 3 (zj < 0)
										if(isLBPUnbounded){//check table 1, row 2
											if(!isUBPUnbounded){
												lb.setQuick(pcol, ub.getQuick(pcol));
												log.debug("found fixed variables: x[" + pcol + "]="+ub.getQuick(pcol));
												//presolvingStack.add(presolvingStack.size(), new LinearDependency(pcol, null, null, ub[pcol]));
												addToPresolvingStack(new LinearDependency(pcol, null, null, ub.getQuick(pcol)));
												pruneFixedVariable(pcol, c, A, b, lb, ub, ylb, yub, zlb, zub);
											}
										}else{//check table 1, row 4
											log.debug("unbounded problem");
											throw new RuntimeException("unbounded problem");
										}
										this.someReductionDone = true;
									}else if(v<=0 && cAlfaC>0){
										zub.setQuick(pcol, 0);//table 3, row 4 (zj > 0)
										if(!isLBPUnbounded){//check table 1, row 1
											if(isUBPUnbounded){
												ub.setQuick(pcol, lb.getQuick(pcol));
												log.debug("found fixed variables: x[" + pcol + "]="+lb.getQuick(pcol));
												//presolvingStack.add(presolvingStack.size(), new LinearDependency(pcol, null, null, lb[pcol]));
												addToPresolvingStack(new LinearDependency(pcol, null, null, lb.getQuick(pcol)));
												pruneFixedVariable(pcol, c, A, b, lb, ub, ylb, yub, zlb, zub);
											}
										}else{//check table 1, row 3
											log.debug("unbounded problem");
											throw new RuntimeException("unbounded problem");
										}
										this.someReductionDone = true;
									}
								}
							}else{//see A. & A. (46)
								//c[j] -v*c[k] = 0 (j=pcol and k=scol)
								//Replacing two duplicate columns by one: 
								//modifies the bounds on variable scol according to Table 4 
								//and removes variable pcol from the problem.
								//that is: the variable xj and the corresponding column j is removed and 
								//the new lower and upper bounds lb[k] and ub[k] on xk are calculated as
								//given in that table
								boolean vp = v > 0;
								boolean vm = v < 0;
								if(vp || vm){
									log.debug("Replaced two duplicate columns ("+pcol+","+scol+") by one ("+scol+")");
									//remove the variable pcol(i.e. j)
									for(int r=0; r<vColPositionsPcol.length; r++){
										short row = vColPositionsPcol[r];
										vRowPositions[row] = removeElementFromSortedArray(vRowPositions[row], pcol);
										A.setQuick(row, pcol, 0.);
										changeRowsLengthPosition(row, vRowPositions[row].length+1, vRowPositions[row].length);
									}
									changeColumnsLengthPosition(pcol, vColPositions[pcol].length, 0);
									vColPositions[pcol] = new short[]{};
									DuplicatedColumn dc = new DuplicatedColumn(pcol, scol, scol, v, lb.getQuick(pcol), ub.getQuick(pcol), lb.getQuick(scol), ub.getQuick(scol));
									addToPresolvingStack(dc);
									if(vp){
										lb.setQuick(scol, lb.getQuick(scol) + v*lb.getQuick(pcol));
										ub.setQuick(scol, ub.getQuick(scol) + v*ub.getQuick(pcol));
									}else if(vm){
										lb.setQuick(scol, lb.getQuick(scol) + v*ub.getQuick(pcol));
										ub.setQuick(scol, ub.getQuick(scol) + v*lb.getQuick(pcol));
									}
									this.someReductionDone = true;
									
									//this is just for testing purpose
									if(expectedSolution!=null){
										//xk = -v*xj + xkPrime with
										//xj=pcol
										//xk=scol
										//xkPrime=scol
										//expectedSolution[scol] = expectedSolution[scol]+v*expectedSolution[pcol];
										dc.preSolve(expectedSolution);
									}
								}
							}
						}
					}
				}
			}
		}
	}
	
	/**
	 * NB: keep this method AFTER any other method that changes lb, ub, ylb, yub, zlb, zub:
	 * these will be recalculated at the nex iteration.
	 * This method causes fill-in.
	 */
	private void removeDoubletonRow(DoubleMatrix1D c,
			DoubleMatrix2D A, DoubleMatrix1D b, 
			DoubleMatrix1D lb, DoubleMatrix1D ub, 
			DoubleMatrix1D ylb, DoubleMatrix1D yub,
			DoubleMatrix1D zlb, DoubleMatrix1D zub) {
		for(short i=0; i< this.vRowPositions.length; i++){
			short[] vRowPositionsI = vRowPositions[i];
			if(vRowPositionsI.length == 2){
				short x = vRowPositionsI[0];
				short y = vRowPositionsI[1];
				//rx + sy = t;
				//x = - sy/r + t/r = my + q
				double r = A.getQuick(i, x);
				double s = A.getQuick(i, y);
				double t = b.getQuick(i);
				double m = -s/r;
				double q = t/r;
				log.debug("found doubleton row "+i+": x[" + x + "]="+m+"*x["+y+"] + " + q);
				addToPresolvingStack(new LinearDependency(x, new short[]{y}, new double[]{m}, q));
				//the bounds on the variable y are modified so	that the feasible region is unchanged even if the bounds on x are removed
				//y = x/m - q/m
				double lbX = lb.getQuick(x);
				double ubX = ub.getQuick(x);
				double lbY = lb.getQuick(y);
				double ubY = ub.getQuick(y);
				boolean isLBXUnbounded = isLBUnbounded(lbX);
				boolean isUBXUnbounded = isLBUnbounded(ubX);
				boolean isLBYUnbounded = isLBUnbounded(lbY);
				boolean isUBYUnbounded = isLBUnbounded(ubY);
				if(m>0){
					if(!isLBXUnbounded){
						double l = lbX/m - q/m;
						lb.setQuick(y, (isLBYUnbounded)? l : Math.max(lbY, l));
					}
					if(!isUBXUnbounded){
						double u = ubX/m - q/m;
						ub.setQuick(y, (isUBYUnbounded)? u : Math.min(ubY, u));
					}
				}else{
					if(!isUBXUnbounded){
						double u = ubX/m - q/m;
						lb.setQuick(y, (isLBYUnbounded)? u : Math.max(lbY, u));
					}
					if(!isLBXUnbounded){
						double l = lbX/m - q/m;
						ub.setQuick(y, (isUBYUnbounded)? l : Math.min(ubY, l));
					}
				}
				
			    //substitution into objective function
				double cc = c.getQuick(x) * s / r;
				c.setQuick(y, c.getQuick(y) -cc);
				
				//substitution: this can cause fill-in
				for(short k=0; k<this.vRowPositions.length; k++){
					if(k!=i){
						short[] vRowPositionsK = vRowPositions[k];
						for(short j=0; j<vRowPositionsK.length; j++){
							if(vRowPositionsK[j]==x){
								double AKx = A.getQuick(k, x);
								double AKy = A.getQuick(k, y);
								double AKyNew = AKy + AKx * m;//this can be 0
								if(!isZero(AKyNew)){
									//fill in
									A.setQuick(k, y, AKyNew);
									if(!ArrayUtils.contains(vRowPositionsK, y)){
										vRowPositions[k] = addToSortedArray(vRowPositionsK, y);
										changeRowsLengthPosition(k, vRowPositions[k].length-1, vRowPositions[k].length);
										changeColumnsLengthPosition(y, vColPositions[y].length, vColPositions[y].length+1);
										vColPositions[y] = addToSortedArray(vColPositions[y], k);
									}
								}else{
									vRowPositions[k] = removeElementFromSortedArray(vRowPositionsK, y);
									changeRowsLengthPosition(k, vRowPositions[k].length+1, vRowPositions[k].length);
									changeColumnsLengthPosition(y, vColPositions[y].length, vColPositions[y].length-1);
									vColPositions[y] = removeElementFromSortedArray(vColPositions[y], k);
									A.setQuick(k, y, 0.);
								}
								b.setQuick(k, b.getQuick(k) - AKx*q);
								A.setQuick(k, x, 0.);
								vRowPositions[k] = removeElementFromSortedArray(vRowPositions[k], x);
								changeRowsLengthPosition(k, vRowPositions[k].length+1, vRowPositions[k].length);
								changeColumnsLengthPosition(x, vColPositions[x].length, vColPositions[x].length-1);
								vColPositions[x] = removeElementFromSortedArray(vColPositions[x], k);
								break;
							}else	if(vRowPositionsK[j]>x){
								break;//the array is sorted
							}
						}
					}
				}
				
				//remove the row and the two columns
				vRowPositions[i] = new short[]{};
				if(vColPositions[x].length != 1 && vColPositions[x][0] != i){
					log.debug("Expected empty column "+x+" but was not empty");
					throw new IllegalStateException("Expected empty column "+x+" but was not empty");
				}
				changeColumnsLengthPosition(x, vColPositions[x].length, 0);
				vColPositions[x] = new short[]{};
				changeColumnsLengthPosition(y, vColPositions[y].length, vColPositions[y].length-1);
				vColPositions[y] = removeElementFromSortedArray(vColPositions[y], i);
				A.setQuick(i, x, 0.);
				A.setQuick(i, y, 0.);			
				b.setQuick(i, 0);
				this.someReductionDone = true;
			}
		}
	}
	
	/**
	 * Given R the row scaling factor and T the column scaling factor, if x is 
	 * the solution of the problem before scaling and x1 is the solution 
	 * of the problem after scaling, we have: 
	 * <br>R.A.T.x1 = Rb and x = T.x1
	 * 
	 * Every scaling needs the adjustment of the other data vectors of the LP problem. 
	 * After scaling, the vectors c, lb, ub become
	 * <br>c -> T.c 	 
	 * <br>lb -> InvT.lb
	 * <br>ub -> InvT.ub
	 * 
	 * The objective value is the same.
	 *  
	 * @see Xin Huang, "Preprocessing and Postprocessing in Linear Optimization" 2.8
	 */
	private void scaling() {
		if(presolvedA instanceof SparseDoubleMatrix2D){
			MatrixRescaler rescaler = new Matrix1NornRescaler();
			DoubleMatrix1D[] UV = rescaler.getMatrixScalingFactors((SparseDoubleMatrix2D) presolvedA);
			this.R = UV[0];
			this.T = UV[1];
			if(log.isDebugEnabled()){
				boolean checkOK = rescaler.checkScaling(presolvedA, R, T);
				if(!checkOK){
					log.warn("Scaling failed (checkScaling = false)");
				}
				double[] cn_00_original = ColtUtils.getConditionNumberRange(new Array2DRowRealMatrix(presolvedA.toArray()), Integer.MAX_VALUE);
				log.debug("cn_00_original A before scaling: " + ArrayUtils.toString(cn_00_original));
				double[] cn_2_original = ColtUtils.getConditionNumberRange(new Array2DRowRealMatrix(presolvedA.toArray()), 2);
				log.debug("cn_2_original A before scaling : " + ArrayUtils.toString(cn_2_original));
			}
			//scaling A -> R.A.T
			presolvedA = ColtUtils.diagonalMatrixMult(R, presolvedA, T);
			
			if(log.isDebugEnabled()){
				double[] cn_00_scaled = ColtUtils.getConditionNumberRange(new Array2DRowRealMatrix(presolvedA.toArray()), Integer.MAX_VALUE);
				log.debug("cn_00_scaled A after scaling : " + ArrayUtils.toString(cn_00_scaled));
				double[] cn_2_scaled = ColtUtils.getConditionNumberRange(new Array2DRowRealMatrix(presolvedA.toArray()), 2);
				log.debug("cn_2_scaled A after scaling  : " + ArrayUtils.toString(cn_2_scaled));
			}
			for(int i=0; i<R.size(); i++){
				double ri = R.getQuick(i);
				presolvedB.setQuick(i,  presolvedB.getQuick(i) * ri);
			}
			
			this.minRescaledLB = Double.MAX_VALUE;
			this.maxRescaledUB = -Double.MAX_VALUE;
			for(int i=0; i<T.size(); i++){
				double ti = T.getQuick(i);
				
				presolvedC.setQuick(i,  presolvedC.getQuick(i) * ti);
				
				double lbi = presolvedLB.getQuick(i) / ti;
				presolvedLB.setQuick(i, lbi);
				this.minRescaledLB = Math.min(this.minRescaledLB, lbi);
				
				double ubi = presolvedUB.getQuick(i) / ti;
				presolvedUB.setQuick(i,  ubi);
				this.maxRescaledUB = Math.max(this.maxRescaledUB, ubi);
			}
		}
	}
	
	private void removeAllEmptyRowsAndColumns(DoubleMatrix1D c,
			DoubleMatrix2D A, DoubleMatrix1D b, 
			DoubleMatrix1D lb, DoubleMatrix1D ub, 
			DoubleMatrix1D ylb, DoubleMatrix1D yub,
			DoubleMatrix1D zlb, DoubleMatrix1D zub){
	}
	
	private void pruneFixedVariable(short x, DoubleMatrix1D c,
			DoubleMatrix2D A, DoubleMatrix1D b, 
			DoubleMatrix1D lb, DoubleMatrix1D ub, 
			DoubleMatrix1D ylb, DoubleMatrix1D yub,
			DoubleMatrix1D zlb, DoubleMatrix1D zub){
		double v = lb.getQuick(x);
		for(short i=0; i< this.vRowPositions.length; i++){
			if(ArrayUtils.contains(vRowPositions[i], x)){
				vRowPositions[i] = removeElementFromSortedArray(this.vRowPositions[i], x);
				changeRowsLengthPosition(i, vRowPositions[i].length+1, vRowPositions[i].length);
				if(vRowPositions[i]==null || vRowPositions[i].length==0){
					//this row contains only x
					if(!isZero(v - b.getQuick(i) / A.getQuick(i, x))){	
						log.debug("infeasible problem");
						throw new RuntimeException("infeasible problem");
					}
					A.setQuick(i, x, 0.);
					b.setQuick(i, 0);
				}else{
					b.setQuick(i, b.getQuick(i) - A.getQuick(i, x) * v);
					A.setQuick(i, x, 0.);
				}
			}
		}
		changeColumnsLengthPosition(x, vColPositions[x].length, 0);
		vColPositions[x] = new short[]{};
		this.someReductionDone = true;
	}
	
	/**
	 * Removes the first occurrence of the specified element from the
   * specified array. All subsequent elements are shifted to the left.
	 */
	private short[] removeElementFromSortedArray(short[] array, short element){
		if(array.length < 2){
			return new short[]{};
		}
		return ArrayUtils.removeElement(array, element);
	}
	
	/**
	 * Removes the first occurrence of the specified element from the
   * specified array. All subsequent elements are shifted to the left.
	 */
	private static int[] removeElementFromSortedArray(int[] array, int element){
		if(array.length < 2){
			return new int[]{};
		}
		return ArrayUtils.removeElement(array, element);
	}
	
	private static short[] addToSortedArray(short[] array, short element){
		short[] ret = new short[array.length+1];
		short cnt = 0;
		boolean goStraight = false;
		for(short i=0; i<array.length; i++){
			short s = array[i];
			if(goStraight){
				ret[cnt] = s;
				cnt++;
			}else{
				if(s < element){
					ret[cnt] = s;
					cnt++;
					continue;
				}
				if(s == element){
					return array;
				}
				if(s > element){
					ret[cnt] = element;
					cnt++;
					ret[cnt] = s;
					cnt++;
					goStraight = true;
				}
			}
		}
		if(cnt < ret.length){
			//to be added at the last position
			ret[cnt] = element;
		}
		return ret;
	}
	
	private static int[] addToSortedArray(int[] array, int element){
		int[] ret = new int[array.length+1];
		short cnt = 0;
		boolean goStraight = false;
		for(short i=0; i<array.length; i++){
			int s = array[i];
			if(goStraight){
				ret[cnt] = s;
				cnt++;
			}else{
				if(s < element){
					ret[cnt] = s;
					cnt++;
					continue;
				}
				if(s == element){
					return array;
				}
				if(s > element){
					ret[cnt] = element;
					cnt++;
					ret[cnt] = s;
					cnt++;
					goStraight = true;
				}
			}
		}
		if(cnt < ret.length){
			//to be added at the last position
			ret[cnt] = element;
		}
		return ret;
	}
	
	private static boolean isSubsetSparsityPattern(short[] subset, short[] superset) {
		short position = 0;
		for(short i=0; i<subset.length; i++){
			short s = subset[i];
			boolean found = false;
			for(short j=position; j<superset.length; j++){
				if(superset[j] == s){
					found = true;
					position = j;
					break;
				}
			}
			if(!found){
				return false;
			}
		}
		return true;
	}
	
	private static boolean isSameSparsityPattern(short[] sp1, short[] sp2) {
		if(sp1.length == sp2.length){
			for(int k=0; k<sp1.length; k++){
				if(sp1[k] != sp2[k]){
					return false;
				}						
			}
		}
		return true;
	}

	public boolean isLBUnbounded(double lb){
		return Double.compare(unboundedLBValue, lb) == 0;
	}
	
	public boolean isUBUnbounded(double ub){
		return Double.compare(unboundedUBValue, ub) == 0;
	}
	
	public int getOriginalN() {
		return this.originalN;
	}
	
	public int getOriginalMeq() {
		return this.originalMeq;
	}

	public int getPresolvedN() {
		return this.presolvedN;
	}
	
	public int getPresolvedMeq() {
		return this.presolvedMeq;
	}

	public DoubleMatrix1D getPresolvedC() {
		return this.presolvedC;
	}
	
	public DoubleMatrix2D getPresolvedA() {
		return this.presolvedA;
	}

	public DoubleMatrix1D getPresolvedB() {
		return this.presolvedB;
	}

	public DoubleMatrix1D getPresolvedLB() {
		return this.presolvedLB;
	}
	
	public DoubleMatrix1D getPresolvedUB() {
		return this.presolvedUB;
	}
	
	public DoubleMatrix1D getPresolvedYlb() {
		return this.presolvedYlb;
	}

	public DoubleMatrix1D getPresolvedYub() {
		return this.presolvedYub;
	}

	public DoubleMatrix1D getPresolvedZlb() {
		return this.presolvedZlb;
	}

	public DoubleMatrix1D getPresolvedZub() {
		return this.presolvedZub;
	}
	
	private boolean isZero(double d){
		//return Double.compare(d, 0.)==0;
		return Math.abs(d) < eps;
		//return Double.compare(d + 1., 1.)==0;
	}
	
	public void setNOfSlackVariables(short nOfSlackVariables) {
		this.nOfSlackVariables = nOfSlackVariables;
	}
	
	private abstract class PresolvingStackElement{
		abstract void postSolve(double[] x);
		abstract void preSolve(double[] x);
	}
	
	/**
	 * x = q + Sum_i[mi * xi]
	 */
	private class LinearDependency extends PresolvingStackElement{
		short x;
		short[] xi;
		double[] mi;
		double q;
		LinearDependency(short x, short[] xi, double[] mi, double q){
			this.x = x;
			this.xi = xi;
			this.mi = mi;
			this.q = q;
		}
		
		@Override
		void postSolve(double[] postsolvedX){
			//es x[5] = m1*x[1] + m3*x[3] + q
			//short[] xi = this.xi;
			for(int k=0; this.xi!=null && k<this.xi.length; k++){
				postsolvedX[this.x] += this.mi[k] * postsolvedX[this.xi[k]];
			}
			postsolvedX[this.x] += this.q;
		}
		
		@Override
		void preSolve(double[] v) {
			//es x[1]=+1.0*x[2]+-5.0
//			for(int k=0; this.xi!=null && k<this.xi.length; k++){
//				v[this.x] += this.mi[k] * v[this.xi[k]];
//			}
//			v[this.x] += this.q;
		}
		
		@Override
		public String toString(){
			StringBuffer sb = new StringBuffer();
			sb.append("x["+x+"]=");
			for(short i=0; xi!=null && i<xi.length; i++){
				sb.append("+"+mi[i]+"*x["+xi[i]+"]");
			}
			sb.append("+"+q);
			return sb.toString();
		}
	}
	
	/**
	 * The presolving stack element relative to the substitution:
	 * xk = -v*xj + xkPrime
	 */
	private class DuplicatedColumn extends PresolvingStackElement{
		short xj = -1;
		short xk = -1;
		short xkPrime = -1;
		double v = Double.NaN;
		double lbj;//the lower bound of the presolved variables xj 
		double ubj;//the upper bound of the presolved variables xj
		double lbk;//the lower bound of the variables xk 
		double ubk;//the upper bound of the variables xk
		DuplicatedColumn(short xj, short xk, short xkPrime, double v, double lbj, double ubj, double lbk, double ubk){
			this.xj = xj;
			this.xk = xk;
			this.xkPrime = xkPrime;
			this.v = v;
			this.lbj = lbj;
			this.ubj = ubj;
			this.lbk = lbk;
			this.ubk = ubk;
		}
		
		@Override
		void postSolve(double[] postsolvedX){
			//getting back the original variables, the original bounds must be respected
			//NB: remember that xj is a dependent variables (taken out from the problem by the presolver)
			
			this.lbk = isLBUnbounded(this.lbk)? -Double.MAX_VALUE : this.lbk;  
			this.lbj = isLBUnbounded(this.lbj)? -Double.MAX_VALUE : this.lbj;
			this.ubk = isLBUnbounded(this.ubk)?  Double.MAX_VALUE : this.ubk;
			this.ubj = isUBUnbounded(this.ubj)?  Double.MAX_VALUE : this.ubj;
			
			if(v>0){
				//we must have:
				//	lbk < xk < ubk
				//but
				//	xk = xkPrime -v*xj
				//and so (-v>0):
				//	xkPrime-v*ubj < xk = xkPrime -v*xj < xkPrime-v*lbj
				//then:
				//	Math.max(lbk, xkPrime-v*ubj) < xk < Math.min(ubk, xkPrime-v*lbj);
				double p = postsolvedX[xkPrime];
				postsolvedX[xk] = Math.max(lbk, p - v*ubj);
				postsolvedX[xj] = (p-postsolvedX[xk])/v;  
			}else if(v<0){
				//we must have:
				//	lbk < xk < ubk
				//but
				//	xk = xkPrime -v*xj
				//and so (-v<0):
				//	xkPrime-v*lbj < xk = xkPrime -v*xj < xkPrime-v*ubj
				//then:
				//	Math.max(lbk, xkPrime-v*lbj) < xk < Math.min(ubk, xkPrime-v*ubj);
				double p = postsolvedX[xkPrime];
				postsolvedX[xk] = Math.max(lbk, p - v*lbj);
				postsolvedX[xj] = (p-postsolvedX[xk])/v; 
			}else{
				throw new IllegalStateException("coefficient v must be >0 or <0");
			}
		}
		
		@Override
		void preSolve(double[] x) {
			//es x[2]=-1.0*x[0] + xPrime[2]
			x[this.xkPrime] = x[this.xk] + this.v * x[this.xj];
		}
		
		@Override
		public String toString(){
			StringBuffer sb = new StringBuffer();
			sb.append("x[" + xk + "]=-" + v + "*x[" + xj + "] + xPrime[" + xkPrime + "]");
			return sb.toString();
		}
	}
	
	private void addToPresolvingStack(LinearDependency linearDependency){
		this.indipendentVariables[linearDependency.x] = false;
		presolvingStack.add(presolvingStack.size(), linearDependency);
	}
	
	private void addToPresolvingStack(DuplicatedColumn duplicatedColumn){
		this.indipendentVariables[duplicatedColumn.xj] = false;
		presolvingStack.add(presolvingStack.size(), duplicatedColumn);
	}
	
	/**
	 * This method is just for testing scope.
	 */
	private void checkProgress(DoubleMatrix1D c,
			DoubleMatrix2D A, DoubleMatrix1D b, 
			DoubleMatrix1D lb, DoubleMatrix1D ub, 
			DoubleMatrix1D ylb, DoubleMatrix1D yub,
			DoubleMatrix1D zlb, DoubleMatrix1D zub){
		
		if(this.expectedSolution==null){
			return;
		}
		
		if(Double.isNaN(this.expectedTolerance)){
			//for this to work properly, this method must be called at least one time before presolving operations start
			RealVector X = MatrixUtils.createRealVector(expectedSolution);
			RealMatrix AMatrix = MatrixUtils.createRealMatrix(A.toArray()); 
			RealVector Bvector = MatrixUtils.createRealVector(b.toArray());
			RealVector Axb = AMatrix.operate(X).subtract(Bvector);
			double norm = Axb.getNorm();
			this.expectedTolerance = Math.max(1.e-7, 1.01 * norm);
		}
		
		double tolerance = this.expectedTolerance;
		log.debug("tolerance: " + tolerance);
		
		RealVector X = MatrixUtils.createRealVector(expectedSolution);
		RealMatrix AMatrix = MatrixUtils.createRealMatrix(A.toArray()); 
		RealVector Bvector = MatrixUtils.createRealVector(b.toArray());
		//logger.debug("A.X-b: " + ArrayUtils.toString(originalA.operate(X).subtract(originalB)));
		
		//nz rows
		for(int i=0; i<vRowPositions.length; i++){
			short[] vRowPositionsI = vRowPositions[i];
			for(short nzJ : vRowPositionsI){
				if(Double.compare(A.getQuick(i, nzJ), 0.) == 0){
					log.debug("entry "+i+"," + nzJ + " est zero: " + A.getQuick(i, nzJ));
					throw new IllegalStateException();
				}
			}
		}
		
		//nz columns
		for(int j=0; j<vColPositions.length; j++){
			short[] vColPositionsJ = vColPositions[j];
			for(short nzI : vColPositionsJ){
				if(Double.compare(A.getQuick(nzI, j), 0.) == 0){
					log.debug("entry ("+nzI+"," + j + ") est zero: " + A.getQuick(nzI, j));
					throw new IllegalStateException();
				}
			}
		}
		
		//nz Aij
		for(int i=0; i<A.rows(); i++){
			short[] vRowPositionsI = vRowPositions[i];
			for(int j=0; j<A.columns(); j++){
				if(Double.compare(Math.abs(A.getQuick(i, j)), 0.) != 0){
					if(!ArrayUtils.contains(vRowPositionsI, (short)j)){
						log.debug("entry "+i+"," + j + " est non-zero: " + A.getQuick(i, j));
						throw new IllegalStateException();
					}
					if(!ArrayUtils.contains(vColPositions[j], (short)i)){
						log.debug("entry "+i+"," + j + " est non-zero: " + A.getQuick(i, j));
						throw new IllegalStateException();
					}
				}
			}
		}
		
		//A.x = b
		RealVector Axb = AMatrix.operate(X).subtract(Bvector); 
		double norm = Axb.getNorm();
		log.debug("|| A.x-b ||: " + norm);
		if(norm > tolerance){
			//where is the error?
			for(int i=0; i<Axb.getDimension(); i++){
				if(Math.abs(Axb.getEntry(i)) > tolerance){
					log.debug("entry "+i+": " + Axb.getEntry(i));
					throw new IllegalStateException();
				}
			}
			throw new IllegalStateException();
		}
		
		//upper e lower
		for(int i=0; i<X.getDimension(); i++){
			if(X.getEntry(i) + tolerance < lb.getQuick(i)){
				log.debug("lower bound "+i+" not respected: lb="+lb.getQuick(i)+ ", value=" + X.getEntry(i));
				throw new IllegalStateException();
			}
			if(X.getEntry(i) > ub.getQuick(i) + tolerance){
				log.debug("upper bound "+i+" not respected: ub="+ub.getQuick(i)+ ", value=" + X.getEntry(i));
				throw new IllegalStateException();
			}
		}
	}
	
	private void changeRowsLengthPosition(short rowIndex, int lengthIndexFrom, int lengthIndexTo){
		if(lengthIndexFrom==0){
			return;
		}
		if(vRowLengthMap[lengthIndexTo]==null){
			vRowLengthMap[lengthIndexTo] = new int[]{};
		}
		vRowLengthMap[lengthIndexTo] = addToSortedArray(vRowLengthMap[lengthIndexTo], rowIndex);
		vRowLengthMap[lengthIndexFrom] = removeElementFromSortedArray(vRowLengthMap[lengthIndexFrom], (int)rowIndex);
	}
	
	private void changeColumnsLengthPosition(short colIndex, int lengthIndexFrom, int lengthIndexTo){
		if(lengthIndexFrom==0){
			return;
		}
		if(vColLengthMap[lengthIndexTo]==null){
			vColLengthMap[lengthIndexTo] = new int[]{};
		}
		vColLengthMap[lengthIndexTo] = addToSortedArray(vColLengthMap[lengthIndexTo], colIndex);
		vColLengthMap[lengthIndexFrom] = removeElementFromSortedArray(vColLengthMap[lengthIndexFrom], (int)colIndex);
	}
	
	/**
	 * Just for testing porpose
	 */
	public void setExpectedSolution(double sol[]){
		this.expectedSolution = Arrays.copyOf(sol, sol.length);
	}

	public double getMinRescaledLB() {
		return minRescaledLB;
	}

	public double getMaxRescaledUB() {
		return maxRescaledUB;
	}
	
	/**
	 * Set the value for zero-comparison: 
	 * <br>if |a - b| < eps then a - b = 0.
	 * <br>Default is the <i>double epsilon machine<i> value.  
	 */
	public void setZeroTolerance(double eps) {
		this.eps = eps;
	}
	
	/**
	 * Just for testing porpose
	 */
//	public void setExpectedTolerance(double expectedTolerance) {
//		this.expectedTolerance = expectedTolerance;
//	}
}
