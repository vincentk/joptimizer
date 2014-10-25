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
import java.util.List;

import com.joptimizer.util.ColtUtils;

import cern.colt.function.IntIntDoubleFunction;
import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;


/**
 * Converts a general LP problem stated in the form (1):
 * <br>min(c) s.t.
 * <br>G.x < h
 * <br>A.x = b
 * <br>lb <= x <= ub
 * <br>
 * <br>to the (strictly)standard form (2)
 * <br>min(c) s.t.
 * <br>A.x = b
 * <br>x >= 0
 * <br>
 * <br>or to the (quasi)standard form (3)
 * <br>min(c) s.t.
 * <br>A.x = b
 * <br>lb <= x <= ub
 * <br>
 * <br>Setting the field <i>strictlyStandardForm</i> to true, the conversion is in the (strictly) standard form (2).
 * 
 * <br>Note 1: (3) it is not exactly the standard LP form (2) because of the more general lower and upper bounds terms.
 * <br>Note 2: if the vector lb is not passed in, all the lower bounds are assumed to be equal to the value of the field <i>unboundedLBValue</i> 
 * <br>Note 3: if the vector ub is not passed in, all the upper bounds are assumed to be equal to the value of the field <i>unboundedUBValue</i>
 * <br>Note 4: unboundedLBValue is the distinctive value of an unbounded lower bound. It must be one of the values:
 *  <ol>
 *   <li>Double.NaN (the default)</li>
 *   <li>Double.NEGATIVE_INFINITY</li>
 *  </ol>
 * <br>Note 5: unboundedUBValue is the distinctive value of an unbounded upper bound. It must be one of the values:
 *  <ol>
 *   <li>Double.NaN (the default)</li>
 *   <li>Double.POSITIVE_INFINITY</li>
 *  </ol> 
 * 
 * @see Converting LPs to standard form, "Convex Optimization", p 147
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 * @TODO: the strict conversion is not yet ready.
 */
public class LPStandardConverter {
	public static final double DEFAULT_UNBOUNDED_LOWER_BOUND = Double.NaN;//NaN because in the tests files the unbounded lb are usually with this value
	public static final double DEFAULT_UNBOUNDED_UPPER_BOUND = Double.NaN;//NaN because in the tests files the unbounded ub are usually with this value
//	public static final double DEFAULT_UNSPECIFIED_LOWER_BOUND = DEFAULT_UNBOUNDED_LOWER_BOUND;
//	public static final double DEFAULT_UNSPECIFIED_UPPER_BOUND = DEFAULT_UNBOUNDED_UPPER_BOUND;
	
	private boolean useSparsity = true;
	private int originalN;//original number of variables
	private int standardN;//final number of variables
	private int standardS;//final number of slack variables for inequalities constraints
	private DoubleMatrix1D standardC;//final objective function
	private DoubleMatrix2D standardA;//final equalities constraints coefficients
	private DoubleMatrix1D standardB;//final equalities constraints limits
	private DoubleMatrix1D standardLB;//final lower bounds
	private DoubleMatrix1D standardUB;//final upper bounds
	private List<Integer> splittedVariablesList = new ArrayList<Integer>();//original variables to split for having positive final variables
	private boolean[] lbSlack;
	private boolean[] ubSlack;
	//private List<Integer> slackLBVariablesList = new ArrayList<Integer>();//original variables that need a slack variable for the lower bound
	//private List<Integer> slackUBVariablesList = new ArrayList<Integer>();//original variables that need a slack variable for the upper bound
//	private double unspecifiedLBValue = DEFAULT_UNSPECIFIED_LOWER_BOUND;
//	private double unspecifiedUBValue = DEFAULT_UNSPECIFIED_UPPER_BOUND;
	private double unboundedLBValue = DEFAULT_UNBOUNDED_LOWER_BOUND;
	private double unboundedUBValue = DEFAULT_UNBOUNDED_UPPER_BOUND;
	/**
	 * if true, convert the problem to the strictly standard form:
	 * min(c) s.t.
	 * A.x = b
	 * x >=0
	 * 
	 * otherwise, convert the problem to the more general (quasi) standard form:
	 * min(c) s.t.
	 * A.x = b
	 * lb <= x <= ub
	 */
	private boolean strictlyStandardForm = false;
	private Algebra ALG = Algebra.DEFAULT;
	private DoubleFactory1D F1 = null;
	private DoubleFactory2D F2 = null;
	//private double[] postconvertedX; 
	
	public LPStandardConverter(){
		 this(false);
	}
	
	public LPStandardConverter(boolean strictlyStandardForm){
		this(strictlyStandardForm, DEFAULT_UNBOUNDED_LOWER_BOUND, DEFAULT_UNBOUNDED_UPPER_BOUND);
	}
	
	public LPStandardConverter(double unboundedLBValue, double unboundedUBValue){
		this(false, unboundedLBValue, unboundedUBValue);
	}
			
	public LPStandardConverter(boolean strictlyStandardForm, double unboundedLBValue, double unboundedUBValue){
		if(!Double.isNaN(unboundedLBValue) && !Double.isInfinite(unboundedLBValue) ){
			throw new IllegalArgumentException("The field unboundedLBValue must be set to Double.NaN or Double.NEGATIVE_INFINITY");
		}
		if(!Double.isNaN(unboundedUBValue) && !Double.isInfinite(unboundedUBValue) ){
			throw new IllegalArgumentException("The field unboundedUBValue must be set to Double.NaN or Double.POSITIVE_INFINITY");
		}
		this.strictlyStandardForm = strictlyStandardForm;
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
	
	/**
	 * Transforms the problem from a general form to the (quasi) standard LP form.
	 * 
	 * @param originalLB if null, all lower bounds default to this.unspecifiedLBValue
	 * @param originalUB if null, all upper bounds default to this.unspecifiedUBValue
	 * 
	 * @see Converting LPs to standard form, "Convex Optimization", p 147
	 */
	public void toStandardForm(double[] originalC, double[][] originalG, double[] originalH,
			double[][] originalA, double[] originalB, double[] originalLB, double[] originalUB) {
		
		this.F1 = (useSparsity)? DoubleFactory1D.sparse : DoubleFactory1D.dense;
		this.F2 = (useSparsity)? DoubleFactory2D.sparse : DoubleFactory2D.dense;
		
		DoubleMatrix1D cVector = F1.make(originalC);
		DoubleMatrix2D GMatrix = null;
		DoubleMatrix1D hVector = null;
		if(originalG!=null){
			//GMatrix = (useSparsity)? new SparseDoubleMatrix2D(G) : F2.make(G);
			GMatrix = F2.make(originalG);
			hVector = F1.make(originalH);
		}
		DoubleMatrix2D AMatrix = null;
		DoubleMatrix1D bVector = null;
		if(originalA!=null){
			//AMatrix = (useSparsity)? new SparseDoubleMatrix2D(A) : F2.make(A);
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
		
		toStandardForm(cVector, GMatrix, hVector, AMatrix, bVector, lbVector, ubVector);
	}
	
	/**
	 * Transforms the problem from a general form to the (quasi) standard LP form.
	 * 
	 * @param originalLB if null, all lower bounds default to this.unspecifiedLBValue
	 * @param originalUB if null, all upper bounds default to this.unspecifiedUBValue
	 * 
	 * @see Converting LPs to standard form, "Convex Optimization", p 147
	 */
	public void toStandardForm(DoubleMatrix1D originalC,
			DoubleMatrix2D originalG, DoubleMatrix1D originalH, 
			DoubleMatrix2D originalA, DoubleMatrix1D originalB,
			DoubleMatrix1D originalLB, DoubleMatrix1D originalUB){
		
		this.F1 = (useSparsity)? DoubleFactory1D.sparse : DoubleFactory1D.dense;
		this.F2 = (useSparsity)? DoubleFactory2D.sparse : DoubleFactory2D.dense;
		
		this.originalN = originalC.size();
		if(originalLB != null && originalUB !=null){
			if(originalLB.size() != originalUB.size()){
				throw new IllegalArgumentException("lower and upper bounds have different size");
			}
		}
		if(originalLB==null){
			//there are no lb, that is they are all unbounded
			originalLB = F1.make(originalN, unboundedLBValue);
		}
		if(originalUB==null){
			//there are no ub, that is they are all unbounded
			originalUB = F1.make(originalN, unboundedUBValue);
		}
		
		if(originalG==null && !strictlyStandardForm){
			//nothing to convert
			this.standardN = originalN;
			this.standardA = originalA;
			this.standardB = originalB;
			this.standardC = originalC;
			this.standardLB = originalLB;
			this.standardUB = originalUB;
			return;
		}
		
		//definition of the elements
		int nOfSlackG = (originalG!=null)? originalG.rows() : 0;//number of slack variables given by G
		int nOfSlackUB = 0;//number of slack variables given by the upper bounds
		int nOfSplittedVariables = 0;//number of variables to split, x = xPlus-xMinus with xPlus and xMinus positive
		int nOfSlackLB = 0;//number of slack variables given by the lower bounds
		lbSlack = new boolean[originalN];
		ubSlack = new boolean[originalN];
		
		if(strictlyStandardForm){
			//record the variables that need the split x = xPlus-xMinus
			for(int i=0; i<originalN; i++){
				double lbi = originalLB.getQuick(i);
				double ubi = originalUB.getQuick(i);
				if(isLbUnbounded(lbi)){
					//no slack (no row in the final A) but split the variable
					//we have lb[i] = -oo, so must split this variable (because it is not forced to be non negative)
					splittedVariablesList.add(splittedVariablesList.size(), i);
				}else{
					int lbCompare = Double.compare(lbi, 0.);
					if(lbCompare < 0){
						//this inequality must become an equality
						nOfSlackLB++;
						lbSlack[i] = true;
						//slackLBVariablesList.add(slackLBVariablesList.size(), i);
						//we have lb[i] < 0, so must split this variable (because it is not forced to be non negative)
						splittedVariablesList.add(splittedVariablesList.size(), i);  
					}else if(lbCompare > 0){
						//we have lb[i] > 0, and this must become a row in the standardA matrix (standard lb limits are all = 0)
						nOfSlackLB++;
						lbSlack[i] = true;
						//slackLBVariablesList.add(slackLBVariablesList.size(), i);
					}
				}
				if(!isUbUnbounded(ubi)){
					//this lb must become a row in the standardA matrix (there are no standard ub limits)
					nOfSlackUB++;
					ubSlack[i] = true;
					//slackUBVariablesList.add(slackUBVariablesList.size(), i);
				}
			}
			nOfSplittedVariables = splittedVariablesList.size();
			//nOfSlackLB = slackLBVariablesList.size();
			//nOfSlackUB = slackUBVariablesList.size();
		}
		
		//The first step: introduce slack variables s[i] for all the inequalities
		this.standardS = nOfSlackG + nOfSlackUB + nOfSlackLB;
		this.standardN = standardS + originalN + nOfSplittedVariables;
		if(standardS==0 && nOfSplittedVariables==0){
			standardA = originalA;
			standardB = originalB;
			standardC = originalC;
			standardLB = originalLB;
			standardUB = originalUB;
		}else{
			//we must build a final A matrix that is different from the original
			
			//standardA = (useSparsity)? new SparseDoubleMatrix2D(standardS + A.rows(), standardN) : F2.make(standardS + A.rows(), standardN);
			if(originalA!=null){
				standardA = F2.make(standardS + originalA.rows(), standardN);
				standardB = F1.make(standardS + originalB.size());
			}else{
				standardA = F2.make(standardS, standardN);
				standardB = F1.make(standardS);
			}
			
			//filling with original G values
			for(int i=0; i<nOfSlackG; i++){
				standardA.set(i, i, 1);//slack variable position
				standardB.setQuick(i, originalH.getQuick(i));
			}
			if(originalG instanceof SparseDoubleMatrix2D){
				originalG.forEachNonZero(new IntIntDoubleFunction() {
					public double apply(int i, int j, double gij) {
						standardA.set(i, standardS + j, gij);
						return gij;
					}
				});
			}else{
				for(int i=0; i<nOfSlackG; i++){
					for(int j=0; j<originalN; j++){
						standardA.set(i, standardS + j, originalG.getQuick(i, j));
					}
				}
			}
			
			//filling for the lower and upper bounds
			int cntSlackLB = 0;
			int cntSlackUB = 0;
			for(int i=0; i<originalN; i++ ){
				if(lbSlack[i]){
					standardA.set(nOfSlackG + cntSlackLB, nOfSlackG + cntSlackLB, 1);//slack variable position
					standardA.set(nOfSlackG + cntSlackLB, standardS + i,  -1);
					standardB.setQuick(nOfSlackG + cntSlackLB, -originalLB.getQuick(i));
					cntSlackLB++;
				}
				if(ubSlack[i]){
					standardA.set(nOfSlackG + nOfSlackLB + cntSlackUB, nOfSlackG + nOfSlackLB + cntSlackUB, 1);//slack variable position
					standardA.set(nOfSlackG + nOfSlackLB + cntSlackUB, standardS + i,  1);
					standardB.setQuick(nOfSlackG + nOfSlackLB + cntSlackUB, originalUB.getQuick(i));
					cntSlackUB++;
				}
			}
			
//			//filling for LB slack variables
//			for(int sv=0; sv<nOfSlackLB; sv++ ){
//				int i = slackLBVariablesList.get(sv);
//				standardA.set(nOfSlackG + nOfSlackUB + sv, nOfSlackG + nOfSlackUB + sv, 1);//slack variable position
//				standardA.set(nOfSlackG + nOfSlackUB + sv, standardS + i, -1);
//				standardB.setQuick(nOfSlackG + nOfSlackUB + sv, -lb.getQuick(i));
//			}
			
			//filling with original A values
			for(int i=0; originalA!=null && i<originalA.rows(); i++){
				for(int j=0; j<originalN; j++){
					standardA.set(standardS + i, standardS + j, originalA.get(i, j));//@TODO: implementation for sparse G
				}
				standardB.setQuick(standardS + i, originalB.getQuick(i));
			}
			
			//filling for splitted variables
			cntSlackLB = 0;
			cntSlackUB = 0;
			int previousSplittedVariables = 0;
			for(int sv=0; sv<nOfSplittedVariables; sv++ ){
				int i = splittedVariablesList.get(sv);
				//variable i was splitted
				for(int r=0; r<nOfSlackG; r++){//n of rows of G
					standardA.set(r, standardS + originalN + sv, -originalG.getQuick(r, i));
				}
				
				if(lbSlack[i]){
					for(int k=previousSplittedVariables; k<i; k++){
						if(lbSlack[k]){
							//for this lb we have a row (and a slack variable) in the standard A
							cntSlackLB++;
						}
					}
					standardA.set(nOfSlackG + cntSlackLB, standardS + originalN + sv, 1);//lower bound
				}
				
				if(ubSlack[i]){
					for(int k=previousSplittedVariables; k<i; k++){
						if(ubSlack[k]){
							//for this ub we have a row (and a slack variable) in the standard A
							cntSlackUB++;
						}
					}
					standardA.set(nOfSlackG + nOfSlackLB + cntSlackUB, standardS + originalN + sv, -1);//upper bound
				}
				
				previousSplittedVariables = i;
				
				//standardA.set(nOfSlackG + nOfSlackUB + pos-1, standardS + originalN + sv, 1);//lower bound
				for(int r=0; originalA!=null && r<originalA.rows(); r++){
					standardA.set(standardS + r, standardS + originalN + sv, -originalA.get(r, i));
				}
			}
			
			standardC = F1.make(standardN);
			standardLB = F1.make(standardN);
			standardUB = F1.make(standardN, unboundedUBValue);//the slacks are upper unbounded
			for(int i=0; i<originalN; i++){
				standardC.setQuick(standardS + i, originalC.getQuick(i));
			}
			for(int i=0; i<standardS; i++){
				standardLB.setQuick(i, 0.);
			}
			for(int i=0; i<originalN; i++){
				standardLB.setQuick(standardS + i, originalLB.getQuick(i));
			}
			for(int i=0; i<originalN; i++){
				standardUB.setQuick(standardS + i, originalUB.getQuick(i));
			}
		}
		
		if(strictlyStandardForm){
			standardLB = F1.make(standardN, 0.);//brand new lb
			standardUB = null;//no ub for the strictly standard form 
		}
	}
	
	/**
	 * Get back the vector in the original components.
	 * @param X vector in the standard variables
	 * @return the original component
	 */
	public double[] postConvert(double[] X) {
		if (X.length != standardN) {
			throw new IllegalArgumentException("wrong array dimension: " + X.length);
		}
		double[] ret = new double[originalN];
		int cntSplitted = 0;
		for (int i = standardS; i < standardN; i++) {
			if (splittedVariablesList.contains(i - standardS)) {
				// this variable was splitted: x = xPlus-xMinus
				ret[i - standardS] = X[i] - X[standardN + cntSplitted];
				cntSplitted++;
			} else {
				ret[i - standardS] = X[i];
			}
		}
		// this.postconvertedX = ret;
		return ret;
	}
	
	/**
	 * Express a vector in the original variables in the final standard variable form
	 * @param x vector in the original variables
	 * @return vector in the standard variables
	 */
	public double[] getStandardComponents(double[] x){
		if(x.length != originalN){
			throw new IllegalArgumentException("wrong array dimension: " + x.length);
		}
		double[] ret = new double[standardN];
		for(int i=0; i<x.length; i++){
			if (splittedVariablesList.contains(i)) {
				// this variable was splitted: x = xPlus-xMinus
				if(x[i] >= 0){
					//value for xPlus
					ret[standardS + i] = x[i];
				}else{
					int pos = -1;
					for(int k=0; k<splittedVariablesList.size(); k++){
						if(splittedVariablesList.get(k)==i){
							pos = k;
							break;
						}
					}
					//value for xMinus
					ret[standardS + x.length + pos] = -x[i];
				}
			}else{
				ret[standardS + i] = x[i];
			}
		}
		if(standardS>0){
			DoubleMatrix1D residuals = ColtUtils.zMult(standardA, F1.make(ret), standardB, -1);
			for(int i=0; i<standardS; i++){
				ret[i] = -residuals.get(i) + ret[i];
			}
		}
		return ret;
	}
	
	public int getOriginalN() {
		return originalN;
	}
	
	public int getStandardN() {
		return standardN;
	}
	
	public int getStandardS() {
		return standardS;
	}
	
	public DoubleMatrix1D getStandardC() {
		return standardC;
	}
	
	public DoubleMatrix2D getStandardA() {
		return standardA;
	}

	public DoubleMatrix1D getStandardB() {
		return standardB;
	}

	/**
	 * This makes sense only if strictlyStandardForm = false (otherwise all lb are 0).
	 */
	public DoubleMatrix1D getStandardLB() {
		return standardLB;
	}

	/**
	 * This makes sense only if strictlyStandardForm = false (otherwise all ub are unbounded).
	 */
	public DoubleMatrix1D getStandardUB() {
		return standardUB;
	}

//	public double[] getPostconvertedX() {
//		return postconvertedX;
//	}
	
	public boolean isLbUnbounded(Double lb){
		return Double.compare(unboundedLBValue, lb)==0;
	}
	
	public boolean isUbUnbounded(Double ub){
		return Double.compare(unboundedUBValue, ub)==0;
	}
	
	public double getUnboundedLBValue() {
		return unboundedLBValue;
	}
	
	public double getUnboundedUBValue() {
		return unboundedUBValue;
	}
}
