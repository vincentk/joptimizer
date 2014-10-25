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
package com.joptimizer.functions;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * Generalized logarithm barrier function for SOCP with constraints: 
 * <br>||Ai.x+bi|| < ci.x+di, i=1,...,m
 *
 * @see "S.Boyd and L.Vandenberghe, Convex Optimization, p. 600"
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class SOCPLogarithmicBarrier implements BarrierFunction {

	private List<SOCPConstraintParameters> socpConstraintParametersList = null;
	private int dim = -1;

	public SOCPLogarithmicBarrier(List<SOCPConstraintParameters> socpConstraintParametersList, int dim) {
		this.socpConstraintParametersList = socpConstraintParametersList;
		this.dim = dim;
	}

	public double value(double[] X) {
		RealVector x = new ArrayRealVector(X);
		
		double ret = 0;
		for(int i=0; i<socpConstraintParametersList.size(); i++){
			SOCPConstraintParameters param = socpConstraintParametersList.get(i);
			double t = this.buildT(param, x);
			if(t < 0){
				return Double.NaN;
			}
			RealVector u = this.buildU(param, x);
			double t2uu = t*t - u.dotProduct(u);
			if(t2uu <= 0){
				return Double.NaN;
			}
			double ret_i = -Math.log(t2uu);
			ret += ret_i;
		}
		
		return ret;
	}

	public double[] gradient(double[] X) {
		RealVector x = new ArrayRealVector(X);
		
		RealVector ret = new ArrayRealVector(dim);
		for(int i=0; i<socpConstraintParametersList.size(); i++){
			SOCPConstraintParameters param = socpConstraintParametersList.get(i);
			double t = this.buildT(param, x);
			RealVector u = this.buildU(param, x);
			double t2uu = t*t - u.dotProduct(u);
			RealMatrix Jacob = this.buildJ(param, x);
			int k = u.getDimension();
			RealVector G = new ArrayRealVector(k+1);
			G.setSubVector(0, u);
			G.setEntry(k, -t);
			RealVector ret_i = Jacob.operate(G).mapMultiply((2./t2uu));
			ret = ret.add(ret_i);
		}
		
		return ret.toArray();
	}
	
	public double[][] hessian(double[] X) {
		RealVector x = new ArrayRealVector(X);
		
		RealMatrix ret = new Array2DRowRealMatrix(dim, dim);
		for(int i=0; i<socpConstraintParametersList.size(); i++){
			SOCPConstraintParameters param = socpConstraintParametersList.get(i);
			double t = this.buildT(param, x);
			RealVector u = this.buildU(param, x);
			double t2uu = t*t - u.dotProduct(u);
			RealVector t2u = u.mapMultiply(-2*t);
			RealMatrix Jacob = this.buildJ(param, x);
			int k = u.getDimension();
			RealMatrix H = new Array2DRowRealMatrix(k+1, k+1);
			RealMatrix ID = MatrixUtils.createRealIdentityMatrix(k);
			H.setSubMatrix(ID.scalarMultiply(t2uu).add(u.outerProduct(u).scalarMultiply(2)).getData(), 0, 0);
			H.setSubMatrix(new double[][]{t2u.toArray()}, k, 0);
			for(int j=0; j<k; j++){
				H.setEntry(j, k, t2u.getEntry(j));
			}
			H.setEntry(k, k, t*t+u.dotProduct(u));
			RealMatrix ret_i = Jacob.multiply(H).multiply(Jacob.transpose()).scalarMultiply(2./Math.pow(t2uu, 2));
			ret = ret.add(ret_i);
		}
		
		return ret.getData();
	}
	
	/**
	 * Create the barrier function for the Phase I.
	 * It is an instance of this class for the constraints: 
	 * <br>||Ai.x+bi|| < ci.x+di+t, i=1,...,m
	 * @see "S.Boyd and L.Vandenberghe, Convex Optimization, 11.6.2"
	 */
	public BarrierFunction createPhase1BarrierFunction(){
		
		final int dimPh1 = dim +1;
		List<SOCPConstraintParameters> socpConstraintParametersPh1List = new ArrayList<SOCPConstraintParameters>();
 		SOCPLogarithmicBarrier bfPh1 = new SOCPLogarithmicBarrier(socpConstraintParametersPh1List, dimPh1);
 		
 		for(int i=0; i<socpConstraintParametersList.size(); i++){
 			SOCPConstraintParameters param = socpConstraintParametersList.get(i);
 			RealMatrix A = param.getA();
 			RealVector b = param.getB();
 			RealVector c = param.getC();
 			double d = param.getD();
 			
 			RealMatrix APh1 = MatrixUtils.createRealMatrix(A.getRowDimension(), dimPh1);
 			APh1.setSubMatrix(A.getData(), 0, 0); 
			RealVector bPh1 = b;
			RealVector cPh1 = new ArrayRealVector(c.getDimension()+1);
			cPh1.setSubVector(0, c);
			cPh1.setEntry(c.getDimension(), 1);
	 		double dPh1 = d;
	 		
	 		SOCPConstraintParameters paramsPh1 = new SOCPConstraintParameters(APh1.getData(), bPh1.toArray(), cPh1.toArray(), dPh1);
	 		socpConstraintParametersPh1List.add(socpConstraintParametersPh1List.size(), paramsPh1);
		}
		
		return bfPh1;
	}
	
	/**
	 * Calculates the initial value for the s parameter in Phase I.
	 * Return s = max(||Ai.x+bi|| - (ci.x+di))
	 * @see "S.Boyd and L.Vandenberghe, Convex Optimization, 11.6.2"
	 */
	public double calculatePhase1InitialFeasiblePoint(double[] originalNotFeasiblePoint, double tolerance){
		double s = -Double.MAX_VALUE;
		RealVector x = new ArrayRealVector(originalNotFeasiblePoint); 
		for(int i=0; i<socpConstraintParametersList.size(); i++){
 			SOCPConstraintParameters param = socpConstraintParametersList.get(i);
 			RealMatrix A = param.getA();
 			RealVector b = param.getB();
 			RealVector c = param.getC();
 			double d = param.getD();
 			s = Math.max(s, (A.operate(x).subtract(b).getNorm() - (c.dotProduct(x)+d))*Math.pow(tolerance,-0.5));
		}
		if(Double.compare(0., s)==0){
			//for the point to be feasible, we must have s<0.
			//s==0 is an ambiguous value the is better to avoid
			s = 2 * tolerance;
		}
		return s;
	}
	
	public int getDim() {
		return this.dim;
	}
	
	/**
	 * 
	 * @param param SOCPConstraintParameters instance
	 * @param X evaluation point
	 * @return t = c.x + d
	 */
	private double buildT(SOCPConstraintParameters param, RealVector X){
		return param.getC().dotProduct(X) + param.getD();
	}
	
	/**
	 * 
	 * @param param SOCPConstraintParameters instance
	 * @param X evaluation point
	 * @return u = A.x + b
	 */
	private RealVector buildU(SOCPConstraintParameters param, RealVector X){
		return param.getA().operate(X).add(param.getB());
	}
	
	private RealMatrix buildJ(SOCPConstraintParameters param, RealVector X){
		RealMatrix J = new Array2DRowRealMatrix(dim, param.getA().getRowDimension()+1);
		J.setSubMatrix(param.getA().transpose().getData(), 0, 0);
		J.setColumnVector(param.getA().getRowDimension(), param.getC());
		return J;
	}
	
	/**
	 * The definition of a socp inequality constraint in the form of:
	 * <br/>||A.x+b|| < c.x+d
	 *
	 */
	public class SOCPConstraintParameters{
		private RealMatrix A = null;
		private RealVector b;
		private RealVector c;
		private double d;
		
		public SOCPConstraintParameters(double[][] AMatrix, double[] bVector, double[] cVector, double d){
			this.A = new Array2DRowRealMatrix(AMatrix);
			this.b = new ArrayRealVector(bVector);
			this.c = new ArrayRealVector(cVector);
			this.d = d;;
		}
		
		public RealMatrix getA() {
			return A;
		}
		public RealVector getB() {
			return b;
		}
		public RealVector getC() {
			return c;
		}
		public double getD() {
			return d;
		}
		
		@Override
		public String toString(){
			StringBuffer sb = new StringBuffer("SOCPConstraintParameters ||A.x+b|| < c.x + d:");
			sb.append("\nA : " + ArrayUtils.toString(A.getData()));
			sb.append("\nb : " + ArrayUtils.toString(b.toArray()));
			sb.append("\nc : " + ArrayUtils.toString(c.toArray()));
			sb.append("\nd : " + d);
			return sb.toString();
		}
	}

	public double getDualityGap(double t) {
		return ((double)this.socpConstraintParametersList.size()) / t;
	}

}
