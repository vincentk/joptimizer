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

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;



public abstract class LPOptimizationRequestHandler extends OptimizationRequestHandler{
	protected LPOptimizationRequestHandler successor = null;
	//private LPOptimizationRequest lpRequest;
	//private LPOptimizationResponse lpResponse;

	public void setLPOptimizationRequest(LPOptimizationRequest lpRequest) {
		this.request = lpRequest;
	}
//	
	protected LPOptimizationRequest getLPOptimizationRequest() {
		return (LPOptimizationRequest)this.request;
	}
	
	protected void setLPOptimizationResponse(LPOptimizationResponse lpResponse) {
		this.response = lpResponse;
	}

	public LPOptimizationResponse getLPOptimizationResponse() {
		return (LPOptimizationResponse)this.response;
	}
	
	@Override
	public void setOptimizationRequest(OptimizationRequest request) {
		if(request instanceof LPOptimizationRequest){
			super.setOptimizationRequest(request);
		}else{
			throw new UnsupportedOperationException("Use the matrix formulation with the class " +LPOptimizationRequest.class.getName()+ " for this linear problem");	
		}
	}
	
//	@Override
//	public void setOptimizationRequest(OptimizationRequest request) {
//		throw new UnsupportedOperationException("Use the matrix formulation with the class " +LPOptimizationRequest.class.getName()+ " for this linear problem");	
//	}
	
//	@Override
//	protected OptimizationRequest getOptimizationRequest() {
//		throw new UnsupportedOperationException("Use the matrix formulation for this linear problem");
//	}
	
	@Override
	protected void setOptimizationResponse(OptimizationResponse response) {
		if(response instanceof LPOptimizationResponse){
			super.setOptimizationResponse(response);
		}else{
			throw new UnsupportedOperationException("Use the matrix formulation with the class " +LPOptimizationRequest.class.getName()+ " for this linear problem");	
		}
	}

//	@Override
//	public OptimizationResponse getOptimizationResponse() {
//		throw new UnsupportedOperationException("Use the matrix formulation for this linear problem");
//	}
	
//	@Override
//	protected int forwardOptimizationRequest() throws Exception {
//		if (successor != null) {
//			successor.setLPOptimizationRequest(request);
//			int retCode = successor.optimize();
//			this.response = successor.getLPOptimizationResponse();
//			return retCode;
//		}
//		throw new Exception("Failed to solve the problem");
//	}
	
//	@Override
//	protected DoubleMatrix1D getInitialPoint(){
//		return F1.make(new double[]{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.0375, 0.01, 0.15, 0.04, 0.15, 0.04, 0.15, 0.04, 0.15, 0.04});
//		//return lpRequest.getInitialPoint();
//	}
	
//	@Override
//	protected DoubleMatrix1D getNotFeasibleInitialPoint(){
//		//return lpRequest.getNotFeasibleInitialPoint();
//		return F1.make(new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0375, 0.01, 0.15, 0.04, 0.15, 0.04, 0.15, 0.04, 0.15, 0.04});
//	}
	
//	@Override
//	protected DoubleMatrix1D getInitialLagrangian(){
//		return lpRequest.getInitialLagrangian();
//	}
	
	protected boolean isDumpProblem() {
		return getLPOptimizationRequest().isDumpProblem();
	}
	
	protected DoubleMatrix1D getC() {
		return getLPOptimizationRequest().getC();
	}
	
	protected DoubleMatrix2D getG() {
		return getLPOptimizationRequest().getG();
	}
	
	protected DoubleMatrix1D getH() {
		return getLPOptimizationRequest().getH();
	}
	
	protected DoubleMatrix1D getLb() {
		return getLPOptimizationRequest().getLb();
	}
	
	protected DoubleMatrix1D getUb() {
		return getLPOptimizationRequest().getUb();
	}
	
	protected DoubleMatrix1D getYlb() {
		return getLPOptimizationRequest().getYlb();
	}
	
	protected DoubleMatrix1D getYub() {
		return getLPOptimizationRequest().getYub();
	}
	
	protected DoubleMatrix1D getZlb() {
		return getLPOptimizationRequest().getZlb();
	}
	
	protected DoubleMatrix1D getZub() {
		return getLPOptimizationRequest().getZub();
	}

}
