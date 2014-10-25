package com.joptimizer.algebra;

import junit.framework.TestCase;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;

import cern.colt.matrix.DoubleFactory2D;

public class CholeskyRCFactorizationTest extends TestCase {

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

		CholeskyRCFactorization myc = new CholeskyRCFactorization(DoubleFactory2D.dense.make(QData));
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
}
