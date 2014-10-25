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
import java.util.Arrays;
import java.util.List;

import junit.framework.TestCase;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

/**
 * MPS parsing test.
 * 
 * @see "http://en.wikipedia.org/wiki/MPS_%28format%29"
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class MPSParserTest extends TestCase {

	private Log log = LogFactory.getLog(this.getClass().getName());

	/**
	 * This is the AFIRO netlib problem.
	 */
	public void testMps1() throws Exception {
		log.debug("testMps1");
		
		String problemId = "1";
		
		File f = Utils.getClasspathResourceAsFile("lp" + File.separator	+ "mps" + File.separator + problemId + ".mps");
		double unspecifiedLBValue = 0;
		double unspecifiedUBValue = 99999;
		double unboundedLBValue = MPSParser.DEFAULT_UNBOUNDED_LOWER_BOUND;
		double unboundedUBValue = Double.POSITIVE_INFINITY;
		MPSParser p = new MPSParser(unspecifiedLBValue, unspecifiedUBValue, unboundedLBValue, unboundedUBValue);
		p.parse(f);
		
		int n = p.getN();
		int meq = p.getMeq();
		int mieq = p.getMieq();
		log.debug("name: " + p.getName());
		log.debug("n   : " + n);
		log.debug("meq : " + meq);
		log.debug("mieq: " + mieq);
		log.debug("rows: " + (meq+mieq));
		log.debug("nz: " + (p.getNzG() + p.getNzA()));
//		log.debug("c   : " + ArrayUtils.toString(p.getC().toArray()));
//		log.debug("G   : " + ArrayUtils.toString(p.getG().toArray()));
//		log.debug("h   : " + ArrayUtils.toString(p.getH().toArray()));
//		log.debug("A   : " + ArrayUtils.toString(p.getA().toArray()));
//		log.debug("b   : " + ArrayUtils.toString(p.getB().toArray()));
		log.debug("lb  : " + ArrayUtils.toString(p.getLb().toArray()));
		log.debug("ub  : " + ArrayUtils.toString(p.getUb().toArray()));
		
		assertEquals(n, 32);
		assertEquals(meq, 8);
		assertEquals(mieq, 19);
		
		//all the lower bound are not explicit in this mps model
		for(int i=0; i<n; i++){
			assertEquals(unspecifiedLBValue, p.getLb().getQuick(i));
		}
		//all the upper bound are not explicit in this mps model
		for(int i=0; i<n; i++){
			assertEquals(unspecifiedUBValue, p.getUb().getQuick(i));
		}
	}
	
	/**
	 * This is the SCORPION netlib problem.
	 */
	public void testMps2() throws Exception {
		log.debug("testMps2");
		
		String problemId = "2";
		
		File f = Utils.getClasspathResourceAsFile("lp" + File.separator	+ "mps" + File.separator + problemId + ".mps");
		MPSParser p = new MPSParser();
		p.parse(f);
		
		int n = p.getN();
		int meq = p.getMeq();
		int mieq = p.getMieq();
		log.debug("name: " + p.getName());
		log.debug("n   : " + n);
		log.debug("meq : " + meq);
		log.debug("mieq: " + mieq);
		log.debug("rows: " + (meq+mieq));
		
		assertEquals(n, 358);
		assertEquals(meq, 280);
		assertEquals(mieq, 108);
	}
	
	/**
	 * This is the PILOT4 netlib problem.
	 */
	public void testMps3() throws Exception {
		log.debug("testMps3");
		
		String problemId = "3";
		
		File f = Utils.getClasspathResourceAsFile("lp" + File.separator	+ "mps" + File.separator + problemId + ".mps");
		double unboundedLBValue = Double.NEGATIVE_INFINITY;
		double unboundedUBValue = Double.POSITIVE_INFINITY;
		double unspecifiedLBValue = 0;
		double unspecifiedUBValue = unboundedUBValue;
		MPSParser p = new MPSParser(unspecifiedLBValue, unspecifiedUBValue, unboundedLBValue, unboundedUBValue);
		p.parse(f);
		
		int n = p.getN();
		int meq = p.getMeq();
		int mieq = p.getMieq();
		log.debug("name: " + p.getName());
		log.debug("n   : " + n);
		log.debug("meq : " + meq);
		log.debug("mieq: " + mieq);
		log.debug("rows: " + (meq+mieq));
		log.debug("lb  : " + ArrayUtils.toString(p.getLb().toArray()));
		log.debug("ub  : " + ArrayUtils.toString(p.getUb().toArray()));
		
		assertEquals(n, 1000);
		assertEquals(meq, 287);
		assertEquals(mieq, 123);
		
		List<String> unboundedVariables = Arrays.asList(new String[]{"XCRO01", "XROP01", "XGAS01", "XELE01", "XAGR01", "XMNG01", "XCMP01", "XFDS01", "XPPR01", "XSCG01", "XMET01", "XTEX01", "XLUM01", "XFAP01", "XMFG01", "XTAW01", "XTRD01", "XFIN01", "XSVC01", "XTRE01", "XMAC01"});
		List<String> variablesNames = p.getVariablesNames();
		for(int i=0; i<n; i++){
			String variable = variablesNames.get(i);
			if(unboundedVariables.contains(variable)){
				//this variables are stated to be unbounded in this mps model
				assertEquals(unboundedLBValue, p.getLb().getQuick(i));
				assertEquals(unboundedUBValue, p.getUb().getQuick(i));
			}else if("PLWU01".equalsIgnoreCase(variable)){
			  //this variables has not explicit bounds in this mps model
				assertEquals(unspecifiedLBValue, p.getLb().getQuick(i));
				assertEquals(unspecifiedUBValue, p.getUb().getQuick(i));
			}
		}
	}
	
	/**
	 * This is the STOCHFOR netlib problem.
	 */
	public void testMps4() throws Exception {
		log.debug("testMps4");
		
		String problemId = "4";
		
		File f = Utils.getClasspathResourceAsFile("lp" + File.separator	+ "mps" + File.separator + problemId + ".mps");
		MPSParser p = new MPSParser();
		p.parse(f);
		
		int n = p.getN();
		int meq = p.getMeq();
		int mieq = p.getMieq();
		log.debug("name: " + p.getName());
		log.debug("n   : " + n);
		log.debug("meq : " + meq);
		log.debug("mieq: " + mieq);
		log.debug("rows: " + (meq+mieq));
		
		assertEquals(n, 111);
		assertEquals(meq, 63);
		assertEquals(mieq, 54);
	}
}
