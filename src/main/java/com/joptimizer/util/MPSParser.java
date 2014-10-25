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

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.net.URI;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;

/**
 * Mathematical Programming System (MPS) Format parser.
 * The output of the parsing is driven by the four fields:
 * <ol>
 *  <li>unboundedLBValue: the distinctive value of a lower bound if the mps file states that it is unbounded,
 *  that is, if the lower bound of a variables is said to be unbounded, it is assigned this value.
 *  Must be one of the values:
 *  <ol>
 *   <li>Double.NaN (the default)</li>
 *   <li>Double.NEGATIVE_INFINITY</li>
 *  </ol>
 *  </li>
 *  <li>unboundedLBValue: the distinctive value of an upper bound if the mps file states that it is unbounded,
 *  that is, if the upper bound of a variables is said to be unbounded, it is assigned this value.
 *  Must be one of the values:
 *  <ol>
 *   <li>Double.NaN (the default)</li>
 *   <li>Double.POSITIVE_INFINITY</li>
 *  </ol>
 *  <li>unspecifiedLBValue: the value of a lower bounds if the mps file does not specify it,
 *  that is, if the lower bound of a variables is not explicit, it is assigned this value.
 *  For example it can be 0 (as is the usual default) or other</li>
 *  <li>unspecifiedUBValue: the value of an upper bounds if the mps file does not specify it,
 *  that is, if the upper bound of a variables is not explicit, it is assigned this value.  
 *  For example it can be the unbounded upper bound value (as is the usual default) or other</li>
 * </ol>
 * 
 * @see "http://en.wikipedia.org/wiki/MPS_%28format%29"
 * @see "http://lpsolve.sourceforge.net/5.0/mps-format.htm"
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 */
public class MPSParser {
	private Log log = LogFactory.getLog(this.getClass().getName());
	private boolean useSparsity = true;
	private int section;
	private String name;
	private static final String NAME = "NAME";
	private static final String ROWS = "ROWS";
	private static final String QSECTION = "QSECTION";
	private static final String COLUMNS = "COLUMNS";
	private static final String RHS = "RHS";
	private static final String RANGES = "RANGES";
	private static final String BOUNDS = "BOUNDS";
	private static final String ENDATA = "ENDATA";
	private static final int SECTION_NAME = 0;
	private static final int SECTION_ROWS = 1;
	private static final int SECTION_COLUMNS = 2;
	private static final int SECTION_RHS = 3;
	private static final int SECTION_RANGES = 4;
	private static final int SECTION_BOUNDS = 5;

	public static final double DEFAULT_UNBOUNDED_LOWER_BOUND = Double.NaN;
	public static final double DEFAULT_UNBOUNDED_UPPER_BOUND = Double.NaN;
	public static final double DEFAULT_UNSPECIFIED_LOWER_BOUND = 0.;
	public static final double DEFAULT_UNSPECIFIED_UPPER_BOUND = DEFAULT_UNBOUNDED_UPPER_BOUND;
	
	private final String OBJECTIVE = "N";//objective function
	private final String LESS_THEN = "L";//less-then
	private final String GREATER_THEN = "G";//greater-then
	private final String EQUAL = "E";//equals
	private final String LOWER_BOUND = "LO";//lower bound
	private final String UPPER_BOUND = "UP";//upper bound
	private final String FX_BOUND = "FX";//not-ignorable bound
	private final String FR_BOUND = "FR";//ignorable bound
	private final String PL_BOUND = "PL";//infinite upper bound   (0 <=) x < +oo
	private final String MI_BOUND = "MI";//infinite lower bound   -oo < x (<= 0)
	
	private double unspecifiedLBValue = DEFAULT_UNSPECIFIED_LOWER_BOUND;
	private double unspecifiedUBValue = DEFAULT_UNSPECIFIED_UPPER_BOUND;
	private double unboundedLBValue = DEFAULT_UNBOUNDED_LOWER_BOUND;
	private double unboundedUBValue = DEFAULT_UNBOUNDED_UPPER_BOUND;
	private Map<String, Integer> columnsIndexMap;//the name of the variables (columns) and their index
	private int n;//number of variables
	private int mieq;//number of inequalities
	private int meq;//number of equalities
	private int nzG;//number of non zero elements in G
	private int nzA;//number of non zero elements in A
	private DoubleMatrix1D c;//objective function
	private DoubleMatrix2D G;//inequalities constraints coefficients
	private DoubleMatrix1D h;//inequalities constraints limits
	private DoubleMatrix2D A;//equalities constraints coefficients
	private DoubleMatrix1D b;//equalities constraints limits
	private DoubleMatrix1D lb;//lower bounds
	private DoubleMatrix1D ub;//upper bounds
	protected Algebra ALG = Algebra.DEFAULT;
	protected DoubleFactory1D F1 = DoubleFactory1D.dense;
	protected DoubleFactory2D F2 = DoubleFactory2D.dense;
	
	public MPSParser(){
		this(DEFAULT_UNSPECIFIED_LOWER_BOUND, DEFAULT_UNSPECIFIED_UPPER_BOUND, 
				DEFAULT_UNBOUNDED_LOWER_BOUND, DEFAULT_UNBOUNDED_UPPER_BOUND);
	}
	
	public MPSParser(double unspecifiedLBValue, double unspecifiedUBValue, double unboundedLBValue, double unboundedUBValue){
		if(!Double.isNaN(unboundedLBValue) && !Double.isInfinite(unboundedLBValue) ){
			throw new IllegalArgumentException("The field unboundedLBValue must be set to Double.NaN or Double.NEGATIVE_INFINITY");
		}
		if(!Double.isNaN(unboundedUBValue) && !Double.isInfinite(unboundedUBValue) ){
			throw new IllegalArgumentException("The field unboundedUBValue must be set to Double.NaN or Double.POSITIVE_INFINITY");
		}
		this.unspecifiedLBValue = unspecifiedLBValue;
		this.unspecifiedUBValue = unspecifiedUBValue;
		this.unboundedLBValue = unboundedLBValue;
		this.unboundedUBValue = unboundedUBValue;
	}
	
	public boolean isUseSparsity() {
		return useSparsity;
	}

	public void setUseSparsity(boolean useSparsity) {
		this.useSparsity = useSparsity;
	}
	
	public void parse(String classpathFileName) throws Exception{
		parse(new File(new URI(Thread.currentThread().getContextClassLoader().getResource(classpathFileName).toString())));
	}
	
	public void parse(File file) throws Exception{
		BufferedReader in = new BufferedReader(new FileReader(file));
		parse(in);
	}

	public void parse(BufferedReader in) throws Exception{
		long t0 = System.currentTimeMillis();
		Map<String, Integer> rowsIndexMap = new HashMap<String, Integer>();//the name of the rows and its 1-based index (<0 for eq, >0 for ineq)
		Map<String, String> rowsTypeMap = new HashMap<String, String>();//the name of the rows and its type (objective|equals|less then|greater then)
		List<String> columnList = new ArrayList<String>();
		List<String> rhsList = new ArrayList<String>();
		List<String> boundsList = new ArrayList<String>();
		String previousColumnName = "NOT_YET_SET";
		columnsIndexMap = new LinkedHashMap<String, Integer>();//the name of the columns and its index
		
		//read the file and set the problem dimensions
		try {
			String line = new String();
			int lineNumber = 0;
			while ((line = in.readLine()) != null){
				lineNumber++;
				line = line.trim();
				if(log.isDebugEnabled()){
					//log.debug("line "+lineNumber+": "+line);
				}
				if(line.startsWith("#") || line.startsWith("*")){
					//this is a commented line
					continue;
				}
				
				if("".equals(line.trim())){
					//this is an empty line
					continue;
				}
				
				if(line.startsWith(NAME)){
					section = SECTION_NAME;
				}else if(line.startsWith(QSECTION)){
					log.error("Quadratic problems parsing not supported");
					throw new RuntimeException("Quadratic problems parsing not supported");
				}else if(line.startsWith(ROWS)){
					section = SECTION_ROWS;
					continue;
				}else if(line.startsWith(COLUMNS)){
					section = SECTION_COLUMNS;
					continue;
				}else if(line.startsWith(RHS)){
					if(section != SECTION_RHS ){
						//the line of this section has again this starting string
						section = SECTION_RHS;
						continue;
					}
				}else if(line.startsWith(RANGES)){
					log.error("Ranges are not supported");
					throw new RuntimeException("Ranges are not supported");
				}else if(line.startsWith(BOUNDS)){
					section = SECTION_BOUNDS;
					continue;
				}else if(line.startsWith(ENDATA)){
					break;
				}
				
				switch(section){
				case SECTION_NAME:
					name = line.substring(4).trim();
					log.info("name: " + name);
					break;
				case SECTION_ROWS:
					String rowType = line.substring(0, 1);
					String rowName = line.substring(1).trim();
					if(rowType.equalsIgnoreCase(EQUAL)){
						meq++;
						rowsIndexMap.put(rowName, -meq);
						rowsTypeMap.put(rowName, EQUAL);
					}else if(rowType.equalsIgnoreCase(LESS_THEN)){
						mieq++;
						rowsIndexMap.put(rowName, mieq);
						rowsTypeMap.put(rowName, LESS_THEN);
					}else if(rowType.equalsIgnoreCase(GREATER_THEN)){
						mieq++;
						rowsIndexMap.put(rowName, mieq);
						rowsTypeMap.put(rowName, GREATER_THEN);
					}else if(rowType.equalsIgnoreCase(OBJECTIVE)){
						rowsTypeMap.put(rowName, OBJECTIVE);
					}
					
					break;
				case SECTION_COLUMNS:
					String columnName = line.substring(0, line.indexOf(" "));
					if(!previousColumnName.equalsIgnoreCase(columnName)){
						n++;
						previousColumnName = columnName;
					}
					columnList.add(columnList.size(), line);
					break;
				case SECTION_RHS:
					rhsList.add(rhsList.size(), line);
					break;
				case SECTION_BOUNDS:
					boundsList.add(boundsList.size(), line);
					break;
				}
				
			}
			in.close();
			
			//define vectors and matrices
			c = F1.make(n);
			//A = new double[meq][n];
			A = (useSparsity)? new SparseDoubleMatrix2D(meq, n) : F2.make(meq, n);
			b = F1.make(meq);
			//G = new double[mieq][n];
			G = (useSparsity)? new SparseDoubleMatrix2D(mieq, n) : F2.make(mieq, n);
			h = F1.make(mieq);
			lb = F1.make(n, unspecifiedLBValue);
			ub = F1.make(n, unspecifiedUBValue);
			//Arrays.fill(lb, unspecifiedLBValue);
			//Arrays.fill(ub, unspecifiedUBValue);
			
			//look into the lines
			previousColumnName = "NOT_YET_SET";
			int colIndex = -1;
			for(int i=0; i<columnList.size(); i++){
				String myline = columnList.get(i);
				StringTokenizer cst = new StringTokenizer(myline, " ");
				int cmax = cst.countTokens();
				String rowType = null;
				for(int j=0; j<cmax; j++){
					String v = cst.nextToken().trim();
					switch(j){
						case 0:
							//v is the column (or variable) name
							if(!previousColumnName.equalsIgnoreCase(v)){
								colIndex++;
								columnsIndexMap.put(v, colIndex);
								previousColumnName = v;
							}
							break;
						case 1:
							//v is the constraint name
							rowType = rowsTypeMap.get(v);
							double d1 = new Double(cst.nextToken());
							if(rowType.equalsIgnoreCase(EQUAL)){
								int rowIndex = -rowsIndexMap.get(v);
								A.setQuick(rowIndex-1, colIndex, d1);
								nzA++;
							}else if(rowType.equalsIgnoreCase(LESS_THEN)){
								int rowIndex = rowsIndexMap.get(v);
								G.setQuick(rowIndex-1, colIndex, d1);
								nzG++;
							}else if(rowType.equalsIgnoreCase(GREATER_THEN)){
								int rowIndex = rowsIndexMap.get(v);
								G.setQuick(rowIndex-1, colIndex, -d1);
								nzG++;
							}else if(rowType.equalsIgnoreCase(OBJECTIVE)){
								c.setQuick(colIndex, d1);
							}
							j++;
							break;
						case 3:
							//v is the constraint name
							rowType = rowsTypeMap.get(v);
							double d2 = new Double(cst.nextToken());
							if(rowType.equalsIgnoreCase(EQUAL)){
								int rowIndex = -rowsIndexMap.get(v);
								A.setQuick(rowIndex-1, colIndex, d2);
								nzA++;
							}else if(rowType.equalsIgnoreCase(LESS_THEN)){
								int rowIndex = rowsIndexMap.get(v);
								G.setQuick(rowIndex-1, colIndex, d2);
								nzG++;
							}else if(rowType.equalsIgnoreCase(GREATER_THEN)){
								int rowIndex = rowsIndexMap.get(v);
								G.setQuick(rowIndex-1, colIndex, -d2);
								nzG++;
							}else if(rowType.equalsIgnoreCase(OBJECTIVE)){
								c.setQuick(colIndex, d2);
							}
							j++;
							break;
					}
				}
			}
			for(int i=0; i<rhsList.size(); i++){
				String myline = rhsList.get(i);
				StringTokenizer rst = new StringTokenizer(myline, " ");
				int rmax = rst.countTokens();
				if(rmax==5 || rmax==3){
					rst.nextToken();//it is the RHS name
					rmax--;
				}
				for(int j=0; j<rmax; j++){
					String v = rst.nextToken().trim();
					switch(j){
						case 0:
							//v is the constraint name
							Double d1 = new Double(rst.nextToken());
							Integer indexObj = rowsIndexMap.get(v);
							if(indexObj==null){
								log.warn("unknown constraint " + v);
								j = rmax;
								break;
							}
							int rowIndex = indexObj.intValue();
							String rowType = rowsTypeMap.get(v);
							if(rowIndex<0){
								//equality
								b.setQuick(-rowIndex-1, d1);
							}else{
								//inequality
								if(LESS_THEN.equals(rowType)){
									h.setQuick(rowIndex-1, d1);
								}else if(GREATER_THEN.equals(rowType)){
									h.setQuick(rowIndex-1, -d1);
								}
							}
							j++;
							break;
						case 2:
							//v is the constraint name
							double d2 = new Double(rst.nextToken());
							int rowIndex2 = rowsIndexMap.get(v);
							String rowType2 = rowsTypeMap.get(v);
							if(rowIndex2<0){
								//equality
								b.setQuick(-rowIndex2-1, d2);
							}else{
								//inequality
								if(LESS_THEN.equals(rowType2)){
									h.setQuick(rowIndex2-1, d2);
								}else if(GREATER_THEN.equals(rowType2)){
									h.setQuick(rowIndex2-1, -d2);
								}
								
							}
							j++;
							break;
					}
				}
			}
			for(int i=0; i<boundsList.size(); i++){
				String myline = boundsList.get(i);
				StringTokenizer bst = new StringTokenizer(myline, " ");
				int nOfTokens = bst.countTokens();
				String[] tokens = new String[nOfTokens];
				for(int j=0; j<nOfTokens; j++){
					tokens[j] = bst.nextToken().trim();
				}
				String type = tokens[0];
				if(LOWER_BOUND.equals(type)){
					String columnName = (nOfTokens>3)? tokens[2] : tokens[1];
					int columnIndex = columnsIndexMap.get(columnName);
					lb.setQuick(columnIndex, new Double(tokens[nOfTokens-1]).doubleValue());
				}else if(UPPER_BOUND.equals(type)){
					String columnName = (nOfTokens>3)? tokens[2] : tokens[1];
					int columnIndex = columnsIndexMap.get(columnName);
					ub.setQuick(columnIndex, new Double(tokens[nOfTokens-1]).doubleValue());
				}else if(FX_BOUND.equals(type)){
					//not-ignorable bound:  b<x<b
					String columnName = (nOfTokens>3)? tokens[2] : tokens[1];
					int columnIndex = columnsIndexMap.get(columnName);
					lb.setQuick(columnIndex, new Double(tokens[nOfTokens-1]).doubleValue());
					ub.setQuick(columnIndex, new Double(tokens[nOfTokens-1]).doubleValue());
				}else if(FR_BOUND.equals(type)){
					//ignorable bound:  -oo<x<+oo
					String columnName = tokens[2];
					int columnIndex = columnsIndexMap.get(columnName);
					lb.setQuick(columnIndex, unboundedLBValue);
					ub.setQuick(columnIndex, unboundedUBValue);
				}else if(MI_BOUND.equals(type)){
					//infinite lower bound   -oo < x (<= 0)
					String columnName = tokens[2];
					int columnIndex = columnsIndexMap.get(columnName);
					lb.setQuick(columnIndex, unboundedLBValue);
					ub.setQuick(columnIndex, 0);
				}else if(PL_BOUND.equals(type)){
					//infinite lower bound  (0 <=) x < +oo
					String columnName = tokens[2];
					int columnIndex = columnsIndexMap.get(columnName);
					lb.setQuick(columnIndex, 0);
					ub.setQuick(columnIndex, unboundedUBValue);
				}else{
					log.warn("unknown bound type: " + type);
				}
			}
			
			//log.debug("Variables: " + columnsIndexMap.keySet());
			
		} catch (Exception e) {
			log.error("Exception", e);
			throw e;
		}
		log.info("parsing time: " + (System.currentTimeMillis()-t0) + " ms");
	}
	
	public String getName() {
		return name;
	}
	
	public int getN() {
		return n;
	}
	
	public List<String> getVariablesNames(){
		return new ArrayList<String>(this.columnsIndexMap.keySet());
	}

	public int getMieq() {
		return mieq;
	}

	public int getMeq() {
		return meq;
	}
	
	public int getNzG() {
		return nzG;
	}

	public int getNzA() {
		return nzA;
	}
	
	public DoubleMatrix1D getC() {
		return c;
	}

	public DoubleMatrix2D getG() {
		return G;
	}

	public DoubleMatrix1D getH() {
		return h;
	}

	public DoubleMatrix2D getA() {
		return A;
	}

	public DoubleMatrix1D getB() {
		return b;
	}

	public DoubleMatrix1D getLb() {
		return lb;
	}

	public DoubleMatrix1D getUb() {
		return ub;
	}
	
	protected boolean isLbUnbounded(Double lb){
		return Double.compare(unboundedLBValue, lb)==0;
	}
	
	protected boolean isUbUnbounded(Double ub){
		return Double.compare(unboundedUBValue, ub)==0;
	}
}
