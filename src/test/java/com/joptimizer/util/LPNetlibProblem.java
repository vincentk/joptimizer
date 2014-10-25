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

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;


/**
 * Represent a Netlib LP problem as contained in the bundle 
 * http://www.math.ufl.edu/~hager/LPTest/MPS/MPS.tar.gz.
 * 
 * @see http://www.math.ufl.edu/~hager/LPTest/MPS/MPS.tar.gz
 * @see http://www.netlib.org/lp/data/readme
 * @author alberto trivellato (<a href="mailto:alberto.trivellato@gmail.com">alberto.trivellato@gmail.com</a>)
 */
public class LPNetlibProblem {

	public String name;
	public int nz;
	public int rows;
	public int columns;
	public double optimalValue;
	
	public static Map<String, LPNetlibProblem> loadAllProblems() throws Exception{
		File f = Utils.getClasspathResourceAsFile("lp" + File.separator	+ "netlib" + File.separator + "problemsList.csv");
		DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(f)));
		Map<String, LPNetlibProblem> problemsMap = new HashMap<String, LPNetlibProblem>();
		String line = new String();
		//Name;NZ;Rows;Columns;Solution
		while ((line = in.readLine()) != null){
			if(line.startsWith("#")){
				//this is a comment
				continue;
			}
			LPNetlibProblem problem = new LPNetlibProblem();
			StringTokenizer st = new StringTokenizer(line, ";");
			problem.name = st.nextToken();
			//System.out.println(result.name);
			problem.nz = Integer.parseInt(st.nextToken());
			String strRows = st.nextToken();
			problem.rows = ("".equals(strRows))? 0 : Integer.parseInt(strRows);
			String strCols = st.nextToken();
			problem.columns = ("".equals(strCols))? 0 : Integer.parseInt(strCols);
			problem.optimalValue = Double.parseDouble(st.nextToken()); 
			problemsMap.put(problem.name, problem);
		}
		return problemsMap;
	}
	
	@Override
	public String toString(){
		return "name: " + name + ", nz: " + nz + ", rows: " + rows + ", cols: " + columns + ", optimalValue: " +optimalValue;
	}
	
	public static List<LPNetlibProblem> getProblemsOrderedByName(Map<String, LPNetlibProblem> resultsMap){
		List<LPNetlibProblem> orderedList = new ArrayList<LPNetlibProblem>();
		orderedList.addAll(resultsMap.values());
		Collections.sort(orderedList, new Comparator<LPNetlibProblem>() {
			public int compare(LPNetlibProblem arg0, LPNetlibProblem arg1) {
				return arg0.name.compareTo(arg1.name);
			}
		});
		return orderedList;
	}
	
	public static List<LPNetlibProblem> getProblemsOrderedByNumberOfRows(Map<String, LPNetlibProblem> resultsMap){
		List<LPNetlibProblem> orderedList = new ArrayList<LPNetlibProblem>();
		orderedList.addAll(resultsMap.values());
		Collections.sort(orderedList, new Comparator<LPNetlibProblem>() {
			public int compare(LPNetlibProblem arg0, LPNetlibProblem arg1) {
				if(arg0.rows  > arg1.rows){
					return 1;
				}else if(arg0.rows  < arg1.rows){
					return -1;
				}else{
					return 0;
				}
			}
		});
		return orderedList;
	}
	
	public static List<LPNetlibProblem> getProblemsOrderedByNumberOfColumns(Map<String, LPNetlibProblem> resultsMap){
		List<LPNetlibProblem> orderedList = new ArrayList<LPNetlibProblem>();
		orderedList.addAll(resultsMap.values());
		Collections.sort(orderedList, new Comparator<LPNetlibProblem>() {
			public int compare(LPNetlibProblem arg0, LPNetlibProblem arg1) {
				if(arg0.columns  > arg1.columns){
					return 1;
				}else if(arg0.columns  < arg1.columns){
					return -1;
				}else{
					return 0;
				}
			}
		});
		return orderedList;
	}
	
	public static List<LPNetlibProblem> getProblemsOrderedBySize(Map<String, LPNetlibProblem> resultsMap){
		List<LPNetlibProblem> orderedList = new ArrayList<LPNetlibProblem>();
		orderedList.addAll(resultsMap.values());
		Collections.sort(orderedList, new Comparator<LPNetlibProblem>() {
			public int compare(LPNetlibProblem arg0, LPNetlibProblem arg1) {
				if(arg0.rows*arg0.columns  > arg1.rows*arg1.columns){
					return 1;
				}else if(arg0.rows*arg0.columns  < arg1.rows*arg1.columns){
					return -1;
				}else{
					return 0;
				}
			}
		});
		return orderedList;
	}
}
