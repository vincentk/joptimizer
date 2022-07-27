package com.joptimizer.util;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.List;
import java.util.Locale;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;

public final class TestUtils {

	
	public static final void writeDoubleArrayToFile(double[] v, String fileName) throws Exception {
		DecimalFormat df = (DecimalFormat) NumberFormat.getInstance(Locale.US);
		df.applyPattern("#");
		df.setMaximumFractionDigits(16);
		String[][] ret = new String[v.length][1];
		for(int j=0; j<v.length; j++){
			if(Double.isNaN(v[j])){
				ret[j][0] = String.valueOf(v[j]);
			}else{
				ret[j][0] = df.format(v[j]);
				//ret[j][0] = String.valueOf(v[j]);
			}
		}
		CSVPrinter csvPrinter = new CSVPrinter(new FileWriter(fileName), CSVFormat.DEFAULT.withDelimiter(','));
		try{
			csvPrinter.printRecords(ret);
		}finally{
			csvPrinter.close();
		}
	}
	
	public static final void writeDoubleMatrixToFile(double[][] m, String fileName) throws Exception {
		DecimalFormat df = (DecimalFormat) NumberFormat.getInstance(Locale.US);
		df.applyPattern("#");
		df.setMaximumFractionDigits(16);
		String[][] ret = new String[m.length][];
		for(int i=0; i<m.length; i++){
			double[] MI = m[i];
			String[] retI = new String[MI.length];
			for(int j=0; j<MI.length; j++){
				if(Double.isNaN(MI[j])){
					retI[j] = String.valueOf(MI[j]);
				}else{
					retI[j] = df.format(MI[j]);
					//retI[j] = String.valueOf(MI[j]);
				}
			}
			ret[i] = retI;
		}
		CSVPrinter csvPrinter = new CSVPrinter(new FileWriter(fileName), CSVFormat.DEFAULT.withDelimiter(' '));
		try{
			csvPrinter.printRecords(ret);
		}finally{
			csvPrinter.close();
		}
	}
	
	public static final double[] loadDoubleArrayFromFile(String classpathFileName) throws Exception {
		//FileReader fr = new FileReader(classpathFileName);
		InputStream is = Thread.currentThread().getContextClassLoader().getResourceAsStream(classpathFileName);
		CSVParser parser = new CSVParser(new InputStreamReader(is), CSVFormat.DEFAULT.withDelimiter(',').withCommentMarker('#'));
		List<CSVRecord> records = parser.getRecords();
		double[] v = new double[records.size()];
		for(int i=0; i<records.size(); i++){
			v[i] = Double.parseDouble(records.get(i).get(0));
		}
		return v;
	}
	
	public static final double[][] loadDoubleMatrixFromFile(String classpathFileName, char fieldSeparator) throws Exception {
		InputStream is = Thread.currentThread().getContextClassLoader().getResourceAsStream(classpathFileName);
		CSVParser parser = new CSVParser(new InputStreamReader(is), CSVFormat.DEFAULT.withDelimiter(fieldSeparator).withCommentMarker('#'));
		List<CSVRecord> records = parser.getRecords();
		double[][] m = new double[records.size()][records.get(0).size()];
		for(int i=0; i<records.size(); i++){
			for(int j=0; j<records.get(0).size(); j++){
				m[i][j] = Double.parseDouble(records.get(i).get(j));
			}
		}
		return m;
	}
	
	public static final double[][] loadDoubleMatrixFromFile(String classpathFileName) throws Exception {
		return loadDoubleMatrixFromFile(classpathFileName, ",".charAt(0));
	}
	
}
