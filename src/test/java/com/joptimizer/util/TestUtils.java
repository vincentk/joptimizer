package com.joptimizer.util;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.text.DecimalFormat;

import com.Ostermiller.util.CSVParser;
import com.Ostermiller.util.CSVPrinter;

public final class TestUtils {

	
	public static final void writeDoubleArrayToFile(double[] v, String fileName) throws Exception {
		CSVPrinter csvPrinter = new CSVPrinter(new FileOutputStream(new File(fileName)));
		DecimalFormat df = new DecimalFormat("#");
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
		csvPrinter.println(ret);
	}
	
	public static final void writeDoubleMatrixToFile(double[][] m, String fileName) throws Exception {
		CSVPrinter csvPrinter = new CSVPrinter(new FileOutputStream(new File(fileName)));
		DecimalFormat df = new DecimalFormat("#");
        df.setMaximumFractionDigits(16);
		csvPrinter.changeDelimiter(" ".charAt(0));
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
		csvPrinter.println(ret);
	}
	
	public static final double[] loadDoubleArrayFromFile(String classpathFileName) throws Exception {
		//FileReader fr = new FileReader(classpathFileName);
		InputStream is = Thread.currentThread().getContextClassLoader().getResourceAsStream(classpathFileName);
		CSVParser parser = new CSVParser(is, ',');
		parser.setCommentStart("#");
		String[][] mapMatrix = parser.getAllValues();
		double[] v = new double[mapMatrix.length];
		for(int i=0; i<mapMatrix.length; i++){
			v[i] = Double.parseDouble(mapMatrix[i][0]);
		}
		return v;
	}
	
	public static final double[][] loadDoubleMatrixFromFile(String classpathFileName, char fieldSeparator) throws Exception {
		InputStream is = Thread.currentThread().getContextClassLoader().getResourceAsStream(classpathFileName);
		CSVParser parser = new CSVParser(is, fieldSeparator);
		parser.setCommentStart("#");
		String[][] mapMatrix = parser.getAllValues();
		double[][] m = new double[mapMatrix.length][mapMatrix[0].length];
		for(int i=0; i<mapMatrix.length; i++){
			for(int j=0; j<mapMatrix[0].length; j++){
				m[i][j] = Double.parseDouble(mapMatrix[i][j]);
			}
		}
		return m;
	}
	
	public static final double[][] loadDoubleMatrixFromFile(String classpathFileName) throws Exception {
		return loadDoubleMatrixFromFile(classpathFileName, ",".charAt(0));
	}
	
}
