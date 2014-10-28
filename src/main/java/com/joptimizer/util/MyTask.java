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

import java.util.concurrent.Callable;

/**
 * Parallelized task.
 * @see "http://embarcaderos.net/2011/01/23/parallel-processing-and-multi-core-utilization-with-java/"
 * @author alberto trivellato (alberto.trivellato@gmail.com)
 *
 */
public class MyTask implements Callable<Integer> {
	private int seq;

	public MyTask() {
	}

	public MyTask(int i) {
		seq = i;
	}

	@Override
	public Integer call() {
		String str = "";
		long begTest = new java.util.Date().getTime();
		System.out.println("start - Task " + seq);
		try {
			// sleep for 1 second to simulate a remote call,
			// just waiting for the call to return
			//Thread.sleep(1000);
			// loop that just concatenate a str to simulate
			// work on the result form remote call
			for (int i = 0; i < 100000000*Math.random(); i++) {
				//str = str + 't';
				for(int k=0; k<10000; k++){
					new java.util.Date().getTime();
				}
			}
		} catch (Exception e) {

		}
		Double secs = new Double((new java.util.Date().getTime() - begTest) * 0.001);
		System.out.println("run time for " +seq + ": " + secs + " secs");
		return seq;
	}
}
