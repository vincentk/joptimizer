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
package com.joptimizer;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import org.architecturerules.AbstractArchitectureRulesConfigurationTest;

public class ArchitectureRulesUnitTest extends AbstractArchitectureRulesConfigurationTest {

	private Log log = LogFactory.getLog(this.getClass().getName());

	@Override
	protected String getConfigurationFileName() {
		return "architecture-rules.xml";
	}

	@Override
	public void testArchitecture() {
		log.debug("testArchitecture");

		assertTrue(doTests());
	}
}
