#!/usr/bin/env python
# Copyright 2017 Alessio Sclocco <a.sclocco@esciencecenter.nl>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions to compute the tuning difficulty score."""

import statistical_analysis

alpha = 0.5
beta = 1.0 - alpha

def difficulty(db_queue, table, benchmark, scenario, peak):
    """Computes the tuning difficulty for the given scenario."""
    last_percentile = statistical_analysis.distribution(db_queue, table, benchmark, scenario)[9]
    metrics = ""
    if benchmark.lower() == "triad":
        metrics = "GBs,"
    elif benchmark.lower() == "reduction":
        metrics = "GBs,"
    elif benchmark.lower() == "stencil":
        metrics = "GFLOPs,"
    elif benchmark.lower() == "md":
        metrics = "GFLOPs,"
    elif benchmark.lower() == "correlator":
        metrics = "GFLOPs,"
    db_queue.execute("SELECT COUNT(id),MIN(" + metrics.rstrip(",") + "),MAX(" + metrics.rstrip(",") + ") FROM " + table + " WHERE " + scenario)
    results = db_queue.fetchall()
    total_confs = results[0][0]
    minimum = results[0][1]
    maximum = results[0][2]
    if last_percentile >= 0.1:
        distribution_component = (0.5 / 0.9) * last_percentile + 1.0 - (0.5 / 0.9)
    else:
        distribution_component = (0.5 / 0.1) * last_percentile
    return (alpha * distribution_component) + beta - ((beta * (maximum - minimum)) / float(peak))