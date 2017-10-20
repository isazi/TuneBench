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

def difficulty(db_queue, table, benchmark, scenario):
    """Computes the tuning difficulty for the given scenario."""
    distribution = list()
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
    percentile_range = (maximum - minimum) / 10
    for percentile in range(1, 11):
        db_queue.execute("SELECT COUNT(id) FROM " + table + " WHERE (" + scenario + " AND " + metrics.rstrip(",") + " >= " + str(minimum + ((percentile - 1) * percentile_range)) + " AND " + metrics.rstrip(",") + " < " + str(minimum + (percentile * percentile_range)) + ")")
        results = db_queue.fetchall()
        distribution.append(results[0][0])
    distribution.append(distribution.pop() + 1)
    for percentile in range(0, 10):
        distribution[percentile] = (distribution[percentile] * 100) / total_confs
    return distribution
