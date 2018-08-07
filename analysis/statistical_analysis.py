#!/usr/bin/env python # Copyright 2016 Alessio Sclocco <a.sclocco@vu.nl>
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
"""Functions to retrieve statistical properties of the data."""

import statistics

def get_quartiles(db_queue, table, benchmark, scenario):
    """Returns the quartiles performance, rounded to integers."""
    results = list()
    metrics = ""
    if benchmark.lower() == "triad":
        metrics = "GBs"
    elif benchmark.lower() == "reduction":
        metrics = "GBs"
    elif benchmark.lower() == "stencil":
        metrics = "GFLOPs"
    elif benchmark.lower() == "md":
        metrics = "GFLOPs"
    elif benchmark.lower() == "correlator":
        metrics = "GFLOPs"
    db_queue.execute("SELECT COUNT(id) FROM " + table + " WHERE " + scenario)
    items = db_queue.fetchall()
    nr_items = items[0][0]
    if int(nr_items) == 0:
        return [-1]
    db_queue.execute("SELECT MIN(" + metrics + ") FROM " + table + " WHERE " + scenario)
    items = db_queue.fetchall()
    results.append(int(items[0][0]))
    for quartile in range(1, 4):
        db_queue.execute("SELECT " + metrics + " FROM " + table + " WHERE (" + scenario + ") ORDER BY " + metrics + " LIMIT " + str(int(nr_items / 4) * quartile) + ",1")
        items = db_queue.fetchall()
        results.append(int(items[0][0]))
    db_queue.execute("SELECT MAX(" + metrics + ") FROM " + table + " WHERE " + scenario)
    items = db_queue.fetchall()
    results.append(int(items[0][0]))
    return results

def get_histogram(db_queue, table, benchmark, scenario):
    """Returns the performance histogram of the configurations."""
    results = dict()
    metrics = ""
    if benchmark.lower() == "triad":
        metrics = "GBs"
    elif benchmark.lower() == "reduction":
        metrics = "GBs"
    elif benchmark.lower() == "stencil":
        metrics = "GFLOPs"
    elif benchmark.lower() == "md":
        metrics = "GFLOPs"
    elif benchmark.lower() == "correlator":
        metrics = "GFLOPs"
    db_queue.execute("SELECT " + metrics + " FROM " + table + " WHERE (" + scenario + ") ORDER BY " + metrics)
    items = db_queue.fetchall()
    for item in items:
        value = int(item[0])
        if value not in results:
            results[value] = 1
        else:
            results[value] = results[value] + 1
    return results

def get_tuning_variability(db_queue, tables, benchmark, scenarios):
    """Return the coefficient of variability of the optimal configurations."""
    extra = ""
    metrics = ""
    if benchmark.lower() == "triad":
        extra = "vector,"
        metrics = "GBs,"
    elif benchmark.lower() == "reduction":
        extra = "nrItemsPerBlock,vector,"
        metrics = "GBs,"
    elif benchmark.lower() == "stencil":
        extra = "localMemory,"
        metrics = "GFLOPs,"
    elif benchmark.lower() == "md":
        metrics = "GFLOPs,"
    elif benchmark.lower() == "correlator":
        extra = "sequentialTime,parallelTime,constantMemory,width,height,"
        metrics = "GFLOPs,"
    parameters = list()
    for table in tables:
        for scenario in scenarios:
            db_queue.execute("SELECT " + extra + "nrThreadsD0,nrThreadsD1,nrThreadsD2,nrItemsD0,nrItemsD1,nrItemsD2 FROM " + table + " WHERE (" + metrics.rstrip(",") + " = (SELECT MAX(" + metrics.rstrip(",") + ") FROM " + table + " WHERE (" + scenario + "))) AND (" + scenario + ")")
            best = db_queue.fetchall()
            if len(parameters) == 0:
                for item in range(0, len(best[0])):
                    parameters.append(list())
                    parameters[item].append(best[0][item])
            else:
                for item in range(0, len(best[0])):
                    parameters[item].append(best[0][item])
    results = list()
    for parameter in parameters:
        if statistics.mean(parameter) == 0:
            results.append(0.0)
        else:
            results.append(statistics.stdev(parameter) / statistics.mean(parameter))
    return results

def distribution(db_queue, table, benchmark, scenario):
    """Computes the distribution over the performance space for the given scenario."""
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