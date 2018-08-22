#!/usr/bin/env python
# Copyright 2018  Alessio Sclocco <a.sclocco@esciencecenter.nl>
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
"""Functions to analyze the performance space of the applications."""

def performance_statistics(db_queue, table, benchmark, scenario):
    """Returns statistics about the performance space."""
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
    db_queue.execute("SELECT COUNT(id),MIN(" + metrics + "),AVG(" + metrics + "),MAX(" + metrics + ") FROM " + table + " WHERE " + scenario)
    items = db_queue.fetchall()
    print(items[0])
        