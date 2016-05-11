#!/usr/bin/env python
# Copyright 2016 Alessio Sclocco <a.sclocco@vu.nl>
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
"""TuneBench analysis script."""

import sys
import pymysql

import config
import management
import tuning
import statistics

if len(sys.argv) == 1:
    print("Supported commmands are: list, create, delete, load, tune, quartiles, histogram")
    print("Type \"" + sys.argv[0] + " <command>\" for specific help.")
    sys.exit(1)

COMMAND = sys.argv[1]
DB_CONNECTION = pymysql.connect(host=config.HOST, port=config.PORT, user=config.USER, password=config.PASSWORD, db=config.DATABASE)
DB_QUEUE = DB_CONNECTION.cursor()

if COMMAND == "list":
    if len(sys.argv) != 2:
        print("Usage: \"" + sys.argv[0] + " list\"")
        print("List all tables in " + config.DATABASE + ".")
    else:
        RESULTS = management.list_tables(DB_QUEUE)
        for item in RESULTS:
            management.print_tuples(item)
elif COMMAND == "create":
    if len(sys.argv) != 4:
        print("Usage: \"" + sys.argv[0] + " create <table> <benchmark>\"")
        print("Create a table in " + config.DATABASE + " for a specific benchmark.")
    else:
        SCENARIO = ""
        EXTRA = ""
        METRICS = ""
        if sys.argv[3].lower() == "triad":
            SCENARIO = "inputSize INTEGER NOT NULL,"
            EXTRA = "vector INTEGER NOT NULL,"
            METRICS = "GBs FLOAT UNSIGNED NOT NULL,"
        elif sys.argv[3].lower() == "reduction":
            SCENARIO = "inputSize INTEGER NOT NULL, outputSize INTEGER NOT NULL,"
            EXTRA = "nrItemsPerBlock INTEGER NOT NULL,"
            METRICS = "GBs FLOAT UNSIGNED NOT NULL,"
        elif sys.argv[3].lower() == "stencil":
            SCENARIO = "matrixWidth INTEGER NOT NULL,"
            EXTRA = "localMemory TINYINT NOT NULL,"
            METRICS = "GFLOPs FLOAT UNSIGNED NOT NULL,"
        elif sys.argv[3].lower() == "md":
            SCENARIO = "nrAtoms INTEGER NOT NULL,"
            METRICS = "GFLOPs FLOAT UNSIGNED NOT NULL,"
        elif sys.argv[3].lower() == "correlator":
            SCENARIO = "nrChannels INTEGER NOT NULL, nrStations INTEGER NOT NULL, nrSamples INTEGER NOT NULL, nrPolarizations INTEGER NOT NULL,"
            METRICS = "GFLOPs FLOAT UNSIGNED NOT NULL,"
        management.create_table(DB_QUEUE, sys.argv[2], SCENARIO, EXTRA, METRICS)
elif COMMAND == "delete":
    if len(sys.argv) != 3:
        print("Usage: \"" + sys.argv[0] + " delete <table>\"")
        print("Delete a table from " + config.DATABASE + ".")
    else:
        management.delete_table(DB_QUEUE, sys.argv[2])
elif COMMAND == "load":
    if len(sys.argv) != 5:
        print("Usage: \"" + sys.argv[0] + " load <table> <input_file> <benchmark>\"")
        print("Load a file containing auto-tuning data into a table of " + config.DATABASE + ".")
    else:
        INPUT_FILE = open(sys.argv[3])
        management.load_file(DB_QUEUE, sys.argv[2], INPUT_FILE, sys.argv[4])
        INPUT_FILE.close()
elif COMMAND == "tune":
    if len(sys.argv) != 5:
        print("Usage: \"" + sys.argv[0] + " tune <table> <benchmark> <scenario>\"")
        print("Returns the optimums values in the table, for a given scenario.")
    else:
        management.print_tuples(tuning.tune(DB_QUEUE, sys.argv[2], sys.argv[3], sys.argv[4]))
elif COMMAND == "quartiles":
    if len(sys.argv) != 5:
        print("Usage: \"" + sys.argv[0] + " quartiles <table> <benchmark> <scenario>\"")
        print("Returns the quartiles for the data in the table, for a given scenario.")
    else:
        management.print_tuples(statistics.get_quartiles(DB_QUEUE, sys.argv[2], sys.argv[3], sys.argv[4]))
elif COMMAND == "histogram":
    if len(sys.argv) != 5:
        print("Usage: \"" + sys.argv[0] + " histogram <table> <benchmark> <scenario>\"")
        print("Returns the histogram of the data in the table, for a given scenario.")
    else:
        RESULTS = statistics.get_histogram(DB_QUEUE, sys.argv[2], sys.argv[3], sys.argv[4])
        ITEMS = sorted(RESULTS.keys())
        for item in ITEMS:
            print(item, RESULTS[item])
else:
    print("Unknown command.")
    print("Type \"" + sys.argv[0]  + "\" for a list of supported commands.")

DB_QUEUE.close()
DB_CONNECTION.commit()
DB_CONNECTION.close()

sys.exit(0)
