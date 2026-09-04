// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Reproduces the dictionary group-key aggregation tax measured in
//! <https://github.com/apache/datafusion/issues/24822>.
//!
//! Builds identical in-memory tables where only the group-key column encoding
//! differs (`Utf8View` vs `Dictionary(Int32, Utf8View)`, produced by casting
//! the exact same batches), then times the same group-by queries against both.
//! Single target partition (single-threaded execution) to isolate kernel cost;
//! 5 timed rounds after 1 warmup, median reported.

use std::sync::Arc;
use std::time::Instant;

use datafusion::arrow::array::{ArrayRef, Int32Array, Int64Array, StringViewArray};
use datafusion::arrow::compute::cast;
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::catalog::MemTable;
use datafusion::error::Result;
use datafusion::execution::context::SessionContext;
use datafusion::prelude::SessionConfig;

const BATCH: usize = 8192;

fn median(vals: &mut [f64]) -> f64 {
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    vals[vals.len() / 2]
}

/// String view column cycling over `distinct` distinct short values (<= 12
/// bytes, inlined in the view) — p_type-like key shape.
fn short_string_col(rows: usize, distinct: usize) -> ArrayRef {
    Arc::new(StringViewArray::from_iter_values(
        (0..rows).map(|i| format!("K{:0>5}", i % distinct)),
    ))
}

/// String view column cycling over `distinct` distinct long values (> 12
/// bytes, heap-escaped in the view) — s_name-like key shape.
fn long_string_col(rows: usize, distinct: usize) -> ArrayRef {
    Arc::new(StringViewArray::from_iter_values(
        (0..rows).map(|i| format!("Supplier#{:0>9}", i % distinct)),
    ))
}

fn int32_mod_col(rows: usize, distinct: i32) -> ArrayRef {
    Arc::new(Int32Array::from_iter_values(
        (0..rows).map(|i| (i as i32) % distinct),
    ))
}

fn measure_col(rows: usize) -> ArrayRef {
    Arc::new(Int64Array::from_iter_values(
        (0..rows).map(|i| ((i % 97) + 1) as i64),
    ))
}

/// Build one (view, dict) batch pair from a row slice of the key arrays.
fn build_batch_pair(
    key_cols: &[(&str, ArrayRef)],
    measure: &ArrayRef,
    start: usize,
    len: usize,
) -> Result<(RecordBatch, RecordBatch)> {
    let slice = |a: &ArrayRef| a.slice(start, len);
    let mut view_fields = Vec::new();
    let mut view_arrays = Vec::new();
    let mut dict_fields = Vec::new();
    let mut dict_arrays = Vec::new();
    for (name, arr) in key_cols {
        let dt = arr.data_type();
        view_fields.push(Field::new(*name, dt.clone(), false));
        view_arrays.push(slice(arr));
        if dt.equals_datatype(&DataType::Utf8View) {
            let dict_dt =
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(dt.clone()));
            dict_fields.push(Field::new(*name, dict_dt.clone(), false));
            dict_arrays.push(cast(&slice(arr), &dict_dt)?);
        } else {
            dict_fields.push(Field::new(*name, dt.clone(), false));
            dict_arrays.push(slice(arr));
        }
    }
    view_fields.push(Field::new("v", DataType::Int64, false));
    view_arrays.push(slice(measure));
    dict_fields.push(Field::new("v", DataType::Int64, false));
    dict_arrays.push(slice(measure));
    Ok((
        RecordBatch::try_new(Arc::new(Schema::new(view_fields)), view_arrays)?,
        RecordBatch::try_new(Arc::new(Schema::new(dict_fields)), dict_arrays)?,
    ))
}

/// Register two tables under `base_view` / `base_dict` from the same batches:
/// identical except every string key column is cast to
/// `Dictionary(Int32, Utf8View)` in the dict variant.
fn register_view_and_dict(
    ctx: &SessionContext,
    base: &str,
    key_cols: &[(&str, ArrayRef)],
    measure: ArrayRef,
    rows: usize,
) -> Result<()> {
    let batches: Vec<(RecordBatch, RecordBatch)> = (0..rows.div_ceil(BATCH))
        .map(|chunk| {
            let start = chunk * BATCH;
            let len = BATCH.min(rows - start);
            build_batch_pair(key_cols, &measure, start, len)
        })
        .collect::<Result<Vec<_>>>()?;

    let (view_batches, dict_batches): (Vec<_>, Vec<_>) = batches.into_iter().unzip();

    let view_schema = view_batches[0].schema();
    let dict_schema = dict_batches[0].schema();
    ctx.register_table(
        format!("{base}_view"),
        Arc::new(MemTable::try_new(view_schema, vec![view_batches])?),
    )?;
    ctx.register_table(
        format!("{base}_dict"),
        Arc::new(MemTable::try_new(dict_schema, vec![dict_batches])?),
    )?;
    Ok(())
}

fn run_query(ctx: &SessionContext, rt: &tokio::runtime::Runtime, sql: &str) -> usize {
    let df = rt.block_on(ctx.sql(sql)).unwrap();
    let batches = rt.block_on(df.collect()).unwrap();
    batches.iter().map(|b| b.num_rows()).sum()
}

/// Warmup once, then 5 timed rounds; returns (median ms, group count).
fn bench(ctx: &SessionContext, rt: &tokio::runtime::Runtime, sql: &str) -> (f64, usize) {
    let groups = run_query(ctx, rt, sql);
    let mut times = [0f64; 5];
    for t in times.iter_mut() {
        let start = Instant::now();
        run_query(ctx, rt, sql);
        *t = start.elapsed().as_secs_f64() * 1000.0;
    }
    (median(&mut times), groups)
}

fn main() -> Result<()> {
    let config = SessionConfig::new().with_target_partitions(1);
    let ctx = SessionContext::new_with_config(config);
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();

    // Q2-like: two keys (string ~150 distinct + int), 6M rows.
    let rows = 6_000_000;
    register_view_and_dict(
        &ctx,
        "q2",
        &[
            ("k1", short_string_col(rows, 150)),
            ("k2", int32_mod_col(rows, 50)),
        ],
        measure_col(rows),
        rows,
    )?;

    // Q19-like: one string key, tiny cardinality (7), 6M rows.
    register_view_and_dict(
        &ctx,
        "q19",
        &[("k", short_string_col(rows, 7))],
        measure_col(rows),
        rows,
    )?;

    // Q20-like: one long string key, high cardinality (100k), 2M rows.
    let rows20 = 2_000_000;
    register_view_and_dict(
        &ctx,
        "q20",
        &[("k", long_string_col(rows20, 100_000))],
        measure_col(rows20),
        rows20,
    )?;

    let cases: &[(&str, &str)] = &[
        (
            "q2",
            "SELECT k1, k2, COUNT(*), SUM(v) FROM t GROUP BY k1, k2",
        ),
        ("q19", "SELECT k, COUNT(*), SUM(v) FROM t GROUP BY k"),
        ("q20", "SELECT k, COUNT(*), SUM(v) FROM t GROUP BY k"),
    ];

    println!("case | Utf8View ms | Dictionary ms | slowdown | groups(v/d)");
    println!("--- | --- | --- | --- | ---");
    for (name, sql) in cases {
        let view_sql = sql.replace("FROM t", &format!("FROM {name}_view"));
        let dict_sql = sql.replace("FROM t", &format!("FROM {name}_dict"));
        let (view_ms, view_groups) = bench(&ctx, &rt, &view_sql);
        let (dict_ms, dict_groups) = bench(&ctx, &rt, &dict_sql);
        let slowdown = (dict_ms - view_ms) / view_ms * 100.0;
        println!(
            "{name} | {view_ms:.1} | {dict_ms:.1} | {slowdown:+.0}% | {view_groups}/{dict_groups}"
        );
    }

    Ok(())
}
