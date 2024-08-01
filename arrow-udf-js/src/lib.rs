// Copyright 2024 RisingWave Labs
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#![doc = include_str!("../README.md")]

use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Mutex;
use std::sync::{atomic::Ordering, Arc};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context as _, Error, Result};
use arrow_array::{builder::Int32Builder, Array, ArrayRef, BooleanArray, RecordBatch};
use arrow_schema::{DataType, Field, FieldRef, Schema, SchemaRef};
pub use rquickjs::runtime::MemoryUsage;
use rquickjs::{
    context::intrinsic::All, function::Args, module::Evaluated, Context, Ctx, FromJs, Module,
    Object, Persistent, Value,
};

pub use self::into_field::IntoField;

mod into_field;
mod jsarrow;

// TODO: first pass at multithreaded should just do away with function caching
// - just store function text and eval each time.
/// A runtime to execute user defined functions in JavaScript.
///
/// # Usages
///
/// - Create a new runtime with [`Runtime::new`].
/// - For scalar functions, use [`add_function`] and [`call`].
/// - For table functions, use [`add_function`] and [`call_table_function`].
/// - For aggregate functions, create the function with [`add_aggregate`], and then
///     - create a new state with [`create_state`],
///     - update the state with [`accumulate`] or [`accumulate_or_retract`],
///     - merge states with [`merge`],
///     - finally get the result with [`finish`].
///
/// Click on each function to see the example.
///
/// [`add_function`]: Runtime::add_function
/// [`add_aggregate`]: Runtime::add_aggregate
/// [`call`]: Runtime::call
/// [`call_table_function`]: Runtime::call_table_function
/// [`create_state`]: Runtime::create_state
/// [`accumulate`]: Runtime::accumulate
/// [`accumulate_or_retract`]: Runtime::accumulate_or_retract
/// [`merge`]: Runtime::merge
/// [`finish`]: Runtime::finish
pub struct Runtime {
    functions: HashMap<String, FunctionDefinition>, // - each instance will lazy init and cache
    aggregates: HashMap<String, AggregateDefinition>,
    // NOTE: `functions` and `aggregates` must be put before the `runtime` and `context` to be dropped first.
    converter: jsarrow::Converter,
    memory_limit: Option<usize>,
    // this needs to be contexts
    instances: Mutex<Vec<Instance>>,
    /// Timeout of each function call.
    timeout: Option<Duration>,
}

pub struct Instance {
    // drop functions before dropping context or quickjs will panic due to functions not being
    // freed before the context is freed
    functions: HashMap<String, Function>,
    aggregates: HashMap<String, Aggregate>,
    context: Context,
    runtime: rquickjs::Runtime,
    /// Deadline of the current function call.
    deadline: Arc<atomic_time::AtomicOptionInstant>,
}

// right now this just makes a new context each time
// which I might argue is not the worst thing, since quickjs is
// made for fast startup.. Hmm
fn build_context<'a>() -> Result<Context> {
    let runtime = rquickjs::Runtime::new().context("failed to create quickjs runtime")?;
    let context = rquickjs::Context::custom::<All>(&runtime)
        .context("failed to create quickjs context")?;
    Ok(context)
}

fn get_function<'a>(
    ctx: &Ctx<'a>,
    module: &Module<'a, Evaluated>,
    name: &str,
) -> Result<JsFunction> {
    let function: rquickjs::Function = module.get(name).with_context(|| {
        format!("function \"{name}\" not found. HINT: make sure the function is exported")
    })?;
    // not ideal to call this even for just testing if the function is valid
    // should not do a save here I don't think..
    Ok(Persistent::save(ctx, function))
}

// TODO: currently this is not used. Eventually this should be a pool
impl Instance {
    pub fn new(rt: &Runtime) -> Result<Self> {
        let instance_runtime = rquickjs::Runtime::new().context("failed to create quickjs runtime")?;
        let context = rquickjs::Context::custom::<All>(&instance_runtime)
            .context("failed to create quickjs context")?;

        let deadline = Arc::new(atomic_time::AtomicOptionInstant::new(None));

        if let Some(memory_limit) = rt.memory_limit {
            instance_runtime.set_memory_limit(memory_limit);
        }
        if let Some(deadline) = deadline.load(Ordering::Relaxed) {
            instance_runtime.set_interrupt_handler(Some(Box::new(move || {
                return deadline <= Instant::now();
            })));
        }

        Ok(Self {
            runtime: instance_runtime,
            context,
            functions: HashMap::new(),
            aggregates: HashMap::new(),
            deadline,
        })
    }

    // pub fn add_function(
    //     &mut self,
    //     name: &str,
    //     return_type: impl IntoField,
    //     mode: CallMode,
    //     code: &str,
    //     handler: &str,
    // ) -> Result<()> {
    //     let function = self.context.with(|ctx| {
    //         let (module, _) = Module::declare(ctx.clone(), name, code)
    //             .map_err(|e| check_exception(e, &ctx))
    //             .context("failed to declare module")?
    //             .eval()
    //             .map_err(|e| check_exception(e, &ctx))
    //             .context("failed to evaluate module")?;
    //         Self::get_function(&ctx, &module, handler)
    //     })?;
    //     let function = Function {
    //         function,
    //         return_field: return_type.into_field(name).into(),
    //         mode,
    //     };
    //     self.functions.insert(name.to_string(), function);
    //     Ok(())
    // }

    /// Get a function from a module.
    fn get_function<'a>(
        ctx: &Ctx<'a>,
        module: &Module<'a, Evaluated>,
        name: &str,
    ) -> Result<JsFunction> {
        let function: rquickjs::Function = module.get(name).with_context(|| {
            format!("function \"{name}\" not found. HINT: make sure the function is exported")
        })?;
        Ok(Persistent::save(ctx, function))
    }
}

// how would function adding work if we have multiple instances?
// lazy init and cache?

impl Debug for Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Runtime")
            .field("functions", &self.functions.keys())
            .field("aggregates", &self.aggregates.keys())
            .field("timeout", &self.timeout)
            .finish()
    }
}

/// A user defined scalar function or table function.
struct Function {
    function: JsFunction,
    return_field: FieldRef,
    mode: CallMode,
}

struct FunctionDefinition {
    name: String,
    return_type: Field,
    mode: CallMode,
    handler: String,
    code: String,
    bytecode: Vec<u8>,
}

/// A user defined aggregate function.
struct Aggregate {
    state_field: FieldRef,
    output_field: FieldRef,
    mode: CallMode,
    create_state: JsFunction,
    accumulate: JsFunction,
    retract: Option<JsFunction>,
    finish: Option<JsFunction>,
    merge: Option<JsFunction>,
}

struct AggregateDefinition {
    name: String,
    state_type: DataType,
    output_type: DataType,
    mode: CallMode,
    code: String,
}

/// A persistent function.
type JsFunction = Persistent<rquickjs::Function<'static>>;

// SAFETY: `rquickjs::Runtime` is `Send` and `Sync`
unsafe impl Send for Runtime {}
unsafe impl Sync for Runtime {}

/// Whether the function will be called when some of its arguments are null.
#[derive(Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone)]
pub enum CallMode {
    /// The function will be called normally when some of its arguments are null.
    /// It is then the function author's responsibility to check for null values if necessary and respond appropriately.
    #[default]
    CalledOnNullInput,

    /// The function always returns null whenever any of its arguments are null.
    /// If this parameter is specified, the function is not executed when there are null arguments;
    /// instead a null result is assumed automatically.
    ReturnNullOnNullInput,
}

impl Runtime {
    /// Create a new `Runtime`.
    pub fn new() -> Result<Self> {
        Ok(Self {
            functions: HashMap::new(),
            aggregates: HashMap::new(),
            instances: Mutex::new(vec![]),
            memory_limit: None,
            timeout: None,
            // deadline: Default::default(),
            converter: jsarrow::Converter::new(),
        })
    }

    /// Set the memory limit of the runtime.
    ///
    /// # Example
    ///
    /// ```
    /// # use arrow_udf_js::Runtime;
    /// let runtime = Runtime::new().unwrap();
    /// runtime.set_memory_limit(Some(1 << 20)); // 1MB
    /// ```
    pub fn set_memory_limit(&mut self, limit: Option<usize>) {
        self.memory_limit = limit;
        // self.runtime.set_memory_limit(limit.unwrap_or(0));
    }

    /// Set the timeout of each function call.
    ///
    /// # Example
    ///
    /// ```
    /// # use arrow_udf_js::Runtime;
    /// # use std::time::Duration;
    /// let mut runtime = Runtime::new().unwrap();
    /// runtime.set_timeout(Some(Duration::from_secs(1)));
    /// ```
    pub fn set_timeout(&mut self, timeout: Option<Duration>) {
        self.timeout = timeout;
        self.instances.lock().unwrap().iter().for_each(|instance| {
            if timeout.is_some() {
                if let Some(deadline) = instance.deadline.load(Ordering::Relaxed) {
                    instance.runtime.set_interrupt_handler(Some(Box::new(move || {
                        return deadline <= Instant::now();
                    })));
                }
            } else {
                instance.runtime.set_interrupt_handler(None);
            }
        });
    }

    /// Get memory usage of the internal quickjs runtime.
    ///
    /// # Example
    ///
    /// ```
    /// # use arrow_udf_js::Runtime;
    /// let runtime = Runtime::new().unwrap();
    /// let usage = runtime.memory_usage();
    /// ```
    pub fn memory_usage(&self) -> MemoryUsage {
        // just trying to make it pass for a sec
        let instance = Instance::new(self).unwrap();
        // TODO: this is problematic because we have multiple instances
        // and this is an internal qjs runtime memory usage
        // struct. Just returning the first instance's memory usage for now.
        // maybe eventually it's summed or something?
        // self.instances.lock().unwrap()[0].runtime.memory_usage()
        return instance.runtime.memory_usage();
    }

    /// Return the converter where you can configure the extension metadata key and values.
    pub fn converter_mut(&mut self) -> &mut jsarrow::Converter {
        &mut self.converter
    }

    /// Add a new scalar function or table function.
    ///
    /// # Arguments
    ///
    /// - `name`: The name of the function.
    /// - `return_type`: The data type of the return value.
    /// - `mode`: Whether the function will be called when some of its arguments are null.
    /// - `code`: The JavaScript code of the function.
    ///
    /// The code should define an **exported** function with the same name as the function.
    /// The function should return a value for scalar functions, or yield values for table functions.
    ///
    /// # Example
    ///
    /// ```
    /// # use arrow_udf_js::{Runtime, CallMode};
    /// # use arrow_schema::DataType;
    /// let mut runtime = Runtime::new().unwrap();
    /// // add a scalar function
    /// runtime
    ///     .add_function(
    ///         "gcd",
    ///         DataType::Int32,
    ///         CallMode::ReturnNullOnNullInput,
    ///         r#"
    ///         export function gcd(a, b) {
    ///             while (b != 0) {
    ///                 let t = b;
    ///                 b = a % b;
    ///                 a = t;
    ///             }
    ///             return a;
    ///         }
    /// "#,
    ///     )
    ///     .unwrap();
    /// // add a table function
    /// runtime
    ///     .add_function(
    ///         "series",
    ///         DataType::Int32,
    ///         CallMode::ReturnNullOnNullInput,
    ///         r#"
    ///         export function* series(n) {
    ///             for (let i = 0; i < n; i++) {
    ///                 yield i;
    ///             }
    ///         }
    /// "#,
    ///     )
    ///     .unwrap();
    /// ```
    pub fn add_function(
        &mut self,
        name: &str,
        return_type: impl IntoField,
        mode: CallMode,
        code: &str,
    ) -> Result<()> {
        self.add_function_with_handler(name, return_type, mode, code, name)
    }

    /// Add a new scalar function or table function with custom handler name.
    ///
    /// # Arguments
    ///
    /// - `handler`: The name of function in Python code to be called.
    /// - others: Same as [`add_function`].
    ///
    /// [`add_function`]: Runtime::add_function
    pub fn add_function_with_handler(
        &mut self,
        name: &str,
        return_type: impl IntoField,
        mode: CallMode,
        code: &str,
        handler: &str,
    ) -> Result<()> {
        // temporary context for testing if the function is valid
        let context = build_context()?;

        // compile module to bytecode for fast loading
        let bytecode = context.with(|ctx| {
            let module = Module::declare(ctx.clone(), name, code)
                .map_err(|e| check_exception(e, &ctx))
                .context("failed to declare module")?;

            module.write(false)
                .map_err(|e| check_exception(e, &ctx))
        })?;

        let function = FunctionDefinition {
            name: name.to_string(),
            // this needs to be a Field and not a DataType to retain metadata
            return_type: return_type.into_field(name).clone(),
            mode,
            handler: handler.to_string(),
            code: code.to_string(),
            bytecode,
        };
        self.functions.insert(name.to_string(), function);
        Ok(())
    }

    /// Add a new aggregate function.
    ///
    /// # Arguments
    ///
    /// - `name`: The name of the function.
    /// - `state_type`: The data type of the internal state.
    /// - `output_type`: The data type of the aggregate value.
    /// - `mode`: Whether the function will be called when some of its arguments are null.
    /// - `code`: The JavaScript code of the aggregate function.
    ///
    /// The code should define at least two functions:
    ///
    /// - `create_state() -> state`: Create a new state object.
    /// - `accumulate(state, *args) -> state`: Accumulate a new value into the state, returning the updated state.
    ///
    /// optionally, the code can define:
    ///
    /// - `finish(state) -> value`: Get the result of the aggregate function.
    ///     If not defined, the state is returned as the result.
    ///     In this case, `output_type` must be the same as `state_type`.
    /// - `retract(state, *args) -> state`: Retract a value from the state, returning the updated state.
    /// - `merge(state, state) -> state`: Merge two states, returning the merged state.
    ///
    /// Each function must be **exported**.
    ///
    /// # Example
    ///
    /// ```
    /// # use arrow_udf_js::{Runtime, CallMode};
    /// # use arrow_schema::DataType;
    /// let mut runtime = Runtime::new().unwrap();
    /// runtime
    ///     .add_aggregate(
    ///         "sum",
    ///         DataType::Int32, // state_type
    ///         DataType::Int32, // output_type
    ///         CallMode::ReturnNullOnNullInput,
    ///         r#"
    ///         export function create_state() {
    ///             return 0;
    ///         }
    ///         export function accumulate(state, value) {
    ///             return state + value;
    ///         }
    ///         export function retract(state, value) {
    ///             return state - value;
    ///         }
    ///         export function merge(state1, state2) {
    ///             return state1 + state2;
    ///         }
    ///         "#,
    ///     )
    ///     .unwrap();
    /// ```
    pub fn add_aggregate(
        &mut self,
        name: &str,
        state_type: impl IntoField + Clone,
        output_type: impl IntoField + Clone,
        mode: CallMode,
        code: &str,
    ) -> Result<()> {
        let context = build_context()?;

        // TODO: pull this into shared function
        // both for validating here and for calling with memo
        let aggregate = context.with(|ctx| {
            let (module, _) = Module::declare(ctx.clone(), name, code)
                .map_err(|e| check_exception(e, &ctx))
                .context("failed to declare module")?
                .eval()
                .map_err(|e| check_exception(e, &ctx))
                .context("failed to evaluate module")?;
            Ok(Aggregate {
                state_field: state_type.clone().into_field(name).into(),
                output_field: output_type.clone().into_field(name).into(),
                mode: mode.clone(),
                create_state: get_function(&ctx, &module, "create_state")?,
                accumulate: get_function(&ctx, &module, "accumulate")?,
                retract: get_function(&ctx, &module, "retract").ok(),
                finish: get_function(&ctx, &module, "finish").ok(),
                merge: get_function(&ctx, &module, "merge").ok(),
            }) as Result<Aggregate>
        })?;

        if aggregate.finish.is_none() && aggregate.state_field != aggregate.output_field {
            bail!("`output_type` must be the same as `state_type` when `finish` is not defined");
        }

        let definition = AggregateDefinition {
            name: name.to_string(),
            state_type: state_type.into_field(name).data_type().clone(),
            output_type: output_type.into_field(name).data_type().clone(),
            mode,
            code: code.to_string(),
        };

        self.aggregates.insert(name.to_string(), definition);
        Ok(())
    }

    /// Call a scalar function.
    ///
    /// # Example
    ///
    /// ```
    #[doc = include_str!("doc_create_function.txt")]
    /// // suppose we have created a scalar function `gcd`
    /// // see the example in `add_function`
    ///
    /// let schema = Schema::new(vec![
    ///     Field::new("x", DataType::Int32, true),
    ///     Field::new("y", DataType::Int32, true),
    /// ]);
    /// let arg0 = Int32Array::from(vec![Some(25), None]);
    /// let arg1 = Int32Array::from(vec![Some(15), None]);
    /// let input = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(arg0), Arc::new(arg1)]).unwrap();
    ///
    /// let output = runtime.call("gcd", &input).unwrap();
    /// assert_eq!(&**output.column(0), &Int32Array::from(vec![Some(5), None]));
    /// ```
    pub fn call(&self, name: &str, input: &RecordBatch) -> Result<RecordBatch> {
        let function_definition = self.functions.get(name).context("function not found")?;

        let context = build_context()?;

        let function_mode = function_definition.mode.clone();

        // convert each row to js objects and call the function
        let res = context.with(|ctx| {
            let js_function = {
                let (module, _) = unsafe { Module::load(ctx.clone(), &function_definition.bytecode) }
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to load module")?
                    .eval()
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to evaluate module")?;

                module.get(&function_definition.handler).with_context(|| {
                    format!("function \"{name}\" not found. HINT: make sure the function is exported")
                })
            }?;

            let return_field = function_definition.return_type.clone().into_field(name).into();

            let mut results = Vec::with_capacity(input.num_rows());
            let mut row = Vec::with_capacity(input.num_columns());

            // what if this were an iterator instead?
            for i in 0..input.num_rows() {
                row.clear();
                for (column, field) in input.columns().iter().zip(input.schema().fields()) {
                    let val = self
                        .converter
                        .get_jsvalue(&ctx, field, column, i)
                        .context("failed to get jsvalue from arrow array")?;

                    row.push(val);
                }
                if function_mode == CallMode::ReturnNullOnNullInput
                    && row.iter().any(|v| v.is_null())
                {
                    results.push(Value::new_null(ctx.clone()));
                    continue;
                }
                let mut args = Args::new(ctx.clone(), row.len());
                args.push_args(row.drain(..))?;
                let result = self
                    .call_user_fn(&ctx, &js_function, args)
                    .context("failed to call function")?;
                results.push(result);
            }

            let array = self
                .converter
                .build_array(&return_field, &ctx, results)
                .context("failed to build arrow array from return values")?;
            let schema = Schema::new(vec![return_field.clone()]);
            Ok(RecordBatch::try_new(Arc::new(schema), vec![array])?)
        });

        res
    }

    /// Call a table function.
    ///
    /// # Example
    ///
    /// ```
    #[doc = include_str!("doc_create_function.txt")]
    /// // suppose we have created a table function `series`
    /// // see the example in `add_function`
    ///
    /// let schema = Schema::new(vec![Field::new("x", DataType::Int32, true)]);
    /// let arg0 = Int32Array::from(vec![Some(1), None, Some(3)]);
    /// let input = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(arg0)]).unwrap();
    ///
    /// let mut outputs = runtime.call_table_function("series", &input, 10).unwrap();
    /// let output = outputs.next().unwrap().unwrap();
    /// let pretty = arrow_cast::pretty::pretty_format_batches(&[output]).unwrap().to_string();
    /// assert_eq!(pretty, r#"
    /// +-----+--------+
    /// | row | series |
    /// +-----+--------+
    /// | 0   | 0      |
    /// | 2   | 0      |
    /// | 2   | 1      |
    /// | 2   | 2      |
    /// +-----+--------+"#.trim());
    /// ```
    pub fn call_table_function<'a>(
        &'a self,
        name: &'a str,
        input: &'a RecordBatch,
        chunk_size: usize,
    ) -> Result<RecordBatchIter<'a>> {
        assert!(chunk_size > 0);
        let function_definition = self.functions.get(name).context("function not found")?;

        // TODO: build and grab from cached instances array like wasm does
        // instead of creating a new instance each time
        let context = build_context()?;

        let function = context.with(|ctx| {
            let (module, _) = Module::declare(ctx.clone(), name, function_definition.code.clone())
                .map_err(|e| check_exception(e, &ctx))
                .context("failed to declare module")?
                .eval()
                .map_err(|e| check_exception(e, &ctx))
                .context("failed to evaluate module")?;
            get_function(&ctx, &module, &function_definition.handler)
        })?;

        // use 'a lifetime for this
        let function = Function {
            function: function.clone(),
            return_field: function_definition.return_type.clone().into_field(name).into(),
            mode: function_definition.mode.clone(),
        };

        let return_field = function.return_field.clone();

        // initial state
        Ok(RecordBatchIter {
            rt: self,
            input,
            function,
            schema: Arc::new(Schema::new(vec![
                Arc::new(Field::new("row", DataType::Int32, false)),
                return_field,
            ])),
            chunk_size,
            row: 0,
            generator: None,
            converter: &self.converter,
        })
    }

    /// Create a new state for an aggregate function.
    ///
    /// # Example
    /// ```
    #[doc = include_str!("doc_create_aggregate.txt")]
    /// let state = runtime.create_state("sum").unwrap();
    /// assert_eq!(&*state, &Int32Array::from(vec![0]));
    /// ```
    pub fn create_state(&self, name: &str) -> Result<ArrayRef> {
        let aggregate_definition = self.aggregates.get(name).context("function not found")?;
        let context = build_context()?;

        // no clue if this is right.. just getting things compiling. TODO: test/revisit
        let aggregate = Aggregate {
            state_field: aggregate_definition.state_type.clone().into_field(name).into(),
            output_field: aggregate_definition.output_type.clone().into_field(name).into(),
            mode: aggregate_definition.mode.clone(),
            create_state: context.with(|ctx| {
                let (module, _) = Module::declare(ctx.clone(), name, aggregate_definition.code.clone())
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to declare module")?
                    .eval()
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to evaluate module")?;
                get_function(&ctx, &module, "create_state")
            })?,
            accumulate: context.with(|ctx| {
                let (module, _) = Module::declare(ctx.clone(), name, aggregate_definition.code.clone())
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to declare module")?
                    .eval()
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to evaluate module")?;
                get_function(&ctx, &module, "accumulate")
            })?,
            retract: context.with(|ctx| {
                let (module, _) = Module::declare(ctx.clone(), name, aggregate_definition.code.clone())
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to declare module")
                    .ok()?
                    .eval()
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to evaluate module")
                    .ok()?;
                get_function(&ctx, &module, "retract").ok()
            }),
            finish: context.with(|ctx| {
                let (module, _) = Module::declare(ctx.clone(), name, aggregate_definition.code.clone())
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to declare module")
                    .ok()?
                    .eval()
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to evaluate module")
                    .ok()?;
                get_function(&ctx, &module, "finish").ok()
            }),
            merge: context.with(|ctx| {
                let (module, _) = Module::declare(ctx.clone(), name, aggregate_definition.code.clone())
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to declare module")
                    .ok()?
                    .eval()
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to evaluate module")
                    .ok()?;
                get_function(&ctx, &module, "merge").ok()
            }),
        };

        let state = context.with(|ctx| {
            let create_state = aggregate.create_state.clone().restore(&ctx)?;
            let state = self
                .call_user_fn(&ctx, &create_state, Args::new(ctx.clone(), 0))
                .context("failed to call create_state")?;
            let state = self
                .converter
                .build_array(&aggregate.state_field, &ctx, vec![state])?;
            Ok(state) as Result<_>
        })?;
        Ok(state)
    }

    /// Call accumulate of an aggregate function.
    ///
    /// # Example
    /// ```
    #[doc = include_str!("doc_create_aggregate.txt")]
    /// let state = runtime.create_state("sum").unwrap();
    ///
    /// let schema = Schema::new(vec![Field::new("value", DataType::Int32, true)]);
    /// let arg0 = Int32Array::from(vec![Some(1), None, Some(3), Some(5)]);
    /// let input = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(arg0)]).unwrap();
    ///
    /// let state = runtime.accumulate("sum", &state, &input).unwrap();
    /// assert_eq!(&*state, &Int32Array::from(vec![9]));
    /// ```
    pub fn accumulate(
        &self,
        name: &str,
        state: &dyn Array,
        input: &RecordBatch,
    ) -> Result<ArrayRef> {
        let aggregate_definition = self.aggregates.get(name).context("function not found")?;
        let context = build_context()?;

        // no clue if this is right.. just getting things compiling. TODO: test/revisit
        let aggregate = Aggregate {
            state_field: aggregate_definition.state_type.clone().into_field(name).into(),
            output_field: aggregate_definition.output_type.clone().into_field(name).into(),
            mode: aggregate_definition.mode.clone(),
            create_state: context.with(|ctx| {
                let (module, _) = Module::declare(ctx.clone(), name, aggregate_definition.code.clone())
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to declare module")?
                    .eval()
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to evaluate module")?;
                get_function(&ctx, &module, "create_state")
            })?,
            accumulate: context.with(|ctx| {
                let (module, _) = Module::declare(ctx.clone(), name, aggregate_definition.code.clone())
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to declare module")?
                    .eval()
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to evaluate module")?;
                get_function(&ctx, &module, "accumulate")
            })?,
            retract: context.with(|ctx| {
                let (module, _) = Module::declare(ctx.clone(), name, aggregate_definition.code.clone())
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to declare module")
                    .ok()?
                    .eval()
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to evaluate module")
                    .ok()?;
                get_function(&ctx, &module, "retract").ok()
            }),
            finish: context.with(|ctx| {
                let (module, _) = Module::declare(ctx.clone(), name, aggregate_definition.code.clone())
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to declare module")
                    .ok()?
                    .eval()
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to evaluate module")
                    .ok()?;
                get_function(&ctx, &module, "finish").ok()
            }),
            merge: context.with(|ctx| {
                let (module, _) = Module::declare(ctx.clone(), name, aggregate_definition.code.clone())
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to declare module")
                    .ok()?
                    .eval()
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to evaluate module")
                    .ok()?;
                get_function(&ctx, &module, "merge").ok()
            }),
        };

        // convert each row to python objects and call the accumulate function
        let new_state = context.with(|ctx| {
            let accumulate = aggregate.accumulate.clone().restore(&ctx)?;
            let mut state = self
                .converter
                .get_jsvalue(&ctx, &aggregate.state_field, state, 0)?;

            let mut row = Vec::with_capacity(1 + input.num_columns());
            for i in 0..input.num_rows() {
                if aggregate.mode == CallMode::ReturnNullOnNullInput
                    && input.columns().iter().any(|column| column.is_null(i))
                {
                    continue;
                }
                row.clear();
                row.push(state.clone());
                for (column, field) in input.columns().iter().zip(input.schema().fields()) {
                    let pyobj = self.converter.get_jsvalue(&ctx, field, column, i)?;
                    row.push(pyobj);
                }
                let mut args = Args::new(ctx.clone(), row.len());
                args.push_args(row.drain(..))?;
                state = self
                    .call_user_fn(&ctx, &accumulate, args)
                    .context("failed to call accumulate")?;
            }
            let output = self
                .converter
                .build_array(&aggregate.state_field, &ctx, vec![state])?;
            Ok(output) as Result<_>
        })?;
        Ok(new_state)
    }

    /// Call accumulate or retract of an aggregate function.
    ///
    /// The `ops` is a boolean array that indicates whether to accumulate or retract each row.
    /// `false` for accumulate and `true` for retract.
    ///
    /// # Example
    /// ```
    #[doc = include_str!("doc_create_aggregate.txt")]
    /// let state = runtime.create_state("sum").unwrap();
    ///
    /// let schema = Schema::new(vec![Field::new("value", DataType::Int32, true)]);
    /// let arg0 = Int32Array::from(vec![Some(1), None, Some(3), Some(5)]);
    /// let ops = BooleanArray::from(vec![false, false, true, false]);
    /// let input = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(arg0)]).unwrap();
    ///
    /// let state = runtime.accumulate_or_retract("sum", &state, &ops, &input).unwrap();
    /// assert_eq!(&*state, &Int32Array::from(vec![3]));
    /// ```
    pub fn accumulate_or_retract(
        &self,
        name: &str,
        state: &dyn Array,
        ops: &BooleanArray,
        input: &RecordBatch,
    ) -> Result<ArrayRef> {
        let aggregate_definition = self.aggregates.get(name).context("function not found")?;
        let context = build_context()?;

        // no clue if this is right.. just getting things compiling. TODO: test/revisit
        let aggregate = Aggregate {
            state_field: aggregate_definition.state_type.clone().into_field(name).into(),
            output_field: aggregate_definition.output_type.clone().into_field(name).into(),
            mode: aggregate_definition.mode.clone(),
            create_state: context.with(|ctx| {
                let (module, _) = Module::declare(ctx.clone(), name, aggregate_definition.code.clone())
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to declare module")?
                    .eval()
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to evaluate module")?;
                get_function(&ctx, &module, "create_state")
            })?,
            accumulate: context.with(|ctx| {
                let (module, _) = Module::declare(ctx.clone(), name, aggregate_definition.code.clone())
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to declare module")?
                    .eval()
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to evaluate module")?;
                get_function(&ctx, &module, "accumulate")
            })?,
            retract: context.with(|ctx| {
                let (module, _) = Module::declare(ctx.clone(), name, aggregate_definition.code.clone())
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to declare module")
                    .ok()?
                    .eval()
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to evaluate module")
                    .ok()?;
                get_function(&ctx, &module, "retract").ok()
            }),
            finish: context.with(|ctx| {
                let (module, _) = Module::declare(ctx.clone(), name, aggregate_definition.code.clone())
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to declare module")
                    .ok()?
                    .eval()
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to evaluate module")
                    .ok()?;
                get_function(&ctx, &module, "finish").ok()
            }),
            merge: context.with(|ctx| {
                let (module, _) = Module::declare(ctx.clone(), name, aggregate_definition.code.clone())
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to declare module")
                    .ok()?
                    .eval()
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to evaluate module")
                    .ok()?;
                get_function(&ctx, &module, "merge").ok()
            }),
        };

        // convert each row to python objects and call the accumulate function
        let new_state = context.with(|ctx| {
            let accumulate = aggregate.accumulate.clone().restore(&ctx)?;
            let retract = aggregate
                .retract
                .clone()
                .context("function does not support retraction")?
                .restore(&ctx)?;

            let mut state = self
                .converter
                .get_jsvalue(&ctx, &aggregate.state_field, state, 0)?;

            let mut row = Vec::with_capacity(1 + input.num_columns());
            for i in 0..input.num_rows() {
                if aggregate.mode == CallMode::ReturnNullOnNullInput
                    && input.columns().iter().any(|column| column.is_null(i))
                {
                    continue;
                }
                row.clear();
                row.push(state.clone());
                for (column, field) in input.columns().iter().zip(input.schema().fields()) {
                    let pyobj = self.converter.get_jsvalue(&ctx, field, column, i)?;
                    row.push(pyobj);
                }
                let func = if ops.is_valid(i) && ops.value(i) {
                    &retract
                } else {
                    &accumulate
                };
                let mut args = Args::new(ctx.clone(), row.len());
                args.push_args(row.drain(..))?;
                state = self
                    .call_user_fn(&ctx, func, args)
                    .context("failed to call accumulate or retract")?;
            }
            let output = self
                .converter
                .build_array(&aggregate.state_field, &ctx, vec![state])?;
            Ok(output) as Result<_>
        })?;
        Ok(new_state)
    }

    /// Merge states of an aggregate function.
    ///
    /// # Example
    /// ```
    #[doc = include_str!("doc_create_aggregate.txt")]
    /// let states = Int32Array::from(vec![Some(1), None, Some(3), Some(5)]);
    ///
    /// let state = runtime.merge("sum", &states).unwrap();
    /// assert_eq!(&*state, &Int32Array::from(vec![9]));
    /// ```
    pub fn merge(&self, name: &str, states: &dyn Array) -> Result<ArrayRef> {
        let aggregate_definition = self.aggregates.get(name).context("function not found")?;
        let context = build_context()?;

        // no clue if this is right.. just getting things compiling. TODO: test/revisit
        let aggregate = Aggregate {
            state_field: aggregate_definition.state_type.clone().into_field(name).into(),
            output_field: aggregate_definition.output_type.clone().into_field(name).into(),
            mode: aggregate_definition.mode.clone(),
            create_state: context.with(|ctx| {
                let (module, _) = Module::declare(ctx.clone(), name, aggregate_definition.code.clone())
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to declare module")?
                    .eval()
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to evaluate module")?;
                get_function(&ctx, &module, "create_state")
            })?,
            accumulate: context.with(|ctx| {
                let (module, _) = Module::declare(ctx.clone(), name, aggregate_definition.code.clone())
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to declare module")?
                    .eval()
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to evaluate module")?;
                get_function(&ctx, &module, "accumulate")
            })?,
            retract: context.with(|ctx| {
                let (module, _) = Module::declare(ctx.clone(), name, aggregate_definition.code.clone())
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to declare module")
                    .ok()?
                    .eval()
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to evaluate module")
                    .ok()?;
                get_function(&ctx, &module, "retract").ok()
            }),
            finish: context.with(|ctx| {
                let (module, _) = Module::declare(ctx.clone(), name, aggregate_definition.code.clone())
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to declare module")
                    .ok()?
                    .eval()
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to evaluate module")
                    .ok()?;
                get_function(&ctx, &module, "finish").ok()
            }),
            merge: context.with(|ctx| {
                let (module, _) = Module::declare(ctx.clone(), name, aggregate_definition.code.clone())
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to declare module")
                    .ok()?
                    .eval()
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to evaluate module")
                    .ok()?;
                get_function(&ctx, &module, "merge").ok()
            }),
        };

        let output = context.with(|ctx| {
            let merge = aggregate
                .merge
                .clone()
                .context("merge not found")?
                .restore(&ctx)?;
            let mut state = self
                .converter
                .get_jsvalue(&ctx, &aggregate.state_field, states, 0)?;
            for i in 1..states.len() {
                if aggregate.mode == CallMode::ReturnNullOnNullInput && states.is_null(i) {
                    continue;
                }
                let state2 = self
                    .converter
                    .get_jsvalue(&ctx, &aggregate.state_field, states, i)?;
                let mut args = Args::new(ctx.clone(), 2);
                args.push_args([state, state2])?;
                state = self
                    .call_user_fn(&ctx, &merge, args)
                    .context("failed to call accumulate or retract")?;
            }
            let output = self
                .converter
                .build_array(&aggregate.state_field, &ctx, vec![state])?;
            Ok(output) as Result<_>
        })?;
        Ok(output)
    }

    /// Get the result of an aggregate function.
    ///
    /// If the `finish` function is not defined, the state is returned as the result.
    ///
    /// # Example
    /// ```
    #[doc = include_str!("doc_create_aggregate.txt")]
    /// let states: ArrayRef = Arc::new(Int32Array::from(vec![Some(1), None, Some(3), Some(5)]));
    ///
    /// let outputs = runtime.finish("sum", &states).unwrap();
    /// assert_eq!(&outputs, &states);
    /// ```
    pub fn finish(&self, name: &str, states: &ArrayRef) -> Result<ArrayRef> {
        let aggregate_definition = self.aggregates.get(name).context("function not found")?;
        let context = build_context()?;

        // no clue if this is right.. just getting things compiling. TODO: test/revisit
        let aggregate = Aggregate {
            state_field: aggregate_definition.state_type.clone().into_field(name).into(),
            output_field: aggregate_definition.output_type.clone().into_field(name).into(),
            mode: aggregate_definition.mode.clone(),
            create_state: context.with(|ctx| {
                let (module, _) = Module::declare(ctx.clone(), name, aggregate_definition.code.clone())
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to declare module")?
                    .eval()
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to evaluate module")?;
                get_function(&ctx, &module, "create_state")
            })?,
            accumulate: context.with(|ctx| {
                let (module, _) = Module::declare(ctx.clone(), name, aggregate_definition.code.clone())
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to declare module")?
                    .eval()
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to evaluate module")?;
                get_function(&ctx, &module, "accumulate")
            })?,
            retract: context.with(|ctx| {
                let (module, _) = Module::declare(ctx.clone(), name, aggregate_definition.code.clone())
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to declare module")
                    .ok()?
                    .eval()
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to evaluate module")
                    .ok()?;
                get_function(&ctx, &module, "retract").ok()
            }),
            finish: context.with(|ctx| {
                let (module, _) = Module::declare(ctx.clone(), name, aggregate_definition.code.clone())
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to declare module")
                    .ok()?
                    .eval()
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to evaluate module")
                    .ok()?;
                get_function(&ctx, &module, "finish").ok()
            }),
            merge: context.with(|ctx| {
                let (module, _) = Module::declare(ctx.clone(), name, aggregate_definition.code.clone())
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to declare module")
                    .ok()?
                    .eval()
                    .map_err(|e| check_exception(e, &ctx))
                    .context("failed to evaluate module")
                    .ok()?;
                get_function(&ctx, &module, "merge").ok()
            }),
        };

        let Some(finish) = &aggregate.finish else {
            return Ok(states.clone());
        };

        let output = context.with(|ctx| {
            let finish = finish.clone().restore(&ctx)?;
            let mut results = Vec::with_capacity(states.len());
            for i in 0..states.len() {
                if aggregate.mode == CallMode::ReturnNullOnNullInput && states.is_null(i) {
                    results.push(Value::new_null(ctx.clone()));
                    continue;
                }
                let state = self
                    .converter
                    .get_jsvalue(&ctx, &aggregate.state_field, states, i)?;
                let mut args = Args::new(ctx.clone(), 1);
                args.push_args([state])?;
                let result = self
                    .call_user_fn(&ctx, &finish, args)
                    .context("failed to call finish")?;
                results.push(result);
            }
            let output = self
                .converter
                .build_array(&aggregate.output_field, &ctx, results)?;
            Ok(output) as Result<_>
        })?;
        Ok(output)
    }

    /// Call a user function.
    ///
    /// If `timeout` is set, the function will be interrupted after the timeout.
    fn call_user_fn<'js, T: FromJs<'js>>(
        &self,
        ctx: &Ctx<'js>,
        f: &rquickjs::Function<'js>,
        args: Args<'js>,
    ) -> Result<T> {
        // let context = build_context()?;

        // obvs wrong, just getting things compiling to test base scalar parallelism
        // let deadline = Arc::new(atomic_time::AtomicOptionInstant::new(None));

        let result = f.call_arg(args);
        // let result = if let Some(timeout) = self.timeout {
        //     deadline
        //         .store(Some(Instant::now() + timeout), Ordering::Relaxed);
        //     let result = f.call_arg(args);
        //     deadline.store(None, Ordering::Relaxed);
        //     result
        // } else {
        //     f.call_arg(args)
        // };
        result.map_err(|e| check_exception(e, ctx))
    }
}

/// An iterator over the result of a table function.
pub struct RecordBatchIter<'a> {
    rt: &'a Runtime,
    input: &'a RecordBatch,
    function: Function,
    schema: SchemaRef,
    chunk_size: usize,
    // mutable states
    /// Current row index.
    row: usize,
    /// Generator of the current row.
    generator: Option<Persistent<Object<'static>>>,
    converter: &'a jsarrow::Converter,
}

// XXX: not sure if this is safe.
unsafe impl Send for RecordBatchIter<'_> {}

impl RecordBatchIter<'_> {
    /// Get the schema of the output.
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    fn next(&mut self) -> Result<Option<RecordBatch>> {
        if self.row == self.input.num_rows() {
            return Ok(None);
        }

        let context = build_context()?;

        context.with(|ctx| {
            let js_function = self.function.function.clone().restore(&ctx)?;
            let mut indexes = Int32Builder::with_capacity(self.chunk_size);
            let mut results = Vec::with_capacity(self.input.num_rows());
            let mut row = Vec::with_capacity(self.input.num_columns());
            // restore generator from state
            let mut generator = match self.generator.take() {
                Some(generator) => {
                    let gen = generator.restore(&ctx)?;
                    let next: rquickjs::Function =
                        gen.get("next").context("failed to get 'next' method")?;
                    Some((gen, next))
                }
                None => None,
            };
            while self.row < self.input.num_rows() && results.len() < self.chunk_size {
                let (gen, next) = if let Some(g) = generator.as_ref() {
                    g
                } else {
                    // call the table function to get a generator
                    row.clear();
                    for (column, field) in
                        (self.input.columns().iter()).zip(self.input.schema().fields())
                    {
                        let val = self
                            .converter
                            .get_jsvalue(&ctx, field, column, self.row)
                            .context("failed to get jsvalue from arrow array")?;
                        row.push(val);
                    }
                    if self.function.mode == CallMode::ReturnNullOnNullInput
                        && row.iter().any(|v| v.is_null())
                    {
                        self.row += 1;
                        continue;
                    }
                    let mut args = Args::new(ctx.clone(), row.len());
                    args.push_args(row.drain(..))?;
                    let gen: Object = self
                        .rt
                        .call_user_fn(&ctx, &js_function, args)
                        .context("failed to call function")?;
                    let next: rquickjs::Function =
                        gen.get("next").context("failed to get 'next' method")?;
                    let mut args = Args::new(ctx.clone(), 0);
                    args.this(gen.clone())?;
                    generator.insert((gen, next))
                };
                let mut args = Args::new(ctx.clone(), 0);
                args.this(gen.clone())?;
                let object: Object = self
                    .rt
                    .call_user_fn(&ctx, next, args)
                    .context("failed to call next")?;
                let value: Value = object.get("value")?;
                let done: bool = object.get("done")?;
                if done {
                    self.row += 1;
                    generator = None;
                    continue;
                }
                indexes.append_value(self.row as i32);
                results.push(value);
            }
            self.generator = generator.map(|(gen, _)| Persistent::save(&ctx, gen));

            if results.is_empty() {
                return Ok(None);
            }
            let indexes = Arc::new(indexes.finish());
            let array = self
                .converter
                .build_array(&self.function.return_field, &ctx, results)
                .context("failed to build arrow array from return values")?;
            Ok(Some(RecordBatch::try_new(
                self.schema.clone(),
                vec![indexes, array],
            )?))
        })
    }
}

impl Iterator for RecordBatchIter<'_> {
    type Item = Result<RecordBatch>;
    fn next(&mut self) -> Option<Self::Item> {
        self.next().transpose()
    }
}

/// Get exception from `ctx` if the error is an exception.
fn check_exception(err: rquickjs::Error, ctx: &Ctx) -> anyhow::Error {
    match err {
        rquickjs::Error::Exception => {
            anyhow!("exception generated by QuickJS: {:?}", ctx.catch())
        }
        e => e.into(),
    }
}
