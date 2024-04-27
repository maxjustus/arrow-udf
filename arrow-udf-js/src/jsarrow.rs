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

//! Convert arrow array from/to js objects.

use anyhow::{Context, Result};
use arrow_array::{array::*, builder::*, ArrowNativeTypeOp};
use arrow_buffer::{OffsetBuffer, i256};
use arrow_schema::DataType;
use rquickjs::{Ctx, FromJs, Error, IntoJs, Function, Object, TypedArray, Value};
use rquickjs::function::Args;
use std::sync::Arc;

macro_rules! get_jsvalue {
    ($array_type: ty, $ctx:expr, $array:expr, $i:expr) => {{
        let array = $array.as_any().downcast_ref::<$array_type>().unwrap();
        array.value($i).into_js($ctx)
    }};
}

macro_rules! get_typed_array {
    ($array_type: ty, $ctx:expr, $array:expr) => {{
        let array = $array.as_any().downcast_ref::<$array_type>().unwrap();
        TypedArray::new($ctx.clone(), array.values().as_ref()).map(|a| a.into_value())
    }};
}

macro_rules! build_array {
    (NullBuilder, $ctx:expr, $values:expr) => {{
        let mut builder = NullBuilder::with_capacity($values.len());
        for val in $values {
            if val.is_null() || val.is_undefined() {
                builder.append_null();
            } else {
                builder.append_empty_value();
            }
        }
        Ok(Arc::new(builder.finish()))
    }};
    // primitive types
    ($builder_type: ty, $ctx:expr, $values:expr) => {{
        let mut builder = <$builder_type>::with_capacity($values.len());
        for val in $values {
            if val.is_null() || val.is_undefined() {
                builder.append_null();
            } else {
                builder.append_value(FromJs::from_js($ctx, val)?);
            }
        }
        Ok(Arc::new(builder.finish()))
    }};
    // string and bytea
    ($builder_type: ty, $elem_type: ty, $ctx:expr, $values:expr) => {{
        let mut builder = <$builder_type>::with_capacity($values.len(), 1024);
        for val in $values {
            if val.is_null() || val.is_undefined() {
                builder.append_null();
            } else {
                builder.append_value(<$elem_type>::from_js($ctx, val)?);
            }
        }
        Ok(Arc::new(builder.finish()))
    }};
}

macro_rules! build_json_array {
    ($array_type: ty, $ctx:expr, $values:expr) => {{
        let mut builder = <$array_type>::with_capacity($values.len(), 1024);
        for val in $values {
            if val.is_null() || val.is_undefined() {
                builder.append_null();
            } else if let Some(s) = $ctx.json_stringify(val)? {
                builder.append_value(s.to_string()?);
            } else {
                builder.append_null();
            }
        }
        Ok(Arc::new(builder.finish()))
    }};
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LargeUtf8ConvertedType {
    JSON,
    String
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LargeBinaryConvertedType {
    JSON,
    Decimal
}

#[derive(Debug, Clone, Copy)]
pub struct Converter{
    pub large_utf8_conversion_type: LargeUtf8ConvertedType,
    pub large_binary_conversion_type: LargeBinaryConvertedType,
}

impl Converter {
    /// Get array element as a JS Value.
    pub fn get_jsvalue<'a>(
        &self,
        ctx: &Ctx<'a>,
        bigdecimal: &rquickjs::Function<'a>,
        array: &dyn Array,
        i: usize,
    ) -> Result<Value<'a>, Error> {
        if array.is_null(i) {
            return Ok(Value::new_null(ctx.clone()));
        }

        match array.data_type() {
            DataType::Null => Ok(Value::new_null(ctx.clone())),
            DataType::Boolean => get_jsvalue!(BooleanArray, ctx, array, i),
            DataType::Int8 => get_jsvalue!(Int8Array, ctx, array, i),
            DataType::Int16 => get_jsvalue!(Int16Array, ctx, array, i),
            DataType::Int32 => get_jsvalue!(Int32Array, ctx, array, i),
            DataType::Int64 => get_jsvalue!(Int64Array, ctx, array, i),
            DataType::UInt8 => get_jsvalue!(UInt8Array, ctx, array, i),
            DataType::UInt16 => get_jsvalue!(UInt16Array, ctx, array, i),
            DataType::UInt32 => get_jsvalue!(UInt32Array, ctx, array, i),
            DataType::UInt64 => get_jsvalue!(UInt64Array, ctx, array, i),
            DataType::Float32 => get_jsvalue!(Float32Array, ctx, array, i),
            DataType::Float64 => get_jsvalue!(Float64Array, ctx, array, i),
            DataType::Utf8 => get_jsvalue!(StringArray, ctx, array, i),
            DataType::Binary => get_jsvalue!(BinaryArray, ctx, array, i),
            DataType::LargeUtf8 => {
                match self.large_utf8_conversion_type {
                    LargeUtf8ConvertedType::JSON => {
                        let array = array.as_any().downcast_ref::<LargeStringArray>().unwrap();
                        let string = array.value(i);
                        ctx.json_parse(string)
                    },
                    LargeUtf8ConvertedType::String=> {
                        get_jsvalue!(LargeStringArray, ctx, array, i)
                    }
                }
            },
            // decimal type
            DataType::LargeBinary => {
                let array = array.as_any().downcast_ref::<LargeBinaryArray>().unwrap();
                let string = std::str::from_utf8(array.value(i))?;

                match self.large_binary_conversion_type {
                    LargeBinaryConvertedType::JSON => {
                        ctx.json_parse(string)
                    },
                    LargeBinaryConvertedType::Decimal => {
                        bigdecimal.call((string,))
                    }
                }
            },
            DataType::Decimal128(_, _) => {
                let array = array.as_any().downcast_ref::<Decimal128Array>().unwrap();
                let decimal_str = array.value_as_string(i);
                bigdecimal.call((decimal_str,))
            },
            DataType::Decimal256(_, _) => {
                let array = array.as_any().downcast_ref::<Decimal256Array>().unwrap();
                let decimal_str = array.value_as_string(i);
                bigdecimal.call((decimal_str,))
            },
            // list
            DataType::List(inner) => {
                let array = array.as_any().downcast_ref::<ListArray>().unwrap();
                let list = array.value(i);
                match inner.data_type() {
                    DataType::Int8 => get_typed_array!(Int8Array, ctx, list),
                    DataType::Int16 => get_typed_array!(Int16Array, ctx, list),
                    DataType::Int32 => get_typed_array!(Int32Array, ctx, list),
                    DataType::Int64 => get_typed_array!(Int64Array, ctx, list),
                    DataType::UInt8 => get_typed_array!(UInt8Array, ctx, list),
                    DataType::UInt16 => get_typed_array!(UInt16Array, ctx, list),
                    DataType::UInt32 => get_typed_array!(UInt32Array, ctx, list),
                    DataType::UInt64 => get_typed_array!(UInt64Array, ctx, list),
                    DataType::Float32 => get_typed_array!(Float32Array, ctx, list),
                    DataType::Float64 => get_typed_array!(Float64Array, ctx, list),
                    _ => {
                        let mut values = Vec::with_capacity(list.len());
                        for j in 0..list.len() {
                            values.push(self.get_jsvalue(ctx, bigdecimal, list.as_ref(), j)?);
                        }
                        values.into_js(ctx)
                    }
                }
            }
            DataType::Struct(fields) => {
                let array = array.as_any().downcast_ref::<StructArray>().unwrap();
                let object = Object::new(ctx.clone())?;
                for (j, field) in fields.iter().enumerate() {
                    let value = self.get_jsvalue(ctx, bigdecimal, array.column(j).as_ref(), i)?;
                    object.set(field.name(), value)?;
                }
                Ok(object.into_value())
            }
            _ => todo!(),
        }
    }

    pub fn build_array<'a>(
        &self,
        data_type: &DataType,
        ctx: &Ctx<'a>,
        values: Vec<Value<'a>>,
    ) -> Result<ArrayRef> {
        match data_type {
            DataType::Null => build_array!(NullBuilder, ctx, values),
            DataType::Boolean => build_array!(BooleanBuilder, ctx, values),
            DataType::Int8 => build_array!(Int8Builder, ctx, values),
            DataType::Int16 => build_array!(Int16Builder, ctx, values),
            DataType::Int32 => build_array!(Int32Builder, ctx, values),
            DataType::Int64 => build_array!(Int64Builder, ctx, values),
            DataType::UInt8 => build_array!(UInt8Builder, ctx, values),
            DataType::UInt16 => build_array!(UInt16Builder, ctx, values),
            DataType::UInt32 => build_array!(UInt32Builder, ctx, values),
            DataType::UInt64 => build_array!(UInt64Builder, ctx, values),
            DataType::Float32 => build_array!(Float32Builder, ctx, values),
            DataType::Float64 => build_array!(Float64Builder, ctx, values),
            DataType::Utf8 => build_array!(StringBuilder, String, ctx, values),
            DataType::Binary => build_array!(BinaryBuilder, Vec::<u8>, ctx, values),
            // json type
            DataType::LargeUtf8 => {
                match self.large_utf8_conversion_type {
                    LargeUtf8ConvertedType::JSON => {
                        build_json_array!(LargeStringBuilder, ctx, values)
                    },
                    LargeUtf8ConvertedType::String => {
                        build_array!(LargeStringBuilder, String, ctx, values)
                    }
                }
            },
            // decimal type
            DataType::LargeBinary => {
                match self.large_binary_conversion_type {
                    LargeBinaryConvertedType::JSON => {
                        build_json_array!(LargeBinaryBuilder, ctx, values)
                    },
                    LargeBinaryConvertedType::Decimal => {
                        let mut builder = LargeBinaryBuilder::with_capacity(values.len(), 1024);
                        let bigdecimal_to_string: Function = ctx
                            .eval("BigDecimal.prototype.toString")
                            .context("failed to get BigDecimal.prototype.string")?;
                        for val in values {
                            if val.is_null() || val.is_undefined() {
                                builder.append_null();
                            } else {
                                let mut args = Args::new(ctx.clone(), 0);
                                args.this(val)?;
                                let string: String = bigdecimal_to_string.call_arg(args).context(
                                    "failed to convert BigDecimal to string. make sure you return a BigDecimal value",
                                    )?;
                                builder.append_value(string);
                            }
                        }
                        Ok(Arc::new(builder.finish()))
                    }
                }
            },
            DataType::Decimal128(precision, scale) => {
                let mut builder = Decimal128Builder::with_capacity(values.len())
                    .with_precision_and_scale(*precision, *scale)?;

                let bigdecimal_to_string: Function = ctx
                    .eval("BigDecimal.prototype.toString")
                    .context("failed to get BigDecimal.prototype.string")?;

                for val in values {
                    if val.is_null() || val.is_undefined() {
                        builder.append_null();
                    } else {
                        let mut args = Args::new(ctx.clone(), 0);
                        args.this(val)?;
                        let string: String = bigdecimal_to_string.call_arg(args).context(
                            "failed to convert BigDecimal to string. make sure you return a BigDecimal value",
                            )?;
                        let decimal_integer = decimal_string_to_i256(&string, *scale)?
                            .to_i128()
                            .ok_or_else(|| anyhow::anyhow!("failed to convert to i128"))?;
                        builder.append_value(decimal_integer);
                    }
                }
                Ok(Arc::new(builder.finish()))
            },
            DataType::Decimal256(precision, scale) => {
                let mut builder = Decimal256Builder::with_capacity(values.len())
                    .with_precision_and_scale(*precision, *scale)?;

                let bigdecimal_to_string: Function = ctx
                    .eval("BigDecimal.prototype.toString")
                    .context("failed to get BigDecimal.prototype.string")?;

                for val in values {
                    if val.is_null() || val.is_undefined() {
                        builder.append_null();
                    } else {
                        let mut args = Args::new(ctx.clone(), 0);
                        args.this(val)?;
                        let string: String = bigdecimal_to_string.call_arg(args).context(
                            "failed to convert BigDecimal to string. make sure you return a BigDecimal value",
                            )?;
                        let decimal_integer = decimal_string_to_i256(&string, *scale)?;
                        builder.append_value(decimal_integer);
                    }
                }
                Ok(Arc::new(builder.finish()))
            },
            // list
            DataType::List(inner) => {
                // flatten lists
                let mut flatten_values = vec![];
                let mut offsets = Vec::<i32>::with_capacity(values.len() + 1);
                offsets.push(0);
                for val in &values {
                    if !val.is_null() && !val.is_undefined() {
                        let array = val.as_array().context("failed to convert to array")?;
                        flatten_values.reserve(array.len());
                        for elem in array.iter() {
                            flatten_values.push(elem?);
                        }
                    }
                    offsets.push(flatten_values.len() as i32);
                }
                let values_array = self.build_array(inner.data_type(), ctx, flatten_values)?;
                let nulls = values
                    .iter()
                    .map(|v| !v.is_null() && !v.is_undefined())
                    .collect();
                Ok(Arc::new(ListArray::new(
                    inner.clone(),
                    OffsetBuffer::new(offsets.into()),
                    values_array,
                    Some(nulls),
                )))
            }
            DataType::Struct(fields) => {
                let mut arrays = Vec::with_capacity(fields.len());
                for field in fields {
                    let mut field_values = Vec::with_capacity(values.len());
                    for val in &values {
                        let v = if val.is_null() || val.is_undefined() {
                            Value::new_null(ctx.clone())
                        } else {
                            let object = val.as_object().context("expect object")?;
                            object.get(field.name())?
                        };
                        field_values.push(v);
                    }
                    arrays.push(self.build_array(field.data_type(), ctx, field_values)?);
                }
                let nulls = values
                    .iter()
                    .map(|v| !v.is_null() && !v.is_undefined())
                    .collect();
                Ok(Arc::new(StructArray::new(
                    fields.clone(),
                    arrays,
                    Some(nulls),
                )))
            }
            _ => todo!(),
        }
    }
}

fn decimal_string_to_i256(s: &str, scale: i8) -> Result<i256> {
    if scale < 0 {
        return Err(anyhow::anyhow!("currently only supports non-negative scale"));
    }

   let parts = s.split('.').collect::<Vec<&str>>();
   let integer = i256::from_string(parts[0]).ok_or_else(|| anyhow::anyhow!("failed to parse integer part"))?;
   let fractional = if parts.len() == 1 {
       i256::ZERO
   } else {
       i256::from_string(parts[1]).ok_or_else(|| anyhow::anyhow!("failed to parse fractional part"))?
   };

   Ok((integer * i256::from_i128(10).pow_checked(scale as u32)?) + fractional)
}
