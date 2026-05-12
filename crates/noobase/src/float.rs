use ndarray::ScalarOperand;
use num_traits::{Float as NumFloat, FromPrimitive};

pub trait Float: NumFloat + FromPrimitive + ScalarOperand + Send + Sync + 'static {}

impl Float for f32 {}
impl Float for f64 {}
