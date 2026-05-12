use ndarray::{Array1, ArrayView1};
use thiserror::Error;

use crate::float::Float;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Spacing {
    /// Arithmetic-midpoint convention. Used for grids that are linearly spaced or that
    /// the caller wishes to treat as such (the values are not required to be strictly
    /// uniform).
    Linear,
    /// Geometric-midpoint convention (arithmetic midpoint in ln-space). All values must
    /// be strictly positive.
    Log,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridKind {
    /// Values represent bin centers; an N-element grid describes N bins.
    Centers,
    /// Values represent bin edges; an N-element grid describes N-1 bins.
    Edges,
}

#[derive(Debug, Error, PartialEq)]
pub enum GridError {
    #[error("grid must contain at least 2 values, got {0}")]
    TooFewValues(usize),
    #[error("grid is not strictly monotonically increasing")]
    NotMonotonic,
    #[error("log-spaced grid requires all values to be strictly positive")]
    LogNonPositive,
}

#[derive(Debug, Clone)]
pub struct Grid<T: Float> {
    values: Array1<T>,
    spacing: Spacing,
    kind: GridKind,
}

impl<T: Float> Grid<T> {
    pub fn new(
        values: Array1<T>,
        spacing: Spacing,
        kind: GridKind,
    ) -> Result<Self, GridError> {
        check_length(values.view())?;
        check_monotonic(values.view())?;
        if spacing == Spacing::Log {
            check_positive(values.view())?;
        }
        Ok(Self {
            values,
            spacing,
            kind,
        })
    }

    /// Construct a grid from raw values, auto-detecting whether the spacing is best
    /// described by `Log` (uniform in ln-space within `rel_tol`) or `Linear` (everything
    /// else). `rel_tol` is the allowed relative deviation of per-step spacing.
    pub fn from_array(
        values: Array1<T>,
        rel_tol: T,
        kind: GridKind,
    ) -> Result<Self, GridError> {
        check_length(values.view())?;
        check_monotonic(values.view())?;
        let all_positive = values.iter().all(|value| *value > T::zero());
        let spacing = if all_positive && is_log_uniform(values.view(), rel_tol) {
            Spacing::Log
        } else {
            Spacing::Linear
        };
        Ok(Self {
            values,
            spacing,
            kind,
        })
    }

    /// Generate `n` linearly spaced values on `[start, end]`. Panics if `n < 2`.
    pub fn linspace(start: T, end: T, n: usize, kind: GridKind) -> Self {
        assert!(n >= 2, "linspace requires n >= 2");
        let denom = T::from_usize(n - 1).expect("n - 1 fits in T");
        let step = (end - start) / denom;
        let values: Array1<T> = (0..n)
            .map(|i| start + step * T::from_usize(i).expect("i fits in T"))
            .collect();
        Self {
            values,
            spacing: Spacing::Linear,
            kind,
        }
    }

    /// Generate `n` log-spaced values on `[start, end]`, where `start` and `end` are the
    /// actual endpoint values (not their logs). Panics if `n < 2` or either endpoint is
    /// non-positive.
    pub fn logspace(start: T, end: T, n: usize, kind: GridKind) -> Self {
        assert!(n >= 2, "logspace requires n >= 2");
        assert!(
            start > T::zero() && end > T::zero(),
            "logspace requires positive endpoints"
        );
        let log_start = start.ln();
        let log_end = end.ln();
        let denom = T::from_usize(n - 1).expect("n - 1 fits in T");
        let step = (log_end - log_start) / denom;
        let values: Array1<T> = (0..n)
            .map(|i| (log_start + step * T::from_usize(i).expect("i fits in T")).exp())
            .collect();
        Self {
            values,
            spacing: Spacing::Log,
            kind,
        }
    }

    pub fn values(&self) -> ArrayView1<'_, T> {
        self.values.view()
    }

    pub fn spacing(&self) -> Spacing {
        self.spacing
    }

    pub fn kind(&self) -> GridKind {
        self.kind
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the underlying values are uniform within `rel_tol` under the grid's
    /// spacing convention. Useful for selecting analytic fast paths in downstream code.
    pub fn is_uniform(&self, rel_tol: T) -> bool {
        match self.spacing {
            Spacing::Linear => is_linear_uniform(self.values.view(), rel_tol),
            Spacing::Log => is_log_uniform(self.values.view(), rel_tol),
        }
    }

    /// Return a grid whose values are bin edges. If `self.kind == Edges`, returns a
    /// clone. If `self.kind == Centers`, computes edges using the spacing's midpoint
    /// convention; end edges are extrapolated half a bin out from the first/last pair.
    pub fn to_edges(&self) -> Grid<T> {
        if self.kind == GridKind::Edges {
            return self.clone();
        }
        let n = self.values.len();
        let mut edges = Array1::<T>::zeros(n + 1);
        let two = T::from_usize(2).expect("2 fits in T");
        let centers = &self.values;
        match self.spacing {
            Spacing::Linear => {
                for i in 1..n {
                    edges[i] = (centers[i - 1] + centers[i]) / two;
                }
                edges[0] = centers[0] - (centers[1] - centers[0]) / two;
                edges[n] = centers[n - 1] + (centers[n - 1] - centers[n - 2]) / two;
            }
            Spacing::Log => {
                for i in 1..n {
                    edges[i] = (centers[i - 1] * centers[i]).sqrt();
                }
                let ratio_left = centers[1] / centers[0];
                let ratio_right = centers[n - 1] / centers[n - 2];
                edges[0] = centers[0] / ratio_left.sqrt();
                edges[n] = centers[n - 1] * ratio_right.sqrt();
            }
        }
        Grid {
            values: edges,
            spacing: self.spacing,
            kind: GridKind::Edges,
        }
    }

    /// Return a grid whose values are bin centers. If `self.kind == Centers`, returns a
    /// clone. If `self.kind == Edges`, computes centers via the spacing's midpoint
    /// convention. The result is the exact inverse of `to_edges` only when the original
    /// grid is strictly uniform under its spacing.
    pub fn to_centers(&self) -> Grid<T> {
        if self.kind == GridKind::Centers {
            return self.clone();
        }
        let n = self.values.len();
        let mut centers = Array1::<T>::zeros(n - 1);
        let two = T::from_usize(2).expect("2 fits in T");
        let edges = &self.values;
        match self.spacing {
            Spacing::Linear => {
                for i in 0..n - 1 {
                    centers[i] = (edges[i] + edges[i + 1]) / two;
                }
            }
            Spacing::Log => {
                for i in 0..n - 1 {
                    centers[i] = (edges[i] * edges[i + 1]).sqrt();
                }
            }
        }
        Grid {
            values: centers,
            spacing: self.spacing,
            kind: GridKind::Centers,
        }
    }
}

fn check_length<T: Float>(values: ArrayView1<T>) -> Result<(), GridError> {
    if values.len() < 2 {
        Err(GridError::TooFewValues(values.len()))
    } else {
        Ok(())
    }
}

fn check_monotonic<T: Float>(values: ArrayView1<T>) -> Result<(), GridError> {
    for i in 1..values.len() {
        if !(values[i - 1] < values[i]) {
            return Err(GridError::NotMonotonic);
        }
    }
    Ok(())
}

fn check_positive<T: Float>(values: ArrayView1<T>) -> Result<(), GridError> {
    if values.iter().all(|value| *value > T::zero()) {
        Ok(())
    } else {
        Err(GridError::LogNonPositive)
    }
}

fn is_linear_uniform<T: Float>(values: ArrayView1<T>, rel_tol: T) -> bool {
    let n = values.len();
    if n < 3 {
        return true;
    }
    let mean_step = (values[n - 1] - values[0]) / T::from_usize(n - 1).expect("n - 1 fits in T");
    if mean_step == T::zero() {
        return false;
    }
    let mut max_dev = T::zero();
    for i in 1..n {
        let step = values[i] - values[i - 1];
        let dev = (step - mean_step).abs();
        if dev > max_dev {
            max_dev = dev;
        }
    }
    max_dev <= rel_tol * mean_step.abs()
}

fn is_log_uniform<T: Float>(values: ArrayView1<T>, rel_tol: T) -> bool {
    let n = values.len();
    if n < 3 {
        return false;
    }
    if !values.iter().all(|value| *value > T::zero()) {
        return false;
    }
    let log_first = values[0].ln();
    let log_last = values[n - 1].ln();
    let mean_step = (log_last - log_first) / T::from_usize(n - 1).expect("n - 1 fits in T");
    if mean_step == T::zero() {
        return false;
    }
    let mut max_dev = T::zero();
    for i in 1..n {
        let step = values[i].ln() - values[i - 1].ln();
        let dev = (step - mean_step).abs();
        if dev > max_dev {
            max_dev = dev;
        }
    }
    max_dev <= rel_tol * mean_step.abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const TOL: f64 = 1e-6;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn new_rejects_too_few_values() {
        let arr = array![1.0_f64];
        assert_eq!(
            Grid::new(arr, Spacing::Linear, GridKind::Centers).unwrap_err(),
            GridError::TooFewValues(1)
        );
    }

    #[test]
    fn new_rejects_non_monotonic() {
        let arr = array![1.0_f64, 3.0, 2.0];
        assert_eq!(
            Grid::new(arr, Spacing::Linear, GridKind::Centers).unwrap_err(),
            GridError::NotMonotonic
        );
    }

    #[test]
    fn new_rejects_equal_neighbors() {
        let arr = array![1.0_f64, 1.0, 2.0];
        assert_eq!(
            Grid::new(arr, Spacing::Linear, GridKind::Centers).unwrap_err(),
            GridError::NotMonotonic
        );
    }

    #[test]
    fn new_rejects_log_non_positive() {
        let arr = array![-1.0_f64, 1.0, 2.0];
        assert_eq!(
            Grid::new(arr, Spacing::Log, GridKind::Centers).unwrap_err(),
            GridError::LogNonPositive
        );
    }

    #[test]
    fn linspace_endpoints_and_step() {
        let grid = Grid::<f64>::linspace(0.0, 10.0, 11, GridKind::Centers);
        assert_eq!(grid.len(), 11);
        let values = grid.values();
        assert!(approx_eq(values[0], 0.0, TOL));
        assert!(approx_eq(values[10], 10.0, TOL));
        assert!(approx_eq(values[5], 5.0, TOL));
        assert_eq!(grid.spacing(), Spacing::Linear);
        assert_eq!(grid.kind(), GridKind::Centers);
    }

    #[test]
    fn logspace_endpoints_and_step() {
        let grid = Grid::<f64>::logspace(1.0, 1000.0, 4, GridKind::Centers);
        assert_eq!(grid.len(), 4);
        let values = grid.values();
        assert!(approx_eq(values[0], 1.0, TOL));
        assert!(approx_eq(values[3], 1000.0, TOL));
        assert!(approx_eq(values[1], 10.0, TOL));
        assert!(approx_eq(values[2], 100.0, TOL));
        assert_eq!(grid.spacing(), Spacing::Log);
    }

    #[test]
    fn from_array_detects_log() {
        let arr = array![1.0_f64, 10.0, 100.0, 1000.0];
        let grid = Grid::from_array(arr, 1e-9, GridKind::Centers).unwrap();
        assert_eq!(grid.spacing(), Spacing::Log);
    }

    #[test]
    fn from_array_detects_linear() {
        let arr = array![1.0_f64, 2.0, 3.0, 4.0];
        let grid = Grid::from_array(arr, 1e-9, GridKind::Centers).unwrap();
        assert_eq!(grid.spacing(), Spacing::Linear);
    }

    #[test]
    fn from_array_irregular_falls_back_to_linear() {
        let arr = array![1.0_f64, 2.0, 5.0, 11.0];
        let grid = Grid::from_array(arr, 1e-9, GridKind::Centers).unwrap();
        assert_eq!(grid.spacing(), Spacing::Linear);
        assert!(!grid.is_uniform(1e-6));
    }

    #[test]
    fn kind_can_be_set_explicitly() {
        let edges_grid = Grid::<f64>::linspace(0.0, 10.0, 11, GridKind::Edges);
        assert_eq!(edges_grid.kind(), GridKind::Edges);
    }

    #[test]
    fn to_edges_on_centers_linear_uniform() {
        let centers = Grid::<f64>::linspace(0.5, 4.5, 5, GridKind::Centers);
        let edges = centers.to_edges();
        assert_eq!(edges.kind(), GridKind::Edges);
        assert_eq!(edges.len(), 6);
        let values = edges.values();
        for (i, expected) in (0..=5).map(|i| i as f64).enumerate() {
            assert!(approx_eq(values[i], expected, TOL));
        }
        assert_eq!(edges.spacing(), Spacing::Linear);
    }

    #[test]
    fn to_edges_on_centers_log_uses_geometric_mean() {
        let arr = array![1.0_f64, 10.0, 100.0];
        let centers = Grid::new(arr, Spacing::Log, GridKind::Centers).unwrap();
        let edges = centers.to_edges();
        let values = edges.values();
        // Interior edges are geometric means: sqrt(1*10), sqrt(10*100)
        assert!(approx_eq(values[1], 10.0_f64.sqrt(), TOL));
        assert!(approx_eq(values[2], 1000.0_f64.sqrt(), TOL));
        // End edges: extrapolate by half a log-step (ratio of 10 -> sqrt(10))
        assert!(approx_eq(values[0], 1.0 / 10.0_f64.sqrt(), TOL));
        assert!(approx_eq(values[3], 100.0 * 10.0_f64.sqrt(), TOL));
        assert_eq!(edges.spacing(), Spacing::Log);
        assert_eq!(edges.kind(), GridKind::Edges);
    }

    #[test]
    fn to_edges_on_edges_is_idempotent() {
        let arr = array![0.0_f64, 1.0, 2.0, 3.0];
        let edges = Grid::new(arr.clone(), Spacing::Linear, GridKind::Edges).unwrap();
        let again = edges.to_edges();
        assert_eq!(again.kind(), GridKind::Edges);
        assert_eq!(again.len(), edges.len());
        for i in 0..edges.len() {
            assert!(approx_eq(again.values()[i], edges.values()[i], TOL));
        }
    }

    #[test]
    fn to_centers_on_edges_linear() {
        let arr = array![0.0_f64, 1.0, 2.0, 3.0];
        let edges = Grid::new(arr, Spacing::Linear, GridKind::Edges).unwrap();
        let centers = edges.to_centers();
        let values = centers.values();
        assert_eq!(centers.kind(), GridKind::Centers);
        assert_eq!(centers.len(), 3);
        assert!(approx_eq(values[0], 0.5, TOL));
        assert!(approx_eq(values[1], 1.5, TOL));
        assert!(approx_eq(values[2], 2.5, TOL));
    }

    #[test]
    fn to_centers_on_edges_log() {
        let arr = array![1.0_f64, 10.0, 100.0];
        let edges = Grid::new(arr, Spacing::Log, GridKind::Edges).unwrap();
        let centers = edges.to_centers();
        let values = centers.values();
        assert!(approx_eq(values[0], 10.0_f64.sqrt(), TOL));
        assert!(approx_eq(values[1], 1000.0_f64.sqrt(), TOL));
    }

    #[test]
    fn to_centers_on_centers_is_idempotent() {
        let centers = Grid::<f64>::linspace(0.0, 10.0, 11, GridKind::Centers);
        let again = centers.to_centers();
        assert_eq!(again.kind(), GridKind::Centers);
        assert_eq!(again.len(), centers.len());
        for i in 0..centers.len() {
            assert!(approx_eq(again.values()[i], centers.values()[i], TOL));
        }
    }

    #[test]
    fn centers_edges_roundtrip_uniform_linear() {
        let original = Grid::<f64>::linspace(0.0, 10.0, 11, GridKind::Centers);
        let recovered = original.to_edges().to_centers();
        assert_eq!(recovered.len(), original.len());
        assert_eq!(recovered.kind(), GridKind::Centers);
        for i in 0..original.len() {
            assert!(approx_eq(recovered.values()[i], original.values()[i], TOL));
        }
    }

    #[test]
    fn centers_edges_roundtrip_uniform_log() {
        let original = Grid::<f64>::logspace(1.0, 1e6, 7, GridKind::Centers);
        let recovered = original.to_edges().to_centers();
        for i in 0..original.len() {
            assert!(approx_eq(recovered.values()[i], original.values()[i], TOL));
        }
    }

    #[test]
    fn centers_edges_roundtrip_irregular_is_not_identity() {
        let arr = array![1.0_f64, 2.0, 5.0, 11.0];
        let original = Grid::new(arr, Spacing::Linear, GridKind::Centers).unwrap();
        let recovered = original.to_edges().to_centers();
        // At least one interior value should differ from the original.
        let differs = (0..original.len())
            .any(|i| (recovered.values()[i] - original.values()[i]).abs() > 1e-9);
        assert!(differs);
    }

    #[test]
    fn is_uniform_linear() {
        assert!(Grid::<f64>::linspace(0.0, 1.0, 11, GridKind::Centers).is_uniform(1e-12));
        let arr = array![0.0_f64, 1.0, 2.0, 4.0];
        let grid = Grid::new(arr, Spacing::Linear, GridKind::Centers).unwrap();
        assert!(!grid.is_uniform(1e-6));
    }

    #[test]
    fn is_uniform_log() {
        assert!(Grid::<f64>::logspace(1.0, 1e6, 7, GridKind::Centers).is_uniform(1e-12));
    }

    #[test]
    fn works_with_f32() {
        let grid = Grid::<f32>::linspace(0.0, 1.0, 5, GridKind::Centers);
        let edges = grid.to_edges();
        assert_eq!(edges.len(), 6);
    }
}
