//! Planar polygon primitives for image reprojection.
//!
//! This module provides a Sutherland-Hodgman polygon clipper specialised
//! against axis-aligned unit cells, plus the signed-area shoelace formula.
//! Both helpers operate in `f64` because the reprojection kernel runs in
//! `f64` regardless of the input image dtype (see `image::reproject`).
//!
//! The clipper is intentionally narrow in scope: it clips a 4-vertex
//! "subject" polygon (an output pixel projected into input-pixel space)
//! against a single unit cell `[x0, x0 + 1] x [y0, y0 + 1]` where the
//! integers `(x0, y0)` index one input pixel after the internal
//! half-pixel shift applied by the reprojection driver. The subject
//! polygon is allowed to be non-convex; Sutherland-Hodgman only requires
//! the clip polygon (the unit cell) to be convex.

use smallvec::SmallVec;

/// A 2-D point in input-pixel coordinates. Crate-internal because the
/// polygon primitives are an implementation detail of `image::reproject`.
pub(crate) type Point = [f64; 2];

/// Inline capacity for clipped polygons. A convex unit-cell clip of a
/// 4-vertex subject polygon yields at most 8 vertices in general (each
/// of the 4 clip edges can introduce at most one extra vertex), so 8
/// is the natural inline budget.
const CLIPPED_INLINE: usize = 8;

/// Vertex buffer used inside the clipper. Kept private so `smallvec`
/// does not leak into the crate's public API.
type ClippedPolygon = SmallVec<[Point; CLIPPED_INLINE]>;

/// Edge of the unit cell, used to drive the four clipping passes.
#[derive(Clone, Copy)]
enum CellEdge {
    Left { x_min: f64 },
    Right { x_max: f64 },
    Bottom { y_min: f64 },
    Top { y_max: f64 },
}

impl CellEdge {
    /// A point is "inside" the half-plane defined by the edge if it is
    /// on the cell's side of the edge (boundary inclusive). Boundary
    /// inclusion is important: vertices that land exactly on an edge of
    /// the cell must survive the clip so adjacent cells produce
    /// watertight area sums.
    fn is_inside(self, point: Point) -> bool {
        match self {
            CellEdge::Left { x_min } => point[0] >= x_min,
            CellEdge::Right { x_max } => point[0] <= x_max,
            CellEdge::Bottom { y_min } => point[1] >= y_min,
            CellEdge::Top { y_max } => point[1] <= y_max,
        }
    }

    /// Intersect the segment `a -> b` with this edge's supporting line.
    /// Caller must guarantee that the two endpoints straddle the line
    /// (one inside, one outside); otherwise the denominator may be
    /// zero. In Sutherland-Hodgman the straddle is enforced by the
    /// surrounding case analysis.
    fn intersect(self, a: Point, b: Point) -> Point {
        match self {
            CellEdge::Left { x_min: x_target } | CellEdge::Right { x_max: x_target } => {
                let denom = b[0] - a[0];
                // Straddle guarantees `denom != 0`.
                let t = (x_target - a[0]) / denom;
                [x_target, a[1] + t * (b[1] - a[1])]
            }
            CellEdge::Bottom { y_min: y_target } | CellEdge::Top { y_max: y_target } => {
                let denom = b[1] - a[1];
                let t = (y_target - a[1]) / denom;
                [a[0] + t * (b[0] - a[0]), y_target]
            }
        }
    }
}

/// Clip a 4-vertex subject polygon against the unit cell with origin
/// `cell_origin = (column, row)` (after the half-pixel shift). The
/// cell occupies `[column, column + 1] x [row, row + 1]` in continuous
/// input-pixel coordinates.
///
/// Returns the clipped polygon as a `SmallVec<[Point; 8]>`. An empty
/// return means the subject polygon does not overlap the cell.
///
/// The subject polygon is allowed to wind in either direction and may
/// be non-convex; the clip polygon (the unit cell) is convex, which is
/// all Sutherland-Hodgman requires.
pub(crate) fn clip_quad_against_unit_cell(
    quad: &[Point; 4],
    cell_origin: (i32, i32),
) -> SmallVec<[Point; 8]> {
    let (column, row) = cell_origin;
    let x_min = column as f64;
    let x_max = x_min + 1.0;
    let y_min = row as f64;
    let y_max = y_min + 1.0;

    let edges = [
        CellEdge::Left { x_min },
        CellEdge::Right { x_max },
        CellEdge::Bottom { y_min },
        CellEdge::Top { y_max },
    ];

    let mut input: ClippedPolygon = SmallVec::new();
    input.extend_from_slice(quad);

    for edge in edges {
        if input.is_empty() {
            return input;
        }
        let mut output: ClippedPolygon = SmallVec::new();
        let vertex_count = input.len();
        for i in 0..vertex_count {
            let current = input[i];
            let previous = input[(i + vertex_count - 1) % vertex_count];
            let current_inside = edge.is_inside(current);
            let previous_inside = edge.is_inside(previous);
            match (previous_inside, current_inside) {
                (true, true) => {
                    output.push(current);
                }
                (true, false) => {
                    output.push(edge.intersect(previous, current));
                }
                (false, true) => {
                    output.push(edge.intersect(previous, current));
                    output.push(current);
                }
                (false, false) => {}
            }
        }
        input = output;
    }

    input
}

/// Signed area of a (possibly degenerate) polygon via the shoelace
/// formula. Positive for counter-clockwise winding, negative for
/// clockwise winding, zero for degenerate (collinear or zero-vertex)
/// inputs. Callers that only need a magnitude should take `.abs()`.
pub(crate) fn signed_area(polygon: &[Point]) -> f64 {
    if polygon.len() < 3 {
        return 0.0;
    }
    let mut accumulator = 0.0;
    let vertex_count = polygon.len();
    for i in 0..vertex_count {
        let current = polygon[i];
        let next = polygon[(i + 1) % vertex_count];
        accumulator += current[0] * next[1] - next[0] * current[1];
    }
    accumulator * 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * a.abs().max(b.abs()).max(1.0)
    }

    // ------------------------------------------------------------------
    // signed_area
    // ------------------------------------------------------------------

    #[test]
    fn signed_area_counter_clockwise_unit_square_is_one() {
        let square: [Point; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        assert!(approx_eq(signed_area(&square), 1.0, TOL));
    }

    #[test]
    fn signed_area_clockwise_unit_square_is_minus_one() {
        let square: [Point; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]];
        assert!(approx_eq(signed_area(&square), -1.0, TOL));
    }

    #[test]
    fn signed_area_degenerate_collinear_is_zero() {
        let triangle: [Point; 3] = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        assert!(approx_eq(signed_area(&triangle), 0.0, TOL));
    }

    #[test]
    fn signed_area_empty_or_short_is_zero() {
        let empty: [Point; 0] = [];
        assert!(approx_eq(signed_area(&empty), 0.0, TOL));
        let two: [Point; 2] = [[0.0, 0.0], [1.0, 1.0]];
        assert!(approx_eq(signed_area(&two), 0.0, TOL));
    }

    #[test]
    fn signed_area_known_triangle() {
        // Triangle with vertices (0,0), (4,0), (0,3) -> area 6.
        let triangle: [Point; 3] = [[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]];
        assert!(approx_eq(signed_area(&triangle), 6.0, TOL));
    }

    // ------------------------------------------------------------------
    // clip_quad_against_unit_cell
    // ------------------------------------------------------------------

    #[test]
    fn clip_subject_completely_inside_cell_returns_subject() {
        // Subject quad entirely inside cell [0, 1] x [0, 1].
        let quad: [Point; 4] = [
            [0.1, 0.1],
            [0.7, 0.2],
            [0.8, 0.9],
            [0.2, 0.8],
        ];
        let clipped = clip_quad_against_unit_cell(&quad, (0, 0));
        // Sutherland-Hodgman preserves vertices wholly inside; expect 4.
        assert_eq!(clipped.len(), 4);
        assert!(approx_eq(signed_area(&clipped).abs(), signed_area(&quad).abs(), TOL));
    }

    #[test]
    fn clip_subject_completely_disjoint_returns_empty() {
        // Subject entirely to the right of cell [0, 1] x [0, 1].
        let quad: [Point; 4] = [
            [2.0, 2.0],
            [3.0, 2.0],
            [3.0, 3.0],
            [2.0, 3.0],
        ];
        let clipped = clip_quad_against_unit_cell(&quad, (0, 0));
        assert!(clipped.is_empty());
        assert!(approx_eq(signed_area(&clipped).abs(), 0.0, TOL));
    }

    #[test]
    fn clip_axis_aligned_partial_overlap() {
        // Subject [0.5, 1.5] x [0.5, 1.5] overlaps cell [0, 1] x [0, 1]
        // in the quarter [0.5, 1.0] x [0.5, 1.0] (area 0.25).
        let quad: [Point; 4] = [
            [0.5, 0.5],
            [1.5, 0.5],
            [1.5, 1.5],
            [0.5, 1.5],
        ];
        let clipped = clip_quad_against_unit_cell(&quad, (0, 0));
        let area = signed_area(&clipped).abs();
        assert!(approx_eq(area, 0.25, TOL));
    }

    #[test]
    fn clip_cell_entirely_inside_subject_returns_cell() {
        // Subject covers a much larger region; cell is fully inside.
        let quad: [Point; 4] = [
            [-5.0, -5.0],
            [5.0, -5.0],
            [5.0, 5.0],
            [-5.0, 5.0],
        ];
        let clipped = clip_quad_against_unit_cell(&quad, (0, 0));
        let area = signed_area(&clipped).abs();
        assert!(approx_eq(area, 1.0, TOL));
    }

    #[test]
    fn clip_one_corner_inside() {
        // Subject [-0.5, 0.5] x [-0.5, 0.5] overlaps cell [0,1] x [0,1]
        // only in the small square [0, 0.5] x [0, 0.5] (area 0.25).
        let quad: [Point; 4] = [
            [-0.5, -0.5],
            [0.5, -0.5],
            [0.5, 0.5],
            [-0.5, 0.5],
        ];
        let clipped = clip_quad_against_unit_cell(&quad, (0, 0));
        let area = signed_area(&clipped).abs();
        assert!(approx_eq(area, 0.25, TOL));
    }

    #[test]
    fn clip_rotated_quad_through_cell() {
        // 45-degree rotated unit square centered at the cell center,
        // with half-diagonal 0.5 -> vertices on the cell midpoints.
        // The intersection is exactly the rotated square itself, with
        // area 0.5.
        let quad: [Point; 4] = [
            [0.5, 0.0],
            [1.0, 0.5],
            [0.5, 1.0],
            [0.0, 0.5],
        ];
        let clipped = clip_quad_against_unit_cell(&quad, (0, 0));
        let area = signed_area(&clipped).abs();
        assert!(approx_eq(area, 0.5, TOL));
    }

    #[test]
    fn clip_degenerate_quad_has_zero_area() {
        // Collinear "quad" along y = 0 -> zero area.
        let quad: [Point; 4] = [
            [0.0, 0.0],
            [0.4, 0.0],
            [0.8, 0.0],
            [0.2, 0.0],
        ];
        let clipped = clip_quad_against_unit_cell(&quad, (0, 0));
        let area = signed_area(&clipped).abs();
        assert!(approx_eq(area, 0.0, TOL));
    }

    #[test]
    fn clip_against_offset_cell() {
        // Same axis-aligned subject, but clipping against cell at
        // (3, 5) -> [3, 4] x [5, 6]. Subject [3.5, 4.5] x [5.5, 6.5]
        // overlaps in [3.5, 4.0] x [5.5, 6.0] (area 0.25).
        let quad: [Point; 4] = [
            [3.5, 5.5],
            [4.5, 5.5],
            [4.5, 6.5],
            [3.5, 6.5],
        ];
        let clipped = clip_quad_against_unit_cell(&quad, (3, 5));
        let area = signed_area(&clipped).abs();
        assert!(approx_eq(area, 0.25, TOL));
    }

    #[test]
    fn clip_clockwise_subject_returns_same_magnitude() {
        // Same overlap geometry as the partial-overlap test but with
        // clockwise winding. Magnitude must match.
        let quad: [Point; 4] = [
            [0.5, 0.5],
            [0.5, 1.5],
            [1.5, 1.5],
            [1.5, 0.5],
        ];
        let clipped = clip_quad_against_unit_cell(&quad, (0, 0));
        let area = signed_area(&clipped).abs();
        assert!(approx_eq(area, 0.25, TOL));
    }
}
