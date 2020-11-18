use nalgebra_sparse::proptest::{coo_with_duplicates, coo_no_duplicates};
use nalgebra::DMatrix;

use proptest::prelude::*;
use itertools::Itertools;

use std::collections::HashSet;
use std::iter::repeat;

#[cfg(feature = "slow-tests")]
use {
    proptest::test_runner::TestRunner,
    proptest::strategy::ValueTree
};
use std::ops::RangeInclusive;

#[cfg(feature = "slow-tests")]
fn generate_all_possible_matrices(value_range: RangeInclusive<i32>,
                                  rows_range: RangeInclusive<usize>,
                                  cols_range: RangeInclusive<usize>)
    -> HashSet<DMatrix<i32>>
{
    // Enumerate all possible combinations
    let mut all_combinations = HashSet::new();
    for nrows in rows_range {
        for ncols in cols_range.clone() {
            // For the given number of rows and columns
            let n_values = nrows * ncols;

            if n_values == 0 {
                // If we have zero rows or columns, the set of matrices with the given
                // rows and columns is a single element: an empty matrix
                all_combinations.insert(DMatrix::from_row_slice(nrows, ncols, &[]));
            } else {
                // Otherwise, we need to sample all possible matrices.
                // To do this, we generate the values as the (multi) Cartesian product
                // of the value sets. For example, for a 2x2 matrices, we consider
                // all possible 4-element arrays that the matrices can take by
                // considering all elements in the cartesian product
                //  V x V x V x V
                // where V is the set of eligible values, e.g. V := -1 ..= 1
                let values_iter = repeat(value_range.clone())
                    .take(n_values)
                    .multi_cartesian_product();
                for matrix_values in values_iter {
                    all_combinations.insert(DMatrix::from_row_slice(nrows, ncols, &matrix_values));
                }
            }
        }
    }
    all_combinations
}

#[cfg(feature = "slow-tests")]
#[test]
fn coo_no_duplicates_samples_all_admissible_outputs() {
    // Note: This test basically mirrors a similar test for `matrix` in the `nalgebra` repo.

    // Test that the proptest generation covers all possible outputs for a small space of inputs
    // given enough samples.

    // We use a deterministic test runner to make the test "stable".
    let mut runner = TestRunner::deterministic();

    // This number needs to be high enough so that we with high probability sample
    // all possible cases
    let num_generated_matrices = 500000;

    let values = -1..=1;
    let rows = 0..=2;
    let cols = 0..=3;
    let strategy = coo_no_duplicates(values.clone(), rows.clone(), cols.clone(), 2 * 3);

    // Enumerate all possible combinations
    let all_combinations = generate_all_possible_matrices(values, rows, cols);

    let mut visited_combinations = HashSet::new();
    for _ in 0..num_generated_matrices {
        let tree = strategy
            .new_tree(&mut runner)
            .expect("Tree generation should not fail");
        let matrix = tree.current();
        visited_combinations.insert(DMatrix::from(&matrix));
    }

    assert_eq!(visited_combinations.len(), all_combinations.len());
    assert_eq!(visited_combinations, all_combinations, "Did not sample all possible values.");
}

#[cfg(feature = "slow-tests")]
#[test]
fn coo_with_duplicates_samples_all_admissible_outputs() {
    // This is almost the same as the test for coo_no_duplicates, except that we need
    // a different "success" criterion, since coo_with_duplicates is able to generate
    // matrices with values outside of the value constraints. See below for details.

    // We use a deterministic test runner to make the test "stable".
    let mut runner = TestRunner::deterministic();

    // This number needs to be high enough so that we with high probability sample
    // all possible cases
    let num_generated_matrices = 500000;

    let values = -1..=1;
    let rows = 0..=2;
    let cols = 0..=3;
    let strategy = coo_with_duplicates(values.clone(), rows.clone(), cols.clone(), 2 * 3, 2);

    // Enumerate all possible combinations that fit the constraints
    // (note: this is only a subset of the matrices that can be generated by
    // `coo_with_duplicates`)
    let all_combinations = generate_all_possible_matrices(values, rows, cols);

    let mut visited_combinations = HashSet::new();
    for _ in 0..num_generated_matrices {
        let tree = strategy
            .new_tree(&mut runner)
            .expect("Tree generation should not fail");
        let matrix = tree.current();
        visited_combinations.insert(DMatrix::from(&matrix));
    }

    // Here we cannot verify that the set of visited combinations is *equal* to
    // all possible outcomes with the given constraints, however the
    // strategy should be able to generate all matrices that fit the constraints.
    // In other words, we need to determine that set of all admissible matrices
    // is contained in the set of visited matrices
    assert!(all_combinations.is_subset(&visited_combinations));
}

#[test]
fn coo_no_duplicates_generates_admissible_matrices() {

}