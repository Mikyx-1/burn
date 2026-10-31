#![allow(unexpected_cfgs)]

use burn_backend::{
    TensorMetadata,
    ops::{ModuleOps, ctc::ctc_loss_default},
};
use burn_flex::{Flex, FlexTensor};
use burn_std::{DType, FloatDType};
use std::hint::black_box;
#[cfg(not(ctc_count_alloc))]
use std::time::{Duration, Instant};

#[cfg(ctc_count_alloc)]
#[path = "allocator.rs"]
mod allocation;
#[path = "fixtures.rs"]
mod scenarios;

type Inputs = [FlexTensor; 4];
const DTYPES: [FloatDType; 4] = [
    FloatDType::F16,
    FloatDType::BF16,
    FloatDType::F32,
    FloatDType::F64,
];
const PARALLEL_SCENARIOS: [usize; 4] = [3, 4, 5, 6];

fn forward(inputs: &Inputs, blank: usize, native: bool) -> FlexTensor {
    let [scores, targets, input_lengths, target_lengths] = inputs.clone();
    if native {
        Flex::ctc_loss(scores, targets, input_lengths, target_lengths, blank)
    } else {
        ctc_loss_default::<Flex>(scores, targets, input_lengths, target_lengths, blank)
    }
}

fn verify(inputs: &Inputs, blank: usize) {
    let before = forward(inputs, blank, false);
    let after = forward(inputs, blank, true);
    assert_eq!(before.shape(), after.shape());
    assert_eq!(before.dtype(), after.dtype());
    let dtype = before.dtype();
    let tolerance = if dtype == DType::F32 {
        2e-5
    } else {
        16.0 * dtype.finfo().unwrap().epsilon
    };
    let exact = matches!(dtype, DType::F16 | DType::BF16);
    let before: Vec<f64> = before.into_data().iter::<f64>().collect();
    let after: Vec<f64> = after.into_data().iter::<f64>().collect();
    for (&a, &b) in before.iter().zip(&after) {
        if a.is_nan() {
            assert!(b.is_nan());
        } else if a.is_infinite() {
            assert_eq!(a, b);
        } else {
            assert!(b.is_finite());
            assert!((a - b).abs() <= tolerance * a.abs().max(1.0), "{a} != {b}");
            if exact || (a == 0.0 && b == 0.0) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
    }
    for _ in 0..3 {
        for native in [false, true] {
            black_box(forward(inputs, blank, native));
        }
    }
}

#[cfg(not(ctc_count_alloc))]
fn timing(inputs: &Inputs, blank: usize, native: bool) -> (f64, u64) {
    let started = Instant::now();
    let mut calls = 0u64;
    while calls < 3 || started.elapsed() < Duration::from_millis(80) {
        let result = forward(inputs, blank, native);
        black_box(result.bytes());
        drop(result);
        calls += 1;
    }
    (
        started.elapsed().as_secs_f64() * 1000.0 / calls as f64,
        calls,
    )
}

fn main() {
    if std::env::args().any(|arg| arg == "--catalog") {
        for s in &scenarios::SCENARIOS {
            let [t, b, c, width] = s.shape;
            println!("{},{:?},{t},{b},{c},{width}", s.id, s.name);
        }
        return;
    }
    #[cfg(ctc_count_alloc)]
    allocation::validate();
    let workers: usize = std::env::var("RAYON_NUM_THREADS").unwrap().parse().unwrap();
    assert_eq!(rayon::current_num_threads(), workers);
    // Finish every worker's startup and initial parking before any measurement window.
    rayon::broadcast(|_| std::hint::black_box(()));
    std::thread::sleep(std::time::Duration::from_millis(100));
    let parallel = std::env::args().any(|arg| arg == "--parallel");
    for dtype in DTYPES {
        for (scenario_idx, scenario) in scenarios::SCENARIOS.iter().enumerate() {
            if parallel && !PARALLEL_SCENARIOS.contains(&scenario_idx) {
                continue;
            }
            black_box(scenario.name);
            let (inputs, blank) = scenario.fixture(dtype);
            let snapshots = inputs.each_ref().map(|input| input.bytes().to_vec());
            verify(&inputs, blank);
            #[cfg(not(ctc_count_alloc))]
            for round in 0..7 {
                for native in [round % 2 != 0, round % 2 == 0] {
                    let (ms, calls) = timing(&inputs, blank, native);
                    println!(
                        "time,{workers},{dtype:?},{},{native},{round},{ms:.9},{calls}",
                        scenario.id
                    );
                }
            }
            #[cfg(ctc_count_alloc)]
            for repeat in 0..3 {
                for native in [false, true] {
                    allocation::begin();
                    let result = forward(&inputs, blank, native);
                    black_box(result.bytes());
                    let stats = allocation::end();
                    drop(result);
                    assert!(!stats.overflow);
                    assert_eq!(stats.untracked_reallocs, 0);
                    println!(
                        "memory,{workers},{dtype:?},{},{native},{repeat},{},{},{},{},{}",
                        scenario.id,
                        stats.peak,
                        stats.live,
                        stats.requests,
                        stats.reallocs,
                        stats.untracked_reallocs
                    );
                }
            }
            for (input, snapshot) in inputs.iter().zip(snapshots) {
                assert_eq!(input.bytes(), snapshot);
            }
        }
    }
}
