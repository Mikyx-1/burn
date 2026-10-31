//! Allocation accounting for the temporary CTC measurement executable.
//!
//! Measures requested bytes belonging to allocations created inside a window.
//! Untracked reallocations are reported separately and must be absent when
//! interpreting the result as extra requested heap above pre-existing inputs.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::{Mutex, MutexGuard};

const CAPACITY: usize = 4096;

#[derive(Clone, Copy, Debug)]
pub struct Stats {
    pub peak: usize,
    pub live: usize,
    pub requests: usize,
    pub reallocs: usize,
    pub untracked_reallocs: usize,
    pub overflow: bool,
}

impl Stats {
    const ZERO: Self = Self {
        peak: 0,
        live: 0,
        requests: 0,
        reallocs: 0,
        untracked_reallocs: 0,
        overflow: false,
    };
}

#[derive(Clone, Copy)]
struct Entry {
    pointer: usize,
    size: usize,
}

impl Entry {
    const EMPTY: Self = Self {
        pointer: 0,
        size: 0,
    };
}

struct Ledger {
    entries: [Entry; CAPACITY],
    len: usize,
    active: bool,
    stats: Stats,
}

impl Ledger {
    const fn new() -> Self {
        Self {
            entries: [Entry::EMPTY; CAPACITY],
            len: 0,
            active: false,
            stats: Stats::ZERO,
        }
    }

    fn find(&self, pointer: usize) -> Option<usize> {
        (0..self.len).find(|&index| self.entries[index].pointer == pointer)
    }

    fn add_live(&mut self, size: usize) {
        match self.stats.live.checked_add(size) {
            Some(live) => self.stats.live = live,
            None => {
                self.stats.overflow = true;
                self.stats.live = usize::MAX;
            }
        }
        self.stats.peak = self.stats.peak.max(self.stats.live);
    }

    fn remove_live(&mut self, size: usize) {
        match self.stats.live.checked_sub(size) {
            Some(live) => self.stats.live = live,
            None => {
                self.stats.overflow = true;
                self.stats.live = 0;
            }
        }
    }

    fn insert(&mut self, pointer: usize, size: usize) {
        if self.len == CAPACITY || self.find(pointer).is_some() {
            self.stats.overflow = true;
            return;
        }
        self.entries[self.len] = Entry { pointer, size };
        self.len += 1;
        self.add_live(size);
    }

    fn remove(&mut self, pointer: usize) {
        if let Some(index) = self.find(pointer) {
            self.remove_live(self.entries[index].size);
            self.len -= 1;
            self.entries[index] = self.entries[self.len];
        }
    }

    fn request(&mut self) {
        match self.stats.requests.checked_add(1) {
            Some(requests) => self.stats.requests = requests,
            None => self.stats.overflow = true,
        }
    }

    fn reallocated(&mut self, old: usize, new: usize, size: usize) {
        self.request();
        match self.stats.reallocs.checked_add(1) {
            Some(reallocs) => self.stats.reallocs = reallocs,
            None => self.stats.overflow = true,
        }
        if let Some(index) = self.find(old) {
            // Replace the entry rather than briefly counting both buffers.
            self.remove_live(self.entries[index].size);
            self.entries[index] = Entry { pointer: new, size };
            self.add_live(size);
        } else {
            match self.stats.untracked_reallocs.checked_add(1) {
                Some(reallocs) => self.stats.untracked_reallocs = reallocs,
                None => self.stats.overflow = true,
            }
            self.insert(new, size);
        }
    }
}

static LEDGER: Mutex<Ledger> = Mutex::new(Ledger::new());

fn lock() -> MutexGuard<'static, Ledger> {
    match LEDGER.lock() {
        Ok(ledger) => ledger,
        Err(poisoned) => poisoned.into_inner(),
    }
}

struct CountingAllocator;

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

// The ledger has fixed storage; none of its callbacks allocate or panic.
// Every System operation and its ledger update use the same lock, including
// realloc, so another thread cannot reuse an old address before replacement.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let mut ledger = lock();
        let pointer = unsafe { System.alloc(layout) };
        if ledger.active && !pointer.is_null() {
            ledger.request();
            ledger.insert(pointer as usize, layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let mut ledger = lock();
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if ledger.active && !pointer.is_null() {
            ledger.request();
            ledger.insert(pointer as usize, layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        let mut ledger = lock();
        unsafe { System.dealloc(pointer, layout) };
        if ledger.active {
            ledger.remove(pointer as usize);
        }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let mut ledger = lock();
        let replacement = unsafe { System.realloc(pointer, layout, size) };
        if ledger.active && !replacement.is_null() {
            ledger.reallocated(pointer as usize, replacement as usize, size);
        }
        replacement
    }
}

pub fn begin() {
    let already_active = {
        let mut ledger = lock();
        let already_active = ledger.active;
        ledger.active = false;
        ledger.len = 0;
        ledger.stats = Stats::ZERO;
        if !already_active {
            ledger.active = true;
        }
        already_active
    };
    // On misuse, disable accounting before the assertion can format a panic.
    assert!(!already_active, "allocation window is already active");
}

pub fn end() -> Stats {
    let (was_active, stats) = {
        let mut ledger = lock();
        let was_active = ledger.active;
        ledger.active = false;
        (was_active, ledger.stats)
    };
    assert!(was_active, "allocation window is not active");
    stats
}

pub fn validate() {
    let small = Layout::from_size_align(64, 8).unwrap();
    let large = Layout::from_size_align(128, 8).unwrap();

    begin();
    let pointer = unsafe { ALLOCATOR.alloc(small) };
    if pointer.is_null() {
        end();
        panic!("allocator validation could not allocate 64 bytes");
    }
    let grown = unsafe { ALLOCATOR.realloc(pointer, small, large.size()) };
    if grown.is_null() {
        unsafe { ALLOCATOR.dealloc(pointer, small) };
        end();
        panic!("allocator validation could not grow to 128 bytes");
    }
    let at_peak = { lock().stats };
    unsafe { ALLOCATOR.dealloc(grown, large) };
    let after_free = end();
    assert_eq!(at_peak.peak, 128);
    assert_eq!(at_peak.live, 128);
    assert_eq!(at_peak.requests, 2);
    assert_eq!(at_peak.reallocs, 1);
    assert_eq!(at_peak.untracked_reallocs, 0);
    assert!(!at_peak.overflow);
    assert_eq!(after_free.peak, 128);
    assert_eq!(after_free.live, 0);
    assert!(!after_free.overflow);

    let pre_existing = unsafe { ALLOCATOR.alloc(small) };
    assert!(!pre_existing.is_null());
    begin();
    unsafe { ALLOCATOR.dealloc(pre_existing, small) };
    let ignored_free = end();
    assert_eq!(ignored_free.peak, 0);
    assert_eq!(ignored_free.live, 0);
    assert_eq!(ignored_free.requests, 0);
    assert_eq!(ignored_free.reallocs, 0);
    assert_eq!(ignored_free.untracked_reallocs, 0);
    assert!(!ignored_free.overflow);
}
