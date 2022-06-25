use core::alloc::Layout;
use core::cell::{Cell, RefCell};
use core::mem;
use core::ptr;
use core::slice;

use crate::hir::HirError;

/// The size of a slab in the arena allocator.
const PAGE: usize = 4096;
const HUGE_PAGE: usize = 2 * 1024 * 1024;

struct Chunk {
    storage: Box<[u8]>,
}

impl Chunk {
    /// Construct a new chunk with the specified length.
    fn new(len: usize) -> Self {
        Self {
            storage: Box::from(vec![0u8; len]),
        }
    }

    /// Get the starting pointer of the chunk.
    fn start(&mut self) -> *mut u8 {
        self.storage.as_mut_ptr()
    }

    /// Get the end pointer of the chunk.
    fn end(&mut self) -> *mut u8 {
        let len = self.storage.len();
        self.storage.as_mut_ptr().wrapping_add(len)
    }
}

/// An arena allocator.
pub struct Arena {
    start: Cell<*mut u8>,
    end: Cell<*mut u8>,
    chunks: RefCell<Vec<Chunk>>,
}

impl Arena {
    /// Construct a new empty arena allocator.
    pub const fn new() -> Self {
        Self {
            start: Cell::new(ptr::null_mut()),
            end: Cell::new(ptr::null_mut()),
            chunks: RefCell::new(Vec::new()),
        }
    }

    /// Allocate a new object of the given type.
    pub(crate) fn alloc<T>(&self, object: T) -> Result<&mut T, HirError> {
        assert!(!mem::needs_drop::<T>());

        let mem = self.alloc_raw(Layout::for_value::<T>(&object)) as *mut T;

        unsafe {
            // Write into uninitialized memory.
            ptr::write(mem, object);
            Ok(&mut *mem)
        }
    }

    /// Allocate a new object of the given type.
    pub(crate) fn alloc_iter<I, T>(&self, iter: I) -> Result<&mut [T], HirError>
    where
        I: IntoIterator<Item = Result<T, HirError>>,
        I::IntoIter: ExactSizeIterator,
    {
        assert!(!mem::needs_drop::<T>(), "cannot allocate drop element");

        let iter = iter.into_iter();

        let len = iter.len();

        if len == 0 {
            return Ok(&mut []);
        }

        let mem = self.alloc_raw(Layout::array::<T>(len).unwrap()) as *mut T;

        unsafe {
            for (i, value) in iter.enumerate() {
                ptr::write(mem.add(i), value?);
            }

            Ok(slice::from_raw_parts_mut(mem, len))
        }
    }

    #[inline]
    fn alloc_raw_without_grow(&self, layout: Layout) -> Option<*mut u8> {
        let start = addr(self.start.get());
        let old_end = self.end.get();
        let end = addr(old_end);

        let align = layout.align();
        let bytes = layout.size();

        let new_end = end.checked_sub(bytes)? & !(align - 1);

        if start > new_end {
            return None;
        }

        let new_end = with_addr(old_end, new_end);
        self.end.set(new_end);
        Some(new_end)
    }

    #[inline]
    pub fn alloc_raw(&self, layout: Layout) -> *mut u8 {
        assert!(layout.size() != 0);

        loop {
            if let Some(a) = self.alloc_raw_without_grow(layout) {
                break a;
            }

            self.grow(layout.size());
        }
    }

    #[cold]
    fn grow(&self, additional: usize) {
        unsafe {
            let mut chunks = self.chunks.borrow_mut();
            let new_cap = additional.max(
                chunks
                    .last()
                    .map(|c| c.storage.len().min(HUGE_PAGE / 2) * 2)
                    .unwrap_or(PAGE),
            );

            let mut chunk = Chunk::new(new_cap);
            self.start.set(chunk.start());
            self.end.set(chunk.end());
            chunks.push(chunk);
        }
    }
}

#[inline]
pub fn addr(this: *mut u8) -> usize {
    this as usize
}

#[inline]
pub fn with_addr(this: *mut u8, a: usize) -> *mut u8 {
    let this_addr = addr(this) as isize;
    let dest_addr = a as isize;
    let offset = dest_addr.wrapping_sub(this_addr);
    this.wrapping_offset(offset)
}
