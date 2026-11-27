//! `TUXBOX_TRACE_GPU_MEMORY=1`: live native D3D12 memory by category.
//!
//! Local diagnostic. Every hook is one cached bool check when the switch is
//! unset. When set, each tracked native heap or resource carries a private
//! data interface whose final release subtracts its bytes, so the counters
//! follow the native object's lifetime (including references the host driver
//! holds), not the guest proxy's.
//!
//! Which numbers add up to device memory the provider asked for:
//!   * `heaps` (default/upload/readback/custom/render), `committed` and
//!     `internal` are separate allocations and ADD UP. Their sum is printed
//!     as `device total`.
//!   * `placed` resources live inside a heap already counted (a native heap
//!     or an imported one). They are reported for attribution only.
//!   * `imported` heaps wrap host memory the guest already owns. They cost
//!     no new memory, but the driver pins those pages (amdgpu userptr/GTT).
//!   * `reserved` is virtual size. Its pages come from heaps via tile
//!     mappings, which are counted as they are made and never unmapped here
//!     (`cumulative`).
//!   * `host ram` is system memory the provider itself allocates to copy
//!     guest memory. It is not a D3D12 object.
//!
//! Sizes are approximate: buffers count their width, textures count
//! `GetResourceAllocationInfo` for the description the native resource
//! reports, falling back to the copyable footprint. Host driver overhead
//! (descriptor heaps, pipelines, per-image metadata, its own suballocator
//! slack and scratch buffers) is not visible from here.

use core::ffi::c_void;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Instant;
use windows::Win32::Foundation::{E_NOINTERFACE, E_POINTER, S_OK};
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{GUID, HRESULT, IUnknown, Interface};

#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(usize)]
pub(crate) enum Slot {
    // Native heaps the guest or the provider created: real device memory.
    HeapDefault,
    HeapUpload,
    HeapReadback,
    HeapCustom,
    /// Provider heaps for images the guest placed at a guest address.
    HeapRender,
    /// Heaps opened over host memory (guest pages or a shadow copy of them).
    HeapImported,
    // Committed resources the guest asked for.
    CommittedTexture,
    CommittedTarget,
    CommittedUav,
    CommittedBuffer,
    /// Guest buffers on upload, readback or custom heaps (CPU-visible memory).
    CommittedCpuBuffer,
    // Committed resources the provider creates for itself.
    /// Native image holding a copy of a texture placed in guest memory.
    GuestTextureCopy,
    /// Upload buffers mirroring guest memory for one CBV or SRV.
    ViewMirror,
    /// Device buffer standing in for a placed buffer that could not be imported.
    Standin,
    /// Upload and readback buffers for texture refresh, publication and copies.
    Staging,
    Blas,
    Internal,
    // Placed in a heap counted above.
    PlacedTexture,
    PlacedTarget,
    PlacedUav,
    PlacedBuffer,
    PlacedImported,
    Reserved,
    // System memory, not D3D12 objects.
    ShadowRam,
    SnapshotRam,
}

const SLOTS: usize = Slot::SnapshotRam as usize + 1;

/// Who asked for a committed resource.
#[derive(Clone, Copy)]
pub(crate) enum Origin {
    Guest,
    ViewMirror,
    Standin,
    Staging,
    Blas,
    Internal,
}

#[derive(Default)]
pub(crate) struct Counter {
    live: AtomicU64,
    bytes: AtomicU64,
    created: AtomicU64,
    destroyed: AtomicU64,
}

impl Counter {
    pub(crate) const fn new() -> Self {
        Self { live: AtomicU64::new(0), bytes: AtomicU64::new(0), created: AtomicU64::new(0), destroyed: AtomicU64::new(0) }
    }

    pub(crate) fn add(&self, bytes: u64) {
        self.live.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(bytes, Ordering::Relaxed);
        self.created.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn sub(&self, bytes: u64) {
        self.live.fetch_sub(1, Ordering::Relaxed);
        self.bytes.fetch_sub(bytes, Ordering::Relaxed);
        self.destroyed.fetch_add(1, Ordering::Relaxed);
    }

    /// Live objects, live bytes, created, destroyed.
    pub(crate) fn read(&self) -> [u64; 4] {
        [&self.live, &self.bytes, &self.created, &self.destroyed].map(|v| v.load(Ordering::Relaxed))
    }
}

static COUNTERS: [Counter; SLOTS] = [const { Counter::new() }; SLOTS];
// Tile mapping calls and tiles (64 KiB each) bound to heap memory. Cumulative.
static TILE_CALLS: AtomicU64 = AtomicU64::new(0);
static TILES: AtomicU64 = AtomicU64::new(0);

pub(crate) fn enabled() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| {
        std::env::var("TUXBOX_TRACE_GPU_MEMORY").as_deref() == Ok("1")
    });
    *ON
}

// Private data key of the marker. One per tracked native object.
const MARKER: GUID = GUID::from_u128(0x5f0d2b6e_93a1_4c57_a2be_71d0c4e8a3f6);

#[repr(C)]
struct MarkerVtable {
    query: unsafe extern "system" fn(*mut Marker, *const GUID, *mut *mut c_void) -> HRESULT,
    add_ref: unsafe extern "system" fn(*mut Marker) -> u32,
    release: unsafe extern "system" fn(*mut Marker) -> u32,
}

/// A bare IUnknown. The native object releases it when it is destroyed.
#[repr(C)]
struct Marker {
    vtable: &'static MarkerVtable,
    refs: AtomicU32,
    slot: Slot,
    bytes: u64,
}

static MARKER_VTABLE: MarkerVtable = MarkerVtable { query: marker_query, add_ref: marker_add_ref, release: marker_release };

unsafe extern "system" fn marker_query(this: *mut Marker, iid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    unsafe { out.write(core::ptr::null_mut()) };
    if iid.is_null() {
        return E_POINTER;
    }
    if unsafe { iid.read_unaligned() } != IUnknown::IID {
        return E_NOINTERFACE;
    }
    unsafe {
        marker_add_ref(this);
        out.write(this.cast());
    }
    S_OK
}

unsafe extern "system" fn marker_add_ref(this: *mut Marker) -> u32 {
    unsafe { (*this).refs.fetch_add(1, Ordering::Relaxed) + 1 }
}

unsafe extern "system" fn marker_release(this: *mut Marker) -> u32 {
    let refs = unsafe { (*this).refs.fetch_sub(1, Ordering::AcqRel) } - 1;
    if refs == 0 {
        // SAFETY: the last reference is gone and the marker came from a Box.
        let marker = unsafe { Box::from_raw(this) };
        COUNTERS[marker.slot as usize].sub(marker.bytes);
    }
    refs
}

fn track(object: &ID3D12Object, slot: Slot, bytes: u64) {
    track_quiet(object, slot, bytes);
    tick();
}

/// Count without reporting, for callers that hold a lock the report takes.
fn track_quiet(object: &ID3D12Object, slot: Slot, bytes: u64) {
    COUNTERS[slot as usize].add(bytes);
    let marker = Box::into_raw(Box::new(Marker { vtable: &MARKER_VTABLE, refs: AtomicU32::new(1), slot, bytes }));
    // SAFETY: the marker is a live COM object with one reference, which this
    // IUnknown owns. The native object takes its own reference on success. On
    // failure the drop below is the final release and undoes the count.
    let marker = unsafe { IUnknown::from_raw(marker.cast()) };
    let _ = unsafe { object.SetPrivateDataInterface(&MARKER, &marker) };
    drop(marker);
}

/// Texture class from the usage flags: (plain, render target or depth, UAV only).
fn class(desc: &D3D12_RESOURCE_DESC, slots: [Slot; 4]) -> Slot {
    let target = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    if desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER {
        slots[3]
    } else if (desc.Flags & target).0 != 0 {
        slots[1]
    } else if desc.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) {
        slots[2]
    } else {
        slots[0]
    }
}

/// Approximate bytes behind a native resource (see the module comment).
fn size(resource: &ID3D12Resource, desc: &D3D12_RESOURCE_DESC) -> u64 {
    if desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER {
        return desc.Width;
    }
    let mut device: Option<ID3D12Device> = None;
    // SAFETY: the resource is live and the output is a valid interface slot.
    if unsafe { resource.GetDevice(&mut device) }.is_err() {
        return 0;
    }
    let Some(device) = device else { return 0 };
    // SAFETY: the description is borrowed for the calls only.
    let info = unsafe { device.GetResourceAllocationInfo(0, &[*desc]) };
    if info.SizeInBytes != u64::MAX {
        return info.SizeInBytes;
    }
    let mips = u32::from(desc.MipLevels.max(1));
    let slices = if desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D { 1 } else { u32::from(desc.DepthOrArraySize.max(1)) };
    let mut total = 0;
    unsafe { device.GetCopyableFootprints(desc, 0, mips * slices, 0, None, None, None, Some(&mut total)) };
    if total == u64::MAX { 0 } else { total }
}

/// A heap created with device memory. `render` marks the provider's own heaps
/// for images placed at guest addresses.
pub(crate) fn native_heap(heap: &ID3D12Heap, render: bool) {
    if !enabled() {
        return;
    }
    // SAFETY: the heap is live.
    let desc = unsafe { heap.GetDesc() };
    let slot = match desc.Properties.Type {
        _ if render => Slot::HeapRender,
        D3D12_HEAP_TYPE_DEFAULT => Slot::HeapDefault,
        D3D12_HEAP_TYPE_UPLOAD => Slot::HeapUpload,
        D3D12_HEAP_TYPE_READBACK => Slot::HeapReadback,
        _ => Slot::HeapCustom,
    };
    track(heap, slot, desc.SizeInBytes);
}

/// A heap opened over existing host memory.
pub(crate) fn imported_heap(heap: &ID3D12Heap) {
    if !enabled() {
        return;
    }
    // SAFETY: the heap is live.
    track(heap, Slot::HeapImported, unsafe { heap.GetDesc() }.SizeInBytes);
}

pub(crate) fn committed(resource: &ID3D12Resource, origin: Origin) {
    if !enabled() {
        return;
    }
    // SAFETY: the resource is live.
    let desc = unsafe { resource.GetDesc() };
    let slot = match origin {
        Origin::Guest => {
            let mut heap = D3D12_HEAP_PROPERTIES::default();
            // SAFETY: the output is a valid record. Failure leaves DEFAULT's zero.
            let _ = unsafe { resource.GetHeapProperties(Some(&mut heap), None) };
            let cpu = desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER && heap.Type != D3D12_HEAP_TYPE_DEFAULT;
            if cpu {
                Slot::CommittedCpuBuffer
            } else {
                class(&desc, [Slot::CommittedTexture, Slot::CommittedTarget, Slot::CommittedUav, Slot::CommittedBuffer])
            }
        }
        Origin::ViewMirror => Slot::ViewMirror,
        Origin::Standin => Slot::Standin,
        Origin::Staging => Slot::Staging,
        Origin::Blas => Slot::Blas,
        Origin::Internal => Slot::Internal,
    };
    track(resource, slot, size(resource, &desc));
}

/// The native image copying a texture placed in guest memory. Returns its
/// approximate bytes when tracing, for the per-image usage record. Does not
/// report: the caller holds the placed image table's lock.
pub(crate) fn guest_texture_copy(resource: &ID3D12Resource) -> Option<u64> {
    if !enabled() {
        return None;
    }
    // SAFETY: the resource is live.
    let bytes = size(resource, &unsafe { resource.GetDesc() });
    track_quiet(resource, Slot::GuestTextureCopy, bytes);
    Some(bytes)
}

/// Live native guest texture copies and their bytes.
pub(crate) fn guest_texture_copies() -> [u64; 2] {
    let [live, bytes, ..] = COUNTERS[Slot::GuestTextureCopy as usize].read();
    [live, bytes]
}

/// Live bytes of one host RAM category (an upper bound for snapshots).
pub(crate) fn host_ram_bytes(slot: Slot) -> u64 {
    COUNTERS[slot as usize].read()[1]
}

/// A resource placed in a heap. Its bytes are already counted with the heap.
pub(crate) fn placed(resource: &ID3D12Resource, imported: bool) {
    if !enabled() {
        return;
    }
    // SAFETY: the resource is live.
    let desc = unsafe { resource.GetDesc() };
    let slot = if imported {
        Slot::PlacedImported
    } else {
        class(&desc, [Slot::PlacedTexture, Slot::PlacedTarget, Slot::PlacedUav, Slot::PlacedBuffer])
    };
    track(resource, slot, size(resource, &desc));
}

/// A reserved resource: virtual size only.
pub(crate) fn reserved(resource: &ID3D12Resource) {
    if !enabled() {
        return;
    }
    // SAFETY: the resource is live.
    let desc = unsafe { resource.GetDesc() };
    track(resource, Slot::Reserved, size(resource, &desc));
}

/// Tiles bound to heap memory by one mapping call. Never decremented.
pub(crate) fn tiles_mapped(tiles: u64) {
    if !enabled() {
        return;
    }
    TILE_CALLS.fetch_add(1, Ordering::Relaxed);
    TILES.fetch_add(tiles, Ordering::Relaxed);
}

/// System memory the provider allocated to copy guest memory.
pub(crate) fn host_ram(slot: Slot, bytes: u64) {
    if enabled() {
        COUNTERS[slot as usize].add(bytes);
    }
}

pub(crate) fn host_ram_released(slot: Slot, bytes: u64) {
    if enabled() {
        COUNTERS[slot as usize].sub(bytes);
    }
}

const PERIOD_MS: u64 = 5000;
static START: LazyLock<Instant> = LazyLock::new(Instant::now);
static LAST_MS: AtomicU64 = AtomicU64::new(0);
static REPORTS: AtomicU64 = AtomicU64::new(0);

/// Milliseconds since the trace started.
pub(crate) fn now_ms() -> u64 {
    START.elapsed().as_millis() as u64
}

/// Called from creation hooks and every queue submission.
pub(crate) fn tick() {
    if !enabled() {
        return;
    }
    let now = START.elapsed().as_millis() as u64;
    let last = LAST_MS.load(Ordering::Relaxed);
    if now.saturating_sub(last) < PERIOD_MS
        || LAST_MS.compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed).is_err()
    {
        return;
    }
    report("periodic");
}

/// Report now, for example when the device is lost. Bounded so a failing
/// loop cannot flood the log.
pub(crate) fn report_now(reason: &str) {
    static FORCED: AtomicU32 = AtomicU32::new(0);
    if enabled() && FORCED.fetch_add(1, Ordering::Relaxed) < 8 {
        report(reason);
    }
}

pub(crate) fn mib(bytes: u64) -> String {
    format!("{}.{}M", bytes >> 20, ((bytes & 0xf_ffff) * 10) >> 20)
}

/// `name=live/bytes` for one category.
fn field(name: &str, slot: Slot) -> String {
    let [live, bytes, ..] = COUNTERS[slot as usize].read();
    format!("{name}={live}/{}", mib(bytes))
}

/// `name=created/destroyed` summed over categories.
fn churn(name: &str, slots: &[Slot]) -> String {
    let (created, destroyed) = slots.iter().fold((0, 0), |(c, d), slot| {
        let [_, _, created, destroyed] = COUNTERS[*slot as usize].read();
        (c + created, d + destroyed)
    });
    format!("{name}={created}/{destroyed}")
}

const HEAPS: [Slot; 5] = [Slot::HeapDefault, Slot::HeapUpload, Slot::HeapReadback, Slot::HeapCustom, Slot::HeapRender];
const COMMITTED: [Slot; 5] = [Slot::CommittedTexture, Slot::CommittedTarget, Slot::CommittedUav, Slot::CommittedBuffer, Slot::CommittedCpuBuffer];
const INTERNAL: [Slot; 6] = [Slot::GuestTextureCopy, Slot::ViewMirror, Slot::Standin, Slot::Staging, Slot::Blas, Slot::Internal];
const PLACED: [Slot; 5] = [Slot::PlacedTexture, Slot::PlacedTarget, Slot::PlacedUav, Slot::PlacedBuffer, Slot::PlacedImported];

fn total(slots: &[Slot]) -> u64 {
    slots.iter().map(|slot| COUNTERS[*slot as usize].read()[1]).sum()
}

// Wine clips long debug strings, so one report is several short lines that
// share a number. Each value is `live objects/MiB` unless labelled otherwise.
fn report(reason: &str) {
    if !crate::diagnostics::logging() {
        return;
    }
    let id = REPORTS.fetch_add(1, Ordering::Relaxed) + 1;
    let seconds = START.elapsed().as_secs();
    let log = |line: String| crate::diagnostics::log(&format!("gpu memory #{id} {line}"));
    let (heaps, committed, internal) = (total(&HEAPS), total(&COMMITTED), total(&INTERNAL));
    log(format!(
        "t={seconds}s {reason}: device total={} (heaps {} + committed {} + internal {}); imported, placed and reserved are not in it",
        mib(heaps + committed + internal), mib(heaps), mib(committed), mib(internal),
    ));
    log(format!(
        "heaps: {} {} {} {} {} | {} (guest pages, pinned not new)",
        field("default", Slot::HeapDefault), field("upload", Slot::HeapUpload),
        field("readback", Slot::HeapReadback), field("custom", Slot::HeapCustom),
        field("render", Slot::HeapRender), field("imported", Slot::HeapImported),
    ));
    log(format!(
        "committed: {} {} {} {} {}",
        field("tex", Slot::CommittedTexture), field("rt_ds", Slot::CommittedTarget),
        field("uav_tex", Slot::CommittedUav), field("buf", Slot::CommittedBuffer),
        field("cpu_buf", Slot::CommittedCpuBuffer),
    ));
    log(format!(
        "internal: {} {} {} {} {} {}",
        field("guest_tex_copy", Slot::GuestTextureCopy), field("view_mirror", Slot::ViewMirror),
        field("standin", Slot::Standin), field("staging", Slot::Staging),
        field("blas", Slot::Blas), field("other", Slot::Internal),
    ));
    log(format!(
        "placed (inside heaps): {} {} {} {} {} | {} (virtual) tiles_mapped cumulative={} calls/{}",
        field("tex", Slot::PlacedTexture), field("rt_ds", Slot::PlacedTarget),
        field("uav_tex", Slot::PlacedUav), field("buf", Slot::PlacedBuffer),
        field("on_imported", Slot::PlacedImported), field("reserved", Slot::Reserved),
        TILE_CALLS.load(Ordering::Relaxed), mib(TILES.load(Ordering::Relaxed) << 16),
    ));
    log(format!(
        "host ram: {} {} (upper bound) | created/destroyed: {} {} {} {} {} {}",
        field("shadow", Slot::ShadowRam), field("tex_snapshot", Slot::SnapshotRam),
        churn("heaps", &HEAPS), churn("imported", &[Slot::HeapImported]),
        churn("committed", &COMMITTED), churn("internal", &INTERNAL),
        churn("placed", &PLACED), churn("reserved", &[Slot::Reserved]),
    ));
    crate::device::placement::usage_report(&log);
}

#[cfg(test)]
#[path = "../tests/diagnostics/gpu_memory.rs"]
mod tests;
