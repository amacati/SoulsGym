//! SoulsGym game-speed hook. Works on native Windows and under Proton/Wine.
//!
//! The Souls games get their frame delta from Win32 timer functions. This DLL rewrites the game
//! module's Import Address Table entries for those functions to scaled versions we control, so the
//! game believes time runs slower or faster. Speed is set at runtime over a loopback TCP socket.
//!
//! IAT hooking only patches the game's own memory, which is convenient under Wine because we do not
//! have to touch internals. The games statically import QueryPerformanceCounter + GetTickCount
//! (KERNEL32) and timeGetTime (WINMM).
//!
//! Loading:
//!   Windows: injected at runtime via CreateRemoteThread + LoadLibrary. See also speedhack.py.
//!   Proton/Wine: loaded as a dinput8.dll proxy via WINEDLLOVERRIDES. Real dinput8 exports are
//!     forwarded to a copied builtin shipped as dinput8_hook.dll.
//! 
//! In general, this whole hack is highly unsafe. This is expected, we are patching a live process's
//! memory after all. Rust does not help us here, but the tooling is much nicer than if we go with
//! C/C++.

use core::ffi::c_void;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use pelite::pe64::imports::Import;
use pelite::pe64::{Pe, PeView, Va};
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::System::Memory::{VirtualProtect, PAGE_EXECUTE_READWRITE};

/// Loopback port the DLL listens on for speed commands.
const CONTROL_PORT: u16 = 51285;

/// Timer scaling state
/// 
/// Timing functions compute `fake = fake0 + (real - real0) * speed`. On every speed change
/// `real0` and `fake0` are set to the current instant, so the fake clock stays continuous across
/// the change instead of jumping. GetTickCount and timeGetTime are 32-bit and widen to i64 here.
struct Scale {
    real0: i64,
    fake0: i64,
    speed: f64,
}

impl Scale {
    const fn new() -> Self {
        Scale { real0: 0, fake0: 0, speed: 1.0 }
    }

    /// Scale a real timestamp to the real clock.
    fn scale(&self, real: i64) -> i64 {
        self.fake0 + ((real - self.real0) as f64 * self.speed) as i64
    }

    /// Re-anchor to the current instant, then switch to `speed`, keeping the fake clock continuous.
    fn reanchor(&mut self, real: i64, speed: f64) {
        self.fake0 = self.scale(real);
        self.real0 = real;
        self.speed = speed;
    }
}

static QPC: Mutex<Scale> = Mutex::new(Scale::new());
static GTC: Mutex<Scale> = Mutex::new(Scale::new());
static TGT: Mutex<Scale> = Mutex::new(Scale::new());

// Original timer functions, captured at install time.
static ORIG_QPC: AtomicUsize = AtomicUsize::new(0);
static ORIG_GTC: AtomicUsize = AtomicUsize::new(0);
static ORIG_TGT: AtomicUsize = AtomicUsize::new(0);

// True if any timer hook failed. Sent to the controller on connect as one byte (0 ok, 1 failed).
static HOOK_FAILED: AtomicBool = AtomicBool::new(false);

unsafe extern "system" fn scaled_qpc(out: *mut i64) -> i32 {
    unsafe {
        let orig: unsafe extern "system" fn(*mut i64) -> i32 =
            core::mem::transmute(ORIG_QPC.load(Ordering::Relaxed));
        let mut real: i64 = 0;
        let r = orig(&mut real as *mut i64);
        *out = QPC.lock().unwrap().scale(real);
        r
    }
}

unsafe extern "system" fn scaled_gtc() -> u32 {
    unsafe {
        let orig: unsafe extern "system" fn() -> u32 =
            core::mem::transmute(ORIG_GTC.load(Ordering::Relaxed));
        GTC.lock().unwrap().scale(orig() as i64) as u32
    }
}

unsafe extern "system" fn scaled_tgt() -> u32 {
    unsafe {
        let orig: unsafe extern "system" fn() -> u32 =
            core::mem::transmute(ORIG_TGT.load(Ordering::Relaxed));
        TGT.lock().unwrap().scale(orig() as i64) as u32
    }
}

/// Re-anchor every timer to the current instant and switch them to `speed`.
unsafe fn set_game_speed(speed: f64) {
    unsafe {
        let qpc: unsafe extern "system" fn(*mut i64) -> i32 =
            core::mem::transmute(ORIG_QPC.load(Ordering::Relaxed));
        let mut real: i64 = 0;
        qpc(&mut real as *mut i64);
        QPC.lock().unwrap().reanchor(real, speed);
        let gtc: unsafe extern "system" fn() -> u32 =
            core::mem::transmute(ORIG_GTC.load(Ordering::Relaxed));
        GTC.lock().unwrap().reanchor(gtc() as i64, speed);
        let tgt: unsafe extern "system" fn() -> u32 =
            core::mem::transmute(ORIG_TGT.load(Ordering::Relaxed));
        TGT.lock().unwrap().reanchor(tgt() as i64, speed);
    }
}

/// Rewrite the IAT slot for `want_dll!want_fn` in module `base` to `new_fn` and return the original.
///
/// pelite yields the IAT entries as shared references into the live module. We take each slot's
/// offset from `base` and rebuild a writable pointer from `base`, so the write does not inherit the
/// shared reference's provenance.
unsafe fn patch_iat(base: usize, want_dll: &str, want_fn: &str, new_fn: usize) -> Option<usize> {
    let view = unsafe { PeView::module(base as *const u8) };
    for desc in view.imports().ok()? {
        let dll: &[u8] = desc.dll_name().ok()?;
        if !dll.eq_ignore_ascii_case(want_dll.as_bytes()) {
            continue;
        }
        for (import, slot) in desc.int().ok()?.zip(desc.iat().ok()?) {
            let Ok(Import::ByName { name, .. }) = import else {
                continue;
            };
            let name: &[u8] = name;
            if name != want_fn.as_bytes() {
                continue;
            }
            // We jump through a few hoops to get a writable pointer to the IAT slot. Provenance
            // keeps us from using slot directly
            let rva = slot as *const Va as usize - base;
            let slot = (base + rva) as *mut u64;
            let orig = unsafe { *slot } as usize;
            let mut old: u32 = 0;
            unsafe {
                VirtualProtect(slot as *const c_void, 8, PAGE_EXECUTE_READWRITE, &mut old);
                *slot = new_fn as u64;
                VirtualProtect(slot as *const c_void, 8, old, &mut old);
            }
            return Some(orig);
        }
    }
    None
}

/// Hook one timer: patch its IAT slot, store the original, capture its baseline. Returns whether
/// the import was found and patched.
unsafe fn hook_timer(
    base: usize,
    dll: &str,
    func: &str,
    replacement: usize,
    orig_slot: &AtomicUsize,
    baseline: impl FnOnce(usize),
) -> bool {
    match unsafe { patch_iat(base, dll, func, replacement) } {
        Some(orig) => {
            orig_slot.store(orig, Ordering::Relaxed);
            baseline(orig);
            true
        }
        None => false,
    }
}

/// Accept a single controller at a time. On connect, send the one-byte hook status, then apply
/// streamed speed values.
fn control_server() {
    let Ok(listener) = TcpListener::bind(("127.0.0.1", CONTROL_PORT)) else {
        return;
    };
    for stream in listener.incoming() {
        let Ok(mut s) = stream else { continue };
        let _ = s.set_nodelay(true);
        if s.write_all(&[HOOK_FAILED.load(Ordering::Relaxed) as u8]).is_err() {
            continue;
        }
        let mut buf = [0u8; 4];
        while s.read_exact(&mut buf).is_ok() {
            let v = f32::from_le_bytes(buf) as f64;
            if v >= 0.0 {
                unsafe { set_game_speed(v) };
            }
        }
    }
}

fn install() {
    // Calling with null gets the base of the current process's main module, which is the game
    let base = unsafe { GetModuleHandleW(core::ptr::null()) } as usize;
    if base == 0 {
        return;
    }
    let all_hooked;
    unsafe {
        let qpc = hook_timer(
            base,
            "KERNEL32.dll",
            "QueryPerformanceCounter",
            scaled_qpc as *const () as usize,
            &ORIG_QPC,
            |orig| {
                let f: unsafe extern "system" fn(*mut i64) -> i32 = core::mem::transmute(orig);
                let mut t0 = 0i64;
                f(&mut t0);
                QPC.lock().unwrap().reanchor(t0, 1.0);
            },
        );
        let gtc = hook_timer(
            base,
            "KERNEL32.dll",
            "GetTickCount",
            scaled_gtc as *const () as usize,
            &ORIG_GTC,
            |orig| {
                let f: unsafe extern "system" fn() -> u32 = core::mem::transmute(orig);
                GTC.lock().unwrap().reanchor(f() as i64, 1.0);
            },
        );
        let tgt = hook_timer(
            base,
            "WINMM.dll",
            "timeGetTime",
            scaled_tgt as *const () as usize,
            &ORIG_TGT,
            |orig| {
                let f: unsafe extern "system" fn() -> u32 = core::mem::transmute(orig);
                TGT.lock().unwrap().reanchor(f() as i64, 1.0);
            },
        );
        all_hooked = qpc && gtc && tgt;
    }
    if !all_hooked {
        HOOK_FAILED.store(true, Ordering::Relaxed);
    }
    std::thread::spawn(control_server);
}

#[unsafe(no_mangle)]
pub extern "system" fn DllMain(_hinst: isize, reason: u32, _reserved: *mut c_void) -> i32 {
    const DLL_PROCESS_ATTACH: u32 = 1;
    if reason == DLL_PROCESS_ATTACH {
        // Do the work off the loader lock.
        std::thread::spawn(install);
    }
    1
}
