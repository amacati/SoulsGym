fn main() {
    // Export the dinput8 API as forwarders so that, when this DLL is used as a dinput8.dll proxy
    // under Proton/Wine, the game's DirectInput calls still reach the real dinput8. Unused when
    // injected as speedhack.dll under Windows.
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let def = std::path::Path::new(&manifest).join("exports.def");
    println!("cargo::rustc-cdylib-link-arg=/DEF:{}", def.display());
    println!("cargo::rerun-if-changed=exports.def");
}
