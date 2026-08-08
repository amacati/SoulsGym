# speedhack DLL

Game-speed hook for SoulsGym, working on native Windows and under Proton/Wine.

## Build

### Windwos
Build the DLL with the native MSVC toolchain and then copy the DLL to the `_bin` folder:
```bash
cargo build --release --target "x86_64-pc-windows-msvc"
cp "target/x86_64-pc-windows-msvc/release/speedhack.dll" ../_bin/speedhack.dll
```

### Linux/Proton
Build the DLL with the `cargo-xwin` cross-compiler and then copy the DLL to the `_bin` folder. First install `cargo-xwin`:

```bash
rustup target add x86_64-pc-windows-msvc
cargo install cargo-xwin
```

Then, analogous to Windows, build the DLL and copy it to `_bin`:

```bash
cargo xwin build --release --target "x86_64-pc-windows-msvc"
cp "target/x86_64-pc-windows-msvc/release/speedhack.dll" ../_bin/speedhack.dll
```

## How it gets loaded

On Windows, we inject at runtime via `CreateRemoteThread` + `LoadLibrary`. On Linux/Proton, we load it as a `dinput8.dll` proxy at game launch using `WINEDLLOVERRIDES="dinput8=n,b"`. The real dinput8 exports are forwarded via `exports.def` to `dinput8_hook.dll`, a copy of the active Proton's builtin dinput8, so DirectInput keeps working. `install_proxy()` sets this up.

## Usage
On connect the DLL sends a one-byte status (`0` ok, `1` a timer hook failed). Speed is then set at runtime over the same loopback TCP socket `127.0.0.1:51285` by sending float32 values.

## Wine compatibility
The speedhack rewrites the game module's `Import Address Table` to our modified timer functions. Because we only patch the game's own memory, this works under Wine/Proton without touching Wine internals.
