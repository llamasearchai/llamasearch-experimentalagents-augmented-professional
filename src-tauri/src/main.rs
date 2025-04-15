#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::Manager;
use tauri_plugin_python::Python;

fn main() {
    // Initialize logging
    env_logger::init();

    tauri::Builder::default()
        .plugin(tauri_plugin_python::init(Python::new(env!("CARGO_MANIFEST_DIR").into()).unwrap()))
        .invoke_handler(tauri::generate_handler![greet]) // Example handler
        .run(tauri::generate_context!("tauri.conf.json"))
        .expect("error while running tauri application");
}

// Example command to test IPC
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
} 