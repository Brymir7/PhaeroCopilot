use futures_util::stream::StreamExt;
use lazy_static::lazy_static;
use macroquad::prelude::*;
use rdev::{listen, simulate, Event, EventType, Key};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Mutex;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread::{self, sleep};
use std::time::Duration;
use tokio::runtime::Runtime;
const MAX_DISPLAYED_KEYS: usize = 10;
const MODEL_NAME: &str = "llama3";
const MODEL_URL: &str = "http://127.0.0.1:11434/api/chat";
use arboard::Clipboard;
use once_cell::sync::Lazy;
type ClipboardOps = (
    Box<dyn Fn() -> String + Send + Sync>,
    Box<dyn Fn(String) + Send + Sync>,
);
static RUNTIME: Lazy<Runtime> =
    Lazy::new(|| Runtime::new().expect("Failed to create Tokio runtime"));
lazy_static! {
    static ref RESPONSE_BUFFER: Mutex<VecDeque<String>> = Mutex::new(VecDeque::new());
    static ref CLIPBOARD_CONTENT: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
    static ref CHAR_TO_KEY: HashMap<char, Key> = {
        let mut m = HashMap::new();
        m.insert('a', Key::KeyA);
        m.insert('b', Key::KeyB);
        m.insert('c', Key::KeyC);
        m.insert('d', Key::KeyD);
        m.insert('e', Key::KeyE);
        m.insert('f', Key::KeyF);
        m.insert('g', Key::KeyG);
        m.insert('h', Key::KeyH);
        m.insert('i', Key::KeyI);
        m.insert('j', Key::KeyJ);
        m.insert('k', Key::KeyK);
        m.insert('l', Key::KeyL);
        m.insert('m', Key::KeyM);
        m.insert('n', Key::KeyN);
        m.insert('o', Key::KeyO);
        m.insert('p', Key::KeyP);
        m.insert('q', Key::KeyQ);
        m.insert('r', Key::KeyR);
        m.insert('s', Key::KeyS);
        m.insert('t', Key::KeyT);
        m.insert('u', Key::KeyU);
        m.insert('v', Key::KeyV);
        m.insert('w', Key::KeyW);
        m.insert('x', Key::KeyX);
        m.insert('y', Key::KeyY);
        m.insert('z', Key::KeyZ);
        m.insert(' ', Key::Space);
        m.insert('1', Key::Num1);
        m.insert('2', Key::Num2);
        m.insert('3', Key::Num3);
        m.insert('4', Key::Num4);
        m.insert('5', Key::Num5);
        m.insert('6', Key::Num6);
        m.insert('7', Key::Num7);
        m.insert('8', Key::Num8);
        m.insert('9', Key::Num9);
        m.insert('0', Key::Num0);
        m
    };
    static ref HOTKEYS: HashMap<KeyCombination, fn(Arc<AtomicBool>, ClipboardOps)> = {
        let mut m = HashMap::new();
        m.insert(
            KeyCombination::new(vec![Key::ControlLeft, Key::Alt, Key::KeyI]),
            insert_text as fn(Arc<AtomicBool>, ClipboardOps),
        );
        m.insert(
            KeyCombination::new(vec![Key::ControlLeft, Key::Alt, Key::KeyC]),
            cancel_operation as fn(Arc<AtomicBool>, ClipboardOps),
        );
        m.insert(
            KeyCombination::new(vec![Key::ControlLeft, Key::Alt, Key::KeyO]),
            call_ollama as fn(Arc<AtomicBool>, ClipboardOps),
        );
        m.insert(
            KeyCombination::new(vec![Key::ControlLeft, Key::Alt, Key::KeyF]),
            format_clipboard as fn(Arc<AtomicBool>, ClipboardOps),
        );
        m
    };
    static ref SHOULD_CONTINUE: Arc<AtomicBool> = Arc::new(AtomicBool::new(true));
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct KeyCombination {
    keys: Vec<Key>,
}

impl KeyCombination {
    fn new(keys: Vec<Key>) -> Self {
        KeyCombination { keys }
    }

    fn is_triggered(&self, event: &Event, last_keys: &Vec<Key>) -> bool {
        if let EventType::KeyPress(trigger_key) = event.event_type {
            if self.keys.contains(&trigger_key) {
                return self.keys.iter().all(|key| last_keys.contains(key));
            }
        }
        false
    }
}
fn send(event_type: &EventType) {
    let delay = Duration::from_millis(20);
    match simulate(event_type) {
        Ok(()) => (),
        Err(_) => {
            println!("We could not send {:?}", event_type);
        }
    }
    thread::sleep(delay);
}

fn cancel_operation(should_continue: Arc<AtomicBool>, _clipboard_ops: ClipboardOps) {
    println!("Operation cancelled");
    should_continue.store(false, Ordering::SeqCst);
}

fn insert_text(should_continue: Arc<AtomicBool>, _clipboard_ops: ClipboardOps) {
    while let Some(word) = RESPONSE_BUFFER.lock().unwrap().pop_front() {
        println!("Inserting word: {}", word);
        for ch in word.chars() {
            if !should_continue.load(Ordering::SeqCst) {
                println!("Operation cancelled");
                return;
            }
            if let Some(&key) = CHAR_TO_KEY.get(&ch) {
                send(&EventType::KeyPress(key));
                send(&EventType::KeyRelease(key));
            }
        }
    }
}
fn format_clipboard(should_continue: Arc<AtomicBool>, clipboard_ops: ClipboardOps) {
    if !should_continue.load(Ordering::SeqCst) {
        println!("Operation cancelled");
        return;
    }
    let system_prompt = "Format the following text neatly with proper indentation, line breaks, and punctuation. Do not include any explanations or comments. Only return the formatted text. Fix any spelling or grammatical errors.";
    let (read_clipboard, update_clipboard) = clipboard_ops;
    let content = read_clipboard();

    thread::spawn(move || {
        RUNTIME.block_on(async {
            tokio::task::spawn_blocking(move || {
                ollama_call(system_prompt, content, Some(update_clipboard));
            })
            .await
            .expect("Task panicked");
        })
    });
}
fn format_text_remove_explanations(text: &str) -> String {
    // Define the prefix that you want to remove
    let prefix = "Here's the formatted text:\n\n";

    // Check if the text starts with the prefix
    if let Some(stripped) = text.strip_prefix(prefix) {
        stripped.to_string()
    } else {
        text.to_string() // If the prefix is not found, return the original text
    }
}
fn ollama_call(
    system_prompt: &'static str,
    user_prompt: String,
    update_clipboard: Option<Box<dyn Fn(String) + Send + Sync>>,
) {
    println!("Starting ollama_call function");
    let loading_indicator = Arc::new(AtomicBool::new(true));
    let loading_indicator_clone = Arc::clone(&loading_indicator);
    thread::spawn(move || {
        let loading_text = "Loading...";
        let mut current_text = String::new();
        while loading_indicator_clone.load(Ordering::SeqCst) {
            for ch in loading_text.chars() {
                if !loading_indicator_clone.load(Ordering::SeqCst) {
                    break;
                }
                current_text.push(ch);
                if let Some(&key) = CHAR_TO_KEY.get(&ch) {
                    send(&EventType::KeyPress(key));
                    send(&EventType::KeyRelease(key));
                }
                sleep(Duration::from_millis(200));
            }
            sleep(Duration::from_millis(500));
            while !current_text.is_empty() && loading_indicator_clone.load(Ordering::SeqCst) {
                send(&EventType::KeyPress(Key::Backspace));
                send(&EventType::KeyRelease(Key::Backspace));
                current_text.pop();
            }
            sleep(Duration::from_millis(300));
        }
        for _ in 0..current_text.len() {
            send(&EventType::KeyPress(Key::Backspace));
            send(&EventType::KeyRelease(Key::Backspace));
        }
    });

    let future = async move {
        println!("Creating HTTP client");
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build();
        if let Err(e) = client {
            eprintln!("Failed to create client: {:?}", e);
            loading_indicator.store(false, Ordering::SeqCst);
            return;
        }
        let client = client.unwrap();

        let request_body = serde_json::json!({
            "model": MODEL_NAME,
            "messages": [
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": user_prompt }
            ]
        });

        println!("Sending request to Ollama API");
        match client
            .post(MODEL_URL)
            .header("Content-Type", "application/json")
            .body(serde_json::to_string(&request_body).unwrap())
            .send()
            .await
        {
            Ok(response) => {
                println!("Received response from Ollama API");
                println!("Response status: {}", response.status());
                let mut stream = response.bytes_stream();
                let mut full_response = String::new();
                while let Some(item) = stream.next().await {
                    match item {
                        Ok(bytes) => {
                            if let Ok(text) = String::from_utf8(bytes.to_vec()) {
                                println!("Received data: {}", text);
                                if let Ok(json) = serde_json::from_str::<Value>(&text) {
                                    if let Some(content) = json["message"]["content"].as_str() {
                                        let mut buffer = RESPONSE_BUFFER.lock().unwrap();
                                        buffer.push_back(content.to_string());
                                        full_response.push_str(content);
                                    } else {
                                        println!("No content field in JSON: {:?}", json);
                                    }
                                } else {
                                    println!("Failed to parse JSON: {}", text);
                                }
                            } else {
                                println!("Received non-UTF8 data");
                            }
                        }
                        Err(err) => {
                            eprintln!("Stream error: {}", err);
                        }
                    }
                }
                if let Some(update_clipboard) = update_clipboard {
                    let formatted_text: String = format_text_remove_explanations(&full_response);
                    update_clipboard(formatted_text);
                }
            }

            Err(err) => {
                eprintln!("Request error: {:?}", err);
                if let Some(status) = err.status() {
                    eprintln!("HTTP Status: {}", status);
                }
                if let Some(url) = err.url() {
                    eprintln!("URL: {}", url);
                }
            }
        }
        println!("ollama_call function completed");
        loading_indicator.store(false, Ordering::SeqCst);
    };

    RUNTIME.spawn(future);
}

fn call_ollama(should_continue: Arc<AtomicBool>, clipboard_ops: ClipboardOps) {
    let (read_clipboard, _) = clipboard_ops;
    if !should_continue.load(Ordering::SeqCst) {
        println!("Operation cancelled");
        return;
    }
    let system_prompt = "You are a helpful AI assistant.";
    let clipboard_content = read_clipboard();
    thread::spawn(move || {
        RUNTIME.block_on(async {
            tokio::task::spawn_blocking(move || {
                ollama_call(system_prompt, clipboard_content, None);
            })
            .await
            .expect("Task panicked");
        })
    });
}

struct PhaeroPredict {
    last_keys: Vec<Key>,
    keyboard_event_rx: Receiver<Event>,
    clipboard: Clipboard,
    clipboard_content: Arc<Mutex<String>>,
    clipboard_tx: Sender<String>,
    clipboard_rx: Receiver<String>,
}

impl PhaeroPredict {
    fn new() -> Arc<Mutex<Self>> {
        let (event_tx, event_rx) = channel();
        let (clipboard_tx, clipboard_rx) = channel();

        thread::spawn(move || {
            listen(move |event| {
                if let EventType::KeyPress(_) | EventType::KeyRelease(_) = event.event_type {
                    let _ = event_tx.send(event);
                }
            })
            .expect("Could not listen for events");
        });

        let mut clipboard = Clipboard::new().unwrap();
        let clipboard_content = Arc::new(Mutex::new(clipboard.get_text().unwrap_or_default()));

        let phaero_predict = Arc::new(Mutex::new(PhaeroPredict {
            last_keys: Vec::new(),
            keyboard_event_rx: event_rx,
            clipboard,
            clipboard_content: Arc::clone(&clipboard_content),
            clipboard_tx,
            clipboard_rx,
        }));

        let clipboard_content_clone = Arc::clone(&clipboard_content);
        thread::spawn(move || loop {
            if let Ok(mut clipboard) = Clipboard::new() {
                if let Ok(content) = clipboard.get_text() {
                    let mut current_content = clipboard_content_clone.lock().unwrap();
                    if *current_content != content {
                        *current_content = content.clone();
                    }
                }
            }
            sleep(Duration::from_millis(100));
        });

        phaero_predict
    }

    fn update(&mut self) {
        while let Ok(content) = self.clipboard_rx.try_recv() {
            let mut clipboard_content = self.clipboard_content.lock().unwrap();
            if *clipboard_content != content {
                *clipboard_content = content.clone();
                let _ = self.clipboard.set_text(content);
            }
        }
        while let Ok(event) = self.keyboard_event_rx.try_recv() {
            match event.event_type {
                EventType::KeyPress(key) => {
                    if let Some(&last_key) = self.last_keys.last() {
                        if last_key == key {
                            continue;
                        }
                    }
                    self.last_keys.push(key);
                    if self.last_keys.len() > MAX_DISPLAYED_KEYS {
                        self.last_keys.remove(0);
                    }
                    SHOULD_CONTINUE.store(true, Ordering::SeqCst);

                    for (hotkey, function) in HOTKEYS.iter() {
                        if hotkey.is_triggered(&event, &self.last_keys) {
                            let should_continue = Arc::clone(&SHOULD_CONTINUE);
                            let clipboard_content = Arc::clone(&self.clipboard_content);
                            let read_clipboard =
                                Box::new(move || clipboard_content.lock().unwrap().clone())
                                    as Box<dyn Fn() -> String + Send + Sync>;

                            let clipboard_tx = Arc::new(self.clipboard_tx.clone());

                            let update_clipboard = {
                                let clipboard_tx = Arc::clone(&clipboard_tx);
                                Box::new(move |content: String| {
                                    let _ = clipboard_tx.send(content.clone());
                                })
                                    as Box<dyn Fn(String) + Send + Sync>
                            };

                            let clipboard_ops = (read_clipboard, update_clipboard);

                            sleep(Duration::from_millis(100));
                            self.last_keys.clear();
                            thread::spawn(move || {
                                function(should_continue, clipboard_ops);
                            });
                        }
                    }
                }
                EventType::KeyRelease(key) => {
                    self.last_keys.retain(|&k| k != key);
                }
                _ => {}
            }
        }
    }

    fn draw(&self) {
        clear_background(BLACK);

        // Draw last keys
        for (i, key) in self.last_keys.iter().enumerate() {
            let text = format!("Key: {:?}", key);
            draw_text(&text, 20.0, 40.0 + i as f32 * 30.0, 30.0, WHITE);
        }

        // Draw response buffer and clipboard content
        let mut y_pos_text = 50.0 + self.last_keys.len() as f32 * 30.0;
        let mut x_pos_text = 20.0;
        let mut y_pos_offset = 0.0;
        const MAX_WIDTH: f32 = 500.0;

        draw_text("Response Buffer:", 20.0, y_pos_text, 30.0, YELLOW);
        y_pos_text += 40.0;

        for word in RESPONSE_BUFFER.lock().unwrap().iter() {
            if x_pos_text > MAX_WIDTH {
                x_pos_text = 20.0;
                y_pos_offset += 45.0;
            }
            draw_text(word, x_pos_text, y_pos_text + y_pos_offset, 30.0, WHITE);
            x_pos_text += word.len() as f32 * 15.0;
        }

        y_pos_text += y_pos_offset + 60.0;
        x_pos_text = 20.0;
        y_pos_offset = 0.0;

        draw_text("Clipboard Content:", 20.0, y_pos_text, 30.0, GREEN);
        y_pos_text += 40.0;

        for content in self
            .clipboard_content
            .lock()
            .unwrap()
            .split_ascii_whitespace()
        {
            if x_pos_text > MAX_WIDTH {
                x_pos_text = 20.0;
                y_pos_offset += 45.0;
            }
            draw_text(content, x_pos_text, y_pos_text + y_pos_offset, 30.0, WHITE);
            x_pos_text += content.len() as f32 * 15.0;
        }
    }
}
#[macroquad::main("Keyboard Input Display")]
async fn main() {
    let phaero_predict = PhaeroPredict::new();
    loop {
        let mut phaero_predict_guard = phaero_predict.lock().unwrap();
        phaero_predict_guard.update();
        phaero_predict_guard.draw();
        next_frame().await;
    }
}
