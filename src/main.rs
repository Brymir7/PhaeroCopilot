use futures_util::stream::StreamExt;
use lazy_static::lazy_static;
use macroquad::prelude::*;
use rdev::{listen, simulate, Event, EventType, Key};
use serde_json::Value;
use std::cmp::max;
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
const MODEL_NAME: &str = "gemma2:27b";
const MODEL_URL: &str = "http://127.0.0.1:11434/api/chat";
static COMPANION_TEXTURE: Lazy<Texture2D> = Lazy::new(|| {
    Texture2D::from_file_with_format(
        include_bytes!("../images/PhaeroCompanion.png"),
        Some(ImageFormat::Png),
    )
});
const SYS_FORMAT_PROMPT: &str = "Format the following text neatly with proper indentation, line breaks, and punctuation. Do not include any explanations or comments. Only return the formatted text. Fix any spelling or grammatical errors.";
const SYS_PREDICT_NEXT_WORD: &str = "You are an AI assistant specialized in predicting the next word in a sequence. Provide only the single most likely next word, with no additional explanation or punctuation.";
const SYS_PREDICT_NEXT_SENTENCE: &str = "You are an AI assistant specialized in predicting the next sentence in a sequence. Provide only the most likely next sentence, with no additional explanation or punctuation.";
use arboard::Clipboard;
use once_cell::sync::Lazy;

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
    static ref KEY_TO_CHAR: HashMap<Key, char> = {
        let mut m = HashMap::new();
        m.insert(Key::KeyA, 'a');
        m.insert(Key::KeyB, 'b');
        m.insert(Key::KeyC, 'c');
        m.insert(Key::KeyD, 'd');
        m.insert(Key::KeyE, 'e');
        m.insert(Key::KeyF, 'f');
        m.insert(Key::KeyG, 'g');
        m.insert(Key::KeyH, 'h');
        m.insert(Key::KeyI, 'i');
        m.insert(Key::KeyJ, 'j');
        m.insert(Key::KeyK, 'k');
        m.insert(Key::KeyL, 'l');
        m.insert(Key::KeyM, 'm');
        m.insert(Key::KeyN, 'n');
        m.insert(Key::KeyO, 'o');
        m.insert(Key::KeyP, 'p');
        m.insert(Key::KeyQ, 'q');
        m.insert(Key::KeyR, 'r');
        m.insert(Key::KeyS, 's');
        m.insert(Key::KeyT, 't');
        m.insert(Key::KeyU, 'u');
        m.insert(Key::KeyV, 'v');
        m.insert(Key::KeyW, 'w');
        m.insert(Key::KeyX, 'x');
        m.insert(Key::KeyY, 'y');
        m.insert(Key::KeyZ, 'z');
        m.insert(Key::Space, ' ');
        m.insert(Key::Num1, '1');
        m.insert(Key::Num2, '2');
        m.insert(Key::Num3, '3');
        m.insert(Key::Num4, '4');
        m.insert(Key::Num5, '5');
        m.insert(Key::Num6, '6');
        m.insert(Key::Num7, '7');
        m.insert(Key::Num8, '8');
        m.insert(Key::Num9, '9');
        m.insert(Key::Num0, '0');
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
struct PredictionHandler {
    context: VecDeque<String>,
    max_context_length: usize,
}

impl PredictionHandler {
    fn new(max_context_length: usize) -> Self {
        PredictionHandler {
            context: VecDeque::new(),
            max_context_length,
        }
    }

    fn add_word(&mut self, word: String) {
        self.context.push_back(word);
        if self.context.len() > self.max_context_length {
            self.context.pop_front();
        }
    }
    fn clear_context(&mut self) {
        self.context.clear();
    }
    fn predict_next_word(&self) -> String {
        let system_prompt = &SYS_PREDICT_NEXT_WORD;
        let context_string = self
            .context
            .iter()
            .cloned()
            .collect::<Vec<String>>()
            .join(" ");
        let user_prompt = format!(
            "Given the following context, predict the single most likely next word:\n\n{}",
            context_string
        );
        println!("User Prompt: {}", user_prompt);
        let prediction = RUNTIME.block_on(async {
            tokio::task::spawn_blocking(move || {
                let (tx, rx) = std::sync::mpsc::channel();
                ollama_call(
                    system_prompt,
                    user_prompt,
                    Some(Box::new(move |response| {
                        tx.send(response).unwrap();
                    })),
                );
                rx.recv().unwrap_or_else(|_| "".to_string())
            })
            .await
            .expect("Task panicked")
        });
        prediction
            .split_whitespace()
            .next()
            .unwrap_or("")
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_string()
    }
    fn predict_next_sentence(&self) -> String {
        let system_prompt = &SYS_PREDICT_NEXT_SENTENCE;
        let context_string = self
            .context
            .iter()
            .cloned()
            .collect::<Vec<String>>()
            .join(" ");

        let user_prompt = format!(
            "Given the following context, predict the most likely next sentence:\n\n{}",
            context_string
        );

        println!("User Prompt: {}", user_prompt);

        let prediction = RUNTIME.block_on(async {
            tokio::task::spawn_blocking(move || {
                let (tx, rx) = std::sync::mpsc::channel();
                ollama_call(
                    system_prompt,
                    user_prompt,
                    Some(Box::new(move |response| {
                        tx.send(response).unwrap();
                    })),
                );
                rx.recv().unwrap_or_else(|_| "".to_string())
            })
            .await
            .expect("Task panicked")
        });

        // Parse the prediction to ensure it's only one sentence
        parse_single_sentence(&prediction)
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

fn format_text_remove_explanations(text: &str) -> String {
    let prefixes = [
        "Here's the formatted text:\n\n",
        "Here is the formatted text:\n\n",
        "Here is:\n\n",
    ];

    for prefix in &prefixes {
        if let Some(stripped) = text.strip_prefix(prefix) {
            return stripped.to_string();
        }
    }

    text.to_string()
}
fn parse_single_sentence(text: &str) -> String {
    let sentence_end_chars = ['.', '!', '?'];
    let mut result = String::new();
    let mut found_end = false;

    for c in text.chars() {
        if found_end {
            if c.is_whitespace() {
                result.push(c);
            } else {
                break;
            }
        } else {
            result.push(c);
            if sentence_end_chars.contains(&c) {
                found_end = true;
            }
        }
    }

    result.trim().to_string()
}
fn ollama_call(
    system_prompt: &'static str,
    user_prompt: String,
    update_clipboard: Option<Box<dyn Fn(String) + Send + Sync>>,
) {
    println!("Starting ollama_call function");
    let future = async move {
        println!("Creating HTTP client");
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build();
        if let Err(e) = client {
            eprintln!("Failed to create client: {:?}", e);
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
                            if !SHOULD_CONTINUE.load(Ordering::SeqCst) {
                                println!("Operation cancelled");
                                return;
                            }
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
    };

    RUNTIME.spawn(future);
}

struct PhaeroPredict {
    last_keys: Vec<Key>,
    keyboard_event_rx: Receiver<Event>,
    clipboard: Clipboard,
    clipboard_content: Arc<Mutex<String>>,
    clipboard_tx: Sender<String>,
    clipboard_rx: Receiver<String>,
    prediction_handler: PredictionHandler,
    current_word: String,
    hotkeys: HashMap<KeyCombination, fn(&mut Self)>,
}

impl PhaeroPredict {
    fn new() -> PhaeroPredict {
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

        let mut phaero_predict = PhaeroPredict {
            last_keys: Vec::new(),
            keyboard_event_rx: event_rx,
            clipboard,
            clipboard_content: Arc::clone(&clipboard_content),
            clipboard_tx,
            clipboard_rx,
            prediction_handler: PredictionHandler::new(50),
            current_word: String::new(),
            hotkeys: HashMap::new(),
        };
        phaero_predict.hotkeys.insert(
            KeyCombination::new(vec![Key::ControlLeft, Key::Alt, Key::KeyI]),
            Self::insert_text,
        );
        phaero_predict.hotkeys.insert(
            KeyCombination::new(vec![Key::ControlLeft, Key::Alt, Key::KeyC]),
            Self::cancel_operation,
        );
        phaero_predict.hotkeys.insert(
            KeyCombination::new(vec![Key::ControlLeft, Key::Alt, Key::KeyO]),
            Self::explain_ollama,
        );
        phaero_predict.hotkeys.insert(
            KeyCombination::new(vec![Key::ControlLeft, Key::Alt, Key::KeyA]),
            Self::answer_question,
        );
        phaero_predict.hotkeys.insert(
            KeyCombination::new(vec![Key::ControlLeft, Key::Alt, Key::KeyF]),
            Self::format_clipboard,
        );
        phaero_predict.hotkeys.insert(
            KeyCombination::new(vec![Key::ControlLeft, Key::Alt, Key::KeyP]),
            Self::predict_next_sentence,
        );
        phaero_predict.hotkeys.insert(
            KeyCombination::new(vec![Key::ControlLeft, Key::KeyC]),
            Self::copy_into_context,
        );
        phaero_predict.hotkeys.insert(
            KeyCombination::new(vec![Key::ControlLeft, Key::KeyV]),
            Self::do_nothing,
        );
        phaero_predict.hotkeys.insert(
            KeyCombination::new(vec![Key::ControlLeft, Key::KeyS]),
            Self::do_nothing,
        );
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
                    self.handle_key_press(&event);
                }
                EventType::KeyRelease(key) => {
                    self.last_keys.retain(|&k| k != key);
                }
                _ => {}
            }
        }
    }
    fn do_nothing(&mut self) {}
    fn copy_into_context(&mut self) {
        let clipboard_content = self.clipboard_content.lock().unwrap().clone();
        self.prediction_handler.clear_context();
        self.prediction_handler.add_word(clipboard_content);
    }
    fn explain_ollama(&mut self) {
        if !SHOULD_CONTINUE.load(Ordering::SeqCst) {
            println!("Operation cancelled");
            return;
        }
        let system_prompt = "You are a helpful AI assistant.";
        let user_prompt_base =
        "Please explain the following text in a clear and concise manner. Only return the explanation.";
        let clipboard_content = self.clipboard_content.lock().unwrap().clone();
        let user_prompt = format!("{} {}", user_prompt_base, clipboard_content);
        thread::spawn(move || {
            RUNTIME.block_on(async {
                tokio::task::spawn_blocking(move || {
                    ollama_call(system_prompt, user_prompt, None);
                })
                .await
                .expect("Task panicked");
            })
        });
    }
    fn handle_key_press(&mut self, event: &Event) {
        for (hotkey, function) in self.hotkeys.iter() {
            if hotkey.is_triggered(event, &self.last_keys) {
                SHOULD_CONTINUE.store(true, Ordering::SeqCst);
                sleep(Duration::from_millis(100));
                self.last_keys.clear();
                RESPONSE_BUFFER.lock().unwrap().clear();
                function(self);
                return;
            }
        }

        match event.event_type {
            EventType::KeyPress(key) => match key {
                Key::Backspace => {
                    if !self.current_word.is_empty() {
                        self.current_word.pop();
                    } else {
                        if let Some(word) = self.prediction_handler.context.pop_back() {
                            self.current_word = word;
                        }
                    }
                }
                Key::Space => {
                    self.prediction_handler.add_word(self.current_word.clone());
                    self.current_word.clear();
                }
                _ => {
                    if let Some(ch) = KEY_TO_CHAR.get(&key) {
                        self.current_word.push(*ch);
                    }
                }
            },
            _ => {}
        }
    }

    fn insert_text(&mut self) {
        while let Some(word) = RESPONSE_BUFFER.lock().unwrap().pop_front() {
            println!("Inserting word: {}", word);
            for ch in word.chars() {
                if !SHOULD_CONTINUE.load(Ordering::SeqCst) {
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

    fn cancel_operation(&mut self) {
        println!("Operation cancelled");
        SHOULD_CONTINUE.store(false, Ordering::SeqCst);
    }

    fn format_clipboard(&mut self) {
        if !SHOULD_CONTINUE.load(Ordering::SeqCst) {
            println!("Operation cancelled");
            return;
        }
        let system_prompt = SYS_FORMAT_PROMPT;
        let content = self.clipboard_content.lock().unwrap().clone();
        let clipboard_tx = Arc::new(self.clipboard_tx.clone());
        let update_clipboard = Box::new(move |formatted_text| {
            let clipboard_tx = Arc::clone(&clipboard_tx);
            let _ = clipboard_tx.send(formatted_text);
        });
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
    fn answer_question(&mut self) {
        let system_prompt = "You are an AI assistant specialized in answering questions.";
        let user_prompt = "Please answer the following question: ";
        let clipboard_content = self.clipboard_content.lock().unwrap().clone();
        let user_prompt = format!("{} {}", user_prompt, clipboard_content);
        let clipboard_tx = Arc::new(self.clipboard_tx.clone());
        let update_clipboard = Box::new(move |formatted_text| {
            let clipboard_tx = Arc::clone(&clipboard_tx);
            let _ = clipboard_tx.send(formatted_text);
        });
        thread::spawn(move || {
            RUNTIME.block_on(async {
                tokio::task::spawn_blocking(move || {
                    ollama_call(system_prompt, user_prompt, Some(update_clipboard));
                })
                .await
                .expect("Task panicked");
            })
        });
    }
    fn predict_next_word(&mut self) {
        let predicted_word = self.prediction_handler.predict_next_word();
        for ch in predicted_word.chars() {
            if let Some(&key) = CHAR_TO_KEY.get(&ch.to_lowercase().next().unwrap()) {
                send(&EventType::KeyPress(key));
                send(&EventType::KeyRelease(key));
            }
        }
        send(&EventType::KeyPress(Key::Space));
        send(&EventType::KeyRelease(Key::Space));
    }
    fn predict_next_sentence(&mut self) {
        let predicted_sentence = self.prediction_handler.predict_next_sentence();
        for ch in predicted_sentence.chars() {
            if let Some(&key) = CHAR_TO_KEY.get(&ch.to_lowercase().next().unwrap()) {
                send(&EventType::KeyPress(key));
                send(&EventType::KeyRelease(key));
            }
        }
        send(&EventType::KeyPress(Key::Space));
        send(&EventType::KeyRelease(Key::Space));
    }
    fn draw(&self) {
        clear_background(Color::new(0.05, 0.05, 0.1, 1.0)); // Dark blue background

        let companion_size = 150.0;
        let max_text_width = max(
            measure_text("Hey I'm Phaero", None, 20, 1.0).width as i32,
            measure_text(
                "I'm here to help you with your text based tasks!",
                None,
                20,
                1.0,
            )
            .width as i32,
        ) as f32;
        self.draw_companion(companion_size);
        let default_x_pos = max_text_width + 20.0 + companion_size;
        let mut y_pos = 20.0;
        y_pos = self.draw_keybinds(default_x_pos, y_pos);
        y_pos = self.draw_clipboard_content(default_x_pos, y_pos, 250.0);
        self.draw_prediction_context(default_x_pos, y_pos, 250.0);
    }

    fn draw_companion(&self, size: f32) {
        draw_texture_ex(
            &COMPANION_TEXTURE,
            20.0,
            (screen_height() / 2.0) - size,
            WHITE,
            DrawTextureParams {
                dest_size: Some(Vec2::new(size, size)),
                ..Default::default()
            },
        );
        const MESSAGE1: &str = "Hello! I'm your PhaeroPredict assistant.";
        const MESSAGE2: &str = "I can help you with text formatting and word prediction.";
        let max_text_width: f32 = max(
            measure_text(MESSAGE1, None, 20, 1.0).width as i32,
            measure_text(MESSAGE2, None, 20, 1.0).width as i32,
        ) as f32;
        let y_pos = screen_height() / 2.0 + 20.0;
        draw_text(MESSAGE1, 20.0, y_pos, 20.0, PINK);
        draw_text(MESSAGE2, 20.0, y_pos + 20.0, 20.0, PINK);
        draw_text("My Response:", 20.0, y_pos + 50.0, 30.0, YELLOW);
        self.draw_response(y_pos + 75.0, max_text_width);
    }
    fn draw_response(&self, mut y_pos: f32, max_width: f32) {
        let text_color = Color::new(0.9, 0.9, 0.95, 1.0);
        let mut x_pos = 20.0;
        for word in RESPONSE_BUFFER.lock().unwrap().iter() {
            if x_pos > max_width {
                x_pos = 20.0;
                y_pos += 35.0;
            }
            draw_text(word, x_pos, y_pos, 25.0, text_color);
            x_pos += word.len() as f32 * 13.0;
        }
    }

    fn draw_keybinds(&self, x_pos: f32, mut y_pos: f32) -> f32 {
        draw_text("Keybinds:", x_pos, y_pos, 30.0, PURPLE);
        y_pos += 40.0;
        let keybinds = [
            "Ctrl+Alt+I: Insert Text",
            "Ctrl+Alt+C: Cancel Operation",
            "Ctrl+Alt+F: Format Clipboard",
            "Ctrl+Alt+P: Predict Next Word",
        ];
        for keybind in &keybinds {
            draw_text(keybind, x_pos, y_pos, 20.0, SKYBLUE);
            y_pos += 25.0;
        }
        y_pos + 20.0
    }

    fn draw_clipboard_content(&self, x_pos: f32, mut y_pos: f32, max_width: f32) -> f32 {
        draw_text("Clipboard Content:", x_pos, y_pos, 30.0, GREEN);
        y_pos += 40.0;
        y_pos = self.draw_wrapped_text(
            &self.clipboard_content.lock().unwrap(),
            x_pos,
            y_pos,
            max_width,
            Color::new(0.9, 0.9, 0.95, 1.0),
        );
        y_pos + 60.0
    }

    fn draw_prediction_context(&self, x_pos: f32, mut y_pos: f32, max_width: f32) {
        draw_text("Current Context for Prediction:", x_pos, y_pos, 30.0, BLUE);
        y_pos += 40.0;
        y_pos = self.draw_wrapped_text(
            &self
                .prediction_handler
                .context
                .iter()
                .take(50)
                .cloned()
                .collect::<Vec<String>>()
                .join(" "),
            x_pos,
            y_pos,
            max_width,
            Color::new(0.9, 0.9, 0.95, 1.0),
        );
        if !self.current_word.is_empty() {
            draw_text(&self.current_word, x_pos, y_pos, 25.0, YELLOW);
        }
    }

    fn draw_wrapped_text(
        &self,
        text: &str,
        mut x_pos: f32,
        mut y_pos: f32,
        max_width: f32,
        color: Color,
    ) -> f32 {
        let starting_x: f32 = x_pos;
        let mut words = text.split_whitespace();
        let mut current_line = String::new();
        while let Some(word) = words.next() {
            let word_width = measure_text(word, None, 20, 1.0).width as f32;
            if x_pos + word_width > (max_width + starting_x) {
                draw_text(&current_line, starting_x, y_pos, 20.0, color);
                y_pos += 25.0;
                x_pos = starting_x;
                current_line.clear();
            }
            current_line.push_str(word);
            current_line.push(' ');
            x_pos += word_width;
        }
        draw_text(&current_line, x_pos, y_pos, 20.0, color);
        y_pos + 25.0
    }
}

#[macroquad::main("PhaeroCopilot")]
async fn main() {
    let mut phaero_predict = PhaeroPredict::new();
    loop {
        phaero_predict.update();
        phaero_predict.draw();
        next_frame().await;
    }
}
