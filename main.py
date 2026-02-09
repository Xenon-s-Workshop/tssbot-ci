import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional
import asyncio
from datetime import datetime
from threading import Lock

import google.generativeai as genai
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
from pdf2image import convert_from_path
from PIL import Image

# Import prompts
from prompts import get_extraction_prompt, get_generation_prompt

# Configuration
class Config:
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    GEMINI_API_KEYS = os.getenv("GEMINI_API_KEYS", "").split(",")
    TEMP_DIR = Path("temp")
    OUTPUT_DIR = Path("output")
    MAX_CONCURRENT_IMAGES = 5
    MAX_QUEUE_SIZE = 10
    
    GEMINI_MODEL = "gemini-2.0-flash-exp"
    
    GENERATION_CONFIG = {
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
    }
    
    SAFETY_SETTINGS = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    
    def __init__(self):
        if not self.TELEGRAM_BOT_TOKEN:
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required!")
        if not self.GEMINI_API_KEYS or self.GEMINI_API_KEYS == ['']:
            raise ValueError("GEMINI_API_KEYS environment variable is required!")
        
        self.TEMP_DIR.mkdir(exist_ok=True)
        self.OUTPUT_DIR.mkdir(exist_ok=True)

config = Config()

# Gemini API Key Rotation
class GeminiAPIRotator:
    def __init__(self, api_keys: List[str]):
        self.api_keys = [key.strip() for key in api_keys if key.strip()]
        self.current_index = 0
        self.lock = Lock()
        
        if not self.api_keys:
            raise ValueError("No valid Gemini API keys provided!")
    
    def get_next_key(self) -> str:
        with self.lock:
            key = self.api_keys[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.api_keys)
            return key

api_rotator = GeminiAPIRotator(config.GEMINI_API_KEYS)

# Task Queue Manager
class TaskQueue:
    def __init__(self):
        self.queue = []
        self.lock = Lock()
        self.processing = set()
    
    def add_task(self, user_id: int, task_data: Dict) -> int:
        with self.lock:
            # Check if user already has task in queue or processing
            if user_id in self.processing:
                return -2  # Already processing
            
            for task in self.queue:
                if task['user_id'] == user_id:
                    return -2  # Already in queue
            
            if len(self.queue) >= config.MAX_QUEUE_SIZE:
                return -1  # Queue full
            
            task = {
                'user_id': user_id,
                'data': task_data,
                'timestamp': datetime.now()
            }
            self.queue.append(task)
            return len(self.queue)
    
    def get_next_task(self) -> Optional[Dict]:
        with self.lock:
            if self.queue:
                return self.queue.pop(0)
            return None
    
    def get_position(self, user_id: int) -> int:
        with self.lock:
            for idx, task in enumerate(self.queue):
                if task['user_id'] == user_id:
                    return idx + 1
            return 0
    
    def is_processing(self, user_id: int) -> bool:
        with self.lock:
            return user_id in self.processing
    
    def set_processing(self, user_id: int, status: bool):
        with self.lock:
            if status:
                self.processing.add(user_id)
            else:
                self.processing.discard(user_id)
    
    def get_queue_size(self) -> int:
        with self.lock:
            return len(self.queue)

task_queue = TaskQueue()

# PDF Processing Functions
class PDFProcessor:
    @staticmethod
    async def pdf_to_images(pdf_path: Path, page_range: Optional[tuple] = None) -> List[Image.Image]:
        try:
            if page_range:
                first_page, last_page = page_range
                images = convert_from_path(
                    pdf_path,
                    first_page=first_page,
                    last_page=last_page,
                    dpi=300
                )
            else:
                images = convert_from_path(pdf_path, dpi=300)
            
            return images
        except Exception as e:
            raise Exception(f"Error converting PDF to images: {str(e)}")
    
    @staticmethod
    async def process_single_image(image: Image.Image, image_idx: int, mode: str, retry_count: int = 3) -> Optional[tuple]:
        for attempt in range(retry_count):
            try:
                api_key = api_rotator.get_next_key()
                genai.configure(api_key=api_key)
                
                model = genai.GenerativeModel(
                    model_name=config.GEMINI_MODEL,
                    generation_config=config.GENERATION_CONFIG,
                    safety_settings=config.SAFETY_SETTINGS
                )
                
                # Select prompt based on mode
                prompt = get_extraction_prompt() if mode == "extraction" else get_generation_prompt()
                
                print(f"Processing image {image_idx} in {mode} mode with {config.GEMINI_MODEL}")
                
                response = model.generate_content([prompt, image])
                response_text = response.text.strip()
                
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.startswith("```"):
                    response_text = response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                
                response_text = response_text.strip()
                questions = json.loads(response_text)
                
                print(f"✅ Successfully processed image {image_idx}")
                return (image_idx, questions)
                
            except json.JSONDecodeError as e:
                print(f"JSON decode error for image {image_idx}, attempt {attempt + 1}: {str(e)}")
                if attempt == retry_count - 1:
                    return (image_idx, None)
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"Error processing image {image_idx}, attempt {attempt + 1}: {str(e)}")
                if attempt == retry_count - 1:
                    return (image_idx, None)
                await asyncio.sleep(2)
        
        return (image_idx, None)
    
    @staticmethod
    async def process_images_parallel(images: List[Image.Image], mode: str, progress_callback=None) -> List[Dict]:
        all_questions = []
        total_images = len(images)
        
        for batch_start in range(0, total_images, config.MAX_CONCURRENT_IMAGES):
            batch_end = min(batch_start + config.MAX_CONCURRENT_IMAGES, total_images)
            batch_images = images[batch_start:batch_end]
            
            tasks = []
            for i, image in enumerate(batch_images):
                image_idx = batch_start + i + 1
                task = PDFProcessor.process_single_image(image, image_idx, mode)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            for image_idx, questions in results:
                if progress_callback:
                    await progress_callback(image_idx, total_images)
                
                if questions:
                    all_questions.extend(questions)
            
            if batch_end < total_images:
                await asyncio.sleep(0.5)
        
        return all_questions

# CSV Generation
class CSVGenerator:
    @staticmethod
    def questions_to_csv(questions: List[Dict], output_path: Path):
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'questions', 'option1', 'option2', 'option3', 'option4', 
                'option5', 'answer', 'explanation', 'type', 'section'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for q in questions:
                options = q.get('options', [])
                while len(options) < 4:
                    options.append('')
                
                correct_index = q.get('correct_answer_index', -1)
                answer = str(correct_index + 1) if correct_index >= 0 else ''
                
                row = {
                    'questions': q.get('question_description', ''),
                    'option1': options[0] if len(options) > 0 else '',
                    'option2': options[1] if len(options) > 1 else '',
                    'option3': options[2] if len(options) > 2 else '',
                    'option4': options[3] if len(options) > 3 else '',
                    'option5': '',
                    'answer': answer,
                    'explanation': q.get('explanation', ''),
                    'type': '1',
                    'section': '1'
                }
                writer.writerow(row)

# Queue Processor (runs in background)
class QueueProcessor:
    def __init__(self, bot):
        self.bot = bot
        self.running = False
        self.task = None
    
    async def start(self):
        """Start processing queue"""
        if self.running:
            return
        
        self.running = True
        print("🔄 Queue processor started")
        
        while self.running:
            try:
                task = task_queue.get_next_task()
                
                if task:
                    user_id = task['user_id']
                    task_data = task['data']
                    
                    task_queue.set_processing(user_id, True)
                    
                    try:
                        await self.bot.process_pdf(
                            user_id=user_id,
                            pdf_path=task_data['pdf_path'],
                            page_range=task_data['page_range'],
                            mode=task_data['mode'],
                            context=task_data['context']
                        )
                    except Exception as e:
                        print(f"Error processing task for user {user_id}: {str(e)}")
                        try:
                            await task_data['context'].bot.send_message(
                                chat_id=user_id,
                                text=f"❌ Error processing your PDF: {str(e)}"
                            )
                        except:
                            pass
                    finally:
                        task_queue.set_processing(user_id, False)
                    
                    await asyncio.sleep(1)
                else:
                    # No tasks, wait a bit
                    await asyncio.sleep(2)
                    
            except Exception as e:
                print(f"Queue processor error: {str(e)}")
                await asyncio.sleep(5)
    
    def stop(self):
        """Stop processing queue"""
        self.running = False
        print("🛑 Queue processor stopped")

# Telegram Bot Handlers
class TelegramBot:
    def __init__(self):
        self.user_states = {}
        self.queue_processor = None
    
    async def post_init(self, application: Application):
        """Initialize queue processor after event loop starts"""
        self.queue_processor = QueueProcessor(self)
        # Create task in the running event loop
        asyncio.create_task(self.queue_processor.start())
        print("✅ Queue processor initialized")
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "Welcome​​​​​​​​​​​​​​​​
