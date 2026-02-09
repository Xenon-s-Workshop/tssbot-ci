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
from telegram.error import RetryAfter, TimedOut
from pdf2image import convert_from_path
from PIL import Image

from prompts import get_extraction_prompt, get_generation_prompt

class Config:
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    GEMINI_API_KEYS = os.getenv("GEMINI_API_KEYS", "").split(",")
    QUIZ_MARKER = os.getenv("QUIZ_MARKER", "[TSS]")  # Marker before question
    EXPLANATION_TAG = os.getenv("EXPLANATION_TAG", "t.me/tss")  # Tag at end of explanation
    
    TEMP_DIR = Path("temp")
    OUTPUT_DIR = Path("output")
    MAX_CONCURRENT_IMAGES = 5
    MAX_QUEUE_SIZE = 10
    GEMINI_MODEL = "gemini-2.0-flash"
    
    # Rate limiting configuration
    POLL_DELAY = 3  # Delay between polls in seconds
    BATCH_SIZE = 20  # Send polls in batches
    BATCH_DELAY = 10  # Delay between batches in seconds
    
    GENERATION_CONFIG = {"temperature": 0.1, "top_p": 0.95, "top_k": 40, "max_output_tokens": 8192}
    SAFETY_SETTINGS = [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}, {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}, {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"}, {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}]
    
    def __init__(self):
        if not self.TELEGRAM_BOT_TOKEN:
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required!")
        if not self.GEMINI_API_KEYS or self.GEMINI_API_KEYS == ['']:
            raise ValueError("GEMINI_API_KEYS environment variable is required!")
        self.TEMP_DIR.mkdir(exist_ok=True)
        self.OUTPUT_DIR.mkdir(exist_ok=True)

config = Config()

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

class TaskQueue:
    def __init__(self):
        self.queue = []
        self.lock = Lock()
        self.processing = set()
    def add_task(self, user_id: int, task_data: Dict) -> int:
        with self.lock:
            if user_id in self.processing:
                return -2
            for task in self.queue:
                if task['user_id'] == user_id:
                    return -2
            if len(self.queue) >= config.MAX_QUEUE_SIZE:
                return -1
            task = {'user_id': user_id, 'data': task_data, 'timestamp': datetime.now()}
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

class QuizPoster:
    """Handle posting quizzes to channels/groups with rate limiting"""
    
    @staticmethod
    def format_question(question_text: str) -> str:
        """Format question with quiz marker and line break"""
        # Add quiz marker before question with line break
        formatted = f"{config.QUIZ_MARKER}\n\n{question_text}"
        
        # Telegram quiz question max length is 300 characters
        if len(formatted) > 300:
            # Truncate if too long
            max_question_len = 300 - len(config.QUIZ_MARKER) - 3  # -3 for "\n\n" and ellipsis
            formatted = f"{config.QUIZ_MARKER}\n\n{question_text[:max_question_len-3]}..."
        
        return formatted
    
    @staticmethod
    def format_explanation(explanation: str) -> str:
        """Format explanation with tag at the end"""
        # Add explanation tag at the end
        formatted = f"{explanation} [{config.EXPLANATION_TAG}]"
        
        # Telegram explanation max length is 200 characters
        if len(formatted) > 200:
            # Truncate explanation but keep tag
            tag_with_brackets = f" [{config.EXPLANATION_TAG}]"
            max_explanation_len = 200 - len(tag_with_brackets) - 3  # -3 for ellipsis
            formatted = f"{explanation[:max_explanation_len]}...{tag_with_brackets}"
        
        return formatted
    
    @staticmethod
    async def send_quiz_with_retry(context: ContextTypes.DEFAULT_TYPE, chat_id: int, question: Dict, message_thread_id: Optional[int] = None, max_retries: int = 5) -> bool:
        """Send a single quiz poll with retry logic for rate limiting"""
        for attempt in range(max_retries):
            try:
                # Get question data
                question_text = question.get('question_description', 'Question')
                options = question.get('options', [])[:10]  # Telegram supports up to 10 options
                correct_option_id = question.get('correct_answer_index', 0)
                explanation = question.get('explanation', '')
                
                # Ensure we have at least 2 options (Telegram requirement)
                if len(options) < 2:
                    print(f"Skipping question with less than 2 options")
                    return False
                
                # Ensure correct_option_id is within valid range
                if correct_option_id < 0 or correct_option_id >= len(options):
                    print(f"Invalid correct_option_id: {correct_option_id} for {len(options)} options")
                    correct_option_id = 0
                
                # Format question and explanation
                formatted_question = QuizPoster.format_question(question_text)
                formatted_explanation = QuizPoster.format_explanation(explanation) if explanation else None
                
                # Send quiz poll
                await context.bot.send_poll(
                    chat_id=chat_id,
                    question=formatted_question,
                    options=options,
                    type='quiz',  # This makes it a quiz (not regular poll)
                    correct_option_id=correct_option_id,
                    explanation=formatted_explanation,
                    is_anonymous=False,
                    message_thread_id=message_thread_id
                )
                
                print(f"✅ Quiz sent successfully")
                return True
                
            except RetryAfter as e:
                wait_time = e.retry_after + 2
                print(f"⏳ Rate limited. Waiting {wait_time} seconds...")
                await asyncio.sleep(wait_time)
                
            except TimedOut:
                print(f"⏱️ Timeout on attempt {attempt + 1}, retrying...")
                await asyncio.sleep(3)
                
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Error sending quiz (attempt {attempt + 1}): {error_msg}")
                
                # Check for specific errors
                if "question is too long" in error_msg.lower():
                    print("Question too long, skipping...")
                    return False
                elif "not enough rights" in error_msg.lower():
                    print("Bot doesn't have permission to send polls")
                    return False
                elif "chat not found" in error_msg.lower():
                    print("Chat not found or bot is not a member")
                    return False
                
                if attempt == max_retries - 1:
                    return False
                    
                await asyncio.sleep(3)
        
        return False
    
    @staticmethod
    async def post_quizzes_batch(context: ContextTypes.DEFAULT_TYPE, chat_id: int, questions: List[Dict], message_thread_id: Optional[int] = None, progress_callback=None) -> Dict:
        """Post quizzes in batches with rate limiting"""
        total = len(questions)
        success_count = 0
        failed_count = 0
        skipped_count = 0
        
        for i in range(0, total, config.BATCH_SIZE):
            batch = questions[i:i + config.BATCH_SIZE]
            batch_num = (i // config.BATCH_SIZE) + 1
            total_batches = (total + config.BATCH_SIZE - 1) // config.BATCH_SIZE
            
            print(f"📦 Processing batch {batch_num}/{total_batches}")
            
            for idx, question in enumerate(batch):
                global_idx = i + idx + 1
                
                if progress_callback:
                    await progress_callback(global_idx, total)
                
                # Check if question has valid data
                if not question.get('question_description') or not question.get('options'):
                    print(f"⏭️ Skipping question {global_idx} - missing data")
                    skipped_count += 1
                    continue
                
                success = await QuizPoster.send_quiz_with_retry(context, chat_id, question, message_thread_id)
                
                if success:
                    success_count += 1
                else:
                    failed_count += 1
                
                # Delay between individual polls
                if global_idx < total:
                    await asyncio.sleep(config.POLL_DELAY)
            
            # Delay between batches
            if i + config.BATCH_SIZE < total:
                print(f"✅ Batch {batch_num} completed. Waiting {config.BATCH_DELAY}s before next batch...")
                await asyncio.sleep(config.BATCH_DELAY)
        
        return {
            'total': total,
            'success': success_count,
            'failed': failed_count,
            'skipped': skipped_count
        }

class ImageProcessor:
    @staticmethod
    def is_image_file(filename: str) -> bool:
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp']
        return any(filename.lower().endswith(ext) for ext in image_extensions)
    @staticmethod
    async def load_image(image_path: Path) -> Image.Image:
        try:
            return Image.open(image_path)
        except Exception as e:
            raise Exception(f"Error loading image: {str(e)}")

class PDFProcessor:
    @staticmethod
    async def pdf_to_images(pdf_path: Path, page_range: Optional[tuple] = None) -> List[Image.Image]:
        try:
            if page_range:
                first_page, last_page = page_range
                images = convert_from_path(pdf_path, first_page=first_page, last_page=last_page, dpi=300)
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
                model = genai.GenerativeModel(model_name=config.GEMINI_MODEL, generation_config=config.GENERATION_CONFIG, safety_settings=config.SAFETY_SETTINGS)
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

class CSVGenerator:
    @staticmethod
    def questions_to_csv(questions: List[Dict], output_path: Path):
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['questions', 'option1', 'option2', 'option3', 'option4', 'option5', 'answer', 'explanation', 'type', 'section']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for q in questions:
                options = q.get('options', [])
                while len(options) < 4:
                    options.append('')
                correct_index = q.get('correct_answer_index', -1)
                answer = str(correct_index + 1) if correct_index >= 0 else ''
                row = {'questions': q.get('question_description', ''), 'option1': options[0] if len(options) > 0 else '', 'option2': options[1] if len(options) > 1 else '', 'option3': options[2] if len(options) > 2 else '', 'option4': options[3] if len(options) > 3 else '', 'option5': '', 'answer': answer, 'explanation': q.get('explanation', ''), 'type': '1', 'section': '1'}
                writer.writerow(row)

class QueueProcessor:
    def __init__(self, bot):
        self.bot = bot
        self.running = False
    async def start(self):
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
                        await self.bot.process_content(user_id=user_id, content_type=task_data['content_type'], content_paths=task_data['content_paths'], page_range=task_data.get('page_range'), mode=task_data['mode'], context=task_data['context'])
                    except Exception as e:
                        print(f"Error processing task for user {user_id}: {str(e)}")
                        try:
                            await task_data['context'].bot.send_message(chat_id=user_id, text=f"❌ Error processing your content: {str(e)}")
                        except:
                            pass
                    finally:
                        task_queue.set_processing(user_id, False)
                    await asyncio.sleep(1)
                else:
                    await asyncio.sleep(2)
            except Exception as e:
                print(f"Queue processor error: {str(e)}")
                await asyncio.sleep(5)

class TelegramBot:
    def __init__(self):
        self.user_states = {}
        self.queue_processor = None
    
    async def post_init(self, application: Application):
        self.queue_processor = QueueProcessor(self)
        asyncio.create_task(self.queue_processor.start())
        print("✅ Queue processor initialized")
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(f"Welcome! 👋\n\nSend me:\n📄 PDF file\n🖼️ Single image\n🖼️🖼️ Multiple images (as media group)\n\n🤖 Powered by Gemini 2.0 Flash\n📢 Quiz Marker: {config.QUIZ_MARKER}\n🔗 Explanation Tag: {config.EXPLANATION_TAG}\n\nCommands:\n/start - Start the bot\n/help - Show help message\n/queue - Check your queue position\n/cancel - Cancel your current task\n/model - Show current AI model info")
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("📚 *How to use:*\n\n1. Send me content:\n   • PDF file (single or multiple pages)\n   • Single image\n   • Multiple images (select multiple)\n\n2. Choose mode:\n   • *Extraction* - Extract existing questions\n   • *Generation* - Generate new questions from textbook\n\n3. Optionally specify page range (PDF only)\n\n4. Get your CSV file\n\n5. Optionally post quizzes to channel/group\n\n*Quiz Format:*\n• Question: MARKER + question\n• Explanation: explanation + [TAG]\n\n*Features:*\n✓ Gemini 2.0 Flash AI\n✓ PDF, single & multiple image support\n✓ Two processing modes\n✓ Post to channels/groups\n✓ Topic support for groups\n✓ Rate limiting protection\n✓ Bengali explanations\n✓ Task queue system\n✓ Parallel processing (5x faster)", parse_mode='Markdown')
    
    async def model_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        queue_size = task_queue.get_queue_size()
        await update.message.reply_text(f"🤖 *AI Model Information:*\n\nModel: `{config.GEMINI_MODEL}`\nTemperature: {config.GENERATION_CONFIG['temperature']}\nMax Tokens: {config.GENERATION_CONFIG['max_output_tokens']}\nParallel Workers: {config.MAX_CONCURRENT_IMAGES}\nAPI Keys: {len(config.GEMINI_API_KEYS)}\nQueue Size: {queue_size}/{config.MAX_QUEUE_SIZE}\n\n*Quiz Settings:*\nMarker: {config.QUIZ_MARKER}\nExplanation Tag: {config.EXPLANATION_TAG}\n\n*Rate Limiting:*\nPoll Delay: {config.POLL_DELAY}s\nBatch Size: {config.BATCH_SIZE}\nBatch Delay: {config.BATCH_DELAY}s", parse_mode='Markdown')
    
    async def queue_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if task_queue.is_processing(user_id):
            await update.message.reply_text("⚙️ Your task is currently being processed!")
        else:
            position = task_queue.get_position(user_id)
            if position > 0:
                await update.message.reply_text(f"📋 Your position in queue: {position}\n⏳ Estimated wait: ~{position * 2} minutes")
            else:
                await update.message.reply_text("❌ You don't have any tasks in queue.")
    
    async def cancel_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if user_id in self.user_states:
            content_paths = self.user_states[user_id].get('content_paths', [])
            for path in content_paths:
                if path.exists():
                    path.unlink(missing_ok=True)
            del self.user_states[user_id]
            task_queue.set_processing(user_id, False)
            await update.message.reply_text("✅ Task cancelled successfully!")
        else:
            await update.message.reply_text("❌ No active task to cancel.")
    
    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        document = update.message.document
        if not document.file_name.endswith('.pdf'):
            await update.message.reply_text("❌ Please send a PDF file only.")
            return
        if user_id in self.user_states or task_queue.is_processing(user_id):
            await update.message.reply_text("⚠️ You already have a task in progress.\nUse /cancel to cancel the current task.")
            return
        processing_msg = await update.message.reply_text("📥 Downloading PDF...")
        try:
            file = await context.bot.get_file(document.file_id)
            pdf_path = config.TEMP_DIR / f"{user_id}_{document.file_name}"
            await file.download_to_drive(pdf_path)
            keyboard = [[InlineKeyboardButton("📤 Extraction Mode", callback_data="mode_extraction")], [InlineKeyboardButton("✨ Generation Mode", callback_data="mode_generation")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            self.user_states[user_id] = {'content_type': 'pdf', 'content_paths': [pdf_path]}
            await processing_msg.edit_text("📄 PDF received!\n\nChoose processing mode:\n\n📤 *Extraction Mode*\nExtract existing MCQs from PDF\n\n✨ *Generation Mode*\nGenerate new MCQs from textbook content", reply_markup=reply_markup, parse_mode='Markdown')
        except Exception as e:
            await processing_msg.edit_text(f"❌ Error downloading PDF: {str(e)}")
    
    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if user_id in self.user_states or task_queue.is_processing(user_id):
            await update.message.reply_text("⚠️ You already have a task in progress.\nUse /cancel to cancel the current task.")
            return
        if update.message.media_group_id:
            if user_id not in self.user_states:
                self.user_states[user_id] = {'content_type': 'images', 'content_paths': [], 'media_group_id': update.message.media_group_id, 'waiting_for_more': True}
                asyncio.create_task(self.process_media_group_delayed(user_id, context))
            photo = update.message.photo[-1]
            file = await context.bot.get_file(photo.file_id)
            image_path = config.TEMP_DIR / f"{user_id}_image_{len(self.user_states[user_id]['content_paths'])}.jpg"
            await file.download_to_drive(image_path)
            self.user_states[user_id]['content_paths'].append(image_path)
        else:
            processing_msg = await update.message.reply_text("📥 Downloading image...")
            try:
                photo = update.message.photo[-1]
                file = await context.bot.get_file(photo.file_id)
                image_path = config.TEMP_DIR / f"{user_id}_single_image.jpg"
                await file.download_to_drive(image_path)
                keyboard = [[InlineKeyboardButton("📤 Extraction Mode", callback_data="mode_extraction")], [InlineKeyboardButton("✨ Generation Mode", callback_data="mode_generation")]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                self.user_states[user_id] = {'content_type': 'images', 'content_paths': [image_path]}
                await processing_msg.edit_text("🖼️ Image received!\n\nChoose processing mode:\n\n📤 *Extraction Mode*\nExtract existing MCQs from image\n\n✨ *Generation Mode*\nGenerate new MCQs from textbook content", reply_markup=reply_markup, parse_mode='Markdown')
            except Exception as e:
                await processing_msg.edit_text(f"❌ Error downloading image: {str(e)}")
    
    async def process_media_group_delayed(self, user_id: int, context: ContextTypes.DEFAULT_TYPE):
        await asyncio.sleep(2)
        if user_id in self.user_states and self.user_states[user_id].get('waiting_for_more'):
            self.user_states[user_id]['waiting_for_more'] = False
            num_images = len(self.user_states[user_id]['content_paths'])
            keyboard = [[InlineKeyboardButton("📤 Extraction Mode", callback_data="mode_extraction")], [InlineKeyboardButton("✨ Generation Mode", callback_data="mode_generation")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await context.bot.send_message(chat_id=user_id, text=f"🖼️ {num_images} images received!\n\nChoose processing mode:\n\n📤 *Extraction Mode*\nExtract existing MCQs from images\n\n✨ *Generation Mode*\nGenerate new MCQs from textbook content", reply_markup=reply_markup, parse_mode='Markdown')
    
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        user_id = update.effective_user.id
        
        if query.data.startswith("mode_"):
            mode = query.data.split("_")[1]
            self.user_states[user_id]['mode'] = mode
            mode_name = "Extraction" if mode == "extraction" else "Generation"
            if self.user_states[user_id]['content_type'] == 'pdf':
                keyboard = [[InlineKeyboardButton("All Pages", callback_data="all_pages")], [InlineKeyboardButton("Specify Range", callback_data="specify_range")]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await query.edit_message_text(f"✅ Mode selected: *{mode_name}*\n\nChoose page range:", reply_markup=reply_markup, parse_mode='Markdown')
            else:
                await query.edit_message_text(f"✅ Mode selected: *{mode_name}*\n\nAdding to queue...", parse_mode='Markdown')
                await self.add_to_queue_direct(user_id, None, context)
        elif query.data == "all_pages":
            await self.add_to_queue_direct(user_id, None, context)
        elif query.data == "specify_range":
            await query.edit_message_text("Please send the page range in format: start-end\nExample: 1-10 or 5-8")
            self.user_states[user_id]['waiting_for_range'] = True
        elif query.data.startswith("post_quizzes_"):
            session_id = query.data.split("_")[2]
            if user_id in self.user_states and self.user_states[user_id].get('session_id') == session_id:
                await query.edit_message_text("📢 *Post Quizzes*\n\nChoose destination:", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📺 Channel", callback_data=f"dest_channel_{session_id}")], [InlineKeyboardButton("👥 Group", callback_data=f"dest_group_{session_id}")]]), parse_mode='Markdown')
        elif query.data.startswith("dest_channel_"):
            session_id = query.data.split("_")[2]
            await query.edit_message_text("📺 Please send the *Channel ID*\n\nExample: -1001234567890\n\nYou can get this by forwarding a message from the channel to @userinfobot", parse_mode='Markdown')
            self.user_states[user_id]['waiting_for'] = 'channel_id'
            self.user_states[user_id]['session_id'] = session_id
        elif query.data.startswith("dest_group_"):
            session_id = query.data.split("_")[2]
            await query.edit_message_text("👥 Please send the *Group ID*\n\nExample: -1001234567890\n\nYou can get this by forwarding a message from the group to @userinfobot", parse_mode='Markdown')
            self.user_states[user_id]['waiting_for'] = 'group_id'
            self.user_states[user_id]['session_id'] = session_id
    
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if user_id not in self.user_states:
            return
        waiting_for = self.user_states[user_id].get('waiting_for')
        if self.user_states[user_id].get('waiting_for_range'):
            try:
                text = update.message.text.strip()
                parts = text.split('-')
                if len(parts) != 2:
                    await update.message.reply_text("❌ Invalid format. Use: start-end (e.g., 1-10)")
                    return
                start_page = int(parts[0])
                end_page = int(parts[1])
                if start_page < 1 or end_page < start_page:
                    await update.message.reply_text("❌ Invalid page range")
                    return
                page_range = (start_page, end_page)
                self.user_states[user_id]['waiting_for_range'] = False
                await self.add_to_queue_direct(user_id, page_range, context)
            except ValueError:
                await update.message.reply_text("❌ Invalid page numbers. Use numbers only.")
        elif waiting_for == 'channel_id':
            try:
                channel_id = int(update.message.text.strip())
                self.user_states[user_id]['channel_id'] = channel_id
                self.user_states[user_id]['waiting_for'] = None
                await self.post_quizzes_to_destination(user_id, channel_id, None, context)
            except ValueError:
                await update.message.reply_text("❌ Invalid Channel ID. Please send a valid number.")
        elif waiting_for == 'group_id':
            try:
                group_id = int(update.message.text.strip())
                self.user_states[user_id]['group_id'] = group_id
                self.user_states[user_id]['waiting_for'] = 'topic_id'
                await update.message.reply_text("🔢 Please send the *Topic ID* (Message Thread ID)\n\nExample: 123\n\nIf the group doesn't have topics enabled, send 0", parse_mode='Markdown')
            except ValueError:
                await update.message.reply_text("❌ Invalid Group ID. Please send a valid number.")
        elif waiting_for == 'topic_id':
            try:
                topic_id = int(update.message.text.strip())
                group_id = self.user_states[user_id]['group_id']
                self.user_states[user_id]['waiting_for'] = None
                message_thread_id = topic_id if topic_id > 0 else None
                await self.post_quizzes_to_destination(user_id, group_id, message_thread_id, context)
            except ValueError:
                await update.message.reply_text("❌ Invalid Topic ID. Please send a valid number or 0.")
    
    async def add_to_queue_direct(self, user_id: int, page_range: Optional[tuple], context: ContextTypes.DEFAULT_TYPE):
        if user_id not in self.user_states:
            return
        mode = self.user_states[user_id].get('mode', 'extraction')
        content_type = self.user_states[user_id]['content_type']
        content_paths = self.user_states[user_id]['content_paths']
        task_data = {'content_type': content_type, 'content_paths': content_paths, 'page_range': page_range, 'mode': mode, 'context': context}
        position = task_queue.add_task(user_id, task_data)
        if position == -1:
            message_text = "❌ Queue is full. Please try again later."
        elif position == -2:
            message_text = "⚠️ You already have a task in queue or being processed."
        else:
            mode_name = "Extraction" if mode == "extraction" else "Generation"
            content_desc = f"{len(content_paths)} image(s)" if content_type == 'images' else "PDF"
            message_text = f"✅ Added to queue!\n📋 Position: {position}\n⏳ Estimated wait: ~{position * 2} minutes\n📄 Content: {content_desc}\n🎯 Mode: {mode_name}\n🤖 {config.GEMINI_MODEL}"
        await context.bot.send_message(chat_id=user_id, text=message_text)
    
    async def process_content(self, user_id: int, content_type: str, content_paths: List[Path], page_range: Optional[tuple], mode: str, context: ContextTypes.DEFAULT_TYPE):
        try:
            mode_name = "Extraction" if mode == "extraction" else "Generation"
            mode_emoji = "📤" if mode == "extraction" else "✨"
            if content_type == 'pdf':
                pdf_path = content_paths[0]
                message = await context.bot.send_message(chat_id=user_id, text=f"🔄 Processing your PDF...\n{mode_emoji} Mode: {mode_name}\n🤖 Using {config.GEMINI_MODEL}")
                await message.edit_text("📄 Converting PDF to images...")
                images = await PDFProcessor.pdf_to_images(pdf_path, page_range)
            else:
                message = await context.bot.send_message(chat_id=user_id, text=f"🔄 Processing your {len(content_paths)} image(s)...\n{mode_emoji} Mode: {mode_name}\n🤖 Using {config.GEMINI_MODEL}")
                await message.edit_text("🖼️ Loading images...")
                images = []
                for img_path in content_paths:
                    img = await ImageProcessor.load_image(img_path)
                    images.append(img)
            await message.edit_text(f"🖼️ Processing {len(images)} image(s) in parallel...\n⚡ Using {config.MAX_CONCURRENT_IMAGES} parallel workers\n{mode_emoji} Mode: {mode_name}\n🤖 AI Model: {config.GEMINI_MODEL}")
            async def update_progress(current: int, total: int):
                try:
                    progress = (current / total) * 100
                    await message.edit_text(f"🔍 Processing: {current}/{total}\n📊 Progress: {progress:.1f}%\n{mode_emoji} {mode_name} Mode\n⚡ Parallel processing enabled")
                except:
                    pass
            all_questions = await PDFProcessor.process_images_parallel(images, mode, update_progress)
            if not all_questions:
                await message.edit_text("❌ No questions found in the content")
                return
            await message.edit_text(f"📊 Generating CSV with {len(all_questions)} questions...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = f"{user_id}_{timestamp}"
            csv_path = config.OUTPUT_DIR / f"questions_{mode}_{session_id}.csv"
            CSVGenerator.questions_to_csv(all_questions, csv_path)
            self.user_states[user_id] = {'questions': all_questions, 'session_id': session_id, 'csv_path': csv_path}
            await message.edit_text("✅ Sending CSV file...")
            keyboard = [[InlineKeyboardButton("📢 Post Quizzes to Channel/Group", callback_data=f"post_quizzes_{session_id}")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            with open(csv_path, 'rb') as csv_file:
                await context.bot.send_document(chat_id=user_id, document=csv_file, filename=f"mcq_{mode}_{timestamp}.csv", caption=f"✅ {len(all_questions)} questions processed!\n{mode_emoji} Mode: {mode_name}\n🤖 {config.GEMINI_MODEL}\n\nYou can now post these quizzes to a channel or group!", reply_markup=reply_markup)
            await message.edit_text(f"✅ Done!\n\n📝 Total questions: {len(all_questions)}\n📄 Content processed: {len(images)} image(s)\n{mode_emoji} Mode: {mode_name}\n⚡ Processing: {config.MAX_CONCURRENT_IMAGES}x speed")
            for path in content_paths:
                if path.exists():
                    path.unlink(missing_ok=True)
        except Exception as e:
            await context.bot.send_message(chat_id=user_id, text=f"❌ Error processing content: {str(e)}")
            for path in content_paths:
                if path.exists():
                    path.unlink(missing_ok=True)
            if user_id in self.user_states:
                del self.user_states[user_id]
    
    async def post_quizzes_to_destination(self, user_id: int, chat_id: int, message_thread_id: Optional[int], context: ContextTypes.DEFAULT_TYPE):
        if user_id not in self.user_states or 'questions' not in self.user_states[user_id]:
            await context.bot.send_message(chat_id=user_id, text="❌ No questions available to post.")
            return
        questions = self.user_states[user_id]['questions']
        destination = "channel" if message_thread_id is None and chat_id < 0 else "group"
        topic_info = f" (Topic ID: {message_thread_id})" if message_thread_id else ""
        status_msg = await context.bot.send_message(chat_id=user_id, text=f"📢 Posting {len(questions)} quizzes to {destination}{topic_info}...\n\n⏳ This may take a while due to rate limiting.\n\n🔄 Progress: 0/{len(questions)}\n\n*Quiz Format:*\n• Marker: {config.QUIZ_MARKER}\n• Tag: [{config.EXPLANATION_TAG}]", parse_mode='Markdown')
        async def update_posting_progress(current: int, total: int):
            try:
                progress = (current / total) * 100
                est_time = ((total - current) * config.POLL_DELAY) // 60
                await status_msg.edit_text(f"📢 Posting quizzes to {destination}{topic_info}...\n\n🔄 Progress: {current}/{total}\n📊 {progress:.1f}% completed\n⏱️ Estimated time: ~{est_time} min\n\n*Quiz Format:*\n• Marker: {config.QUIZ_MARKER}\n• Tag: [{config.EXPLANATION_TAG}]", parse_mode='Markdown')
            except:
                pass
        try:
            result = await QuizPoster.post_quizzes_batch(context=context, chat_id=chat_id, questions=questions, message_thread_id=message_thread_id, progress_callback=update_posting_progress)
            await status_msg.edit_text(f"✅ Posting Complete!\n\n📊 Total: {result['total']}\n✅ Success: {result['success']}\n❌ Failed: {result['failed']}\n⏭️ Skipped: {result['skipped']}\n\nDestination: {destination}{topic_info}")
            csv_path = self.user_states[user_id].get('csv_path')
            if csv_path and csv_path.exists():
                csv_path.unlink(missing_ok=True)
            if user_id in self.user_states:
                del self.user_states[user_id]
        except Exception as e:
            await status_msg.edit_text(f"❌ Error posting quizzes: {str(e)}")

def main():
    bot = TelegramBot()
    application = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
    application.post_init = bot.post_init
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("help", bot.help_command))
    application.add_handler(CommandHandler("model", bot.model_command))
    application.add_handler(CommandHandler("queue", bot.queue_command))
    application.add_handler(CommandHandler("cancel", bot.cancel_command))
    application.add_handler(MessageHandler(filters.Document.PDF, bot.handle_document))
    application.add_handler(MessageHandler(filters.PHOTO, bot.handle_photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_text))
    application.add_handler(CallbackQueryHandler(bot.button_callback))
    print("🤖 Bot started successfully!")
    print(f"🤖 AI Model: {config.GEMINI_MODEL}")
    print(f"⚡ Parallel processing: {config.MAX_CONCURRENT_IMAGES} concurrent images")
    print(f"📋 Max queue size: {config.MAX_QUEUE_SIZE} tasks")
    print(f"🔑 API Keys: {len(config.GEMINI_API_KEYS)}")
    print(f"📄 Supports: PDF, Single Image, Multiple Images")
    print(f"📢 Quiz posting: Channels & Groups with rate limiting")
    print(f"🏷️ Quiz Marker: {config.QUIZ_MARKER}")
    print(f"🔗 Explanation Tag: {config.EXPLANATION_TAG}")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
