import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional
import asyncio
from datetime import datetime
from threading import Lock
import io
import traceback

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
from pymongo import MongoClient
from bson import ObjectId

from prompts import get_extraction_prompt, get_generation_prompt

class Config:
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    GEMINI_API_KEYS = os.getenv("GEMINI_API_KEYS", "").split(",")
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    TEMP_DIR = Path("temp")
    OUTPUT_DIR = Path("output")
    MAX_CONCURRENT_IMAGES = 5
    MAX_QUEUE_SIZE = 10
    GEMINI_MODEL = "gemini-2.0-flash"
    POLL_DELAY = 3
    BATCH_SIZE = 20
    BATCH_DELAY = 10
    GENERATION_CONFIG = {"temperature": 0.1, "top_p": 0.95, "top_k": 40, "max_output_tokens": 8192}
    SAFETY_SETTINGS = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
    ]
    
    def __init__(self):
        if not self.TELEGRAM_BOT_TOKEN:
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required!")
        if not self.GEMINI_API_KEYS or self.GEMINI_API_KEYS == ['']:
            raise ValueError("GEMINI_API_KEYS environment variable is required!")
        self.TEMP_DIR.mkdir(exist_ok=True)
        self.OUTPUT_DIR.mkdir(exist_ok=True)

config = Config()

class MongoDB:
    def __init__(self):
        self.client = MongoClient(config.MONGODB_URI)
        self.db = self.client['telegram_quiz_bot']
        self.users = self.db['users']
        self.channels = self.db['channels']
        self.groups = self.db['groups']
        print("✅ MongoDB connected")
    
    def get_user_settings(self, user_id: int) -> Dict:
        user = self.users.find_one({'user_id': user_id})
        if not user:
            default_settings = {
                'user_id': user_id,
                'quiz_marker': os.getenv("QUIZ_MARKER", "[TSS]"),
                'explanation_tag': os.getenv("EXPLANATION_TAG", "t.me/tss"),
                'created_at': datetime.now()
            }
            self.users.insert_one(default_settings)
            return default_settings
        return user
    
    def update_user_settings(self, user_id: int, key: str, value: str):
        self.users.update_one({'user_id': user_id}, {'$set': {key: value, 'updated_at': datetime.now()}}, upsert=True)
    
    def add_channel(self, user_id: int, channel_id: int, channel_name: str):
        channel = {'user_id': user_id, 'channel_id': channel_id, 'channel_name': channel_name, 'type': 'channel', 'created_at': datetime.now()}
        existing = self.channels.find_one({'user_id': user_id, 'channel_id': channel_id})
        if existing:
            self.channels.update_one({'_id': existing['_id']}, {'$set': {'channel_name': channel_name, 'updated_at': datetime.now()}})
        else:
            self.channels.insert_one(channel)
    
    def add_group(self, user_id: int, group_id: int, group_name: str):
        group = {'user_id': user_id, 'group_id': group_id, 'group_name': group_name, 'type': 'group', 'created_at': datetime.now()}
        existing = self.groups.find_one({'user_id': user_id, 'group_id': group_id})
        if existing:
            self.groups.update_one({'_id': existing['_id']}, {'$set': {'group_name': group_name, 'updated_at': datetime.now()}})
        else:
            self.groups.insert_one(group)
    
    def get_user_channels(self, user_id: int) -> List[Dict]:
        return list(self.channels.find({'user_id': user_id}))
    
    def get_user_groups(self, user_id: int) -> List[Dict]:
        return list(self.groups.find({'user_id': user_id}))
    
    def delete_channel(self, channel_doc_id: str):
        self.channels.delete_one({'_id': ObjectId(channel_doc_id)})
    
    def delete_group(self, group_doc_id: str):
        self.groups.delete_one({'_id': ObjectId(group_doc_id)})

db = MongoDB()

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

class CSVParser:
    @staticmethod
    def parse_csv_file(file_content: bytes) -> List[Dict]:
        questions = []
        try:
            content = file_content.decode('utf-8')
            csv_reader = csv.DictReader(io.StringIO(content))
            for row in csv_reader:
                if not row.get('questions'):
                    continue
                options = []
                for i in range(1, 5):
                    option_key = f'option{i}'
                    if option_key in row and row[option_key]:
                        options.append(row[option_key])
                if len(options) < 2:
                    continue
                try:
                    answer_num = int(row.get('answer', '1'))
                    correct_answer_index = answer_num - 1
                    if correct_answer_index < 0 or correct_answer_index >= len(options):
                        correct_answer_index = 0
                except (ValueError, TypeError):
                    correct_answer_index = 0
                question = {
                    'question_description': row.get('questions', '').strip(),
                    'options': options,
                    'correct_answer_index': correct_answer_index,
                    'correct_option': chr(65 + correct_answer_index),
                    'explanation': row.get('explanation', '').strip()
                }
                questions.append(question)
            return questions
        except Exception as e:
            print(f"Error parsing CSV: {str(e)}")
            raise Exception(f"Failed to parse CSV file: {str(e)}")

class QuizPoster:
    @staticmethod
    def format_question(question_text: str, quiz_marker: str) -> str:
        formatted = f"{quiz_marker}\n\n{question_text}"
        if len(formatted) > 300:
            max_question_len = 300 - len(quiz_marker) - 3
            formatted = f"{quiz_marker}\n\n{question_text[:max_question_len-3]}..."
        return formatted
    
    @staticmethod
    def format_explanation(explanation: str, explanation_tag: str) -> str:
        if not explanation:
            return None
        formatted = f"{explanation} [{explanation_tag}]"
        if len(formatted) > 200:
            tag_with_brackets = f" [{explanation_tag}]"
            max_explanation_len = 200 - len(tag_with_brackets) - 3
            formatted = f"{explanation[:max_explanation_len]}...{tag_with_brackets}"
        return formatted
    
    @staticmethod
    async def send_quiz_with_retry(context: ContextTypes.DEFAULT_TYPE, chat_id: int, question: Dict, quiz_marker: str, explanation_tag: str, message_thread_id: Optional[int] = None, max_retries: int = 5) -> bool:
        for attempt in range(max_retries):
            try:
                question_text = question.get('question_description', 'Question')
                options = question.get('options', [])[:10]
                correct_option_id = question.get('correct_answer_index', 0)
                explanation = question.get('explanation', '')
                
                if len(options) < 2:
                    print(f"⏭️ Skipping question with less than 2 options")
                    return False
                
                if correct_option_id < 0 or correct_option_id >= len(options):
                    print(f"⚠️ Invalid correct_option_id: {correct_option_id} for {len(options)} options, defaulting to 0")
                    correct_option_id = 0
                
                formatted_question = QuizPoster.format_question(question_text, quiz_marker)
                formatted_explanation = QuizPoster.format_explanation(explanation, explanation_tag) if explanation else None
                
                print(f"📤 Sending quiz to chat_id={chat_id}, thread={message_thread_id}")
                print(f"   Question: {formatted_question[:50]}...")
                print(f"   Options: {len(options)}, Correct: {correct_option_id}")
                
                await context.bot.send_poll(
                    chat_id=chat_id,
                    question=formatted_question,
                    options=options,
                    type='quiz',
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
                
                if "question is too long" in error_msg.lower():
                    print("Question too long, skipping...")
                    return False
                elif "not enough rights" in error_msg.lower() or "have no rights" in error_msg.lower():
                    print("❌ Bot doesn't have permission to send polls in this chat")
                    print(f"   Chat ID: {chat_id}")
                    print(f"   Please make sure the bot is an admin with 'Post Messages' and 'Send Polls' permissions")
                    return False
                elif "chat not found" in error_msg.lower():
                    print("❌ Chat not found or bot is not a member")
                    return False
                elif "forbidden" in error_msg.lower():
                    print("❌ Bot is forbidden from posting (might be banned or kicked)")
                    return False
                
                if attempt == max_retries - 1:
                    print(f"❌ Max retries reached, giving up on this quiz")
                    return False
                
                await asyncio.sleep(3)
        
        return False
    
    @staticmethod
    async def post_quizzes_batch(context: ContextTypes.DEFAULT_TYPE, chat_id: int, questions: List[Dict], quiz_marker: str, explanation_tag: str, message_thread_id: Optional[int] = None, progress_callback=None) -> Dict:
        total = len(questions)
        success_count = 0
        failed_count = 0
        skipped_count = 0
        
        print(f"📦 Starting batch posting: {total} questions to chat_id={chat_id}")
        
        for i in range(0, total, config.BATCH_SIZE):
            batch = questions[i:i + config.BATCH_SIZE]
            batch_num = (i // config.BATCH_SIZE) + 1
            total_batches = (total + config.BATCH_SIZE - 1) // config.BATCH_SIZE
            
            print(f"📦 Processing batch {batch_num}/{total_batches}")
            
            for idx, question in enumerate(batch):
                global_idx = i + idx + 1
                
                if progress_callback:
                    await progress_callback(global_idx, total)
                
                if not question.get('question_description') or not question.get('options'):
                    print(f"⏭️ Skipping question {global_idx} - missing data")
                    skipped_count += 1
                    continue
                
                success = await QuizPoster.send_quiz_with_retry(
                    context, chat_id, question, quiz_marker, explanation_tag, message_thread_id
                )
                
                if success:
                    success_count += 1
                else:
                    failed_count += 1
                
                if global_idx < total:
                    await asyncio.sleep(config.POLL_DELAY)
            
            if i + config.BATCH_SIZE < total:
                print(f"✅ Batch {batch_num} completed. Waiting {config.BATCH_DELAY}s before next batch...")
                await asyncio.sleep(config.BATCH_DELAY)
        
        result = {
            'total': total,
            'success': success_count,
            'failed': failed_count,
            'skipped': skipped_count
        }
        
        print(f"📊 Batch posting complete: {result}")
        return result

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
                client = genai.Client(api_key=api_key)
                
                model_config = types.GenerateContentConfig(
                    temperature=config.GENERATION_CONFIG['temperature'],
                    top_p=config.GENERATION_CONFIG['top_p'],
                    top_k=config.GENERATION_CONFIG['top_k'],
                    max_output_tokens=config.GENERATION_CONFIG['max_output_tokens'],
                    safety_settings=[
                        types.SafetySetting(category='HARM_CATEGORY_HARASSMENT', threshold='BLOCK_NONE'),
                        types.SafetySetting(category='HARM_CATEGORY_HATE_SPEECH', threshold='BLOCK_NONE'),
                        types.SafetySetting(category='HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold='BLOCK_NONE'),
                        types.SafetySetting(category='HARM_CATEGORY_DANGEROUS_CONTENT', threshold='BLOCK_NONE')
                    ]
                )
                
                prompt = get_extraction_prompt() if mode == "extraction" else get_generation_prompt()
                
                print(f"Processing image {image_idx} in {mode} mode with {config.GEMINI_MODEL}")
                
                response = client.models.generate_content(
                    model=config.GEMINI_MODEL,
                    contents=[prompt, image],
                    config=model_config
                )
                
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
                        await self.bot.process_content(
                            user_id=user_id,
                            content_type=task_data['content_type'],
                            content_paths=task_data['content_paths'],
                            page_range=task_data.get('page_range'),
                            mode=task_data['mode'],
                            context=task_data['context']
                        )
                    except Exception as e:
                        print(f"Error processing task for user {user_id}: {str(e)}")
                        try:
                            await task_data['context'].bot.send_message(
                                chat_id=user_id,
                                text=f"❌ Error processing your content: {str(e)}"
                            )
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
        user_id = update.effective_user.id
        settings = db.get_user_settings(user_id)
        await update.message.reply_text(
            f"Welcome! 👋\n\n"
            f"Send me:\n"
            f"📄 PDF file\n"
            f"🖼️ Single image\n"
            f"🖼️🖼️ Multiple images\n"
            f"📊 CSV file (to post quizzes)\n\n"
            f"🤖 Powered by Gemini 2.0 Flash\n"
            f"📢 Quiz Marker: {settings['quiz_marker']}\n"
            f"🔗 Explanation Tag: {settings['explanation_tag']}\n\n"
            f"Commands:\n"
            f"/start - Start the bot\n"
            f"/help - Show help message\n"
            f"/settings - Configure channels, groups & markers\n"
            f"/info - Get chat information\n"
            f"/queue - Check your queue position\n"
            f"/cancel - Cancel your current task\n"
            f"/model - Show current AI model info"
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "📚 *How to use:*\n\n"
            "*Generate Quizzes:*\n"
            "1. Send PDF/images\n"
            "2. Choose mode (Extraction/Generation)\n"
            "3. Get CSV file\n"
            "4. Post to saved channels/groups\n\n"
            "*Post from CSV:*\n"
            "1. Send CSV file\n"
            "2. Choose destination\n"
            "3. Quizzes posted automatically\n\n"
            "*Settings:*\n"
            "Use /settings to:\n"
            "• Add/remove channels\n"
            "• Add/remove groups\n"
            "• Change quiz marker\n"
            "• Change explanation tag\n\n"
            "*CSV Format:*\n"
            "questions,option1,option2,option3,option4,option5,answer,explanation,type,section\n\n"
            "*Features:*\n"
            "✓ Gemini 2.0 Flash AI\n"
            "✓ PDF, images, CSV support\n"
            "✓ Multi-channel/group posting\n"
            "✓ Topic support for groups\n"
            "✓ Rate limiting protection\n"
            "✓ Bengali explanations",
            parse_mode='Markdown'
        )
    
    async def settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        settings = db.get_user_settings(user_id)
        channels = db.get_user_channels(user_id)
        groups = db.get_user_groups(user_id)
        
        keyboard = [
            [InlineKeyboardButton("➕ Add Channel", callback_data="settings_add_channel")],
            [InlineKeyboardButton("➕ Add Group", callback_data="settings_add_group")],
            [InlineKeyboardButton("📺 Manage Channels", callback_data="settings_manage_channels")],
            [InlineKeyboardButton("👥 Manage Groups", callback_data="settings_manage_groups")],
            [InlineKeyboardButton("🏷️ Change Quiz Marker", callback_data="settings_change_marker")],
            [InlineKeyboardButton("🔗 Change Explanation Tag", callback_data="settings_change_tag")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            f"⚙️ *Settings*\n\n"
            f"📢 Quiz Marker: `{settings['quiz_marker']}`\n"
            f"🔗 Explanation Tag: `{settings['explanation_tag']}`\n\n"
            f"📺 Channels: {len(channels)}\n"
            f"👥 Groups: {len(groups)}\n\n"
            f"Choose an option:",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    async def info_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat = update.effective_chat
        info_text = f"📊 *Chat Information*\n\n"
        info_text += f"🆔 Chat ID: `{chat.id}`\n"
        info_text += f"📛 Title: {chat.title or 'N/A'}\n"
        info_text += f"📝 Type: {chat.type}\n"
        info_text += f"👤 Username: @{chat.username or 'N/A'}\n"
        
        try:
            if chat.type in ['supergroup', 'group', 'channel']:
                chat_full = await context.bot.get_chat(chat.id)
                info_text += f"\n🔧 *Additional Info:*\n"
                
                if hasattr(chat_full, 'description') and chat_full.description:
                    info_text += f"📄 Description: {chat_full.description[:100]}...\n"
                
                if hasattr(chat_full, 'permissions') and chat_full.permissions:
                    info_text += f"✅ Can send polls: {chat_full.permissions.can_send_polls}\n"
                
                if chat.type == 'supergroup':
                    try:
                        forums_enabled = getattr(chat_full, 'is_forum', False)
                        info_text += f"📑 Topics enabled: {forums_enabled}\n"
                        
                        if forums_enabled:
                            info_text += f"\n💡 *Tip:* This group has topics enabled!\n"
                            info_text += f"To get topic IDs:\n"
                            info_text += f"1. Send a message in the desired topic\n"
                            info_text += f"2. Forward it to @userinfobot\n"
                            info_text += f"3. Look for 'message_thread_id'\n"
                    except:
                        pass
                
                # Check bot permissions
                try:
                    bot_member = await context.bot.get_chat_member(chat.id, context.bot.id)
                    info_text += f"\n🤖 *Bot Status:*\n"
                    info_text += f"Status: {bot_member.status}\n"
                    
                    if hasattr(bot_member, 'can_post_messages'):
                        info_text += f"Can post: {bot_member.can_post_messages}\n"
                    if hasattr(bot_member, 'can_send_polls'):
                        info_text += f"Can send polls: {bot_member.can_send_polls}\n"
                except Exception as e:
                    info_text += f"\n⚠️ Could not check bot permissions: {str(e)}\n"
        
        except Exception as e:
            info_text += f"\n⚠️ Could not fetch additional info: {str(e)}\n"
        
        await update.message.reply_text(info_text, parse_mode='Markdown')
    
    async def model_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        queue_size = task_queue.get_queue_size()
        await update.message.reply_text(
            f"🤖 *AI Model Information:*\n\n"
            f"Model: `{config.GEMINI_MODEL}`\n"
            f"Temperature: {config.GENERATION_CONFIG['temperature']}\n"
            f"Max Tokens: {config.GENERATION_CONFIG['max_output_tokens']}\n"
            f"Parallel Workers: {config.MAX_CONCURRENT_IMAGES}\n"
            f"API Keys: {len(config.GEMINI_API_KEYS)}\n"
            f"Queue Size: {queue_size}/{config.MAX_QUEUE_SIZE}\n\n"
            f"*Rate Limiting:*\n"
            f"Poll Delay: {config.POLL_DELAY}s\n"
            f"Batch Size: {config.BATCH_SIZE}\n"
            f"Batch Delay: {config.BATCH_DELAY}s",
            parse_mode='Markdown'
        )
    
    async def queue_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        
        if task_queue.is_processing(user_id):
            await update.message.reply_text("⚙️ Your task is currently being processed!")
        else:
            position = task_queue.get_position(user_id)
            if position > 0:
                await update.message.reply_text(
                    f"📋 Your position in queue: {position}\n"
                    f"⏳ Estimated wait: ~{position * 2} minutes"
                )
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
        
        if document.file_name.endswith('.csv'):
            await self.handle_csv(update, context)
            return
        
        if not document.file_name.endswith('.pdf'):
            await update.message.reply_text("❌ Please send a PDF or CSV file only.")
            return
        
        if user_id in self.user_states or task_queue.is_processing(user_id):
            await update.message.reply_text("⚠️ You already have a task in progress.\nUse /cancel to cancel the current task.")
            return
        
        processing_msg = await update.message.reply_text("📥 Downloading PDF...")
        
        try:
            file = await context.bot.get_file(document.file_id)
            pdf_path = config.TEMP_DIR / f"{user_id}_{document.file_name}"
            await file.download_to_drive(pdf_path)
            
            keyboard = [
                [InlineKeyboardButton("📤 Extraction Mode", callback_data="mode_extraction")],
                [InlineKeyboardButton("✨ Generation Mode", callback_data="mode_generation")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            self.user_states[user_id] = {'content_type': 'pdf', 'content_paths': [pdf_path]}
            
            await processing_msg.edit_text(
                "📄 PDF received!\n\n"
                "Choose processing mode:\n\n"
                "📤 *Extraction Mode*\n"
                "Extract existing MCQs from PDF\n\n"
                "✨ *Generation Mode*\n"
                "Generate new MCQs from textbook content",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        except Exception as e:
            await processing_msg.edit_text(f"❌ Error downloading PDF: {str(e)}")
    
    async def handle_csv(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        document = update.message.document
        
        if user_id in self.user_states or task_queue.is_processing(user_id):
            await update.message.reply_text("⚠️ You already have a task in progress.\nUse /cancel to cancel the current task.")
            return
        
        processing_msg = await update.message.reply_text("📊 Processing CSV file...")
        
        try:
            file = await context.bot.get_file(document.file_id)
            file_content = await file.download_as_bytearray()
            
            await processing_msg.edit_text("📊 Parsing CSV questions...")
            questions = CSVParser.parse_csv_file(bytes(file_content))
            
            if not questions:
                await processing_msg.edit_text(
                    "❌ No valid questions found in CSV file.\n\n"
                    "Make sure your CSV has:\n"
                    "• questions column\n"
                    "• option1, option2, option3, option4 columns\n"
                    "• answer column (1-4)\n"
                    "• explanation column (optional)"
                )
                return
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = f"csv_{user_id}_{timestamp}"
            settings = db.get_user_settings(user_id)
            
            self.user_states[user_id] = {
                'questions': questions,
                'session_id': session_id,
                'source': 'csv'
            }
            
            print(f"CSV processed - User: {user_id}, Session: {session_id}, Questions: {len(questions)}")
            
            keyboard = [[InlineKeyboardButton("📢 Post Quizzes", callback_data=f"post_{session_id}")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await processing_msg.edit_text(
                f"✅ CSV Processed!\n\n"
                f"📊 Total questions: {len(questions)}\n"
                f"📢 Quiz Marker: {settings['quiz_marker']}\n"
                f"🔗 Explanation Tag: {settings['explanation_tag']}\n\n"
                f"Ready to post quizzes!",
                reply_markup=reply_markup
            )
        except Exception as e:
            await processing_msg.edit_text(f"❌ Error processing CSV: {str(e)}\n\nMake sure your CSV follows the correct format.")
    
    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        
        if user_id in self.user_states or task_queue.is_processing(user_id):
            await update.message.reply_text("⚠️ You already have a task in progress.\nUse /cancel to cancel the current task.")
            return
        
        if update.message.media_group_id:
            if user_id not in self.user_states:
                self.user_states[user_id] = {
                    'content_type': 'images',
                    'content_paths': [],
                    'media_group_id': update.message.media_group_id,
                    'waiting_for_more': True
                }
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
                
                keyboard = [
                    [InlineKeyboardButton("📤 Extraction Mode", callback_data="mode_extraction")],
                    [InlineKeyboardButton("✨ Generation Mode", callback_data="mode_generation")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                self.user_states[user_id] = {'content_type': 'images', 'content_paths': [image_path]}
                
                await processing_msg.edit_text(
                    "🖼️ Image received!\n\n"
                    "Choose processing mode:\n\n"
                    "📤 *Extraction Mode*\n"
                    "Extract existing MCQs from image\n\n"
                    "✨ *Generation Mode*\n"
                    "Generate new MCQs from textbook content",
                    reply_markup=reply_markup,
                    parse_mode='Markdown'
                )
            except Exception as e:
                await processing_msg.edit_text(f"❌ Error downloading image: {str(e)}")
    
    async def process_media_group_delayed(self, user_id: int, context: ContextTypes.DEFAULT_TYPE):
        await asyncio.sleep(2)
        
        if user_id in self.user_states and self.user_states[user_id].get('waiting_for_more'):
            self.user_states[user_id]['waiting_for_more'] = False
            num_images = len(self.user_states[user_id]['content_paths'])
            
            keyboard = [
                [InlineKeyboardButton("📤 Extraction Mode", callback_data="mode_extraction")],
                [InlineKeyboardButton("✨ Generation Mode", callback_data="mode_generation")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await context.bot.send_message(
                chat_id=user_id,
                text=f"🖼️ {num_images} images received!\n\n"
                     f"Choose processing mode:\n\n"
                     f"📤 *Extraction Mode*\n"
                     f"Extract existing MCQs from images\n\n"
                     f"✨ *Generation Mode*\n"
                     f"Generate new MCQs from textbook content",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
    
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        user_id = update.effective_user.id
        callback_data = query.data
        
        print(f"Button clicked - User: {user_id}, Data: {callback_data}")
        
        # Settings callbacks
        if callback_data == "settings_add_channel":
            await query.edit_message_text(
                "📺 Please send the *Channel ID and Name* in format:\n\n"
                "`channel_id channel_name`\n\n"
                "Example: `-1001234567890 My Channel`\n\n"
                "You can get the ID by forwarding a message from the channel to @userinfobot",
                parse_mode='Markdown'
            )
            self.user_states[user_id] = {'waiting_for': 'add_channel'}
        
        elif callback_data == "settings_add_group":
            await query.edit_message_text(
                "👥 Please send the *Group ID and Name* in format:\n\n"
                "`group_id group_name`\n\n"
                "Example: `-1001234567890 My Group`\n\n"
                "You can get the ID by forwarding a message from the group to @userinfobot",
                parse_mode='Markdown'
            )
            self.user_states[user_id] = {'waiting_for': 'add_group'}
        
        elif callback_data == "settings_manage_channels":
            channels = db.get_user_channels(user_id)
            if not channels:
                await query.edit_message_text("❌ No channels saved.\n\nUse /settings to add channels.")
                return
            
            keyboard = []
            for ch in channels:
                keyboard.append([InlineKeyboardButton(
                    f"❌ {ch['channel_name']}",
                    callback_data=f"del_ch_{str(ch['_id'])}"
                )])
            keyboard.append([InlineKeyboardButton("⬅️ Back", callback_data="settings_back")])
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                "📺 *Manage Channels*\n\nClick to delete:",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        
        elif callback_data == "settings_manage_groups":
            groups = db.get_user_groups(user_id)
            if not groups:
                await query.edit_message_text("❌ No groups saved.\n\nUse /settings to add groups.")
                return
            
            keyboard = []
            for gr in groups:
                keyboard.append([InlineKeyboardButton(
                    f"❌ {gr['group_name']}",
                    callback_data=f"del_gr_{str(gr['_id'])}"
                )])
            keyboard.append([InlineKeyboardButton("⬅️ Back", callback_data="settings_back")])
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                "👥 *Manage Groups*\n\nClick to delete:",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        
        elif callback_data.startswith("del_ch_"):
            channel_id = callback_data[7:]
            db.delete_channel(channel_id)
            await query.answer("✅ Channel deleted!")
            
            channels = db.get_user_channels(user_id)
            if not channels:
                await query.edit_message_text("❌ No channels saved.\n\nUse /settings to add channels.")
                return
            
            keyboard = []
            for ch in channels:
                keyboard.append([InlineKeyboardButton(
                    f"❌ {ch['channel_name']}",
                    callback_data=f"del_ch_{str(ch['_id'])}"
                )])
            keyboard.append([InlineKeyboardButton("⬅️ Back", callback_data="settings_back")])
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                "📺 *Manage Channels*\n\nClick to delete:",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        
        elif callback_data.startswith("del_gr_"):
            group_id = callback_data[7:]
            db.delete_group(group_id)
            await query.answer("✅ Group deleted!")
            
            groups = db.get_user_groups(user_id)
            if not groups:
                await query.edit_message_text("❌ No groups saved.\n\nUse /settings to add groups.")
                return
            
            keyboard = []
            for gr in groups:
                keyboard.append([InlineKeyboardButton(
                    f"❌ {gr['group_name']}",
                    callback_data=f"del_gr_{str(gr['_id'])}"
                )])
            keyboard.append([InlineKeyboardButton("⬅️ Back", callback_data="settings_back")])
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                "👥 *Manage Groups*\n\nClick to delete:",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        
        elif callback_data == "settings_change_marker":
            current_marker = db.get_user_settings(user_id)['quiz_marker']
            await query.edit_message_text(
                f"🏷️ Please send the new *Quiz Marker*\n\n"
                f"Example: `[TSS]`\n\n"
                f"Current: {current_marker}",
                parse_mode='Markdown'
            )
            self.user_states[user_id] = {'waiting_for': 'change_marker'}
        
        elif callback_data == "settings_change_tag":
            current_tag = db.get_user_settings(user_id)['explanation_tag']
            await query.edit_message_text(
                f"🔗 Please send the new *Explanation Tag*\n\n"
                f"Example: `t.me/tss`\n\n"
                f"Current: {current_tag}",
                parse_mode='Markdown'
            )
            self.user_states[user_id] = {'waiting_for': 'change_tag'}
        
        elif callback_data == "settings_back":
            settings = db.get_user_settings(user_id)
            channels = db.get_user_channels(user_id)
            groups = db.get_user_groups(user_id)
            
            keyboard = [
                [InlineKeyboardButton("➕ Add Channel", callback_data="settings_add_channel")],
                [InlineKeyboardButton("➕ Add Group", callback_data="settings_add_group")],
                [InlineKeyboardButton("📺 Manage Channels", callback_data="settings_manage_channels")],
                [InlineKeyboardButton("👥 Manage Groups", callback_data="settings_manage_groups")],
                [InlineKeyboardButton("🏷️ Change Quiz Marker", callback_data="settings_change_marker")],
                [InlineKeyboardButton("🔗 Change Explanation Tag", callback_data="settings_change_tag")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                f"⚙️ *Settings*\n\n"
                f"📢 Quiz Marker: `{settings['quiz_marker']}`\n"
                f"🔗 Explanation Tag: `{settings['explanation_tag']}`\n\n"
                f"📺 Channels: {len(channels)}\n"
                f"👥 Groups: {len(groups)}\n\n"
                f"Choose an option:",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        
        # Mode selection
        elif callback_data.startswith("mode_"):
            mode = callback_data.split("_")[1]
            if user_id not in self.user_states:
                await query.edit_message_text("❌ Session expired. Please upload the file again.")
                return
            
            self.user_states[user_id]['mode'] = mode
            mode_name = "Extraction" if mode == "extraction" else "Generation"
            
            if self.user_states[user_id]['content_type'] == 'pdf':
                keyboard = [
                    [InlineKeyboardButton("All Pages", callback_data="all_pages")],
                    [InlineKeyboardButton("Specify Range", callback_data="specify_range")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await query.edit_message_text(
                    f"✅ Mode selected: *{mode_name}*\n\nChoose page range:",
                    reply_markup=reply_markup,
                    parse_mode='Markdown'
                )
            else:
                await query.edit_message_text(f"✅ Mode selected: *{mode_name}*\n\nAdding to queue...", parse_mode='Markdown')
                await self.add_to_queue_direct(user_id, None, context)
        
        elif callback_data == "all_pages":
            if user_id not in self.user_states:
                await query.edit_message_text("❌ Session expired. Please upload the file again.")
                return
            await self.add_to_queue_direct(user_id, None, context)
        
        elif callback_data == "specify_range":
            if user_id not in self.user_states:
                await query.edit_message_text("❌ Session expired. Please upload the file again.")
                return
            await query.edit_message_text("Please send the page range in format: start-end\nExample: 1-10 or 5-8")
            self.user_states[user_id]['waiting_for_range'] = True
        
        # Post quizzes
        elif callback_data.startswith("post_"):
            session_id = callback_data[5:]
            print(f"Post button clicked - User: {user_id}, Session ID: {session_id}")
            
            if user_id not in self.user_states:
                print(f"User {user_id} not in user_states")
                await query.edit_message_text("❌ Session expired. Please upload your file again.")
                return
            
            stored_session = self.user_states[user_id].get('session_id')
            print(f"Stored session: {stored_session}, Requested: {session_id}")
            
            if stored_session != session_id:
                print(f"Session mismatch for user {user_id}")
                await query.edit_message_text("❌ Session expired. Please try again.")
                return
            
            if 'questions' not in self.user_states[user_id]:
                print(f"No questions for user {user_id}")
                await query.edit_message_text("❌ No questions available. Please try again.")
                return
            
            print(f"Showing destination selection for user {user_id}")
            
            channels = db.get_user_channels(user_id)
            groups = db.get_user_groups(user_id)
            
            if not channels and not groups:
                await query.edit_message_text("❌ No channels or groups saved.\n\nUse /settings to add channels and groups first.")
                return
            
            keyboard = []
            for ch in channels:
                keyboard.append([InlineKeyboardButton(
                    f"📺 {ch['channel_name']}",
                    callback_data=f"dest_ch_{ch['channel_id']}_{session_id}"
                )])
            for gr in groups:
                keyboard.append([InlineKeyboardButton(
                    f"👥 {gr['group_name']}",
                    callback_data=f"dest_gr_{gr['group_id']}_{session_id}"
                )])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                "📢 *Post Quizzes*\n\nSelect destination:",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        
        elif callback_data.startswith("dest_ch_"):
            parts = callback_data.split("_")
            channel_id = int(parts[2])
            session_id = "_".join(parts[3:])
            
            print(f"Channel selected - User: {user_id}, Channel: {channel_id}, Session: {session_id}")
            
            if user_id not in self.user_states or self.user_states[user_id].get('session_id') != session_id:
                await query.edit_message_text("❌ Session expired. Please try again.")
                return
            
            await query.edit_message_text("📺 Posting to channel...")
            await self.post_quizzes_to_destination(user_id, channel_id, None, context, query.message)
        
        elif callback_data.startswith("dest_gr_"):
            parts = callback_data.split("_")
            group_id = int(parts[2])
            session_id = "_".join(parts[3:])
            
            print(f"Group selected - User: {user_id}, Group: {group_id}, Session: {session_id}")
            
            if user_id not in self.user_states or self.user_states[user_id].get('session_id') != session_id:
                await query.edit_message_text("❌ Session expired. Please try again.")
                return
            
            self.user_states[user_id]['selected_group'] = group_id
            await query.edit_message_text(
                "🔢 Please send the *Topic ID* (Message Thread ID)\n\n"
                "Example: 123\n\n"
                "If the group doesn't have topics enabled, send 0",
                parse_mode='Markdown'
            )
            self.user_states[user_id]['waiting_for'] = 'topic_id'
    
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        
        if user_id not in self.user_states:
            return
        
        waiting_for = self.user_states[user_id].get('waiting_for')
        text = update.message.text.strip()
        
        if waiting_for == 'add_channel':
            try:
                parts = text.split(" ", 1)
                if len(parts) < 2:
                    await update.message.reply_text("❌ Invalid format. Use: `channel_id channel_name`", parse_mode='Markdown')
                    return
                
                channel_id = int(parts[0])
                channel_name = parts[1]
                
                db.add_channel(user_id, channel_id, channel_name)
                await update.message.reply_text(f"✅ Channel added!\n\n📺 {channel_name}\n🆔 {channel_id}")
                del self.user_states[user_id]
            except ValueError:
                await update.message.reply_text("❌ Invalid channel ID. Please use a valid number.")
        
        elif waiting_for == 'add_group':
            try:
                parts = text.split(" ", 1)
                if len(parts) < 2:
                    await update.message.reply_text("❌ Invalid format. Use: `group_id group_name`", parse_mode='Markdown')
                    return
                
                group_id = int(parts[0])
                group_name = parts[1]
                
                db.add_group(user_id, group_id, group_name)
                await update.message.reply_text(f"✅ Group added!\n\n👥 {group_name}\n🆔 {group_id}")
                del self.user_states[user_id]
            except ValueError:
                await update.message.reply_text("❌ Invalid group ID. Please use a valid number.")
        
        elif waiting_for == 'change_marker':
            db.update_user_settings(user_id, 'quiz_marker', text)
            await update.message.reply_text(f"✅ Quiz marker updated to: {text}")
            del self.user_states[user_id]
        
        elif waiting_for == 'change_tag':
            db.update_user_settings(user_id, 'explanation_tag', text)
            await update.message.reply_text(f"✅ Explanation tag updated to: {text}")
            del self.user_states[user_id]
        
        elif self.user_states[user_id].get('waiting_for_range'):
            try:
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
        
        elif waiting_for == 'topic_id':
            try:
                topic_id = int(text)
                group_id = self.user_states[user_id]['selected_group']
                self.user_states[user_id]['waiting_for'] = None
                message_thread_id = topic_id if topic_id > 0 else None
                
                status_msg = await update.message.reply_text("👥 Posting to group...")
                await self.post_quizzes_to_destination(user_id, group_id, message_thread_id, context, status_msg)
            except ValueError:
                await update.message.reply_text("❌ Invalid Topic ID. Please send a valid number or 0.")
    
    async def add_to_queue_direct(self, user_id: int, page_range: Optional[tuple], context: ContextTypes.DEFAULT_TYPE):
        if user_id not in self.user_states:
            return
        
        mode = self.user_states[user_id].get('mode', 'extraction')
        content_type = self.user_states[user_id]['content_type']
        content_paths = self.user_states[user_id]['content_paths']
        
        task_data = {
            'content_type': content_type,
            'content_paths': content_paths,
            'page_range': page_range,
            'mode': mode,
            'context': context
        }
        
        position = task_queue.add_task(user_id, task_data)
        
        if position == -1:
            message_text = "❌ Queue is full. Please try again later."
        elif position == -2:
            message_text = "⚠️ You already have a task in queue or being processed."
        else:
            mode_name = "Extraction" if mode == "extraction" else "Generation"
            content_desc = f"{len(content_paths)} image(s)" if content_type == 'images' else "PDF"
            message_text = (
                f"✅ Added to queue!\n"
                f"📋 Position: {position}\n"
                f"⏳ Estimated wait: ~{position * 2} minutes\n"
                f"📄 Content: {content_desc}\n"
                f"🎯 Mode: {mode_name}\n"
                f"🤖 {config.GEMINI_MODEL}"
            )
        
        await context.bot.send_message(chat_id=user_id, text=message_text)
    
    async def process_content(self, user_id: int, content_type: str, content_paths: List[Path], page_range: Optional[tuple], mode: str, context: ContextTypes.DEFAULT_TYPE):
        try:
            mode_name = "Extraction" if mode == "extraction" else "Generation"
            mode_emoji = "📤" if mode == "extraction" else "✨"
            
            if content_type == 'pdf':
                pdf_path = content_paths[0]
                message = await context.bot.send_message(
                    chat_id=user_id,
                    text=f"🔄 Processing your PDF...\n{mode_emoji} Mode: {mode_name}\n🤖 Using {config.GEMINI_MODEL}"
                )
                await message.edit_text("📄 Converting PDF to images...")
                images = await PDFProcessor.pdf_to_images(pdf_path, page_range)
            else:
                message = await context.bot.send_message(
                    chat_id=user_id,
                    text=f"🔄 Processing your {len(content_paths)} image(s)...\n{mode_emoji} Mode: {mode_name}\n🤖 Using {config.GEMINI_MODEL}"
                )
                await message.edit_text("🖼️ Loading images...")
                images = []
                for img_path in content_paths:
                    img = await ImageProcessor.load_image(img_path)
                    images.append(img)
            
            await message.edit_text(
                f"🖼️ Processing {len(images)} image(s) in parallel...\n"
                f"⚡ Using {config.MAX_CONCURRENT_IMAGES} parallel workers\n"
                f"{mode_emoji} Mode: {mode_name}\n"
                f"🤖 AI Model: {config.GEMINI_MODEL}"
            )
            
            async def update_progress(current: int, total: int):
                try:
                    progress = (current / total) * 100
                    await message.edit_text(
                        f"🔍 Processing: {current}/{total}\n"
                        f"📊 Progress: {progress:.1f}%\n"
                        f"{mode_emoji} {mode_name} Mode\n"
                        f"⚡ Parallel processing enabled"
                    )
                except:
                    pass
            
            all_questions = await PDFProcessor.process_images_parallel(images, mode, update_progress)
            
            if not all_questions:
                await message.edit_text("❌ No questions found in the content")
                return
            
            await message.edit_text(f"📊 Generating CSV with {len(all_questions)} questions...")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = f"gen_{user_id}_{timestamp}"
            csv_path = config.OUTPUT_DIR / f"questions_{mode}_{session_id}.csv"
            
            CSVGenerator.questions_to_csv(all_questions, csv_path)
            
            settings = db.get_user_settings(user_id)
            
            self.user_states[user_id] = {
                'questions': all_questions,
                'session_id': session_id,
                'csv_path': csv_path,
                'source': 'generated'
            }
            
            print(f"Content processed - User: {user_id}, Session: {session_id}, Questions: {len(all_questions)}")
            
            await message.edit_text("✅ Sending CSV file...")
            
            keyboard = [[InlineKeyboardButton("📢 Post Quizzes", callback_data=f"post_{session_id}")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            with open(csv_path, 'rb') as csv_file:
                await context.bot.send_document(
                    chat_id=user_id,
                    document=csv_file,
                    filename=f"mcq_{mode}_{timestamp}.csv",
                    caption=f"✅ {len(all_questions)} questions processed!\n{mode_emoji} Mode: {mode_name}\n🤖 {config.GEMINI_MODEL}\n\nReady to post!",
                    reply_markup=reply_markup
                )
            
            await message.edit_text(
                f"✅ Done!\n\n"
                f"📝 Total questions: {len(all_questions)}\n"
                f"📄 Content processed: {len(images)} image(s)\n"
                f"{mode_emoji} Mode: {mode_name}\n"
                f"⚡ Processing: {config.MAX_CONCURRENT_IMAGES}x speed"
            )
            
            for path in content_paths:
                if path.exists():
                    path.unlink(missing_ok=True)
        
        except Exception as e:
            print(f"❌ Error in process_content: {str(e)}")
            traceback.print_exc()
            
            await context.bot.send_message(
                chat_id=user_id,
                text=f"❌ Error processing content: {str(e)}"
            )
            
            for path in content_paths:
                if path.exists():
                    path.unlink(missing_ok=True)
            
            if user_id in self.user_states:
                del self.user_states[user_id]
    
    async def post_quizzes_to_destination(self, user_id: int, chat_id: int, message_thread_id: Optional[int], context: ContextTypes.DEFAULT_TYPE, status_msg):
        print(f"🚀 Starting post_quizzes_to_destination - User: {user_id}, Chat: {chat_id}, Thread: {message_thread_id}")
        
        if user_id not in self.user_states or 'questions' not in self.user_states[user_id]:
            print(f"❌ No questions for user {user_id}")
            await context.bot.send_message(chat_id=user_id, text="❌ No questions available to post.")
            return
        
        questions = self.user_states[user_id]['questions']
        print(f"📊 Found {len(questions)} questions to post")
        
        source = self.user_states[user_id].get('source', 'generated')
        settings = db.get_user_settings(user_id)
        quiz_marker = settings['quiz_marker']
        explanation_tag = settings['explanation_tag']
        
        print(f"⚙️ Settings - Marker: {quiz_marker}, Tag: {explanation_tag}")
        
        destination = "channel" if message_thread_id is None else "group"
        topic_info = f" (Topic: {message_thread_id})" if message_thread_id else ""
        source_info = "📊 CSV" if source == 'csv' else "🤖 Generated"
        
        await status_msg.edit_text(
            f"📢 Posting {len(questions)} quizzes to {destination}{topic_info}...\n\n"
            f"{source_info}\n"
            f"⏳ This may take a while\n\n"
            f"🔄 Progress: 0/{len(questions)}\n\n"
            f"📢 Marker: {quiz_marker}\n"
            f"🔗 Tag: {explanation_tag}"
        )
        
        async def update_posting_progress(current: int, total: int):
            try:
                progress = (current / total) * 100
                est_time = ((total - current) * config.POLL_DELAY) // 60
                await status_msg.edit_text(
                    f"📢 Posting to {destination}{topic_info}...\n\n"
                    f"{source_info}\n"
                    f"🔄 Progress: {current}/{total}\n"
                    f"📊 {progress:.1f}%\n"
                    f"⏱️ ~{est_time} min left\n\n"
                    f"📢 {quiz_marker}\n"
                    f"🔗 {explanation_tag}"
                )
            except Exception as e:
                print(f"⚠️ Progress update error: {e}")
        
        try:
            print(f"📤 Calling QuizPoster.post_quizzes_batch...")
            
            result = await QuizPoster.post_quizzes_batch(
                context=context,
                chat_id=chat_id,
                questions=questions,
                quiz_marker=quiz_marker,
                explanation_tag=explanation_tag,
                message_thread_id=message_thread_id,
                progress_callback=update_posting_progress
            )
            
            print(f"✅ Posting result: {result}")
            
            await status_msg.edit_text(
                f"✅ Posting Complete!\n\n"
                f"{source_info}\n"
                f"📊 Total: {result['total']}\n"
                f"✅ Success: {result['success']}\n"
                f"❌ Failed: {result['failed']}\n"
                f"⏭️ Skipped: {result['skipped']}\n\n"
                f"📍 {destination}{topic_info}"
            )
            
            csv_path = self.user_states[user_id].get('csv_path')
            if csv_path and csv_path.exists():
                csv_path.unlink(missing_ok=True)
            
            if user_id in self.user_states:
                del self.user_states[user_id]
        
        except Exception as e:
            print(f"❌ Error in post_quizzes_to_destination: {str(e)}")
            traceback.print_exc()
            await status_msg.edit_text(f"❌ Error posting quizzes: {str(e)}")

def main():
    bot = TelegramBot()
    application = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
    
    application.post_init = bot.post_init
    
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("help", bot.help_command))
    application.add_handler(CommandHandler("settings", bot.settings_command))
    application.add_handler(CommandHandler("info", bot.info_command))
    application.add_handler(CommandHandler("model", bot.model_command))
    application.add_handler(CommandHandler("queue", bot.queue_command))
    application.add_handler(CommandHandler("cancel", bot.cancel_command))
    application.add_handler(MessageHandler(filters.Document.ALL, bot.handle_document))
    application.add_handler(MessageHandler(filters.PHOTO, bot.handle_photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_text))
    application.add_handler(CallbackQueryHandler(bot.button_callback))
    
    print("🤖 Bot started successfully!")
    print(f"🤖 AI Model: {config.GEMINI_MODEL}")
    print(f"⚡ Parallel processing: {config.MAX_CONCURRENT_IMAGES} concurrent images")
    print(f"📋 Max queue size: {config.MAX_QUEUE_SIZE} tasks")
    print(f"🔑 API Keys: {len(config.GEMINI_API_KEYS)}")
    print(f"📄 Supports: PDF, Images, CSV")
    print(f"📢 Multi-channel/group posting with MongoDB")
    print(f"🗄️ MongoDB: Connected")
    
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
