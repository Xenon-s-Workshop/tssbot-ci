import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional
import asyncio
from datetime import datetime
from queue import Queue
from threading import Lock
import concurrent.futures

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

# Configuration
class Config:
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    GEMINI_API_KEYS = os.getenv("GEMINI_API_KEYS", "").split(",")
    TEMP_DIR = Path("temp")
    OUTPUT_DIR = Path("output")
    MAX_CONCURRENT_IMAGES = 5  # Process 5 images at once
    MAX_QUEUE_SIZE = 10  # Maximum number of tasks in queue
    
    # Gemini Model Configuration
    GEMINI_MODEL = "gemini-2.5-flash"  # Latest Gemini 2.0 Flash (experimental)
    # Alternative models:
    # "gemini-2.0-flash" - Stable Gemini 2.0 Flash (when available)
    # "gemini-1.5-flash" - Previous generation
    # "gemini-1.5-flash-8b" - Smaller, faster model
    # "gemini-1.5-pro" - More powerful but slower
    
    # Generation Configuration
    GENERATION_CONFIG = {
        "temperature": 0.1,  # Lower temperature for more consistent JSON output
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
    }
    
    # Safety Settings (optional - adjust as needed)
    SAFETY_SETTINGS = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        },
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
        """Get next API key in rotation (thread-safe)"""
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
        self.processing = {}
    
    def add_task(self, user_id: int, task_data: Dict) -> int:
        """Add task to queue and return position"""
        with self.lock:
            if len(self.queue) >= config.MAX_QUEUE_SIZE:
                return -1
            
            task = {
                'user_id': user_id,
                'data': task_data,
                'timestamp': datetime.now()
            }
            self.queue.append(task)
            return len(self.queue)
    
    def get_next_task(self) -> Optional[Dict]:
        """Get next task from queue"""
        with self.lock:
            if self.queue:
                return self.queue.pop(0)
            return None
    
    def get_position(self, user_id: int) -> int:
        """Get user's position in queue"""
        with self.lock:
            for idx, task in enumerate(self.queue):
                if task['user_id'] == user_id:
                    return idx + 1
            return 0
    
    def is_processing(self, user_id: int) -> bool:
        """Check if user's task is being processed"""
        with self.lock:
            return user_id in self.processing
    
    def set_processing(self, user_id: int, status: bool):
        """Set processing status for user"""
        with self.lock:
            if status:
                self.processing[user_id] = True
            else:
                self.processing.pop(user_id, None)

task_queue = TaskQueue()

# Prompt for question extraction
def get_prompt():
    """Return the prompt text for quiz extraction"""
    return """You are an expert at converting multiple choice questions (MCQs) from images into JSON format. You have special expertise in detecting and preserving mathematical expressions, chemical equations, and complex notations exactly as they appear. For each image:

1. Extract all visible MCQ questions
2. Format as JSON array with objects containing:
   - "question_description": Full question text only (without any extraneous information)
   - "options": Array of 4 possible answers
   - "correct_answer_index": Index of the correct answer (0-3)
   - "correct_option": Letter of the correct option (A, B, C, or D)
   - "explanation": Concise explanation in Bengali (maximum 165 characters)

CRITICAL INSTRUCTIONS FOR ANSWER DETECTION:
1. RED CIRCLE DETECTION (HIGHEST PRIORITY):
   a) Primary Detection:
      - Carefully scan each option for any red marking (circle, dot, checkmark, underline)
      - Pay special attention to both filled and outlined red circles
      - Check for red marks that may be faint, partial, or slightly offset from the option
      - Verify the red mark is clearly associated with a specific option
   
   b) Verification Steps:
      - Confirm the red mark is actually red (not another color)
      - Ensure the mark is intentional (not a smudge or artifact)
      - Check if the mark is properly aligned with an option
      - Verify the mark is complete and not partially visible
   
   c) Ambiguity Handling:
      - If multiple options have red marks: Set "correct_answer_index": -1 and "correct_option": "?"
      - If a red mark overlaps two options: Set as ambiguous
      - If a red mark is unclear or partially visible: Set as ambiguous
      - If red marks appear inconsistent across questions: Set as ambiguous
   
   d) Quality Checks:
      - Verify the red mark is not a printing artifact
      - Check if the mark is consistent with other marked answers
      - Ensure the mark is not a stray mark or highlight
      - Confirm the mark is not part of the question text or diagram

2. CAREFULLY SCAN the entire image to find answer keys, with special attention to:
   - Answer marked by red circle in the options
   - Answer keys at the BOTTOM of the page (these are the most authoritative source)
   - Answer tables with question numbers and corresponding letters (e.g., "1 2 3 4 5" with "B B C B D" below)
   - Answer grid/matrix formats with numbers in one row and letters (A/B/C/D) in another row
   - Serial numbers with answer options (e.g., "[1] B, [2] A, [3] C...")

3. For FORMAT TYPE 1 (Answer grid at bottom):
   - Look for a grid or table at the bottom of the page
   - There will typically be numbered columns (1, 2, 3, 4...) with letters (A, B, C, D) below them
   - Match each question number to its corresponding letter answer
   - Example: If question 5 has "B" below it in the grid, set correct_option: "B" and correct_answer_index: 1

4. For FORMAT TYPE 2 (Answer below each question):
   - Look for text like "উত্তর" or "Answer" followed by the letter (a, b, c, d) directly under the question

5. Answer Indexes (VERY IMPORTANT):
   - Convert answer letter to ZERO-BASED index: A=0, B=1, C=2, D=3
   - Example: If answer is B, correct_answer_index should be 1 (not 2)
   - Be extremely precise with this conversion to ensure correct quiz functionality

6. If multiple answer formats exist, PRIORITIZE in this order:
   - Bottom-of-page answer grids/keys (highest priority)
   - "উত্তর" / "Answer" notations below individual questions
   - Any official marking or indication in the document

7. If no correct answer can be determined:
   - Set "correct_answer_index" to -1
   - Set "correct_option" to "?"

CRITICAL INSTRUCTIONS FOR QUESTION EXTRACTION:
1. Remove any option text from the question description
2. Remove any reference codes from the question description
3. Remove any attribution notes from the question description
4. Remove any question numbers or prefixes
5. Ensure the question description contains only the actual question text
6. Place all answer choices in the options array, not in the question text
7. If a question contains multiple parts, treat them as separate questions
8. Remove any hints, explanations, or notes that appear with the question
9. PRESERVE EXAM TAGS: Keep exam tags at the end of the question description if present in the image

CRITICAL INSTRUCTIONS FOR POLL-FRIENDLY OPTIONS:
1. For fractions: Convert LaTeX fractions to simple text format (a/b)
2. For chemical equations: Preserve subscripts using Unicode (H₂O)
3. For superscripts: Use Unicode or caret notation (10⁶ or 10^6)
4. For square roots: Use √ symbol with parentheses for compound expressions
5. For special symbols: Use appropriate Unicode characters (±, ×, ÷, →)
6. Keep options concise and readable
7. Remove option labels from the beginning of options
8. Ensure each option is unique

CRITICAL INSTRUCTIONS FOR MATHEMATICAL AND CHEMICAL CONTENT:
1. Preserve ALL mathematical and chemical expressions EXACTLY as they appear
2. Maintain proper spacing and alignment
3. Preserve decimal points, negative signs, and charges
4. Keep all sub/superscripts in their exact positions
5. DO NOT modify or simplify any expressions

EXPLANATION GENERATION REQUIREMENTS:
1. Explains why the correct answer is correct
2. Written in Bengali language only
3. Maximum 165 characters long
4. Does NOT mention answer options by letter
5. Focuses only on the concept or reasoning
6. Is only one sentence
7. Includes essential equations/formulas for science questions

CRITICAL INSTRUCTIONS FOR ERROR HANDLING:
1. If text is unclear, mark with "[UNCLEAR]"
2. If a question has fewer than 4 options, do not process it
3. If mathematical expressions are cut off, mark with "[INCOMPLETE]"
4. Preserve exactly as shown if ambiguous

EXAMPLE OF ANSWER KEY DETECTION:
For an image with answer key:
1 | 2 | 3 | 4 | 5
B | B | C | B | D

For question #3: correct_answer_index: 2, correct_option: "C"
For question #5: correct_answer_index: 3, correct_option: "D"

Example structure:
[
    {
        "question_description": "মাইটোকন্ড্রিয়ার প্রধান কাজ কী?",
        "options": ["কোষের শক্তি উৎপাদন করা", "জেনেটিক তথ্য সংরক্ষণ করা", "প্রোটিন পরিবহন করা", "বর্জ্য পদার্থ ভাঙা"],
        "correct_answer_index": 0,
        "correct_option": "A",
        "explanation": "মাইটোকন্ড্রিয়া কোষের পাওয়ারহাউস হিসেবে ATP উৎপাদন করে যা কোষের শক্তির প্রধান উৎস।"
    }
]

Return complete, valid JSON that can be parsed without modification."""

# PDF Processing Functions
class PDFProcessor:
    @staticmethod
    async def pdf_to_images(pdf_path: Path, page_range: Optional[tuple] = None) -> List[Image.Image]:
        """Convert PDF pages to images"""
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
    async def process_single_image(image: Image.Image, image_idx: int, retry_count: int = 3) -> Optional[tuple]:
        """Process a single image and return (image_idx, questions)"""
        for attempt in range(retry_count):
            try:
                # Get API key for this request
                api_key = api_rotator.get_next_key()
                genai.configure(api_key=api_key)
                
                # Use Gemini 2.0 Flash model with enhanced configuration
                model = genai.GenerativeModel(
                    model_name=config.GEMINI_MODEL,
                    generation_config=config.GENERATION_CONFIG,
                    safety_settings=config.SAFETY_SETTINGS
                )
                
                print(f"Processing image {image_idx} with {config.GEMINI_MODEL}")
                
                # Generate content
                response = model.generate_content([get_prompt(), image])
                
                # Parse JSON response
                response_text = response.text.strip()
                
                # Remove markdown code blocks if present
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.startswith("```"):
                    response_text = response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                
                response_text = response_text.strip()
                
                # Parse JSON
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
    async def process_images_parallel(images: List[Image.Image], progress_callback=None) -> List[Dict]:
        """Process multiple images in parallel"""
        all_questions = []
        total_images = len(images)
        
        # Process images in batches
        for batch_start in range(0, total_images, config.MAX_CONCURRENT_IMAGES):
            batch_end = min(batch_start + config.MAX_CONCURRENT_IMAGES, total_images)
            batch_images = images[batch_start:batch_end]
            
            # Create tasks for this batch
            tasks = []
            for i, image in enumerate(batch_images):
                image_idx = batch_start + i + 1
                task = PDFProcessor.process_single_image(image, image_idx)
                tasks.append(task)
            
            # Process batch concurrently
            results = await asyncio.gather(*tasks)
            
            # Collect results
            for image_idx, questions in results:
                if progress_callback:
                    await progress_callback(image_idx, total_images)
                
                if questions:
                    all_questions.extend(questions)
            
            # Small delay between batches
            if batch_end < total_images:
                await asyncio.sleep(0.5)
        
        return all_questions

# CSV Generation
class CSVGenerator:
    @staticmethod
    def questions_to_csv(questions: List[Dict], output_path: Path):
        """Convert questions to CSV format"""
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
                    'type': '',
                    'section': ''
                }
                writer.writerow(row)

# Telegram Bot Handlers
class TelegramBot:
    def __init__(self):
        self.user_states = {}
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start command handler"""
        await update.message.reply_text(
            "Welcome! 👋\n\n"
            "Send me a PDF file to extract MCQ questions.\n\n"
            "🤖 Powered by Gemini 2.0 Flash\n\n"
            "Commands:\n"
            "/start - Start the bot\n"
            "/help - Show help message\n"
            "/queue - Check your queue position\n"
            "/cancel - Cancel your current task\n"
            "/model - Show current AI model info"
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Help command handler"""
        await update.message.reply_text(
            "📚 *How to use:*\n\n"
            "1. Send me a PDF file\n"
            "2. Optionally specify page range (e.g., 1-5)\n"
            "3. I'll extract all MCQ questions and generate a CSV file\n\n"
            "*Features:*\n"
            "✓ Gemini 2.0 Flash AI\n"
            "✓ Automatic answer detection\n"
            "✓ Bengali explanations\n"
            "✓ Mathematical & chemical notation support\n"
            "✓ CSV export in standard format\n"
            "✓ Task queue system\n"
            "✓ Parallel image processing (5x faster)",
            parse_mode='Markdown'
        )
    
    async def model_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show model information"""
        await update.message.reply_text(
            f"🤖 *AI Model Information:*\n\n"
            f"Model: `{config.GEMINI_MODEL}`\n"
            f"Temperature: {config.GENERATION_CONFIG['temperature']}\n"
            f"Max Tokens: {config.GENERATION_CONFIG['max_output_tokens']}\n"
            f"Parallel Workers: {config.MAX_CONCURRENT_IMAGES}\n"
            f"API Keys: {len(config.GEMINI_API_KEYS)}",
            parse_mode='Markdown'
        )
    
    async def queue_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Check queue position"""
        user_id = update.effective_user.id
        
        if task_queue.is_processing(user_id):
            await update.message.reply_text("⚙️ Your task is currently being processed!")
        else:
            position = task_queue.get_position(user_id)
            if position > 0:
                await update.message.reply_text(f"📋 Your position in queue: {position}")
            else:
                await update.message.reply_text("❌ You don't have any tasks in queue.")
    
    async def cancel_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Cancel user's task"""
        user_id = update.effective_user.id
        
        if user_id in self.user_states:
            pdf_path = self.user_states[user_id].get('pdf_path')
            if pdf_path and pdf_path.exists():
                pdf_path.unlink(missing_ok=True)
            del self.user_states[user_id]
            task_queue.set_processing(user_id, False)
            await update.message.reply_text("✅ Task cancelled successfully!")
        else:
            await update.message.reply_text("❌ No active task to cancel.")
    
    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle PDF document upload"""
        user_id = update.effective_user.id
        document = update.message.document
        
        if not document.file_name.endswith('.pdf'):
            await update.message.reply_text("❌ Please send a PDF file only.")
            return
        
        # Check if user already has a task
        if user_id in self.user_states or task_queue.is_processing(user_id):
            await update.message.reply_text(
                "⚠️ You already have a task in progress.\n"
                "Use /cancel to cancel the current task."
            )
            return
        
        processing_msg = await update.message.reply_text("📥 Downloading PDF...")
        
        try:
            file = await context.bot.get_file(document.file_id)
            pdf_path = config.TEMP_DIR / f"{user_id}_{document.file_name}"
            await file.download_to_drive(pdf_path)
            
            keyboard = [
                [InlineKeyboardButton("All Pages", callback_data="all_pages")],
                [InlineKeyboardButton("Specify Range", callback_data="specify_range")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            self.user_states[user_id] = {'pdf_path': pdf_path}
            
            await processing_msg.edit_text(
                "📄 PDF received!\n\nChoose an option:",
                reply_markup=reply_markup
            )
            
        except Exception as e:
            await processing_msg.edit_text(f"❌ Error downloading PDF: {str(e)}")
    
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button callbacks"""
        query = update.callback_query
        await query.answer()
        
        user_id = update.effective_user.id
        
        if query.data == "all_pages":
            await self.add_to_queue(update, context, user_id, None)
        elif query.data == "specify_range":
            await query.edit_message_text(
                "Please send the page range in format: start-end\n"
                "Example: 1-10 or 5-8"
            )
            self.user_states[user_id]['waiting_for_range'] = True
    
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages (page range)"""
        user_id = update.effective_user.id
        
        if user_id not in self.user_states:
            return
        
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
                
                await self.add_to_queue(update, context, user_id, page_range)
                
            except ValueError:
                await update.message.reply_text("❌ Invalid page numbers. Use numbers only.")
    
    async def add_to_queue(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                          user_id: int, page_range: Optional[tuple]):
        """Add task to queue"""
        if user_id not in self.user_states:
            return
        
        task_data = {
            'pdf_path': self.user_states[user_id]['pdf_path'],
            'page_range': page_range,
            'context': context
        }
        
        position = task_queue.add_task(user_id, task_data)
        
        if position == -1:
            if update.callback_query:
                await update.callback_query.message.edit_text(
                    "❌ Queue is full. Please try again later."
                )
            else:
                await update.message.reply_text("❌ Queue is full. Please try again later.")
            return
        
        if update.callback_query:
            await update.callback_query.message.edit_text(
                f"✅ Added to queue!\n"
                f"📋 Position: {position}\n"
                f"⏳ Estimated wait: ~{position * 2} minutes\n"
                f"🤖 Using {config.GEMINI_MODEL}"
            )
        else:
            await update.message.reply_text(
                f"✅ Added to queue!\n"
                f"📋 Position: {position}\n"
                f"⏳ Estimated wait: ~{position * 2} minutes\n"
                f"🤖 Using {config.GEMINI_MODEL}"
            )
        
        # Process queue
        asyncio.create_task(self.process_queue())
    
    async def process_queue(self):
        """Process tasks from queue"""
        while True:
            task = task_queue.get_next_task()
            
            if not task:
                break
            
            user_id = task['user_id']
            task_data = task['data']
            
            task_queue.set_processing(user_id, True)
            
            try:
                await self.process_pdf(
                    user_id=user_id,
                    pdf_path=task_data['pdf_path'],
                    page_range=task_data['page_range'],
                    context=task_data['context']
                )
            except Exception as e:
                print(f"Error processing task for user {user_id}: {str(e)}")
            finally:
                task_queue.set_processing(user_id, False)
            
            # Small delay between tasks
            await asyncio.sleep(1)
    
    async def process_pdf(self, user_id: int, pdf_path: Path, 
                         page_range: Optional[tuple], context: ContextTypes.DEFAULT_TYPE):
        """Process PDF and extract questions"""
        try:
            # Send initial message
            message = await context.bot.send_message(
                chat_id=user_id,
                text=f"🔄 Processing your PDF...\n🤖 Using {config.GEMINI_MODEL}"
            )
            
            # Convert PDF to images
            await message.edit_text("📄 Converting PDF to images...")
            images = await PDFProcessor.pdf_to_images(pdf_path, page_range)
            
            await message.edit_text(
                f"🖼️ Processing {len(images)} pages in parallel...\n"
                f"⚡ Using {config.MAX_CONCURRENT_IMAGES} parallel workers\n"
                f"🤖 AI Model: {config.GEMINI_MODEL}"
            )
            
            # Progress callback
            async def update_progress(current: int, total: int):
                try:
                    progress = (current / total) * 100
                    await message.edit_text(
                        f"🔍 Processing pages: {current}/{total}\n"
                        f"📊 Progress: {progress:.1f}%\n"
                        f"⚡ Parallel processing enabled\n"
                        f"🤖 {config.GEMINI_MODEL}"
                    )
                except:
                    pass
            
            # Process images in parallel
            all_questions = await PDFProcessor.process_images_parallel(images, update_progress)
            
            if not all_questions:
                await message.edit_text("❌ No questions found in the PDF")
                return
            
            await message.edit_text(f"📊 Generating CSV with {len(all_questions)} questions...")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = config.OUTPUT_DIR / f"questions_{user_id}_{timestamp}.csv"
            
            CSVGenerator.questions_to_csv(all_questions, csv_path)
            
            await message.edit_text("✅ Sending CSV file...")
            
            with open(csv_path, 'rb') as csv_file:
                await context.bot.send_document(
                    chat_id=user_id,
                    document=csv_file,
                    filename=f"mcq_questions_{timestamp}.csv",
                    caption=f"✅ Extracted {len(all_questions)} questions successfully!\n🤖 Powered by {config.GEMINI_MODEL}"
                )
            
            await message.edit_text(
                f"✅ Done!\n\n"
                f"📝 Total questions: {len(all_questions)}\n"
                f"📄 Pages processed: {len(images)}\n"
                f"⚡ Parallel processing: {config.MAX_CONCURRENT_IMAGES}x speed\n"
                f"🤖 AI Model: {config.GEMINI_MODEL}"
            )
            
            # Cleanup
            pdf_path.unlink(missing_ok=True)
            csv_path.unlink(missing_ok=True)
            if user_id in self.user_states:
                del self.user_states[user_id]
            
        except Exception as e:
            await context.bot.send_message(
                chat_id=user_id,
                text=f"❌ Error processing PDF: {str(e)}"
            )
            # Cleanup on error
            if pdf_path.exists():
                pdf_path.unlink(missing_ok=True)
            if user_id in self.user_states:
                del self.user_states[user_id]

# Main application
def main():
    """Start the bot"""
    bot = TelegramBot()
    application = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
    
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("help", bot.help_command))
    application.add_handler(CommandHandler("model", bot.model_command))
    application.add_handler(CommandHandler("queue", bot.queue_command))
    application.add_handler(CommandHandler("cancel", bot.cancel_command))
    application.add_handler(MessageHandler(filters.Document.PDF, bot.handle_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_text))
    application.add_handler(CallbackQueryHandler(bot.button_callback))
    
    print("🤖 Bot started successfully!")
    print(f"🤖 AI Model: {config.GEMINI_MODEL}")
    print(f"⚡ Parallel processing: {config.MAX_CONCURRENT_IMAGES} concurrent images")
    print(f"📋 Max queue size: {config.MAX_QUEUE_SIZE} tasks")
    print(f"🔑 API Keys: {len(config.GEMINI_API_KEYS)}")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
