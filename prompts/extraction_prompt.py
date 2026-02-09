def get_extraction_prompt():
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

2. For FORMAT TYPE 1 (Answer grid at bottom):
   - Look for a grid or table at the bottom of the page
   - There will typically be numbered columns (1, 2, 3, 4...) with letters (A, B, C, D) below them
   - Match each question number to its corresponding letter answer
   - Example: If question 5 has "B" below it in the grid, set correct_option: "B" and correct_answer_index: 1

3. For FORMAT TYPE 2 (Answer below each question):
   - Look for text like "উত্তর" or "Answer" followed by the letter (a, b, c, d) directly under the question

4. Answer Indexes (VERY IMPORTANT):
   - Convert answer letter to ZERO-BASED index: A=0, B=1, C=2, D=3
   - Example: If answer is B, correct_answer_index should be 1 (not 2)
   - Be extremely precise with this conversion to ensure correct quiz functionality

5. If multiple answer formats exist, PRIORITIZE in this order:
   - Bottom-of-page answer grids/keys (highest priority)
   - "উত্তর" / "Answer" notations below individual questions
   - Any official marking or indication in the document

6. If no correct answer can be determined:
   - Set "correct_answer_index" to -1
   - Set "correct_option" to "?"

CRITICAL INSTRUCTIONS FOR QUESTION EXTRACTION:
1. Remove any option text (like "(a) ২ নং-এ (b) ৩ নং-এ (c) ৪ নং-এ (d) ৫ নং-এ") from the question description
2. Remove any reference codes like "[BB: '17'; Din.B: '17']" from the question description
3. Remove any attribution notes like "[আলীম স্যার]" from the question description
4. Remove any question numbers or prefixes (e.g., "1.", "Q1.", "#1")
5. Ensure the question description contains only the actual question text
6. Place all answer choices in the options array, not in the question text
7. If a question contains multiple parts (a, b, c), treat them as separate questions
8. Remove any hints, explanations, or notes that appear with the question
9. **PRESERVE EXAM TAGS**: Keep exam tags at the end of the question description if present in the image.
    Exam tags are typically in format like [DU (C) 04-05], [CU D 12-13], [CU (C) 09-10], 
    [CU (B7) 14-15], [HSTU (C) 11-12], etc. (University code, category, year range)
    These tags indicate which exam the question appeared in and should be preserved.
    Example: If question has "text. [DU (C) 04-05]", keep the tag in the output.
10. For questions with tables or matrices, format as follows:
   - Matrices: Use proper mathematical brackets (⎡⎤, ⎢⎥, ⎣⎦) with center-aligned elements
     Example: 
     ⎡ a₁₁ a₁₂ a₁₃ ⎤
     ⎢ a₂₁ a₂₂ a₂₃ ⎥
     ⎣ a₃₁ a₃₂ a₃₃ ⎦
   - For matrices with numeric values like [1 2 3; 4 5 6; 7 8 9], format as:
     ⎡ 1 2 3 ⎤
     ⎢ 4 5 6 ⎥
     ⎣ 7 8 9 ⎦
   - Tables: Use box-drawing characters (┌─┐, │ │, └─┘) for clarity
11. For questions with images, include "[IMAGE]" placeholder and describe key visual elements

CRITICAL INSTRUCTIONS FOR POLL-FRIENDLY OPTIONS:
1. For fractions: Convert LaTeX fractions (e.g., \frac{a}{b}) to simple text format (a/b)
2. For chemical equations: Preserve subscripts using Unicode (H₂O, not H2O)
3. For superscripts: Use Unicode or caret notation (10⁶ or 10^6)
4. For square roots: 
   - Use √ symbol followed by the expression
   - ALWAYS use parentheses for compound expressions: √(2x+1), not √2x+1
   - Single terms don't need parentheses: √x, √2, etc.
   - Nested roots: ∛(√(x+1))
   - For complex expressions: √(x²+2x+1)
5. For special symbols: Use appropriate Unicode characters (±, ×, ÷, →)
6. Keep options concise and readable for Telegram polls
7. Ensure all mathematical and chemical expressions are formatted consistently
8. If an option is "All of the above" or "None of the above", place it as the last option
9. Remove any option labels (a), b), c), d)) from the beginning of options
10. Ensure each option is unique and not a duplicate
11. For long options, preserve essential information while keeping under 100 characters
12. For options with multiple equations, separate them with semicolons
13. For branched structures in organic compounds, clearly show hydrogen or other substituents attached to carbon atoms using the notation C(H) or C(CH₃), etc. For example, represent a carbon with a hydrogen branch as CH(H) or C(H)

EXAM TAG EXTRACTION RULES:
- **IMPORTANT**: Preserve exam tags that appear in the image
- Exam tags are usually at the end of questions, in brackets like [University Code (Category) Year-Year]
- Common formats: [DU (C) 04-05], [CU D 12-13], [CU (C) 09-10], [CU (B7) 14-15], [JnU (B) 06-07], [HSTU (C) 11-12]
- These indicate the source exam and are valuable information - KEEP THEM in the question description
- DO NOT remove them or treat them as noise

CRITICAL INSTRUCTIONS FOR MATHEMATICAL AND CHEMICAL CONTENT:
1. Preserve ALL mathematical and chemical expressions EXACTLY as they appear in the image
2. Pay special attention to:
   - Complex Fractions (e.g., 1/2, a+b/c-d, nested fractions)
   - Chemical Equations (e.g., H₂SO₄ + NaOH → Na₂SO₄ + H₂O)
   - Chemical Subscripts and States (e.g., H₂O(l), Fe²⁺(aq))
   - Branched structures in organic compounds (e.g., represent as C(H) for hydrogen attached to carbon, or C(CH₃) for methyl group attachment)
   - Equations with division signs (÷)
   - Multiplication signs (×, *, •)
   - Exponents and powers (x², x³, x^n)
   - Square/Cube roots (√, ∛)
   - Plus/minus signs (±)
   - Greek letters (α, β, γ, Δ, etc.)
   - Subscripts and superscripts (x₁, y², etc.)
   - Integration and differentiation symbols (∫, d/dx)
   - Summation notation (Σ)
   - Vectors and unit vectors (î, ĵ, k̂)
   - Parentheses, brackets, and braces ({[()]})
   - Special symbols (∞, ∝, ≈, ≠, ≤, ≥)
   - Chemical bonds (single, double, triple bonds)
   - Resonance arrows (↔)
   - Equilibrium arrows (⇌)
   - Electron configurations
   - Orbital diagrams
   - pH values and logarithmic expressions
   - Units and their combinations (m/s², kg⋅m/s², etc.)
   - Quantum numbers and atomic notations
   - Crystal structure notations
   - Organic chemistry structures and bonds
   - Genetic sequences and biological notations
   - Matrix notations and determinants
   - Set theory symbols (∈, ∉, ∪, ∩, etc.)
   - Logic symbols (∧, ∨, ¬, ⇒, etc.)
   - Probability notations (P(A), P(A|B), etc.)
   - Statistical symbols (μ, σ, χ², etc.)

3. DO NOT modify or simplify any expressions
4. Keep all mathematical and chemical symbols exactly as shown
5. Maintain proper spacing and alignment in expressions
6. Preserve decimal points, negative signs, and charges
7. Keep all sub/superscripts in their exact positions
8. Preserve significant figures in numerical answers
9. Maintain proper isotope notations (e.g., ¹⁴C, ²³⁵U)
10. Keep stereochemistry notations intact (R/S, E/Z, cis/trans)
11. Preserve coordinate system notations (xyz, rθφ, etc.)
12. Maintain vector notations (bold, arrow, hat as available)
13. For organic compounds with branches or substituents:
    - Clearly indicate hydrogen or other atoms attached to carbon using parentheses: CH₃(H), C(H), C(OH), etc.
    - For multiple substituents on the same carbon, list them within the same parentheses: C(H,OH)
    - For complex branched structures, maintain proper connectivity using this notation
    - Preserve all structural information visible in the image, including hydrogen atoms that might be shown explicitly

EXPLANATION GENERATION REQUIREMENTS:
For each question, generate a concise explanation in Bengali that:
1. Explains why the correct answer is correct
2. Focuses only on the concept or reasoning behind the correct answer
3. Is written in Bengali language only
4. Is maximum 165 characters long
5. Does NOT use phrases like:
   - "The answer is" or "উত্তর হল"
   - "option A/B/C/D is correct" or "অপ্শন এ/বি/সি/ডি সঠিক"
   - "তাই উত্তর এ/বি/সি/ডি" (therefore answer A/B/C/D)
   - "উত্তর এ/বি/সি/ডি সঠিক" (answer A/B/C/D is correct)
6. Does NOT mention wrong options or why other options are incorrect
7. For physics, chemistry, or math questions, includes essential equations/formulas
8. Is only one sentence
9. Does NOT include character count or meta-information
10. Explains the scientific concept or reasoning clearly and concisely

CRITICAL INSTRUCTIONS FOR ERROR HANDLING:
1. If an image is blurry or text is unclear, mark it with "[UNCLEAR]" in the affected part
2. If a question has fewer than 4 options, do not process it
3. If mathematical expressions are partially visible or cut off, mark with "[INCOMPLETE]"
4. If chemical structures are ambiguous, preserve exactly as shown without interpretation
5. If text is in multiple languages, preserve the original language
6. If diagrams or figures are referenced, include "[DIAGRAM]" in the question text
7. If handwritten text is present, mark with "[HANDWRITTEN]" if legible, "[ILLEGIBLE]" if not
8. If there are formatting inconsistencies, standardize while preserving meaning
9. If tables are cut off, mark with "[TABLE_INCOMPLETE]"
10. If question context is missing, mark with "[CONTEXT_MISSING]"

EXAMPLE OF ANSWER KEY DETECTION (CRITICALLY IMPORTANT):
For an image with an answer key at the bottom that looks like:
```
1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10
B | B | C | B | D | C | B | C | D | D
```

For question #3, you would set:
- "correct_answer_index": 2  (because C is at index 2, zero-based)
- "correct_option": "C"

And for question #5, you would set:
- "correct_answer_index": 3  (because D is at index 3, zero-based)
- "correct_option": "D"

Example structure:
[
    {
        "question_description": "মাইটোকন্ড্রিয়ার প্রধান কাজ কী?",
        "options": ["কোষের শক্তি উৎপাদন করা", "জেনেটিক তথ্য সংরক্ষণ করা", "প্রোটিন পরিবহন করা", "বর্জ্য পদার্থ ভাঙা"],
        "correct_answer_index": 0,
        "correct_option": "A",
        "explanation": "মাইটোকন্ড্রিয়া কোষের পাওয়ারহাউস হিসেবে ATP উৎপাদন করে যা কোষের শক্তির প্রধান উৎস।"
    },
    {
        "question_description": "If the reaction H₂(g) + I₂(g) ⇌ 2HI(g) has Kc = 4 at 700K, what is the equilibrium concentration of HI when [H₂] = 2M and [I₂] = 1M?",
        "options": ["4M", "2M", "√8M", "8M"],
        "correct_answer_index": 2,
        "correct_option": "C",
        "explanation": "Kc = [HI]²/[H₂][I₂] = 4, so [HI]² = 4×2×1 = 8, therefore [HI] = √8M।"
    }
]

Requirements:
- Exact text as shown in image
- Preserve ALL notations exactly as they appear
- 4 options per question
- Preserve special characters/non-English text
- Valid JSON with proper escaping
- No additional fields/metadata except explanation
- Return only JSON data
- Maximum 100 characters per option
- Clean, unambiguous formatting
- Accurate detection of correct answers
- Both index and letter format for correct answers
- Generate explanations for ALL questions using the exact format specified above

Return complete, valid JSON that can be parsed without modification.""" 
