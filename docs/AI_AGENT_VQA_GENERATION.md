# AI Agent-Based VQA Dataset Generation

## Why Use AI Agents? 🤖

### Problems with Rule-Based Generation ❌

**Current approach (`data/create_vqa_pairs.py`):**
```python
# Rule-based templates
QUESTION_TEMPLATES = {
    "general": ["What items are visible in this X-ray scan?"],
    "specific": ["Is there a {item_type} in this scan?"],
}

# Rigid answer generation
answer = f"Detected items: {', '.join(items)}."
```

**Limitations:**
1. **Repetitive questions** - Same templates over and over
2. **Unnatural language** - Sounds robotic
3. **Limited variety** - Hard to capture edge cases
4. **Manual maintenance** - Need to update templates manually
5. **Poor quality** - Doesn't capture nuanced descriptions

### Benefits of AI Agent Generation ✅

1. **Natural Language**
   - Human-like questions and answers
   - Diverse phrasing
   - Context-aware descriptions

2. **Higher Quality**
   - Better item descriptions
   - Nuanced explanations
   - Realistic scenarios

3. **Scalability**
   - Generate 10x more diverse data
   - Easy to expand categories
   - Automatic adaptation to new items

4. **Less Maintenance**
   - No template updates needed
   - Self-improving with better prompts
   - Handles edge cases automatically

## Recommended AI Agents

### Option 1: GPT-4V / GPT-4o (BEST) ⭐

**Advantages:**
- Can actually SEE the X-ray images
- Generates high-quality VQA pairs
- Understands spatial relationships
- Natural language output

**Cost:**
- ~$0.01-0.03 per image
- For 10k images: $100-300
- Worth it for quality!

### Option 2: Claude 3 Opus/Sonnet with Vision

**Advantages:**
- Excellent reasoning
- Detailed descriptions
- Good spatial understanding
- Similar quality to GPT-4V

**Cost:**
- Similar to GPT-4V
- Good alternative

### Option 3: Qwen2.5-VL-72B (Zero-shot)

**Advantages:**
- Can use existing large Qwen model
- Free (if you have hardware)
- Good quality

**Disadvantages:**
- Requires 8x A100 GPUs
- Slower than API-based

## Implementation Approaches

### Approach 1: VLM-Based Generation (Recommended)

Use GPT-4V or Claude 3 Opus to generate VQA pairs:

```python
# data/ai_agent_vqa_generator.py

import anthropic  # or openai
import base64
from pathlib import Path

class AIAgentVQAGenerator:
    def __init__(self, model="claude-3-opus-20240229"):
        self.client = anthropic.Anthropic()
        self.model = model
    
    def generate_vqa_pairs(self, image_path: str, annotations: dict) -> list:
        """
        Generate diverse VQA pairs using AI agent.
        
        Args:
            image_path: Path to X-ray image
            annotations: Ground truth annotations (categories, bboxes)
        
        Returns:
            List of VQA pairs
        """
        # Encode image
        image_data = base64.standard_b64encode(
            open(image_path, "rb").read()
        ).decode("utf-8")
        
        # Create prompt with ground truth context
        prompt = self._create_generation_prompt(annotations)
        
        # Call AI agent
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ],
        )
        
        # Parse response into VQA pairs
        vqa_pairs = self._parse_response(response.content[0].text)
        
        return vqa_pairs
    
    def _create_generation_prompt(self, annotations: dict) -> str:
        """Create prompt for AI agent."""
        
        # Extract ground truth
        categories = annotations.get("categories", [])
        bboxes = annotations.get("bboxes", [])
        occlusion_info = annotations.get("occlusion", [])
        
        prompt = f"""
You are an expert X-ray security analyst. Analyze this X-ray baggage scan image.

Ground Truth Annotations:
- Detected items: {', '.join(categories)}
- Number of items: {len(categories)}
- Occlusion information: {occlusion_info}

Generate 3-5 diverse Question-Answer pairs for training a VQA model:

Requirements:
1. Questions should be natural and varied
2. Answers should focus on ITEM RECOGNITION only (no risk assessment)
3. Include different question types:
   - General: "What items are visible?"
   - Specific: "Is there a knife?"
   - Location: "Where are the items located?"
   - Occlusion: "Are any items concealed?"
   - Detailed: "Describe all items with locations"

4. Answer format:
   - Natural language
   - Mention item names
   - Include location when relevant
   - Note if items are concealed/occluded
   - DO NOT include risk levels or actions

5. Use the ground truth annotations to ensure accuracy

Output format (JSON):
[
  {{
    "question": "What items are visible in this X-ray scan?",
    "answer": "I can see a folding knife in the center-left area of the baggage.",
    "question_type": "general"
  }},
  ...
]
"""
        return prompt
    
    def _parse_response(self, response_text: str) -> list:
        """Parse AI agent response into structured VQA pairs."""
        import json
        
        # Extract JSON from response
        # (handle markdown code blocks, etc.)
        try:
            # Try direct JSON parse
            vqa_pairs = json.loads(response_text)
        except:
            # Extract from markdown code block
            import re
            json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', 
                                  response_text, re.DOTALL)
            if json_match:
                vqa_pairs = json.loads(json_match.group(1))
            else:
                raise ValueError("Could not parse AI agent response")
        
        return vqa_pairs
```

**Usage:**
```python
# Generate VQA pairs with AI agent
generator = AIAgentVQAGenerator()

for image_path, annotations in dataset:
    vqa_pairs = generator.generate_vqa_pairs(image_path, annotations)
    
    # Save to JSONL
    for pair in vqa_pairs:
        output_file.write(json.dumps({
            "image_path": image_path,
            "question": pair["question"],
            "answer": pair["answer"],
            "metadata": {
                "question_type": pair["question_type"],
                "generated_by": "ai_agent",
                ...
            }
        }) + "\n")
```

### Approach 2: Hybrid (Best Quality + Scale)

Combine AI agent with rule-based filtering:

```python
class HybridVQAGenerator:
    def __init__(self):
        self.ai_generator = AIAgentVQAGenerator()
        self.rule_based = RuleBasedGenerator()  # Current approach
    
    def generate_vqa_pairs(self, image_path, annotations):
        # Generate with AI agent (high quality)
        ai_pairs = self.ai_generator.generate_vqa_pairs(
            image_path, annotations
        )
        
        # Generate with rules (coverage)
        rule_pairs = self.rule_based.generate_vqa_pairs(
            image_path, annotations
        )
        
        # Combine and deduplicate
        all_pairs = ai_pairs + rule_pairs
        
        # Quality filter
        filtered_pairs = self.quality_filter(all_pairs)
        
        return filtered_pairs
    
    def quality_filter(self, pairs):
        """Filter low-quality pairs."""
        filtered = []
        for pair in pairs:
            # Check answer quality
            if len(pair["answer"]) < 10:  # Too short
                continue
            if "UNKNOWN" in pair["answer"]:  # Uncertain
                continue
            if pair["answer"].startswith("I cannot"):  # Refusal
                continue
            
            filtered.append(pair)
        
        return filtered
```

### Approach 3: Self-Training with Synthetic Data

Use the base Qwen2.5-VL model to generate training data:

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

class SelfTrainingGenerator:
    def __init__(self):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct"
        )
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct"
        )
    
    def generate_vqa_pairs(self, image_path, annotations):
        # Generate questions
        questions = self._generate_questions(annotations)
        
        # Generate answers using base model
        answers = []
        for question in questions:
            answer = self._generate_answer(image_path, question)
            
            # Verify against ground truth
            if self._verify_answer(answer, annotations):
                answers.append(answer)
        
        return list(zip(questions, answers))
```

## Comparison: Rule-Based vs AI Agent

| Aspect | Rule-Based | AI Agent (GPT-4V) | Hybrid |
|--------|------------|-------------------|--------|
| **Quality** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Diversity** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Cost** | Free | $100-300/10k | $50-150/10k |
| **Speed** | Fast | Medium | Medium |
| **Maintenance** | High | Low | Medium |
| **Accuracy** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## Example Output Comparison

### Rule-Based (Current)
```json
{
  "question": "What items are visible in this X-ray scan?",
  "answer": "Detected items: a folding knife."
}
```

**Problem:** Repetitive, robotic

### AI Agent (GPT-4V)
```json
{
  "question": "Can you identify any prohibited items in this baggage?",
  "answer": "Yes, I can see a folding knife positioned in the center-left area of the bag, partially obscured by what appears to be clothing items."
}
```

**Better:** Natural, detailed, context-aware

### Multiple AI-Generated Examples
```json
[
  {
    "question": "What do you observe in this X-ray security scan?",
    "answer": "The scan reveals a folding knife located in the central-left portion of the luggage. The knife appears to be partially concealed beneath layers of fabric."
  },
  {
    "question": "Are there any security concerns visible in this image?",
    "answer": "Yes, there is a folding knife present in the left-center area of the baggage."
  },
  {
    "question": "Describe the items visible in this X-ray scan, paying special attention to any prohibited objects.",
    "answer": "The primary concern in this scan is a folding knife visible in the left-central region. It appears to be partially hidden, possibly intentionally, among other baggage contents."
  }
]
```

**Much better:** Diverse, natural, realistic

## Implementation Plan

### Phase 1: Proof of Concept (Week 1)
```bash
# Create AI agent generator
python data/ai_agent_vqa_generator.py \
  --model gpt-4o \
  --input data/opixray/images/ \
  --annotations data/opixray/annotations/train.json \
  --output data/opixray_ai_vqa.jsonl \
  --max-samples 100  # Test with 100 images first
```

### Phase 2: Quality Evaluation (Week 2)
```bash
# Compare quality
python evaluation/compare_vqa_quality.py \
  --rule-based data/opixray_vqa_train.jsonl \
  --ai-generated data/opixray_ai_vqa.jsonl

# Metrics:
# - Diversity (unique questions/answers)
# - Fluency (language model perplexity)
# - Accuracy (match with ground truth)
```

### Phase 3: Full Generation (Week 3)
```bash
# Generate full dataset with AI agent
python data/ai_agent_vqa_generator.py \
  --model gpt-4o \
  --input data/opixray/images/ \
  --annotations data/opixray/annotations/ \
  --output data/opixray_ai_vqa_full.jsonl \
  --samples-per-image 3  # 3-5 diverse pairs per image
```

### Phase 4: Hybrid Approach (Week 4)
```bash
# Best of both worlds
python data/hybrid_vqa_generator.py \
  --ai-model gpt-4o \
  --rule-based-templates configs/question_templates.yaml \
  --output data/opixray_hybrid_vqa.jsonl
```

## Cost Estimation

### GPT-4o Pricing (as of 2024)
- Input: $5 / 1M tokens
- Output: $15 / 1M tokens

**For 10,000 images:**
- Avg input per image: ~1,500 tokens (image + prompt)
- Avg output per image: ~500 tokens (3-5 VQA pairs)
- Total cost: ~$200-300

**ROI:**
- Better model accuracy (+10-15%)
- Reduced manual template work
- More natural language
- **Worth the investment!**

## Recommended Tools

### 1. LangChain + GPT-4V
```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

model = ChatOpenAI(model="gpt-4o", temperature=0.7)

def generate_vqa(image_path):
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": image_path}
        ]
    )
    response = model([message])
    return response.content
```

### 2. Anthropic Claude (Direct)
```python
import anthropic

client = anthropic.Anthropic()

def generate_vqa(image_path):
    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=2000,
        messages=[...]
    )
    return response.content
```

### 3. Batch Processing with Ray
```python
import ray

@ray.remote
def generate_vqa_batch(images):
    results = []
    for image in images:
        vqa = generator.generate_vqa_pairs(image)
        results.append(vqa)
    return results

# Parallel processing
ray.init()
futures = [generate_vqa_batch.remote(batch) for batch in image_batches]
results = ray.get(futures)
```

## Quality Assurance

### Validation Pipeline
```python
def validate_ai_generated_vqa(vqa_pair, ground_truth):
    """Validate AI-generated VQA against ground truth."""
    
    # Check 1: Answer mentions correct items
    detected_items = extract_items(vqa_pair["answer"])
    gt_items = ground_truth["categories"]
    
    if not set(detected_items) == set(gt_items):
        return False, "Item mismatch"
    
    # Check 2: Answer is descriptive
    if len(vqa_pair["answer"]) < 20:
        return False, "Answer too short"
    
    # Check 3: No hallucination
    if mentions_items_not_in_gt(vqa_pair["answer"], gt_items):
        return False, "Hallucination detected"
    
    # Check 4: Natural language
    fluency_score = calculate_fluency(vqa_pair["answer"])
    if fluency_score < 0.7:
        return False, "Poor fluency"
    
    return True, "Valid"
```

## Conclusion

### Use AI Agents When:
✅ You need high-quality, natural language
✅ You want diverse question-answer pairs
✅ Budget allows ($200-500 for full dataset)
✅ You're building production systems

### Use Rule-Based When:
✅ Quick prototyping / proof of concept
✅ Limited budget
✅ Simple, consistent templates needed
✅ Full control over output format

### Recommended Approach:
**Hybrid** - Use AI agents for quality, rules for coverage
- 70% AI-generated (diverse, high-quality)
- 30% rule-based (ensure coverage of all patterns)
- Validate all with ground truth
- Filter low-quality examples

**Next Steps:**
1. Implement `data/ai_agent_vqa_generator.py`
2. Test with 100 images
3. Compare quality with rule-based
4. Scale to full dataset if quality is good
