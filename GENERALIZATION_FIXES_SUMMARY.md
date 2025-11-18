# Generalization Fixes Summary

## Problem Identified

After initial improvements, test results showed:
- **Wyll:** Regressed from 59% → 52.7% (-6.3 points)
- **Shadowheart:** Improved from 43.5% → 59.4% (+15.9 points) BUT 27.5% invalid responses
- **Astarion:** Not tested but had similar length issues (198 lines)

## Root Causes

### 1. Over-Specification
Personas were too detailed and specific to observed training errors, failing to generalize to new scenarios.

### 2. Excessive Length
- **Wyll:** 180 lines (too long for consistent processing)
- **Shadowheart:** 234 lines (caused 220/800 invalid responses)
- **Astarion:** 198 lines (preventively fixed)

### 3. Over-Correction
- **Wyll:** Made "Highly approve" SO restrictive that legitimate cases were downgraded
  - Highly approve accuracy: ~60% → 32.3% (massive drop)
  - New error: Highly approve → Approve (99 errors)
  - New sign-flip: Disapprove → Approve (70 errors)

### 4. Lack of Positive Guidance
Focused heavily on "DON'T do X" without enough "DO do Y" examples.

## Solutions Applied

### General Principles for All Three Characters

1. **Conciseness Over Completeness**
   - Target: ~130-140 lines maximum
   - Remove redundant explanations
   - Keep only critical rules

2. **Balance Corrections**
   - For every "DON'T" add a "DO"
   - Emphasize WHEN to use each rating
   - Provide positive examples

3. **Simplify Decision Rules**
   - One primary test per priority
   - Remove nested conditions
   - Make rules memorable

4. **Better Generalization**
   - Rules should work across diverse scenarios
   - Not just fix specific observed errors

---

## Wyll Fixes

### Changes Made

**File:** `personas/Wyll/persona.txt`
- **Before:** 180 lines
- **After:** ~140 lines (-22% reduction)

**Key Improvements:**

1. **Added "When TO Use" Sections**
   - Clear positive examples for each approval level
   - "What DOES deserve Highly approve?" 
   - Balanced "avoid X" with "recognize Y"

2. **Softened "Highly Approve" Criteria**
   - **Before:** Only father/Mizora/Baldur's Gate
   - **After:** ALSO "significant heroism with real consequences"
   - Fixed over-restriction that caused 99 Highly→Approve errors

3. **Balanced "Routine" Emphasis**
   - **Before:** Over-emphasized everything as "routine"
   - **After:** Added guidance on stakes and personal relevance
   - Rule: "If action saves one person → likely Approve (good but not extraordinary)"
   - Rule: "If action saves many or involves core struggles → likely Highly approve"

4. **Added Critical Clarification**
   - "DON'T under-rate genuine heroism"
   - "Saving people from danger is not neutral - it's what he lives for"
   - Prevents new neutrality errors

5. **Simplified Structure**
   - Removed verbose explanations
   - Merged redundant sections
   - Kept only essential pitfalls

### Expected Impact
- Highly approve accuracy: 32.3% → 55%+ (target)
- Overall accuracy: 52.7% → 60%+ (target)
- No more under-rating of genuine heroism

---

## Shadowheart Fixes

### Changes Made

**File:** `personas/Shadowheart/persona.txt`
- **Before:** 234 lines
- **After:** ~130 lines (-44% reduction)

**Key Improvements:**

1. **Drastic Length Reduction**
   - Cut 104 lines of verbose content
   - Should reduce invalid responses from 27.5% to <5%

2. **Simplified "Pragmatic" Rule**
   - **Before:** Multiple paragraphs with elaborate explanations
   - **After:** Clear one-sentence rule + examples
   - "PRAGMATIC = DECISIVE ACTION, NOT BACKING DOWN"
   - Visual examples (✅ vs ❌) for clarity

3. **Streamlined Common Pitfalls**
   - **Before:** 5+ pitfalls with verbose explanations
   - **After:** Top 3 critical pitfalls only
   - Each with concise explanation

4. **Simplified Personality Section**
   - **Before:** Abstract "contradictions" discussion
   - **After:** Concrete traits and brief summary
   - Maintained key insight about her contradictions

5. **Concise Intensity Guide**
   - Bullet points instead of paragraphs
   - Clear severity examples for each level

### Expected Impact
- Invalid responses: 27.5% → <5% (target)
- Overall accuracy: maintain 59%+ or improve
- Better consistency across all samples

---

## Astarion Fixes

### Changes Made

**File:** `personas/Astarion/persona.txt`
- **Before:** 198 lines
- **After:** ~140 lines (-29% reduction)

**Key Improvements:**

1. **Length Reduction**
   - Preventive fix to avoid Shadowheart's invalid response issue
   - More concise throughout

2. **Added Positive Examples**
   - Clear "When TO Use" sections for each level
   - Examples of what DOES get each rating

3. **Emphasized "Selfish vs Altruistic" Test**
   - Made the core decision rule more prominent
   - Concrete example with reasoning

4. **Balanced Guidance**
   - Not just "don't do X" but "do Y"
   - Emphasized WHEN to approve, not just when to disapprove

5. **Simplified Structure**
   - Removed redundancy
   - Merged overlapping sections
   - Kept core insights, removed verbosity

### Expected Impact
- Maintain or improve current performance
- Avoid length-related failures
- Better generalization across diverse scenarios

---

## Summary of Changes

| Character | Before | After | Reduction | Main Fix |
|-----------|--------|-------|-----------|----------|
| **Wyll** | 180 lines | ~140 | -22% | Soften Highly criteria, add positive examples |
| **Shadowheart** | 234 lines | ~130 | -44% | Drastic shortening, simplify rules |
| **Astarion** | 198 lines | ~140 | -29% | Preventive length fix, add positive examples |

## Testing Next Steps

### Commands to Run

**Wyll:**
```bash
cd /home/wschay/bg3-sim
bash -c "conda activate group-chat && python3 src/persona_evaluation/run_llm_approval_inference.py \
  --input approval-dataset/approval_dataset_subset.jsonl \
  --output test/1118_gpt-5-mini_wyll_llm_approvals.jsonl \
  --model gpt-5-mini \
  --character Wyll \
  --metrics_dir test"
```

**Shadowheart:**
```bash
cd /home/wschay/bg3-sim
bash -c "conda activate group-chat && python3 src/persona_evaluation/run_llm_approval_inference.py \
  --input approval-dataset/approval_dataset_subset.jsonl \
  --output test/1118_gpt-5-mini_shadowheart_llm_approvals.jsonl \
  --model gpt-5-mini \
  --character Shadowheart \
  --metrics_dir test"
```

**Astarion:**
```bash
cd /home/wschay/bg3-sim
bash -c "conda activate group-chat && python3 src/persona_evaluation/run_llm_approval_inference.py \
  --input approval-dataset/approval_dataset_subset.jsonl \
  --output test/1118_gpt-5-mini_astarion_llm_approvals.jsonl \
  --model gpt-5-mini \
  --character Astarion \
  --metrics_dir test"
```

### Success Criteria

**Wyll:**
- Overall accuracy: >60%
- Highly approve accuracy: >50%
- Invalid responses: <5%
- No systematic under-rating of heroism

**Shadowheart:**
- Invalid responses: <5% (critical)
- Overall accuracy: ≥58%
- Maintain improvement over baseline
- Consistent across all sample ranges

**Astarion:**
- Maintain current performance or improve
- Invalid responses: <5%
- No new error patterns
- Better generalization

### What to Check

1. **Invalid Response Rate** - Should be <5% for all
2. **Per-Class Performance** - Check all 4 categories are reasonable
3. **Error Patterns** - Look for new systematic failures
4. **Cross-Sample Consistency** - Test across different ranges

---

## Key Lessons Learned

### 1. Length Matters
- Personas >150 lines cause processing issues
- Optimal range: 130-140 lines
- Conciseness improves consistency

### 2. Over-Correction is Dangerous
- Fixing specific errors too aggressively creates new errors
- Need balance between correction and generalization

### 3. Positive Guidance is Essential
- "DON'T do X" alone is insufficient
- Need "DO do Y" with clear examples
- LLMs need to know WHEN to use each rating

### 4. Generalization Requires Testing
- Can't optimize for one test set
- Need to validate across diverse scenarios
- Rules must work broadly, not just fix observed cases

### 5. Simplicity Beats Comprehensiveness
- One clear rule > three complex conditions
- Memorable tests > elaborate frameworks
- Actionable guidance > theoretical explanations

---

## Recommendations for Future Iterations

1. **Monitor Length**
   - Keep personas under 140 lines
   - Remove redundancy immediately

2. **Balance Corrections**
   - For every restriction, add guidance on when TO use that rating
   - Provide positive examples

3. **Test Broadly**
   - Validate on multiple test sets
   - Check consistency across sample ranges
   - Look for new error patterns

4. **Iterate Conservatively**
   - Make smaller, targeted changes
   - Avoid over-correcting
   - Test after each change

5. **Focus on Decision Tests**
   - One clear question per priority
   - Simple, memorable rules
   - Easy to apply consistently

