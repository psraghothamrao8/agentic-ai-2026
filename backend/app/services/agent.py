import ollama
import json
import re
import numpy as np
from backend.app.services.data import detect_issues, get_dataset_stats

def query_llama3(prompt):
    """
    Queries Ollama with fallback support (Llama3 -> Gemma3:27b).
    """
    models_to_try = ['llama3', 'gemma3:27b']
    
    for model_name in models_to_try:
        try:
            print(f"Agent: Connecting to {model_name}...")
            response = ollama.chat(model=model_name, messages=[
              {
                'role': 'user',
                'content': prompt,
              },
            ], options={'num_predict': 500}) # Limit output size to prevent long hangs
            return response['message']['content']
        except Exception as e:
            print(f"[WARNING] Failed to connect to {model_name}: {e}")
            continue
            
    return "Error communicating with Ollama: All models failed."

def extract_json(text):
    """Robustly extracts JSON from LLM response."""
    # 1. Try finding a markdown block
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try: 
            return json.loads(match.group(1)) 
        except: pass
    
    # 2. Try identifying the first outer bracket pair
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except: pass
        
    return None

def analyze_situation_and_decide():
    """
    Analyzes the current state and returns a decision.
    """
    print("Agent: Analyzing dataset health...")
    
    # 1. Get Stats
    stats = get_dataset_stats()
    
    # 2. Check for Issues
    detection_result = detect_issues()
    
    # Handle new structure (dict) vs old structure (list)
    if isinstance(detection_result, dict):
        if "error" in detection_result:
             return {
                "decision": "ERROR",
                "reason": detection_result["error"],
                "action": "none"
            }
        # Extract issues list from new structure
        issues = detection_result.get("issues", [])
        # We can also use class_summary later if needed, but for now focus on compatibility
    else:
        # Backward compatibility for list
        issues = detection_result

    num_issues = len(issues)
    total_train = stats.get('train', {}).get('count', 0)
    
    # Construct Prompt for Llama3
    prompt = f"""
    You are an expert ML Engineer Agent.
    System Status:
    - Dataset Split: {stats}
    - Total Training Images: {total_train}
    - Detected Potential Label Issues/Outliers: {num_issues}
    
    Goal: Build a high-performance model.
    
    Rules:
    1. If there are likely label issues (>0), priority is 'data_cleaning' to ensure highest model quality.
    2. Only if issues are 0, recommend 'start_training'.
    3. Even if issues are few, cleaning is safer than training on noise.
    
    Respond in JSON format:
    {{
        "analysis": "Your thought process...",
        "recommended_action": "data_cleaning" | "hyperparameter_tuning" | "start_training"
    }}
    """
    
    print("Agent: Asking Llama3...")
    response_text = query_llama3(prompt)
    
    # 3. Rule-based Fallback if AI fails
    if "Error communicating with Ollama" in response_text:
        print("[INFO] AI unavailable, using rule-based fallback.")
        if num_issues > 0:
            return {
                "analysis": f"Rule-based analysis: Label issues detected ({num_issues}). Cleaning required before training.",
                "recommended_action": "data_cleaning",
                "issues_list": issues,
                "raw_issues_count": num_issues,
                "is_fallback": True
            }
        else:
            return {
                "analysis": "Rule-based analysis: Dataset looks healthy. Recommended to start training.",
                "recommended_action": "start_training",
                "issues_list": issues,
                "raw_issues_count": num_issues,
                "is_fallback": True
            }

    analysis_text = response_text
    
    # Attempt to parse JSON
    decision = extract_json(analysis_text)
    
    if decision:
        try:
            decision['raw_issues_count'] = num_issues
            decision['issues_list'] = issues
            
            # FORCE OVERRIDE: If issues exist, ensuring cleaning is recommended regardless of LLM "opinion"
            if num_issues > 0 and decision.get('recommended_action') != 'data_cleaning':
                 decision['recommended_action'] = 'data_cleaning'
                 decision['analysis'] += " [System Note: Enforcing data cleaning due to detected issues.]"

            if isinstance(detection_result, dict) and "strategy" in detection_result:
                strategy_info = f"\n\n[Analysis Strategy: {detection_result['strategy']}]"
                decision['analysis'] = decision.get('analysis', '') + strategy_info
            
            return decision
        except Exception as e:
             print(f"[WARNING] Failed to process parsed JSON: {e}")

    # Fallback (AI Failed or JSON Invalid)
    print(f"[INFO] Using fallback decision logic. AI Response snippet: {analysis_text[:100]}...")
    
    fallback_analysis = f"AI Analysed (Raw): {analysis_text[:200]}...\n\nSystem: Label issues detected ({num_issues}). Cleaning recommended."
    fallback_action = "data_cleaning"
    
    if num_issues == 0:
        fallback_analysis = f"AI Analysed (Raw): {analysis_text[:200]}...\n\nSystem: Dataset looks healthy. Training recommended."
        fallback_action = "start_training"

    return {
        "analysis": fallback_analysis,
        "recommended_action": fallback_action,
        "issues_list": issues,
        "raw_issues_count": num_issues,
        "is_fallback": True
    }

# ============== Phase 4: Post-Training Diagnosis ==============

def diagnose_after_exploration(exploration_results):
    """
    Diagnoses root cause after exhausting automated exploration.
    
    Args:
        exploration_results: dict with status, best_result, all_results
    
    Returns:
        dict with diagnosis, reasoning, and recommended_action
    """
    import numpy as np
    
    best = exploration_results["best_result"]
    all_results = exploration_results["all_results"]
    
    # Compute per-class accuracy from confusion matrix
    cm = np.array(best["confusion_matrix"])
    per_class_acc = {}
    class_names = list(best["per_class_metrics"].keys()) if "per_class_metrics" in best else ["Class_" + str(i) for i in range(len(cm))]
    
    for i, name in enumerate(class_names):
        total = cm[i].sum()
        if total > 0:
            per_class_acc[name] = cm[i, i] / total
        else:
            per_class_acc[name] = 0.0
    
    # Find problematic classes (< 60% accuracy)
    problematic_classes = [cls for cls, acc in per_class_acc.items() if acc < 0.6]
    
    # Check for overfitting
    train_val_gap = best["train_acc"] - best["val_acc"]
    is_overfitting = train_val_gap > 0.15
    
    # Build prompt for Llama3
    prompt = f"""
    You are an ML expert analyzing training results after exhaustive hyperparameter exploration.
    
    Exploration Summary:
    - Configurations tried: {len(all_results)}
    - Best validation accuracy: {best['val_acc']:.2%}
    - Balanced Accuracy: {best.get('balanced_acc', 0.0):.2%}
    - Macro F1 Score: {best.get('macro_f1', 0.0):.2%}
    - Best training accuracy: {best['train_acc']:.2%}
    - Train-Val gap: {train_val_gap:.2%}
    - Target accuracy: 90%
    
    Per-class Performance:
    {', '.join([f'{cls}: {acc:.1%}' for cls, acc in per_class_acc.items()])}
    
    Confusion Matrix (rows=true, cols=pred):
    {cm}
    
    Question: What is the root cause of underperformance?
    
    Analysis Guidelines:
    1. If specific classes have very low accuracy (< 60%) AND there's class confusion → Likely DATA QUALITY issue (mislabeling)
    2. If overfitting (train >> val by 15%+) → Need MORE REGULARIZATION (but we already tried weight_decay variations)
    3. If all classes perform poorly uniformly → May need DIFFERENT MODEL ARCHITECTURE
    4. If performance is close to target (85%+) → Just need MINOR TUNING
    
    Respond in JSON:
    {{
        "diagnosis": "data_quality" | "need_different_model" | "nearly_there" | "need_more_exploration",
        "conclusion": "A concise summary of why the model failed to reach the target.",
        "dataset_analysis": "Specific observations about dataset quality, class imbalance, or tough classes.",
        "problematic_classes": [...],
        "reasoning": "Detailed technical explanation...",
        "recommended_action": "recheck_labels" | "try_resnet50" | "fine_tune" | "continue_exploration",
        "next_steps": [
            {{ "label": "Filter Dataset (Hybrid Approach)", "action": "filter_dataset" }},
            {{ "label": "More Hyperparameter Tuning", "action": "more_tuning" }}
        ]
    }}
    """
    
    print("🤖 Agent: Diagnosing root cause...")
    response = query_llama3(prompt)
    
    # Parse response
    diagnosis = extract_json(response)
    
    if diagnosis:
        diagnosis["best_result"] = best
        diagnosis["all_results_summary"] = [
            {"config": r["config_name"], "val_acc": r["val_acc"]} 
            for r in all_results
        ]
        return diagnosis
    else:
        return {
            "diagnosis": "error",
            "reasoning": "Failed to parse agent response",
            "raw_response": response
        }
