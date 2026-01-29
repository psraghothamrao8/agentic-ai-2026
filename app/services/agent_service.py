import json
import re
import ollama
from app.services.data_service import get_dataset_stats, detect_issues

def query_llama3(prompt):
    try:
        return ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])['message']['content']
    except:
        return "Error"

def extract_json(text):
    try:
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        return json.loads(match.group(0)) if match else None
    except: return None

def analyze_situation_and_decide():
    stats = get_dataset_stats()
    issues = detect_issues().get("issues", [])
    
    prompt = f"""
    Dataset: {stats}
    Issues: {len(issues)}
    Goal: High performance model.
    If Issues > 0 -> "data_cleaning", else "start_training".
    JSON Response: {{ "recommended_action": "..." }}
    """
    
    resp = query_llama3(prompt)
    decision = extract_json(resp)
    
    if not decision: 
        # Fallback
        action = "data_cleaning" if len(issues) > 0 else "start_training"
        return {"recommended_action": action, "issues_list": issues}
        
    decision["issues_list"] = issues
    return decision

def diagnose_after_exploration(results):
    # Minimal diagnosis pass-through
    best = results.get("best_result", {})
    return {"diagnosis": "optimization_needed", "best_acc": best.get("val_acc", 0.0)}
