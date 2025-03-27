from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import ast
import requests
import re
from collections import defaultdict
import random
import uuid
import time
import json
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_fixed
import subprocess

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VALID_COMPLEXITIES = {"O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)", "O(2^n)", "O(n!)"}
AI_SERVICE_URL = "http://127.0.0.1:11434/api/generate"
AI_MODEL = "mistral"
AI_TIMEOUT = 120

# Auto-start Ollama if not running
try:
    requests.get("http://127.0.0.1:11434", timeout=2)
    logger.info("Ollama is already running")
except:
    logger.info("Starting Ollama in background...")
    subprocess.Popen(["ollama", "serve"])
    time.sleep(3)  # Wait for initialization

# Combined single list of 30 questions
questions = [
    {"id": 1, "code": "def is_positive(n):\n    if n > 0:\n        return True\n    return False"},
    {"id": 2, "code": "def max_of_two(a, b):\n    if a > b:\n        return a\n    else:\n        return b"},
    {"id": 3, "code": "def count_vowels(s):\n    count = 0\n    for char in s:\n        if char in 'aeiouAEIOU':\n            count += 1\n    return count"},
    {"id": 4, "code": "def reverse_words(s):\n    words = s.split()\n    words = words[::-1]\n    return ' '.join(words)"},
    {"id": 5, "code": "def factorial(n):\n    if n == 0:\n        return 1\n    result = 1\n    for i in range(1, n + 1):\n        result *= i\n    return result"},
    {"id": 6, "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b"},
    {"id": 7, "code": "def is_palindrome(s):\n    s = s.lower()\n    for i in range(len(s) // 2):\n        if s[i] != s[len(s) - 1 - i]:\n            return False\n    return True"},
    {"id": 8, "code": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1"},
    {"id": 9, "code": "def sum_of_squares(n):\n    total = 0\n    for i in range(1, n + 1):\n        total += i * i\n    return total"},
    {"id": 10, "code": "def shortest_path(graph, start, end):\n    visited = set()\n    queue = [(start, [start])]\n    while queue:\n        node, path = queue.pop(0)\n        if node == end:\n            return path\n        if node not in visited:\n            visited.add(node)\n            for neighbor in graph[node]:\n                queue.append((neighbor, path + [neighbor]))\n    return None"},
    {"id": 11, "code": "def is_even(n):\n    if n % 2 == 0:\n        return True\n    return False"},
    {"id": 12, "code": "def min_of_two(a, b):\n    if a < b:\n        return a\n    else:\n        return b"},
    {"id": 13, "code": "def count_consonants(s):\n    count = 0\n    for char in s:\n        if char.isalpha() and char not in 'aeiouAEIOU':\n            count += 1\n    return count"},
    {"id": 14, "code": "def capitalize_words(s):\n    words = s.split()\n    for i in range(len(words)):\n        words[i] = words[i].capitalize()\n    return ' '.join(words)"},
    {"id": 15, "code": "def power(base, exp):\n    result = 1\n    for _ in range(exp):\n        result *= base\n    return result"},
    {"id": 16, "code": "def tribonacci(n):\n    if n <= 1:\n        return 0\n    if n == 2:\n        return 1\n    a, b, c = 0, 0, 1\n    for _ in range(3, n + 1):\n        a, b, c = b, c, a + b + c\n    return c"},
    {"id": 17, "code": "def is_anagram(s1, s2):\n    s1 = s1.lower().replace(' ', '')\n    s2 = s2.lower().replace(' ', '')\n    if len(s1) != len(s2):\n        return False\n    for char in s1:\n        if char not in s2:\n            return False\n    return True"},
    {"id": 18, "code": "def linear_search(arr, target):\n    for i in range(len(arr)):\n        if arr[i] == target:\n            return i\n    return -1"},
    {"id": 19, "code": "def product_of_numbers(n):\n    result = 1\n    for i in range(1, n + 1):\n        result *= i\n    return result"},
    {"id": 20, "code": "def longest_path(graph, start, end):\n    visited = set()\n    queue = [(start, [start])]\n    longest = None\n    while queue:\n        node, path = queue.pop(0)\n        if node == end and (longest is None or len(path) > len(longest)):\n            longest = path\n        if node not in visited:\n            visited.add(node)\n            for neighbor in graph[node]:\n                queue.append((neighbor, path + [neighbor]))\n    return longest"},
    {"id": 21, "code": "def is_negative(n):\n    if n < 0:\n        return True\n    return False"},
    {"id": 22, "code": "def avg_of_two(a, b):\n    total = a + b\n    return total / 2"},
    {"id": 23, "code": "def count_chars(s):\n    count = 0\n    for _ in s:\n        count += 1\n    return count"},
    {"id": 24, "code": "def swap_case(s):\n    result = ''\n    for char in s:\n        if char.isupper():\n            result += char.lower()\n        else:\n            result += char.upper()\n    return result"},
    {"id": 25, "code": "def sum_up_to(n):\n    total = 0\n    for i in range(1, n + 1):\n        total += i\n    return total"},
    {"id": 26, "code": "def lucas(n):\n    if n == 0:\n        return 2\n    if n == 1:\n        return 1\n    a, b = 2, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b"},
    {"id": 27, "code": "def is_sorted(arr):\n    for i in range(len(arr) - 1):\n        if arr[i] > arr[i + 1]:\n            return False\n    return True"},
    {"id": 28, "code": "def find_max(arr):\n    max_val = arr[0]\n    for val in arr:\n        if val > max_val:\n            max_val = val\n    return max_val"},
    {"id": 29, "code": "def sum_of_evens(n):\n    total = 0\n    for i in range(2, n + 1, 2):\n        total += i\n    return total"},
    {"id": 30, "code": "def depth_first_search(graph, start, end):\n    visited = set()\n    stack = [(start, [start])]\n    while stack:\n        node, path = stack.pop()\n        if node == end:\n            return path\n        if node not in visited:\n            visited.add(node)\n            for neighbor in graph[node]:\n                stack.append((neighbor, path + [neighbor]))\n    return None"}
]

test_cases = {
    1: [((5,), True), ((-3,), False), ((0,), False), ((10**6,), True), ((-10**6,), False)],
    2: [((3, 7), 7), ((10, 2), 10), ((-1, -5), -1)],
    3: [(("hello",), 2), (("aeiou",), 5), (("xyz",), 0)],
    4: [(("hello world",), "world hello"), (("a b c",), "c b a"), (("single",), "single")],
    5: [((0,), 1), ((3,), 6), ((5,), 120)],
    6: [((1,), 1), ((3,), 2), ((5,), 5)],
    7: [(("radar",), True), (("hello",), False), (("Aba",), True)],
    8: [(([1, 2, 3, 4], 3), 2), (([1, 5, 9], 6), -1), (([], 1), -1)],
    9: [((3,), 14), ((4,), 30), ((1,), 1)],
    10: [(({"A": ["B"], "B": ["C"], "C": []}, "A", "C"), ["A", "B", "C"]), (({"A": ["B"], "B": []}, "A", "C"), None)],
    11: [((4,), True), ((3,), False), ((0,), True)],
    12: [((3, 7), 3), ((10, 2), 2), ((-1, -5), -5)],
    13: [(("hello",), 3), (("aeiou",), 0), (("xyz",), 3)],
    14: [(("hello world",), "Hello World"), (("a b",), "A B"), (("test",), "Test")],
    15: [((2, 3), 8), ((3, 2), 9), ((5, 0), 1)],
    16: [((1,), 0), ((3,), 1), ((5,), 4)],
    17: [(("listen", "silent"), True), (("hello", "world"), False), (("abc", "cba"), True)],
    18: [(([1, 2, 3], 2), 1), (([4, 5, 6], 7), -1), (([], 1), -1)],
    19: [((3,), 6), ((4,), 24), ((1,), 1)],
    20: [(({"A": ["B"], "B": ["C"], "C": []}, "A", "C"), ["A", "B", "C"]), (({"A": ["B", "C"], "B": ["C"], "C": []}, "A", "C"), ["A", "B", "C"])],
    21: [((-5,), True), ((3,), False), ((0,), False)],
    22: [((2, 4), 3.0), ((1, 5), 3.0), ((-2, -4), -3.0)],
    23: [(("hello",), 5), (("abc",), 3), (("",), 0)],
    24: [(("Hello",), "hELLO"), (("AbC",), "aBc"), (("x",), "X")],
    25: [((3,), 6), ((5,), 15), ((1,), 1)],
    26: [((0,), 2), ((1,), 1), ((4,), 4)],
    27: [(([1, 2, 3],), True), (([3, 1, 2],), False), (([],), True)],
    28: [(([1, 5, 3],), 5), (([-1, -2, -3],), -1), (([0],), 0)],
    29: [((4,), 6), ((6,), 12), ((2,), 2)],
    30: [(({"A": ["B"], "B": ["C"], "C": []}, "A", "C"), ["A", "B", "C"]), (({"A": ["B"], "B": []}, "A", "C"), None)]
}

players = {}
current_round = str(uuid.uuid4())

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def call_ai_service(prompt):
    """Call AI service with robust error handling and logging"""
    try:
        logger.debug(f"Sending to AI: {prompt[:100]}...")  # Log truncated prompt
        
        response = requests.post(
            AI_SERVICE_URL,
            json={
                "model": AI_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7}  # Added for better responses
            },
            timeout=AI_TIMEOUT
        )
        
        # Debug raw response
        logger.debug(f"Raw AI response: {response.status_code} {response.text[:200]}...")
        
        response.raise_for_status()
        data = response.json()
        
        # Validate response structure
        if not isinstance(data, dict) or "response" not in data:
            raise ValueError("Invalid AI response format")
            
        ai_response = data["response"].strip()
        logger.debug(f"Received valid AI response: {ai_response[:100]}...")
        return ai_response
        
    except requests.exceptions.ConnectionError:
        logger.warning("Ollama connection failed - attempting to start...")
        subprocess.Popen(["ollama", "serve"])
        time.sleep(5)
        raise  # Will retry due to @retry
    except json.JSONDecodeError as e:
        logger.error(f"AI response JSON decode failed: {str(e)}")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"AI connection failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected AI error: {str(e)}")
        raise

@lru_cache(maxsize=1000)
def get_complexity(code):
    try:
        prompt = f"""
        Analyze the time complexity of the following Python code and return ONLY the Big-O notation (e.g., O(1), O(n), O(n^2), etc.).
        Do not include any explanations or additional text.

        Code:
        ```python
        {code}
        ```

        Return format:
        Complexity: <O-notation>
        """
        ai_response = call_ai_service(prompt)
        match = re.search(r"Complexity:\s*(O\([^)]+\))", ai_response)
        if match and match.group(1) in VALID_COMPLEXITIES:
            return match.group(1)
        else:
            return "O(n)"
    except Exception as e:
        logger.error(f"AI complexity analysis failed: {str(e)}")
        return "O(n)"

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    username = data.get("username", "").strip()
    
    if not username:
        return jsonify({"error": "Username required"}), 400
    if username in players:
        return jsonify({"error": "Username taken"}), 400
    
    # Create a copy of all questions and shuffle them
    all_questions = questions.copy()
    random.shuffle(all_questions)
    
    # Select first 10 questions for this player
    player_questions = all_questions[:10]
    
    players[username] = {
        "session_id": current_round,
        "score": 0,
        "questions_answered": 0,
        "answered_ids": set(),
        "attempted_scores": {},
        "current_set": player_questions,
        "start_time": time.time()
    }
    
    return jsonify({
        "session_id": current_round,
        "time_limit": 900,
        "total_questions": 10,
        "status": "registered"
    })

@app.route("/get_challenge", methods=["POST"])
def get_challenge():
    try:
        data = request.get_json()
        username = data.get("username", "").strip()
        
        if not username or username not in players:
            return jsonify({"error": "Invalid session"}), 401

        player = players[username]
        
        # Only complete after 10 questions
        if len(player["answered_ids"]) >= 10:
            return jsonify({
                "completed": True,
                "final_score": player["score"],
                "progress": {
                    "answered": 10,
                    "total": 10
                }
            })

        unanswered = [q for q in player['current_set'] 
                     if q["id"] not in player["answered_ids"]]
        
        if not unanswered:
            return jsonify({"error": "No challenges available"}), 404

        next_challenge = unanswered[0]
        
        return jsonify({
            "challenge": {
                "id": next_challenge["id"],
                "code": next_challenge["code"]
            },
            "progress": {
                "answered": len(player["answered_ids"]),
                "total": 10
            },
            "time_remaining": max(0, 900 - (time.time() - player["start_time"]))
        })

    except Exception as e:
        logger.error(f"Challenge error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/get_all_questions", methods=["POST"])
def get_all_questions():
    data = request.get_json()
    username = data.get("username", "").strip()
    
    if not username or username not in players:
        return jsonify({"error": "Invalid session"}), 401

    player = players[username]
    return jsonify({
        "questions": player["current_set"],
        "status": "success"
    })

@app.route("/evaluate", methods=["POST"])
def evaluate():
    try:
        data = request.get_json()
        username = data.get("username", "").strip()
        code = data.get("code", "").strip()
        challenge_code = data.get("challenge_code", "").strip()
        challenge_id = data.get("challenge_id")
        
        if username not in players:
            return jsonify({"error": "Invalid session"}), 400
        
        player = players[username]
        response_data = {
            "feedback": "",
            "score": 0,
            "total_score": player["score"],
            "is_completed": len(player["answered_ids"]) >= 10
        }
        
        if code.strip() == challenge_code.strip():
            if challenge_id not in player["answered_ids"]:
                player["score"] += 5
                player["questions_answered"] += 1
                player["answered_ids"].add(challenge_id)
                player["attempted_scores"][challenge_id] = 5
                response_data["total_score"] = player["score"]
            
            response_data.update({
                "feedback": "score: 5\n⚠️ No changes made - You submitted the original challenge code verbatim (5/10 for base correctness)",
                "score": 5
            })
            return jsonify(response_data)
        
        try:
            ast.parse(code)
        except SyntaxError as e:
            response_data.update({
                "feedback": "score: 0\nSyntax Error: " + str(e),
                "score": 0
            })
            return jsonify(response_data)
        
        challenge_func = _extract_function_name(challenge_code)
        user_func = _extract_function_name(code)
        if not user_func or user_func != challenge_func:
            response_data.update({
                "feedback": f"score: 0\nFunction name must be '{challenge_func}'",
                "score": 0
            })
            return jsonify(response_data)
        
        test_case_key = challenge_id
        if test_case_key not in test_cases:
            return jsonify({"error": "Invalid challenge"}), 400
        
        try:
            namespace = {}
            exec(compile(code, "<string>", "exec"), namespace)
            user_func = namespace[challenge_func]
            
            for args, expected in test_cases[test_case_key]:
                result = user_func(*args)
                if result != expected:
                    response_data.update({
                        "feedback": f"score: 0\nTest failed for input {args}",
                        "score": 0
                    })
                    return jsonify(response_data)
            
            ai_opt_code = _get_ai_optimized_code(challenge_code)
            user_complexity = get_complexity(code)
            challenge_complexity = get_complexity(challenge_code)
            score, feedback = _calculate_score(
                code, challenge_code, ai_opt_code, 
                user_complexity, challenge_complexity
            )
            
            if challenge_id not in player["answered_ids"]:
                player["score"] += score
                player["questions_answered"] += 1
                player["answered_ids"].add(challenge_id)
                player["attempted_scores"][challenge_id] = score
            
            response_data.update({
                "feedback": feedback,
                "score": score,
                "total_score": player["score"],
                "is_completed": len(player["answered_ids"]) >= 10
            })
            return jsonify(response_data)
            
        except Exception as e:
            response_data.update({
                "feedback": f"score: 0\nRuntime error: {str(e)}",
                "score": 0
            })
            return jsonify(response_data)
            
    except Exception as e:
        logger.error(f"Evaluation endpoint error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500       

@app.route("/get_solution", methods=["POST"])
def get_solution():
    try:
        data = request.get_json()
        code = data.get("code", "").strip()
        username = data.get("username", "").strip()
        
        if not code:
            return jsonify({"error": "No code provided"}), 400
            
        if username not in players:
            return jsonify({"error": "Invalid session"}), 400

        prompt = f"""Return ONLY the raw optimized Python code with:
        - No explanations
        - No introductory text
        - No markdown formatting
        Original code:
        ```python
        {code}
        ```"""
        
        response = requests.post(
            AI_SERVICE_URL,
            json={
                "model": AI_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3}
            },
            timeout=30
        )
        response.raise_for_status()
        
        ai_response = response.json().get("response", "").strip()
        solution = re.sub(r'```python|```', '', ai_response).strip()
        
        return jsonify({
            "solution": solution if solution else code,
            "status": "success"
        })

    except requests.exceptions.RequestException as e:
        logger.error(f"AI service error: {str(e)}")
        return jsonify({
            "solution": "# Error generating solution\n" + code,
            "status": "error"
        }), 500
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            "solution": "# Error generating solution\n" + code,
            "status": "error"
        }), 500

def _extract_function_name(code):
    try:
        return next(
            (node.name for node in ast.walk(ast.parse(code)) 
             if isinstance(node, ast.FunctionDef)), None
        )
    except Exception:
        return None

def _get_ai_optimized_code(code):
    prompt = """Optimize this Python code with these STRICT REQUIREMENTS:
1. You MUST return a different implementation than the original
2. The optimized version MUST have better time or space complexity
3. If the original is already optimal (like O(1)), improve readability or conciseness
4. Return ONLY the raw optimized code with NO explanations
5. Never return identical code to the original

Original code:
```python
{code}
```"""

    max_attempts = 3
    attempt = 0

    while attempt < max_attempts:
        try:
            ai_response = call_ai_service(prompt)
            optimized_code = re.sub(r'```python|```', '', ai_response).strip()
            
            # Verify the optimized code is different from original
            if optimized_code and optimized_code != code.strip():
                return optimized_code
                
            attempt += 1
            logger.warning(f"AI returned same code as original (attempt {attempt})")
            time.sleep(1)  # Brief delay before retry
            
        except Exception as e:
            logger.error(f"Optimization error on attempt {attempt}: {str(e)}")
            attempt += 1

    # Fallback: Return a slightly modified version if AI fails
    logger.warning("Using fallback optimization after max attempts")
    return _fallback_optimize(code)


def _fallback_optimize(code):
    """Fallback optimization when AI fails to provide a proper optimized version"""
    try:
        # Try to make minimal safe changes
        tree = ast.parse(code)

        # Example: Change variable names or simple refactors
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
                node.id = f"opt_{node.id}"
        
        return ast.unparse(tree)
    except:
        # If all else fails, return original but with a warning
        logger.error("Fallback optimization failed - returning original")
        return code


def _calculate_score(user_code, challenge_code, ai_code, user_complexity, challenge_complexity):
    # First check if user submitted the original challenge code
    if user_code.strip() == challenge_code.strip():
        return 5, "score: 5\n🔍 Submitted original code (5/10 for correctness - try optimizing for higher scores!)"
    
    complexity_order = {"O(1)": 1, "O(log n)": 2, "O(n)": 3, 
                       "O(n log n)": 4, "O(n^2)": 5, "O(2^n)": 6, "O(n!)": 7}
    
    user_rank = complexity_order.get(user_complexity, 4)
    challenge_rank = complexity_order.get(challenge_complexity, 4)
    
    # Check if AI code is actually different and better
    ai_is_optimized = (ai_code.strip() != challenge_code.strip())
    ai_is_better = False
    
    if ai_is_optimized:
        ai_complexity = get_complexity(ai_code)
        ai_rank = complexity_order.get(ai_complexity, 4)
        ai_is_better = ai_rank < challenge_rank
    
    if ai_is_optimized and ai_is_better and user_code.strip() == ai_code.strip():
        return 10, "SCORE: 10\n✅ Perfect! Matches AI-optimized solution!"
    elif user_rank < challenge_rank:
        return 9, f"SCORE: 9\n🌟 Excellent! Improved complexity ({user_complexity} vs original {challenge_complexity})"
    elif user_rank == challenge_rank:
        if len(user_code.splitlines()) < len(challenge_code.splitlines()):
            return 8, f"SCORE: 8\n💡 Great! Same complexity but more concise ({user_complexity})"
        return 7, f"SCORE: 7\n👍 Good! Correct with same complexity ({user_complexity})"
    else:
        return 6, f"SCORE: 6\n⚠️ Works but less efficient ({user_complexity} vs original {challenge_complexity})"
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)    