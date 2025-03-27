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

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VALID_COMPLEXITIES = {"O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)", "O(2^n)", "O(n!)"}
AI_SERVICE_URL = "http://localhost:11434/api/generate"
AI_MODEL = "mistral"
AI_TIMEOUT = 120

question_sets = [
    [
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
    ],
    [
        {"id": 1, "code": "def is_even(n):\n    if n % 2 == 0:\n        return True\n    return False"},
        {"id": 2, "code": "def min_of_two(a, b):\n    if a < b:\n        return a\n    else:\n        return b"},
        {"id": 3, "code": "def count_consonants(s):\n    count = 0\n    for char in s:\n        if char.isalpha() and char not in 'aeiouAEIOU':\n            count += 1\n    return count"},
        {"id": 4, "code": "def capitalize_words(s):\n    words = s.split()\n    for i in range(len(words)):\n        words[i] = words[i].capitalize()\n    return ' '.join(words)"},
        {"id": 5, "code": "def power(base, exp):\n    result = 1\n    for _ in range(exp):\n        result *= base\n    return result"},
        {"id": 6, "code": "def tribonacci(n):\n    if n <= 1:\n        return 0\n    if n == 2:\n        return 1\n    a, b, c = 0, 0, 1\n    for _ in range(3, n + 1):\n        a, b, c = b, c, a + b + c\n    return c"},
        {"id": 7, "code": "def is_anagram(s1, s2):\n    s1 = s1.lower().replace(' ', '')\n    s2 = s2.lower().replace(' ', '')\n    if len(s1) != len(s2):\n        return False\n    for char in s1:\n        if char not in s2:\n            return False\n    return True"},
        {"id": 8, "code": "def linear_search(arr, target):\n    for i in range(len(arr)):\n        if arr[i] == target:\n            return i\n    return -1"},
        {"id": 9, "code": "def product_of_numbers(n):\n    result = 1\n    for i in range(1, n + 1):\n        result *= i\n    return result"},
        {"id": 10, "code": "def longest_path(graph, start, end):\n    visited = set()\n    queue = [(start, [start])]\n    longest = None\n    while queue:\n        node, path = queue.pop(0)\n        if node == end and (longest is None or len(path) > len(longest)):\n            longest = path\n        if node not in visited:\n            visited.add(node)\n            for neighbor in graph[node]:\n                queue.append((neighbor, path + [neighbor]))\n    return longest"},
    ],
    [
        {"id": 1, "code": "def is_negative(n):\n    if n < 0:\n        return True\n    return False"},
        {"id": 2, "code": "def avg_of_two(a, b):\n    total = a + b\n    return total / 2"},
        {"id": 3, "code": "def count_chars(s):\n    count = 0\n    for _ in s:\n        count += 1\n    return count"},
        {"id": 4, "code": "def swap_case(s):\n    result = ''\n    for char in s:\n        if char.isupper():\n            result += char.lower()\n        else:\n            result += char.upper()\n    return result"},
        {"id": 5, "code": "def sum_up_to(n):\n    total = 0\n    for i in range(1, n + 1):\n        total += i\n    return total"},
        {"id": 6, "code": "def lucas(n):\n    if n == 0:\n        return 2\n    if n == 1:\n        return 1\n    a, b = 2, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b"},
        {"id": 7, "code": "def is_sorted(arr):\n    for i in range(len(arr) - 1):\n        if arr[i] > arr[i + 1]:\n            return False\n    return True"},
        {"id": 8, "code": "def find_max(arr):\n    max_val = arr[0]\n    for val in arr:\n        if val > max_val:\n            max_val = val\n    return max_val"},
        {"id": 9, "code": "def sum_of_evens(n):\n    total = 0\n    for i in range(2, n + 1, 2):\n        total += i\n    return total"},
        {"id": 10, "code": "def depth_first_search(graph, start, end):\n    visited = set()\n    stack = [(start, [start])]\n    while stack:\n        node, path = stack.pop()\n        if node == end:\n            return path\n        if node not in visited:\n            visited.add(node)\n            for neighbor in graph[node]:\n                stack.append((neighbor, path + [neighbor]))\n    return None"},
    ]
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

# Middleware
@app.before_request
def validate_session():
    """Validate user session before processing requests"""
    exempt_endpoints = ['register', 'static', 'health']
    if request.endpoint in exempt_endpoints:
        return
    
    if request.method == 'OPTIONS' or not request.is_json:
        return
    
    try:
        data = request.get_json(silent=True) or {}
        username = data.get('username', '').strip()
        
        if not username or username not in players:
            return jsonify({"error": "Invalid session"}), 401
            
        # Check session timeout
        if time.time() - players[username].get('last_active', 0) > SESSION_TIMEOUT:
            del players[username]
            return jsonify({"error": "Session expired"}), 401
            
        # Update last active time
        players[username]['last_active'] = time.time()
        
    except Exception as e:
        logger.error(f"Session validation failed: {str(e)}")
        return jsonify({"error": "Invalid request"}), 400
# Routes
@app.route("/health", methods=["GET"])
def health_check():
    """Endpoint for health checks"""
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "active_players": len(players)
    })

@app.route("/register", methods=["POST"])

@app.route("/get_challenge", methods=["POST"])
def get_challenge():
    """Get the next challenge for a player"""
    try:
        data = request.get_json()
        username = data.get("username", "").strip()
        
        if username not in players:
            return jsonify({"error": "Register first"}), 400
            
        player = players[username]
        
        # Validate question set
        if 'current_set' not in player or not isinstance(player['current_set'], list):
            return jsonify({"error": "Challenge set not initialized"}), 500
            
        unanswered = [q for q in player['current_set'] if q["id"] not in player["answered_ids"]]
        
        if not unanswered:
            return jsonify({
                "completed": True, 
                "final_score": player["score"],
                "message": "All challenges completed!"
            })
        
        next_challenge = random.choice(unanswered)
        
        return jsonify({
            "challenge": {
                "id": next_challenge["id"],
                "code": next_challenge["code"]
            },
            "progress": {
                "answered": player["questions_answered"],
                "total": MAX_QUESTIONS
            },
            "time_remaining": max(0, GAME_DURATION - (time.time() - player["start_time"]))
        })
        
    except Exception as e:
        logger.error(f"Get challenge failed: {str(e)}")
        return jsonify({"error": "Failed to get challenge"}), 500

# Helper functions
def _extract_function_name(code):
    """Extract function name from Python code"""
    try:
        tree = ast.parse(code)
        return next(
            (node.name for node in ast.walk(tree) 
             if isinstance(node, ast.FunctionDef)),
            None
        )
    except Exception:
        return None

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def call_ai_service(prompt):
    """Call AI service with error handling"""
    try:
        logger.debug(f"Sending to AI: {prompt[:100]}...")
        
        response = requests.post(
            AI_SERVICE_URL,
            json={
                "model": AI_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7}
            },
            timeout=AI_TIMEOUT
        )
        
        response.raise_for_status()
        data = response.json()
        
        if not isinstance(data, dict) or "response" not in data:
            raise ValueError("Invalid AI response format")
            
        return data["response"].strip()
        
    except Exception as e:
        logger.error(f"AI service error: {str(e)}")
        raise

@lru_cache(maxsize=1000)
def get_complexity(code):
    """Get time complexity of code using AI"""
    try:
        prompt = f"""Analyze time complexity and return ONLY Big-O notation:
```python
{code}
Complexity: """
        ai_response = call_ai_service(prompt)
        match = re.search(r"(O[^ ]+)", ai_response)
        return match.group(1) if match and match.group(1) in VALID_COMPLEXITIES else "O(n)"
    except Exception:
        return "O(n)"

def _get_ai_optimized_code(code):
    """Get optimized version of code from AI"""
    prompt = f"""Optimize this code:

    Use Python best practices

    Maintain same functionality

    Return ONLY raw code

    Original:
    {code}"""
    return call_ai_service(prompt)

def _calculate_score(user_code, challenge_code, ai_code, user_complexity, challenge_complexity):
    """Calculate score based on code quality"""
    if user_code.strip() == challenge_code.strip():
        return 5, "⚠️ Submitted original code (5/10)"

    complexity_order = {"O(1)": 1, "O(log n)": 2, "O(n)": 3, 
                       "O(n log n)": 4, "O(n^2)": 5, "O(2^n)": 6, "O(n!)": 7}

    user_rank = complexity_order.get(user_complexity, 4)
    challenge_rank = complexity_order.get(challenge_complexity, 4)

    if user_code.strip() == ai_code.strip():
        return 10, "✅ Perfect! Matches AI-optimized solution"
    elif user_rank < challenge_rank:
        return 9, f"🌟 Improved complexity ({user_complexity})"
    elif user_rank == challenge_rank:
        return 8 if len(user_code.splitlines()) < len(challenge_code.splitlines()) else 7
    else:
        return 6, f"⚠️ Works but less efficient ({user_complexity})"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
# Helper functions
def _extract_function_name(code):
    """Extract function name from Python code"""
    try:
        tree = ast.parse(code)
        return next(
            (node.name for node in ast.walk(tree) 
             if isinstance(node, ast.FunctionDef)),
            None
        )
    except Exception:
        return None

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def call_ai_service(prompt):
    """Call AI service with error handling"""
    try:
        logger.debug(f"Sending to AI: {prompt[:100]}...")
        
        response = requests.post(
            AI_SERVICE_URL,
            json={
                "model": AI_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7}
            },
            timeout=AI_TIMEOUT
        )
        
        response.raise_for_status()
        data = response.json()
        
        if not isinstance(data, dict) or "response" not in data:
            raise ValueError("Invalid AI response format")
            
        return data["response"].strip()
        
    except Exception as e:
        logger.error(f"AI service error: {str(e)}")
        raise

@lru_cache(maxsize=1000)
def get_complexity(code):
    """Get time complexity of code using AI"""
    try:
        prompt = f"""Analyze time complexity and return ONLY Big-O notation:
```python
{code}
Complexity: """
        ai_response = call_ai_service(prompt)
        match = re.search(r"(O[^ ]+)", ai_response)
        return match.group(1) if match and match.group(1) in VALID_COMPLEXITIES else "O(n)"
    except Exception:
        return "O(n)"

def _get_ai_optimized_code(code):
    """Get optimized version of code from AI"""
    prompt = f"""Optimize this code:

    Use Python best practices

    Maintain same functionality

    Return ONLY raw code

    Original:
    {code}"""
    return call_ai_service(prompt)

def _calculate_score(user_code, challenge_code, ai_code, user_complexity, challenge_complexity):
    """Calculate score based on code quality"""
    if user_code.strip() == challenge_code.strip():
        return 5, "⚠️ Submitted original code (5/10)"

    complexity_order = {"O(1)": 1, "O(log n)": 2, "O(n)": 3, 
                       "O(n log n)": 4, "O(n^2)": 5, "O(2^n)": 6, "O(n!)": 7}

    user_rank = complexity_order.get(user_complexity, 4)
    challenge_rank = complexity_order.get(challenge_complexity, 4)

    if user_code.strip() == ai_code.strip():
        return 10, "✅ Perfect! Matches AI-optimized solution"
    elif user_rank < challenge_rank:
        return 9, f"🌟 Improved complexity ({user_complexity})"
    elif user_rank == challenge_rank:
        return 8 if len(user_code.splitlines()) < len(challenge_code.splitlines()) else 7
    else:
        return 6, f"⚠️ Works but less efficient ({user_complexity})"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)