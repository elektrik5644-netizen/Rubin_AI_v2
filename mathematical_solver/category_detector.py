import re

class MathematicalCategoryDetector:
    def _find_variable(self, message: str) -> str:
        """Finds the first alphabetic character to use as a variable."""
        for char in message:
            if char.isalpha():
                return char
        return 'x' # Default to x if no letter is found

    def is_mathematical_request(self, message: str) -> bool:
        """Checks if the message is a mathematical query."""
        message_lower = message.lower()
        # Check for simple arithmetic
        if re.search(r'^\s*\d+(\.\d+)?\s*[+\-*/]\s*\d+(\.\d+)?\s*$', message_lower):
            return True
        # Check for equations containing a letter and an equals sign
        if '=' in message_lower and any(c.isalpha() for c in message_lower):
            return True
        return False

    def detect_math_category(self, message: str) -> str:
        """Detects the category of the mathematical problem."""
        message_lower = message.lower()
        if '=' in message_lower and any(c.isalpha() for c in message_lower):
            return "equation"
        if re.search(r'^\s*\d+(\.\d+)?\s*[+\-*/]\s*\d+(\.\d+)?\s*$', message_lower):
            return "arithmetic"
        return "general"

    def extract_math_data(self, message: str, category: str) -> dict:
        """Extracts data from the message based on the category."""
        if category == "equation":
            variable = self._find_variable(message)
            return {
                "question_type": "equation_solving",
                "equation_string": message,
                "variable": variable
            }
        if category == "arithmetic":
            match = re.search(r'^\s*(\d+(\.\d+)?)\s*([+\-*/])\s*(\d+(\.\d+)?)\s*$', message)
            if match:
                # Use float for numbers to handle decimals
                num1 = float(match.group(1))
                op = match.group(3)
                num2 = float(match.group(4))
                return {
                    "question_type": "simple_arithmetic",
                    "numbers": [num1, num2],
                    "operator": op
                }
        return {}