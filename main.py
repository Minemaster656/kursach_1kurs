import sympy as sp
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import re
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Optional, Tuple, Any
from colorama import Fore, Style, init
import warnings
import sys
import textwrap

warnings.filterwarnings('ignore')

# Инициализация colorama
init(autoreset=True)

# Проверка и импорт ollama
try:
    import ollama

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


def check_ollama_installation():
    """Проверка доступности Ollama"""
    if not OLLAMA_AVAILABLE:
        print(f"{Fore.RED}❌ Библиотека ollama не установлена!")
        print(f"{Fore.YELLOW}Установите её командой: pip install ollama")
        print(f"{Fore.CYAN}Также убедитесь, что Ollama установлена в системе:")
        print(f"{Fore.CYAN}Инструкции: https://ollama.com/download")
        return False

    try:
        ollama.list()
        return True
    except Exception as e:
        print(f"{Fore.RED}❌ Ollama сервис недоступен!")
        print(f"{Fore.YELLOW}Убедитесь, что Ollama запущена в системе")
        print(f"{Fore.CYAN}Инструкции по установке: https://ollama.com/download")
        print(f"{Fore.RED}Ошибка: {e}")
        return False


def parse_with_implicit_multiplication(expression):
    transformations = standard_transformations + (implicit_multiplication_application,)
    return parse_expr(expression, transformations=transformations)


def render_latex_output(expression, step_name="", enable_latex=False):
    """Рендеринг выражения в LaTeX формат."""
    if not enable_latex:
        return expression

    try:
        from sympy import latex, sympify
        from sympy.parsing.sympy_parser import parse_expr

        try:
            sympy_expr = parse_expr(expression)
            latex_expr = latex(sympy_expr)
            if step_name:
                return f"{step_name}: {expression}"
            else:
                return f"{expression}"
        except:
            if step_name:
                return f"{step_name}: {expression}"
            else:
                return expression
    except ImportError:
        if step_name:
            return f"{step_name}: {expression}"
        else:
            return expression


def create_latex_steps_log():
    """Создание лога шагов для LaTeX."""
    return []


def add_latex_step(steps_log, step_name, expression, enable_latex=False):
    """Добавление шага в лог LaTeX."""
    formatted_step = render_latex_output(expression, step_name, enable_latex)
    steps_log.append(formatted_step)


def export_latex_steps(steps_log, filename="processing_steps.md"):
    """Экспорт шагов в Markdown с LaTeX."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# Шаги обработки\n\n")
            for i, step in enumerate(steps_log, 1):
                f.write(f"{i}. {step}\n")
        print(f"Шаги сохранены в {filename}")
    except Exception as e:
        print(f"Ошибка сохранения: {e}")


class AISettings:
    """Настройки ИИ"""

    def __init__(self):
        self.show_steps = True
        self.ai_error_explanations = False
        self.ai_step_explanations = False
        self.model = "gemma3:4b-it-qat"
        self.skip_model_check = False


class OllamaAI:
    """Класс для работы с Ollama ИИ"""

    def __init__(self, settings: AISettings):
        self.settings = settings
        self.conversation_history = []

        self.system_prompts = {
            'dialog': """Ты математический помощник, специализирующийся на составлении и решении алгебраических задач с помощью SymPy. 
            Помогай пользователю составлять правильные математические выражения для SymPy. 
            Отвечай кратко и по делу. Если пользователь просит помощь с формулой, предложи несколько вариантов записи.
            Используй стандартные обозначения: sqrt() для корня, log() для натурального логарифма, 
            sin(), cos(), tan() для тригонометрических функций, pi для числа π, E для числа e.""",

            'error': """Ты помощник для объяснения ошибок парсинга математических выражений в SymPy.
            Анализируй ошибку и объясни пользователю простым языком, что пошло не так и как исправить.
            Будь краток и конструктивен. Предложи правильный вариант записи.""",

            'explanation': """Ты объясняешь этапы решения математических задач в SymPy.
            Для каждого этапа дай краткое математическое обоснование того, что происходит.
            Используй простой язык, подходящий для студентов."""
        }

    def check_model_availability(self):
        """Проверка доступности модели"""
        if self.settings.skip_model_check:
            return True

        try:
            models_response = ollama.list()
            available_models = []
            if 'models' in models_response:
                for model in models_response['models']:
                    model_name = model.get('name') or model.get('model', '')
                    if model_name:
                        available_models.append(model_name)

            if self.settings.model not in available_models:
                print(f"{Fore.YELLOW}⚠️ Модель {self.settings.model} не найдена")
                print(f"{Fore.CYAN}Доступные модели: {available_models}")
                print(f"{Fore.CYAN}Скачиваю модель...")
                try:
                    ollama.pull(self.settings.model)
                    print(f"{Fore.GREEN}✅ Модель {self.settings.model} успешно скачана")
                    return True
                except Exception as e:
                    print(f"{Fore.RED}❌ Ошибка скачивания модели: {e}")
                    print(f"{Fore.YELLOW}💡 Попробуйте вручную: ollama pull {self.settings.model}")
                    return False
            return True
        except Exception as e:
            print(f"{Fore.RED}❌ Ошибка проверки модели: {e}")
            print(f"{Fore.YELLOW}💡 Можно отключить проверку модели в настройках")
            return False

    def chat(self, message: str, prompt_type: str = 'dialog', include_history: bool = True):
        """Общение с ИИ"""
        if not check_ollama_installation():
            return "Ollama недоступна"

        if not self.check_model_availability():
            return "Модель недоступна"

        try:
            messages = []

            if prompt_type in self.system_prompts:
                messages.append({
                    'role': 'system',
                    'content': self.system_prompts[prompt_type]
                })

            if include_history and prompt_type == 'dialog':
                messages.extend(self.conversation_history)

            messages.append({
                'role': 'user',
                'content': message
            })

            response = ollama.chat(
                model=self.settings.model,
                messages=messages
            )

            ai_response = response['message']['content']

            if include_history and prompt_type == 'dialog':
                self.conversation_history.append({'role': 'user', 'content': message})
                self.conversation_history.append({'role': 'assistant', 'content': ai_response})

                if len(self.conversation_history) > 20:
                    self.conversation_history = self.conversation_history[-20:]

            return ai_response

        except Exception as e:
            return f"Ошибка ИИ: {e}"

    def explain_error(self, error_msg: str, expression: str):
        """Объяснение ошибки"""
        if not self.settings.ai_error_explanations:
            return None

        prompt = f"Выражение: '{expression}'\nОшибка: {error_msg}\nОбъясни ошибку и предложи исправление:"
        return self.chat(prompt, 'error', include_history=False)

    def explain_step(self, step_name: str, input_expr: str, output_expr: str):
        """Объяснение этапа решения"""
        if not self.settings.ai_step_explanations:
            return None

        prompt = f"Этап: {step_name}\nВход: {input_expr}\nВыход: {output_expr}\nОбъясни этот этап решения:"
        return self.chat(prompt, 'explanation', include_history=False)

    def interactive_mode(self):
        """Интерактивный режим с ИИ"""
        print(f"{Fore.CYAN}{'=' * 60}")
        print(f"{Fore.CYAN}🤖 Математический ИИ помощник (Ollama)")
        print(f"{Fore.CYAN}Модель: {self.settings.model}")
        print(f"{Fore.CYAN}{'=' * 60}")
        print(f"{Fore.WHITE}Для многострочного ввода используйте пустую строку для завершения")
        print(f"{Fore.WHITE}Команды: 'выход', 'exit', 'quit', 'очистить' для очистки истории")
        print(f"{Fore.CYAN}{'=' * 60}")

        if not check_ollama_installation() or not self.check_model_availability():
            return

        while True:
            try:
                print(f"{Fore.GREEN}Вы:", end=" ")

                lines = []
                while True:
                    line = input()
                    if line.strip() == "":
                        break
                    lines.append(line)

                user_input = "\n".join(lines).strip()

                if not user_input:
                    continue

                if user_input.lower() in ['выход', 'exit', 'quit']:
                    print(f"{Fore.YELLOW}До свидания! 👋")
                    break

                if user_input.lower() in ['очистить', 'clear', 'история']:
                    self.conversation_history = []
                    print(f"{Fore.YELLOW}История диалога очищена")
                    continue

                print(f"{Fore.BLUE}🤖 ИИ: {Fore.WHITE}", end="")
                response = self.chat(user_input, 'dialog')

                wrapped_response = textwrap.fill(response, width=80)
                for line in wrapped_response.split('\n'):
                    print(f"{Fore.WHITE}{line}")

                print()

            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}До свидания! 👋")
                break
            except Exception as e:
                print(f"{Fore.RED}Ошибка: {e}")


class AdvancedMathProcessor:
    """Продвинутый процессор математических выражений с поддержкой SymPy и ИИ"""

    def __init__(self):
        """Инициализация процессора с настройками SymPy и ИИ"""
        sp.init_printing(use_unicode=True, wrap_line=False)

        print(f"{Fore.CYAN}🔧 Настройка ИИ помощника")
        self.ai_settings = AISettings()
        self.setup_ai_settings()

        self.ai = OllamaAI(self.ai_settings) if OLLAMA_AVAILABLE else None

        # ИСПРАВЛЕНО: Возвращаем правильные словари замен из processor2.py
        self.cyrillic_replacements = {
            'х': 'x', 'у': 'y', 'з': 'z', 'а': 'a', 'б': 'b', 'в': 'c',
            'п': 'pi', 'е': 'E'
        }

        self.symbol_replacements = {
            '^': '**',
            '√': 'sqrt',
            '∞': 'oo',
            '±': '+/-',
            '×': '*',
            '÷': '/',
        }

        self.function_replacements = {
            'ln': 'log',
            'lg': 'log10',
            'arctg': 'atan',
            'arcsin': 'asin',
            'arccos': 'acos',
            'sh': 'sinh',
            'ch': 'cosh',
            'th': 'tanh'
        }

    def setup_ai_settings(self):
        """Настройка параметров ИИ"""
        print(f"{Fore.YELLOW}Настройки ИИ помощника:")

        while True:
            show_steps = input(f"{Fore.CYAN}1. Показывать этапы решения? (y/n, по умолчанию y): ").strip().lower()
            if show_steps in ['', 'y', 'yes', 'да', 'д']:
                self.ai_settings.show_steps = True
                break
            elif show_steps in ['n', 'no', 'нет', 'н']:
                self.ai_settings.show_steps = False
                break
            else:
                print(f"{Fore.RED}Введите y/n")

        while True:
            ai_errors = input(
                f"{Fore.CYAN}2. Включить объяснения ошибок от ИИ? (y/n, по умолчанию n): ").strip().lower()
            if ai_errors in ['y', 'yes', 'да', 'д']:
                self.ai_settings.ai_error_explanations = True
                break
            elif ai_errors in ['', 'n', 'no', 'нет', 'н']:
                self.ai_settings.ai_error_explanations = False
                break
            else:
                print(f"{Fore.RED}Введите y/n")

        if self.ai_settings.show_steps:
            while True:
                ai_steps = input(
                    f"{Fore.CYAN}3. Включить объяснения этапов от ИИ? (y/n, по умолчанию n): ").strip().lower()
                if ai_steps in ['y', 'yes', 'да', 'д']:
                    self.ai_settings.ai_step_explanations = True
                    break
                elif ai_steps in ['', 'n', 'no', 'нет', 'н']:
                    self.ai_settings.ai_step_explanations = False
                    break
                else:
                    print(f"{Fore.RED}Введите y/n")
        else:
            self.ai_settings.ai_step_explanations = False

        while True:
            skip_check = input(
                f"{Fore.CYAN}4. Пропустить проверку модели Ollama? (y/n, по умолчанию n): ").strip().lower()
            if skip_check in ['y', 'yes', 'да', 'д']:
                self.ai_settings.skip_model_check = True
                break
            elif skip_check in ['', 'n', 'no', 'нет', 'н']:
                self.ai_settings.skip_model_check = False
                break
            else:
                print(f"{Fore.RED}Введите y/n")

        print(f"{Fore.GREEN}✅ Настройки ИИ сохранены")

    def detect_input_type(self, expression: str) -> str:
        """Определение типа входного выражения"""
        # LaTeX паттерны
        latex_patterns = [
            r'\\[a-zA-Z]+',  # LaTeX команды
            r'\{.*\}',  # Фигурные скобки
            r'\\frac',  # Дроби
            r'\\sqrt',  # Корни
            r'\\int',  # Интегралы
        ]

        if any(re.search(pattern, expression) for pattern in latex_patterns):
            return 'latex'
        return 'mathematical'

    def preprocess_text(self, expression: str) -> str:
        """ИСПРАВЛЕННАЯ предобработка текста с правильным порядком замен"""
        # Убираем лишние пробелы
        expression = re.sub(r'\s+', ' ', expression.strip())

        # ИСПРАВЛЕНИЕ: СНАЧАЛА замена математических функций (до неявного умножения!)
        function_patterns = [
            (r'\blg\b', 'log10'),  # lg -> log10
            (r'\bln\b', 'log'),  # ln -> log
            (r'\barctg\b', 'atan'),  # arctg -> atan
            (r'\barcsin\b', 'asin'),  # arcsin -> asin
            (r'\barccos\b', 'acos'),  # arccos -> acos
            (r'\bsh\b', 'sinh'),  # sh -> sinh
            (r'\bch\b', 'cosh'),  # ch -> cosh
            (r'\bth\b', 'tanh'),  # th -> tanh
        ]

        for pattern, replacement in function_patterns:
            expression = re.sub(pattern, replacement, expression)

        # Замена кириллических символов
        for cyrillic, latin in self.cyrillic_replacements.items():
            expression = expression.replace(cyrillic, latin)

        # Замена специальных символов
        for symbol, replacement in self.symbol_replacements.items():
            expression = expression.replace(symbol, replacement)

        return expression

    def handle_mathematical_notation(self, expression: str) -> sp.Expr:
        """ИСПРАВЛЕННАЯ обработка математической нотации БЕЗ variable-width lookbehind"""

        # Список защищенных математических констант и функций
        protected_words = [
            'pi', 'E', 'oo', 'I',  # Константы
            'sin', 'cos', 'tan', 'cot', 'sec', 'csc',  # Тригонометрические
            'asin', 'acos', 'atan', 'acot', 'asec', 'acsc',  # Обратные тригонометрические
            'sinh', 'cosh', 'tanh', 'coth', 'sech', 'csch',  # Гиперболические
            'log', 'log10', 'ln', 'exp', 'sqrt', 'cbrt',  # Логарифмы и корни
            'factorial', 'gamma', 'beta',  # Специальные функции
            'Abs', 'abs', 'Max', 'Min',  # Другие функции
            'Sum', 'Product', 'Integral', 'Derivative'  # SymPy конструкции
        ]

        # Обрабатываем функции с аргументами
        function_patterns = [
            (r'log10\(([^)]+)\)', lambda m: str(sp.log(sp.sympify(m.group(1)), 10))),
            (r'log\(([^)]+)\)', lambda m: str(sp.log(sp.sympify(m.group(1))))),
            (r'sqrt\(([^)]+)\)', lambda m: str(sp.sqrt(sp.sympify(m.group(1))))),
            (r'sin\(([^)]+)\)', lambda m: str(sp.sin(sp.sympify(m.group(1))))),
            (r'cos\(([^)]+)\)', lambda m: str(sp.cos(sp.sympify(m.group(1))))),
            (r'tan\(([^)]+)\)', lambda m: str(sp.tan(sp.sympify(m.group(1))))),
            (r'factorial\(([^)]+)\)', lambda m: str(sp.factorial(sp.sympify(m.group(1))))),
            (r'exp\(([^)]+)\)', lambda m: str(sp.exp(sp.sympify(m.group(1))))),
        ]

        # Вычисляем функции если аргументы - числа
        for pattern, replacement in function_patterns:
            def replace_func(match):
                try:
                    arg = match.group(1)
                    try:
                        arg_value = sp.sympify(arg)
                        if arg_value.is_number or not arg_value.free_symbols:
                            return replacement(match)
                        else:
                            return match.group(0)
                    except:
                        return match.group(0)
                except:
                    return match.group(0)

            expression = re.sub(pattern, replace_func, expression)

        # Обработка степенных функций: sin^2(x) -> sin(x)**2
        expression = re.sub(r'(\w+)\^(\d+)\(([^)]+)\)', r'\1(\3)**\2', expression)

        # ИСПРАВЛЕННОЕ неявное умножение БЕЗ variable-width lookbehind
        # Метод 1: Защищаем специальные слова временными маркерами

        # Словарь для временных замен
        temp_replacements = {}
        temp_counter = 0

        # Заменяем защищенные слова на временные маркеры
        for word in protected_words:
            if word in expression:
                temp_marker = f"__TEMP_{temp_counter}__"
                temp_replacements[temp_marker] = word
                expression = expression.replace(word, temp_marker)
                temp_counter += 1

        # Теперь применяем неявное умножение безопасно
        # 2x -> 2*x (число + буква)
        expression = re.sub(r'(\d+)([a-zA-Z_])', r'\1*\2', expression)

        # x2 -> x*2 (буква + число)
        expression = re.sub(r'([a-zA-Z_])(\d+)', r'\1*\2', expression)

        # 2(x+1) -> 2*(x+1) (число + скобка)
        expression = re.sub(r'(\d+)\(', r'\1*(', expression)

        # )(x -> )*(x (скобка + буква)
        expression = re.sub(r'\)([a-zA-Z_])', r')*\1', expression)

        # )2 -> )*2 (скобка + число)
        expression = re.sub(r'\)(\d+)', r')*\1', expression)

        # Возвращаем защищенные слова обратно
        for temp_marker, original_word in temp_replacements.items():
            expression = expression.replace(temp_marker, original_word)

        # Обработка модулей: |x| -> Abs(x)
        expression = re.sub(r'\|([^|]+)\|', r'Abs(\1)', expression)

        return parse_with_implicit_multiplication(expression)

    def parse_latex_expression(self, latex_expr: str) -> sp.Expr:
        """Парсинг LaTeX выражений"""
        try:
            return parse_latex(latex_expr)
        except Exception as e:
            print(f"Ошибка парсинга LaTeX: {e}")
            # Fallback: убираем LaTeX команды и парсим как обычное выражение
            cleaned = re.sub(r'\\[a-zA-Z]+\{([^}]+)\}', r'\1', latex_expr)
            cleaned = cleaned.replace('{', '').replace('}', '')
            return self.parse_mathematical_expression(cleaned)

    def parse_mathematical_expression(self, expression: str) -> sp.Expr:
        """Исправленный парсинг математических выражений"""
        try:
            # Проверяем, есть ли знак равенства
            if '=' in expression and not any(op in expression for op in ['<=', '>=', '==']):
                parts = expression.split('=')
                if len(parts) == 2:
                    left = self.preprocess_text(parts[0].strip())
                    right = self.preprocess_text(parts[1].strip())
                    left_expr = self.handle_mathematical_notation(left)
                    right_expr = self.handle_mathematical_notation(right)
                    return sp.Eq(left_expr, right_expr)

            # ИСПРАВЛЕНИЕ: Специальная обработка функций SymPy
            processed = self.preprocess_text(expression)

            # Обработка integrate() - создаем объект Integral
            if 'integrate(' in processed:
                # Парсим как функцию SymPy без немедленного вычисления
                match = re.search(r'integrate\(([^,]+),\s*([^)]+)\)', processed)
                if match:
                    func_expr = sp.sympify(match.group(1))
                    var = sp.Symbol(match.group(2).strip())
                    return sp.Integral(func_expr, var)

            # Обработка diff() - создаем объект Derivative
            if 'diff(' in processed:
                match = re.search(r'diff\(([^,]+),\s*([^)]+)\)', processed)
                if match:
                    func_expr = sp.sympify(match.group(1))
                    var = sp.Symbol(match.group(2).strip())
                    return sp.Derivative(func_expr, var)

            return self.handle_mathematical_notation(processed)

        except Exception as e:
            raise ValueError(f"Не удалось распарсить выражение '{expression}': {e}")

    def validate_expression(self, expr: sp.Expr) -> bool:
        """Валидация математического выражения"""
        try:
            # Проверка на бесконечности и NaN
            if expr.has(sp.zoo) or expr.has(sp.nan):
                return False

            # Попытка упрощения для проверки корректности
            simplified = sp.simplify(expr)
            return True
        except Exception:
            return False

    def step1_parse_and_validate(self, user_input: str) -> Dict[str, Any]:
        """Этап 1: Преобразование и валидация"""
        result = {
            'stage': 'Этап 1: Преобразование и валидация',
            'success': False,
            'input_type': None,
            'original_input': user_input,
            'preprocessed': None,
            'parsed_expression': None,
            'validation_result': False,
            'errors': []
        }

        try:
            # Определение типа ввода
            input_type = self.detect_input_type(user_input)
            result['input_type'] = input_type

            # Парсинг в зависимости от типа
            if input_type == 'latex':
                expr = self.parse_latex_expression(user_input)
            else:
                expr = self.parse_mathematical_expression(user_input)

            result['parsed_expression'] = expr
            result['preprocessed'] = str(expr)

            # Валидация
            validation_result = self.validate_expression(expr)
            result['validation_result'] = validation_result
            result['success'] = validation_result

        except Exception as e:
            error_msg = str(e)
            result['errors'].append(error_msg)

            # ИИ объяснение ошибки
            if self.ai and self.ai_settings.ai_error_explanations:
                print(f"{Fore.MAGENTA}🤖 ИИ объясняет ошибку:")
                ai_explanation = self.ai.explain_error(error_msg, user_input)
                if ai_explanation and "Модель недоступна" not in ai_explanation:
                    print(f"{Fore.YELLOW}{ai_explanation}")
                print()

        return result

    def step2_simplify(self, expr: sp.Expr) -> Dict[str, Any]:
        """Этап 2: Упрощение выражения с сохранением типа объекта"""
        result = {
            'stage': 'Этап 2: Упрощение выражения',
            'success': False,
            'original_expression': expr,
            'simplified_expression': None,
            'simplification_steps': {},
            'errors': []
        }

        try:
            steps = {}

            # ИСПРАВЛЕНИЕ: Сохраняем тип объекта для Derivative и Integral
            if isinstance(expr, (sp.Derivative, sp.Integral)):
                # Для производных и интегралов НЕ вычисляем автоматически
                result['simplified_expression'] = expr
                result['simplification_steps'] = {'preserved': 'Тип объекта сохранен для корректного решения'}
            else:
                # ИСПРАВЛЕНИЕ: Специальная обработка log(exp(x)) ПЕРЕД обычным упрощением
                expr_str = str(expr)

                # Прямая обработка основных логарифмических тождеств
                if 'log(exp(' in expr_str:
                    # log(exp(x)) -> x
                    import re
                    pattern = r'log\(exp\(([^)]+)\)\)'
                    match = re.search(pattern, expr_str)
                    if match:
                        var_content = match.group(1)
                        if var_content.strip():  # Проверяем что переменная не пустая
                            try:
                                simplified_expr = sp.sympify(var_content)
                                steps['log_exp_identity'] = simplified_expr
                            except:
                                pass

                # Проверка других логарифмических тождеств
                if 'exp(log(' in expr_str:
                    # exp(log(x)) -> x
                    pattern = r'exp\(log\(([^)]+)\)\)'
                    match = re.search(pattern, expr_str)
                    if match:
                        var_content = match.group(1)
                        if var_content.strip():
                            try:
                                simplified_expr = sp.sympify(var_content)
                                steps['exp_log_identity'] = simplified_expr
                            except:
                                pass

                # Обычное упрощение для других выражений
                try:
                    steps['basic'] = sp.simplify(expr)
                except:
                    steps['basic'] = expr

                try:
                    steps['expanded'] = sp.expand(expr)
                except:
                    steps['expanded'] = expr

                try:
                    steps['factored'] = sp.factor(expr)
                except:
                    steps['factored'] = expr

                try:
                    steps['cancelled'] = sp.cancel(expr)
                except:
                    steps['cancelled'] = expr

                # Приведение подобных
                if expr.free_symbols:
                    try:
                        steps['collected'] = sp.collect(expr, list(expr.free_symbols))
                    except:
                        steps['collected'] = expr

                # Тригонометрическое упрощение
                if any(func in expr_str for func in ['sin', 'cos', 'tan', 'sec', 'csc', 'cot']):
                    try:
                        steps['trigsimp'] = sp.trigsimp(expr)
                    except:
                        steps['trigsimp'] = expr

                # Улучшенное логарифмическое упрощение для SymPy
                if any(func in expr_str for func in ['log', 'exp', 'ln']):
                    try:
                        # Комбинация различных методов
                        steps['logsimp'] = sp.logcombine(expr, force=True)
                        steps['powersimp'] = sp.powsimp(expr, force=True)
                        steps['expand_log'] = sp.expand_log(expr, force=True)

                        # Принудительное упрощение через замену
                        if 'log(exp(' in expr_str:
                            x = sp.Symbol('x')
                            pattern = sp.log(sp.exp(x))
                            replacement = x
                            steps['log_exp_replace'] = expr.replace(sp.log(sp.exp(sp.Wild('x'))), sp.Wild('x'))
                    except Exception as e:
                        pass

                # Выбор лучшего результата
                candidates = []
                for step_name, step_result in steps.items():
                    if step_result is not None and step_result != expr:
                        candidates.append((step_name, step_result))

                if candidates:
                    # Приоритет для специальных тождеств
                    priority_steps = ['log_exp_identity', 'exp_log_identity', 'log_exp_replace']

                    # Сначала проверяем приоритетные шаги
                    result['simplified_expression'] = expr
                    for priority in priority_steps:
                        for step_name, step_result in candidates:
                            if step_name == priority:
                                result['simplified_expression'] = step_result
                                break
                        if result['simplified_expression'] != expr:
                            break

                    # Если приоритетных нет, выбираем лучший по количеству операций
                    if result['simplified_expression'] == expr:
                        valid_candidates = []
                        for step_name, step_result in candidates:
                            try:
                                ops_count = sp.count_ops(step_result)
                                valid_candidates.append((ops_count, step_result))
                            except:
                                valid_candidates.append((len(str(step_result)), step_result))

                        if valid_candidates:
                            best = min(valid_candidates, key=lambda x: x[0])
                            result['simplified_expression'] = best[1]
                        else:
                            result['simplified_expression'] = expr
                else:
                    result['simplified_expression'] = expr

            # Фильтруем шаги для отображения
            result['simplification_steps'] = {
                k: v for k, v in steps.items()
                if v is not None and v != expr and str(v) != str(expr)
            }
            result['success'] = True

        except Exception as e:
            result['errors'].append(str(e))
            result['simplified_expression'] = expr

        return result

    def step3_solve(self, expr: sp.Expr) -> Dict[str, Any]:
        """Исправленное решение задач"""
        result = {
            'stage': 'Этап 3: Решение задачи',
            'success': False,
            'problem_type': None,
            'solutions': None,
            'additional_info': {},
            'errors': []
        }

        try:
            free_symbols = expr.free_symbols

            # ИСПРАВЛЕНИЕ: Проверка типа объекта SymPy ПЕРВООЧЕРЕДНО
            if isinstance(expr, sp.Derivative):
                # Это производная - вычисляем её
                result['problem_type'] = 'derivative'
                result['solutions'] = expr.doit()
            elif isinstance(expr, sp.Integral):
                # Это интеграл - вычисляем его
                result['problem_type'] = 'integral'
                result['solutions'] = expr.doit()
            elif isinstance(expr, sp.Eq):
                # Уравнение
                result['problem_type'] = 'equation'
                result['solutions'] = sp.solve(expr, free_symbols)
            elif any(op in str(expr) for op in ['>', '<', '>=', '<=']):
                # Неравенство
                result['problem_type'] = 'inequality'
                if len(free_symbols) == 1:
                    var = list(free_symbols)[0]
                    try:
                        result['solutions'] = sp.solve_univariate_inequality(expr, var, relational=False)
                    except:
                        result['solutions'] = sp.solve(expr, var)
            elif not free_symbols:
                # Числовое выражение
                result['problem_type'] = 'numerical_evaluation'
                result['solutions'] = expr.evalf()
            else:
                # ИСПРАВЛЕНИЕ: Улучшенная логика определения типа задачи
                # Проверяем, можно ли упростить выражение
                simplified = sp.simplify(expr)

                # Дополнительная проверка для логарифмических выражений
                if any(func in str(expr) for func in ['log', 'exp']):
                    log_simplified = sp.powsimp(sp.logcombine(expr, force=True), force=True)
                    if sp.count_ops(log_simplified) < sp.count_ops(simplified):
                        simplified = log_simplified

                # НОВОЕ ИСПРАВЛЕНИЕ: Проверка на результат упрощения
                is_simple_variable = (
                        len(free_symbols) == 1 and
                        len(str(expr).strip()) <= 3 and  # x, y, z и т.д.
                        str(expr) in [str(sym) for sym in free_symbols] and
                        isinstance(expr, sp.Symbol)  # Убеждаемся что это именно символ
                )

                # Проверка на упрощение выражения
                expression_simplified = (
                        simplified != expr and
                        (sp.count_ops(simplified) < sp.count_ops(expr) or str(simplified) != str(expr))
                )

                if is_simple_variable:
                    # Это простая переменная - скорее всего результат упрощения
                    result['problem_type'] = 'simplification'
                    result['solutions'] = expr
                elif expression_simplified:
                    # Выражение упростилось - это задача на упрощение
                    result['problem_type'] = 'simplification'
                    result['solutions'] = simplified
                else:
                    # Пытаемся решить как уравнение = 0
                    result['problem_type'] = 'equation_zero'
                    if len(free_symbols) == 1:
                        var = list(free_symbols)[0]
                        result['solutions'] = sp.solve(expr, var)
                    else:
                        result['solutions'] = sp.solve(expr, free_symbols)

            result['success'] = True

        except Exception as e:
            result['errors'].append(str(e))
            # Fallback: упрощение
            try:
                result['solutions'] = sp.simplify(expr)
                result['problem_type'] = 'simplification'
                result['success'] = True
            except:
                pass

        return result

    def step4_format_output(self, solutions: Any, problem_type: str) -> Dict[str, Any]:
        """Этап 4: Форматирование вывода"""
        result = {
            'stage': 'Этап 4: Форматирование вывода',
            'success': False,
            'numerical_output': None,
            'latex_output': None,
            'pretty_output': None,
            'errors': []
        }

        try:
            if solutions is not None:
                # Численный вывод
                if hasattr(solutions, 'evalf'):
                    result['numerical_output'] = str(solutions.evalf())
                else:
                    result['numerical_output'] = str(solutions)

                # LaTeX вывод
                try:
                    if hasattr(solutions, '__iter__') and not isinstance(solutions, (str, sp.Basic)):
                        latex_parts = []
                        for sol in solutions:
                            try:
                                latex_parts.append(sp.latex(sol))
                            except:
                                latex_parts.append(str(sol))
                        result['latex_output'] = ', '.join(latex_parts)
                    else:
                        result['latex_output'] = sp.latex(solutions)
                except:
                    result['latex_output'] = str(solutions)

                # Красивый вывод
                try:
                    result['pretty_output'] = sp.pretty(solutions)
                except:
                    result['pretty_output'] = str(solutions)

                result['success'] = True

        except Exception as e:
            result['errors'].append(str(e))

        return result

    def process_user_input(self, user_input: str, show_steps: bool = True) -> Dict[str, Any]:
        """Основная функция обработки пользовательского ввода"""
        if show_steps:
            print(f"\n{Fore.CYAN}{'=' * 60}")
            print(f"{Fore.CYAN}ОБРАБОТКА МАТЕМАТИЧЕСКОГО ВЫРАЖЕНИЯ")
            print(f"{Fore.CYAN}{'=' * 60}")
            print(f"{Fore.WHITE}Входное выражение: {Fore.YELLOW}{user_input}")

        # Инициализация результата
        final_result = {
            'success': False,
            'original_input': user_input,
            'stages': [],
            'final_answer': None,
            'problem_type': None
        }

        try:
            # Этап 1: Парсинг и валидация
            stage1 = self.step1_parse_and_validate(user_input)
            final_result['stages'].append(stage1)

            if show_steps:
                print(f"\n{Fore.GREEN}✓ {stage1['stage']}")
                print(f" Тип ввода: {stage1['input_type']}")
                if stage1['success']:
                    print(f" Обработанное выражение: {stage1['parsed_expression']}")

                    # ИИ объяснение этапа
                    if self.ai and self.ai_settings.ai_step_explanations:
                        ai_explanation = self.ai.explain_step(
                            "Парсинг", user_input, str(stage1['parsed_expression'])
                        )
                        if ai_explanation and "Модель недоступна" not in ai_explanation:
                            print(f"{Fore.MAGENTA}🤖 ИИ: {Fore.CYAN}{ai_explanation}")
                else:
                    print(f" {Fore.RED}Ошибки: {stage1['errors']}")
                    return final_result

            expr = stage1['parsed_expression']

            # Этап 2: Упрощение
            stage2 = self.step2_simplify(expr)
            final_result['stages'].append(stage2)

            if show_steps:
                print(f"\n{Fore.GREEN}✓ {stage2['stage']}")
                print(f" Упрощенное выражение: {stage2['simplified_expression']}")
                if stage2['simplification_steps']:
                    print(f" Промежуточные шаги:")
                    for step_name, step_result in stage2['simplification_steps'].items():
                        print(f"   {step_name}: {step_result}")

                # ИИ объяснение этапа
                if self.ai and self.ai_settings.ai_step_explanations:
                    ai_explanation = self.ai.explain_step(
                        "Упрощение", str(expr), str(stage2['simplified_expression'])
                    )
                    if ai_explanation and "Модель недоступна" not in ai_explanation:
                        print(f"{Fore.MAGENTA}🤖 ИИ: {Fore.CYAN}{ai_explanation}")

            simplified_expr = stage2['simplified_expression']

            # Этап 3: Решение
            stage3 = self.step3_solve(simplified_expr)
            final_result['stages'].append(stage3)
            final_result['problem_type'] = stage3['problem_type']

            if show_steps:
                print(f"\n{Fore.GREEN}✓ {stage3['stage']}")
                print(f" Тип задачи: {stage3['problem_type']}")
                print(f" Решение: {stage3['solutions']}")

                # ИИ объяснение этапа
                if self.ai and self.ai_settings.ai_step_explanations:
                    ai_explanation = self.ai.explain_step(
                        f"Решение ({stage3['problem_type']})",
                        str(simplified_expr), str(stage3['solutions'])
                    )
                    if ai_explanation and "Модель недоступна" not in ai_explanation:
                        print(f"{Fore.MAGENTA}🤖 ИИ: {Fore.CYAN}{ai_explanation}")

            # Этап 4: Форматирование
            stage4 = self.step4_format_output(stage3['solutions'], stage3['problem_type'])
            final_result['stages'].append(stage4)

            if show_steps:
                print(f"\n{Fore.GREEN}✓ {stage4['stage']}")
                print(f" Численный результат: {stage4['numerical_output']}")
                print(f" LaTeX формат: {stage4['latex_output']}")

            final_result['final_answer'] = stage4
            final_result['success'] = True

        except Exception as e:
            if show_steps:
                print(f"{Fore.RED}Критическая ошибка: {e}")
            final_result['error'] = str(e)

            # ИИ объяснение критической ошибки
            if self.ai and self.ai_settings.ai_error_explanations:
                print(f"{Fore.MAGENTA}🤖 ИИ объясняет критическую ошибку:")
                ai_explanation = self.ai.explain_error(str(e), user_input)
                if ai_explanation and "Модель недоступна" not in ai_explanation:
                    print(f"{Fore.YELLOW}{ai_explanation}")

        if show_steps:
            print(f"\n{Fore.CYAN}{'=' * 60}")

        return final_result


def print_usage_instructions():
    """Инструкции по использованию"""
    instructions = f"""

{Fore.CYAN}╔══════════════════════════════════════════════════════════════════════════════════╗
{Fore.CYAN}║ ИНСТРУКЦИЯ ПО ВВОДУ ВЫРАЖЕНИЙ                                                   ║
{Fore.CYAN}╠══════════════════════════════════════════════════════════════════════════════════╣
{Fore.CYAN}║                                                                                  ║
{Fore.CYAN}║ 1. ПОДДЕРЖИВАЕМЫЕ ФОРМАТЫ ВВОДА:                                                ║
{Fore.CYAN}║                                                                                  ║
{Fore.CYAN}║   • Plain Text (обычный текст)                                                   ║
{Fore.CYAN}║   • LaTeX (математическая разметка)                                              ║
{Fore.CYAN}║                                                                                  ║
{Fore.CYAN}║ 2. ПРИМЕРЫ ВВОДА В ФОРМАТЕ PLAIN TEXT:                                          ║
{Fore.CYAN}║                                                                                  ║
{Fore.CYAN}║   Уравнения:                                                                     ║
{Fore.CYAN}║   • x^2 + 4*x = -8                                                               ║
{Fore.CYAN}║   • 2*x + 3 = 7                                                                  ║
{Fore.CYAN}║   • sin(x) = 0.5                                                                 ║
{Fore.CYAN}║   • x^2 - 4 = 0                                                                  ║
{Fore.CYAN}║                                                                                  ║
{Fore.CYAN}║   Неравенства:                                                                   ║
{Fore.CYAN}║   • x^2 - 4 > 0                                                                  ║
{Fore.CYAN}║   • 2*x + 1 <= 5                                                                 ║
{Fore.CYAN}║                                                                                  ║
{Fore.CYAN}║   Выражения для упрощения:                                                       ║
{Fore.CYAN}║   • x^2 + 2*x + 1                                                                ║
{Fore.CYAN}║   • sin^2(x) + cos^2(x)                                                          ║
{Fore.CYAN}║   • (x+1)*(x-1)                                                                  ║
{Fore.CYAN}║                                                                                  ║
{Fore.CYAN}║   Функции SymPy:                                                                 ║
{Fore.CYAN}║   • diff(x^3, x) (производная)                                                   ║
{Fore.CYAN}║   • integrate(x^2, x) (интеграл)                                                 ║
{Fore.CYAN}║   • factorial(5) (факториал)                                                     ║
{Fore.CYAN}║   • sqrt(16) (квадратный корень)                                                 ║
{Fore.CYAN}║   • log(e^x) (логарифм)                                                          ║
{Fore.CYAN}║                                                                                  ║
{Fore.CYAN}║ 4. ИИ КОМАНДЫ:                                                                   ║
{Fore.CYAN}║   • llm, ollama, ai, ии, ллм - диалог с ИИ помощником                           ║
{Fore.CYAN}║                                                                                  ║
{Fore.CYAN}║ 5. КОМАНДЫ:                                                                      ║
{Fore.CYAN}║   • help - эта справка                                                           ║
{Fore.CYAN}║   • quit/exit - выход                                                            ║
{Fore.CYAN}║                                                                                  ║
{Fore.CYAN}╚══════════════════════════════════════════════════════════════════════════════════╝

"""
    print(instructions)


def interactive_mode():
    """Интерактивный режим работы с программой"""
    processor = AdvancedMathProcessor()

    print(f"{Fore.CYAN}{'=' * 80}")
    print(f"{Fore.CYAN}ИНТЕРАКТИВНЫЙ РЕШАТЕЛЬ МАТЕМАТИЧЕСКИХ ЗАДАЧ")
    print(f"{Fore.CYAN}{'=' * 80}")
    print(f"{Fore.WHITE}Введите 'help' для получения инструкций")
    print(f"{Fore.WHITE}Введите 'quit' или 'exit' для выхода")
    print(f"{Fore.WHITE}ИИ команды: llm, ollama, ai, ии, ллм")
    print(f"{Fore.CYAN}{'=' * 80}")

    while True:
        try:
            user_input = input(f"\n{Fore.GREEN}Введите математическое выражение: {Fore.WHITE}")

            if user_input.lower() in ['quit', 'exit', 'выход']:
                print(f"{Fore.YELLOW}Программа завершена. До свидания!")
                break
            elif user_input.lower() in ['help', 'помощь']:
                print_usage_instructions()
                continue
            elif user_input.lower() in ['llm', 'ollama', 'ai', 'ии', 'ллм']:
                if processor.ai:
                    processor.ai.interactive_mode()
                else:
                    print(f"{Fore.RED}ИИ недоступен (Ollama не установлена)")
                continue
            elif user_input.strip() == '':
                continue

            # Обработка выражения
            result = processor.process_user_input(user_input, show_steps=True)

            if result['success']:
                print(f"\n{Fore.GREEN}🎯 ИТОГОВЫЙ РЕЗУЛЬТАТ:")
                print(f"{Fore.WHITE}Тип задачи: {Fore.CYAN}{result['problem_type']}")

                final_answer = result['final_answer']
                if final_answer['numerical_output']:
                    print(f"{Fore.WHITE}Численный результат: {Fore.YELLOW}{final_answer['numerical_output']}")
                if final_answer['latex_output']:
                    print(f"{Fore.WHITE}LaTeX формат: {Fore.MAGENTA}{final_answer['latex_output']}")
                if final_answer['pretty_output']:
                    print(f"{Fore.WHITE}Форматированный вывод:")
                    print(f"{Fore.CYAN}{final_answer['pretty_output']}")
            else:
                print(f"{Fore.RED}❌ Не удалось обработать выражение")
                if 'error' in result:
                    print(f"{Fore.RED}Ошибка: {result['error']}")

        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Программа прервана пользователем")
            break
        except Exception as e:
            print(f"{Fore.RED}Непредвиденная ошибка: {e}")


def main():
    """Главная функция демонстрации"""
    processor = AdvancedMathProcessor()

    # Исправленные примеры использования
    test_cases = [
        "x^2 + 4*x - 8",  # Квадратное выражение
        "sin^2(x) + cos^2(x)",  # Тригонометрическое тождество
        "integrate(x^2, x)",  # Интеграл
        "diff(x^3 + 2*x, x)",  # Производная
        "x^2 - 4 = 0",  # Квадратное уравнение
        "2*x + 3 = 7",  # Линейное уравнение
        "sqrt(16)",  # Квадратный корень
        "factorial(5)",  # Факториал
        "log(E)",  # Логарифм
        "pi + E"  # Константы
    ]

    print(f"\n{Fore.MAGENTA}{'=' * 80}")
    print(f"{Fore.MAGENTA}ДЕМОНСТРАЦИЯ РАБОТЫ ИСПРАВЛЕННОЙ СИСТЕМЫ")
    print(f"{Fore.MAGENTA}{'=' * 80}")

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{Fore.CYAN}Пример {i}:")
        result = processor.process_user_input(test_case, show_steps=True)

        if result['success']:
            print(f"{Fore.GREEN}✓ Обработка завершена успешно")
            if result['final_answer']['latex_output']:
                print(f"{Fore.YELLOW}LaTeX результат: {result['final_answer']['latex_output']}")
        else:
            print(f"{Fore.RED}✗ Ошибка обработки")

        print("-" * 60)


if __name__ == "__main__":
    print(f"{Fore.CYAN}🚀 ЗАПУСК МАТЕМАТИЧЕСКОГО ПРОЦЕССОРА")
    print("1. Тестирование")
    print("2. Интерактивный режим")

    choice = input("Выберите режим (1/2): ").strip()

    if choice == "1":
        main()
    elif choice == "2":
        interactive_mode()
    else:
        print("Неверный выбор, запускаю интерактивный режим...")
        interactive_mode()
