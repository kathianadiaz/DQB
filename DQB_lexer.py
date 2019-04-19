import ply.lex as lex

#DQB Lexer

#List of Tokens
tokens = [

    'INT',
    'BOOLEAN',
    'FLOAT',
    'CHARACTER',
    'COMMA',
    'CLOSE_PAREN',
    'OPEN_PAREN',
    'CLOSE_BRACKET',
    'OPEN_BRACKET',
    'DOS_PUNTITOS',
    'EQUALS',
    'DOT',
    'MAIN',
    'ENVIRONMENT',
    'PONG',
    'BRICK_BREAKER',
    'MODEL_PARAMETERS',
    'NETWORK',
    'TRAINING',
    'PREDICT_MOVES',
    'DISPLAY',
    'CALCULATE_Q_VALUES',
    'MODEL_CURRENT_STATUS',
    'FIND_PROBABILITIES',
    'FIT',
    'LEARNING_RATE',
    'EPSILON_START',
    'EPSILON_END',
    'EXPLORATION_STEPS',
    'BATCH_SIZE',
    'DISCOUNT_FACTOR',
    'NO_STEPS',
    'ADD',
    'ACTION_SIZE',
    'PREPARE',
    'EXECUTE',
    'CONV_LAYER',
    'PREDICTIVE_LAYER'

]

#Regular Expression Rules

t_CHARACTER = r'[a-zA-Z]+_[a-zA-Z]+'

t_INT = r'[[0-9]+'

t_FLOAT = r'[[0-9]+[.][0-9]+'

t_COMMA = r','

t_OPEN_PAREN = r'\('

t_CLOSE_PAREN = r'\)'

t_OPEN_BRACKET = r'\{'

t_CLOSE_BRACKET = r'\}'

t_DOS_PUNTITOS = r':'

t_EQUALS = r'='

t_DOT = r'.'

t_ignore = '\t\n'

def t_error(t):
    print('lol')

def t_MAIN(t):
    r"""main"""
    t.value = 'main'
    return t

def t_ENVIRONMENT(t):
    r"""ENVIRONMENT"""
    t.value = 'ENVIRONMENT'
    return t
    
def t_PONG(t):
    r"""Pong"""
    t.value = 'Pong'
    return t

def t_BRICK_BREAKER(t):
    r"""BrickBreaker"""
    t.value = 'BrickBreaker'
    return t

def t_MODEL_PARAMETERS(t):
    r"""MODEL_PARAMETERS"""
    t.value = 'MODEL_PARAMETERS'
    return t

def t_NETWORK(t):
    r"""NETWORK"""
    t.value = 'NETWORK'
    return t

def t_TRAINING(t):
    r"""TRAINING"""
    t.value = 'TRAINING'
    return t

def t_PREDICT_MOVES(t):
    r"""predict_Moves"""
    t.value = 'predict_Moves'
    return t

def t_DISPLAY(t):
    r"""DISPLAY"""
    t.value = 'DISPLAY'
    return t

def t_CALCULATE_Q_VALUES(t):
    r"""calculateQ_Values"""
    t.value = 'calculateQ_Values'
    return t

def t_MODEL_CURRENT_STATUS(t):
    r"""MODEL_CURRENT_STATUS"""
    t.value = 'MODEL_CURRENT_STATUS'
    return t

def t_FIND_PROBABILITIES(t):
    r"""FIND_PROBABILITIES"""
    t.value = 'FIND_PROBABILITIES'
    return t

def t_FIT(t):
    r"""FIT"""
    t.value = 'FIT'
    return t

def t_LEARNING_RATE(t):
    r"""LEARNING_RATE"""
    t.value = 'Learning_Rate'
    return t

def t_EPSILON_START(t):
    r"""EPSILON_START"""
    t.value = 'Epsilon_Start'
    return t

def t_EPSILON_END(t):
    r"""EPSILON_END"""
    t.value = 'Epsilon_End'
    return t

def t_EXPLORATION(t):
    r"""EXPLORATION"""
    t.value = 'Exploration_Steps'
    return t

def t_BATCH_SIZE(t):
    r"""BATCH_SIZE"""
    t.value = 'Batch_Size'
    return t

def t_DISCOUNT_FACTOR(t):
    r"""DISCOUNT_FACTOR"""
    t.value = 'Discount_Factor'
    return t

def t_NO_STEPS(t):
    r"""NO_STEPS"""
    t.value = 'No_Steps'
    return t

def t_ADD(t):
    r"""ADD"""
    t.value = 'add'
    return t

def t_ACTION_SIZE(t):
    r"""ACTION_SIZE"""
    t.value = 'Action_Size'
    return t

def t_PREPARE(t):
    r"""PREPARE"""
    t.value = 'PREPARE'
    return t

def t_EXECUTE(t):
    r"""EXECUTE"""
    t.value = 'EXECUTE'
    return t

def t_CONV_LAYER(t):
    r"""CONV_LAYER"""
    t.value = 'ConvolutionalLayers'
    return t

def t_PREDICTIVE_LAYER(t):
    r"""PREDICTIVE_LAYER"""
    t.value = 'PredictiveLayers'
    return t


#Initializes lexer

lexer = lex.lex()

try:
    DQB_test = open("script.txt", 'r')
except IOError:
   print("Error opening file")
   exit()

fileText = DQB_test.read()
lexer.input(fileText)

 # Tokenize

while True:
  tok = lexer.token()
  if not tok:
    print(tok)
    break


