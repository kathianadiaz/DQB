import ply.lex as lex

#DQB Lexer

#List of Tokens
tokens = [

    'INT',
    'FLOAT',
    'CLOSE_PAREN',
    'OPEN_PAREN',
    'CLOSE_BRACKET',
    'OPEN_BRACKET',
    'DOS_PUNTITOS',
    'EQUALS',
    'MAIN',
    'ENVIRONMENT',
    'PONG',
    'BRICK_BREAKER',
    'MODEL_PARAMETERS',
    'AGENT',
    'NETWORK',
    'TRAINING',
    'PREDICT_MOVES',
    'DISPLAY_GAME',
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
    'EXECUTE',
    'CONV_LAYER',
    'PREDICTIVE_LAYER',
    'SHOW_MODEL_SUMMARY',
    'NAME'
]

#Regular Expression Rules

t_INT = r'[[0-9]+'

t_FLOAT = r'[[0-9]+[.][0-9]+'

t_OPEN_PAREN = r'\('

t_CLOSE_PAREN = r'\)'

t_OPEN_BRACKET = r'\{'

t_CLOSE_BRACKET = r'\}'

t_DOS_PUNTITOS = r':'

t_EQUALS = r'='

t_ignore = ' \t\n'

def t_error(t):
    print('Syntax error in source code. Try again!')

def t_MAIN(t):
    r"""main"""
    t.value = 'main'
    return t

def t_ENVIRONMENT(t):
    r"""ENVIRONMENT"""
    t.value = 'ENVIRONMENT'
    return t

def t_AGENT(t):
    r"""AGENT"""
    t.value = 'AGENT'
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
    r"""predict_moves"""
    t.value = 'predict_moves'
    return t

def t_DISPLAY_GAME(t):
    r"""displayGame"""
    t.value = 'DISPLAY_GAME'
    return t

def t_CALCULATE_Q_VALUES(t):
    r"""calculateQ_Values"""
    t.value = 'calculateQ_Values'
    return t

def t_FIND_PROBABILITIES(t):
    r"""find_probabilities"""
    t.value = 'find_probabilities'
    return t

def t_FIT(t):
    r"""fit"""
    t.value = 'fit'
    return t

def t_LEARNING_RATE(t):
    r"""Learning_Rate"""
    t.value = 'Learning_Rate'
    return t

def t_EPSILON_START(t):
    r"""Epsilon_Start"""
    t.value = 'Epsilon_Start'
    return t

def t_EPSILON_END(t):
    r"""Epsilon_End"""
    t.value = 'Epsilon_End'
    return t

def t_EXPLORATION_STEPS(t):
    r"""Exploration_Steps"""
    t.value = 'Exploration_Steps'
    return t

def t_BATCH_SIZE(t):
    r"""Batch_Size"""
    t.value = 'Batch_Size'
    return t

def t_DISCOUNT_FACTOR(t):
    r"""Discount_Factor"""
    t.value = 'Discount_Factor'
    return t

def t_NO_STEPS(t):
    r"""No_Steps"""
    t.value = 'No_Steps'
    return t

def t_ADD(t):
    r"""add"""
    t.value = 'ADD'
    return t

def t_ACTION_SIZE(t):
    r"""Action_Size"""
    t.value = 'Action_Size'
    return t


def t_EXECUTE(t):
    r"""Execute"""
    t.value = 'Execute'
    return t

def t_CONV_LAYER(t):
    r"""ConvolutionalLayers"""
    t.value = 'ConvolutionalLayers'
    return t

def t_PREDICTIVE_LAYER(t):
    r"""PredictiveLayers"""
    t.value = 'PredictiveLayers'
    return t

def t_SHOW_MODEL_SUMMARY(t):
    r"""showModelSummary"""
    t.value = 'showModelSummary'
    return t

def t_MODEL_CURRENT_STATUS(t):
    r"""modelCurrentStatus"""
    t.value = 'modelCurrentStatus'
    return t

def t_NAME(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    t.type = 'NAME'
    return t


def CONDITIONAL(t):
    r"""CONDITIONAL"""
    t.value = 'CONDITIONAL'
    return t

#Initializes lexer

lexer = lex.lex()

try:
    DQB_test = open("DQB_script.txt", 'r')
except IOError:
   print("Error opening file")
   exit()

fileText = DQB_test.read()
lexer.input(fileText)

 # Tokenize

while True:
  tok = lexer.token()
  if not tok:
    break


