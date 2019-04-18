import ply.lex as lex

#DQB Lexer

#List of Tokens
tokens = [

    'DIGIT' 
    'BOOLEAN'
    'FLOAT'
    'CHARACTER'
    'COMMA' 
    'CLOSE_PAREN' 
    'OPEN_PAREN' 
    'CLOSE_BRACKET' 
    'OPEN_BRACKET' 
    'DOS_PUNTITOS'
    'MAIN'
    'ENVIRONMENT' 
    'PONG'
    'BRICK_BREAKER'
    'MODEL_PARAMETERS' 
    'NETWORK'
    'TRAINING' 
    'PREDICT_MOVES' 
    'DISPLAY' 
    'CALCULATE_Q_VALUES' 
    'MODEL_CURRENT_STATUS' 
    'FIND_PROB'
    'FIT' 
    'LEARNING_RATE' 
    'EPSILON_START' 
    'EPSILON_END' 
    'EXPLORATION' 
    'BATCH_SIZE' 
    'DISCOUNT_FACTOR' 
    'NO_STEPS' 
    'ADD'
    'ACTION_SIZE' 
    'PREPARE' 
    'EXECUTE' 
    'CONV_LAYER' 
    'PREDICTIVE_LAYER' 

]

#Regular Expression Rules

def t_DIGIT(t):
    r'DIGIT/INT' 
    t.value = 'DIGIT/INT'
    return t

def BOOLEAN(t):
    r'BOOLEAN' 
    t.value = 'BOOLEAN'
    return t

def t_FLOAT(t):
    r'FLOAT' 
    t.value = 'FLOAT'
    return t
    
def t_CHARACTER(t):
    r'CHARACTER' 
    t.value = 'CHARACTER'
    return t

def t_COMMA(t):
    r'COMMA' 
    t.value = 'COMMA'
    return t

def t_CLOSE_PAREN(t):
    r'CLOSE_PAREN' 
    t.value = 'CLOSE_PAREN'
    return t

def t_OPEN_PAREN(t):
    r'OPEN_PAREN' 
    t.value = 'OPEN_PAREN'
    return t

def t_CLOSE_BRACKET(t):
    r'CLOSE_BRACKET' 
    t.value = 'CLOSE_BRACKET'
    return t

def t_OPEN_BRACKET(t):
    r'OPEN_BRACKET' 
    t.value = 'OPEN_BRACKET'
    return t

def t_DOS_PUNTITOS(t):
    r'DOS_PUNTITOS' 
    t.value = 'DOS_PUNTITOS'
    return t

def t_MAIN(t):
    r'MAIN' 
    t.value = 'MAIN'
    return t

def t_ENVIRONMENT(t):
    r'ENVIRONMENT' 
    t.value = 'ENVIRONMENT'
    return t
    
def t_PONG(t):
    r'PONG' 
    t.value = 'PONG'
    return t

def t_BRICK_BREAKER(t):
    r'BRICK_BREAKER' 
    t.value = 'BRICK_BREAKER'
    return t

def t_MODEL_PARAMETERS(t):
    r'MODEL_PARAMETERS' 
    t.value = 'MODEL_PARAMETERS'
    return t

def t_NETWORK(t):
    r'NETWORK' 
    t.value = 'NETWORK'
    return t

def t_TRAINING(t):
    r'TRAINING' 
    t.value = 'TRAINING'
    return t

def t_PREDICT_MOVES(t):
    r'PREDICT_MOVES' 
    t.value = 'PREDICT_MOVES'
    return t

def t_DISPLAY(t):
    r'DISPLAY' 
    t.value = 'DISPLAY'
    return t

def t_CALCULATE_Q_VALUES(t):
    r'CALCULATE_Q_VALUES' 
    t.value = 'CALCULATE_Q_VALUES'
    return t

def t_MODEL_CURRENT_STATUS(t):
    r'MODEL_CURRENT_STATUS' 
    t.value = 'MODEL_CURRENT_STATUS'
    return t

def t_FIND_PROB(t):
    r'FIND_PROB' 
    t.value = 'FIND_PROB'
    return t

def t_FIT(t):
    r'FIT' 
    t.value = 'FIT'
    return t

def t_LEARNING_RATE(t):
    r'LEARNING_RATE' 
    t.value = 'LEARNING_RATE'
    return t

def t_EPSILON_START(t):
    r'EPSILON_START' 
    t.value = 'EPSILON_START'
    return t

def t_EPSILON_END(t):
    r'EPSILON_END' 
    t.value = 'EPSILON_END'
    return t

def t_EXPLORATION(t):
    r'EXPLORATION' 
    t.value = 'EXPLORATION'
    return t

def t_BATCH_SIZE(t):
    r'BATCH_SIZE' 
    t.value = 'BATCH_SIZE'
    return t

def t_DISCOUNT_FACTOR(t):
    r'DISCOUNT_FACTOR' 
    t.value = 'DISCOUNT_FACTOR'
    return t

def t_NO_STEPS(t):
    r'NO_STEPS' 
    t.value = 'NO_STEPS'
    return t

def t_ADD(t):
    r'ADD' 
    t.value = 'ADD'
    return t

def t_ACTION_SIZE(t):
    r'ACTION_SIZE' 
    t.value = 'ACTION_SIZE'
    return t

def t_PREPARE(t):
    r'PREPARE' 
    t.value = 'PREPARE'
    return t

def t_EXECUTE(t):
    r'EXECUTE' 
    t.value = 'EXECUTE'
    return t

def t_CONV_LAYER(t):
    r'CONV_LAYER' 
    t.value = 'CONV_LAYER'
    return t

def t_PREDICTIVE_LAYER(t):
    r'PREDICTIVE_LAYER' 
    t.value = 'PREDICTIVE_LAYER'
    return t


#Initializes lexer

lexer = lex.lex()

try:
   DQB_test = open("test.txt", 'r')
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


