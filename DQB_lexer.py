import ply.lex as lex

#DBQ Lexer

#List of Tokens
tokens = [


    'AGENT',

    'NUMBER_OF_ACTIONS',

    'NETWORK',

    'LEARNING_RATE',

    'REWARD',

    'TRAINING',

    'BATCH_SIZE',

    'EPISODES',

    'UPDATE_RATE',

]

#Regular Expression Rules

def t_AGENT(t):
    r'AGENT' 
    t.value = 'AGENT'
    return t

def t_NUMBER_OF_ACTIONS(t):
    r'NUMBER_OF_ACTIONS' 
    t.value = 'NUMBER_OF_ACTIONS'
    return t

def t_NETWORK(t):
    r'NETWORK' 
    t.value = 'NETWORK'
    return t
    
def t_LEARNING_RATE(t):
    r'LEARNING_RATE' 
    t.value = 'LEARNING_RATE'
    return t

def t_REWARD(t):
    r'REWARD' 
    t.value = 'REWARD'
    return t
    
def t_TRAINING(t):
    r'TRAINING' 
    t.value = 'TRAINING'
    return t

def t_BATCH_SIZE(t):
    r'BATCH_SIZE' 
    t.value = 'BATCH_SIZE'
    return t

def t_EPISODES(t):
    r'EPISODES' 
    t.value = 'EPISODES'
    return t

def t_UPDATE_RATE(t):
    r'UPDATE_RATE' 
    t.value = 'UPDATE_RATE'
    return t

