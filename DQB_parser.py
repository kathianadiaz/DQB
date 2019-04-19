import MLCodeGenerator as ml
import DQB_lexer as lex
import ply.yacc as yac

tokens = lex.tokens


def p_main(p):
    """main : mainP
            | mainBB"""


def p_mainP(p):
    """mainP : MAIN OPEN_PAREN CLOSE_PAREN OPEN_BRACKET enviromentP agentP calls CLOSE_BRACKET"""
    #gen code


def p_mainBB(p):
    """mainBB : MAIN OPEN_PAREN CLOSE_PAREN OPEN_BRACKET enviromentBB agentBB calls CLOSE_BRACKET"""
    #gen code


def p_environmentP(p):
    """environmentP : ENVIRONMENT DOS_PUNTITOS PONG"""
    #CODE SET ALGRTHM

def p_environmentBB(p):
    """environmentBB : ENVIRONMENT DOS_PUNTITOS BRICK_BREAKER"""
    #code set algrthm

def p_agentBB(p):
    """agentBB : id OPEN_BRACKET modelparamBB network trainingBB CLOSE_BRACKET"""

def p_agentP(p):
    """agentP : id OPEN_BRACKET modelparamP network trainingP CLOSE_BRACKET"""

def p_id(p):
    """id : CHARACTER"""

def p_modelparamBB(p):
    """modelparamBB : MODEL_PARAMETERS DOS_PUNTITOS OPEN_BRACKET propertiesBB CLOSE_BRACKET"""

def p_propertiesBB(p):
    """propertiesBB : LEARNING_RATE EQUALS FLOAT EPSILON_START EQUALS FLOAT EPSILON_END EQUALS FLOAT EXPLORATION_STEPS EQUALS FLOAT BATCH_SIZE EQUALS FLOAT DISCOUNT_FACTOR EQUALS FLOAT NO_STEPS EQUALS FLOAT ACTION_SIZE EQUALS FLOAT """

def p_modelparamP(p):
    """modelparamP : MODEL_PARAMETERS DOS_PUNTITOS OPEN_BRACKET propertiesP CLOSE_BRACKET"""

def p_propertiesP(p):
    """propertiesP : LEARNING_RATE EQUALS FLOAT DISCOUNT_FACTOR EQUALS FLOAT"""

def p_network(p):
    """network : NETWORK DOS_PUNITOS OPEN_BRACKET invocations CLOSE_BRACKET"""

def p_invocations(p):
    """invocations : ADD OPEN_PAREN CONV_LAYER CLOSE_PAREN ADD OPEN_PAREN PREDICTIVE_LAYER CLOSE_PAREN"""

def p_trainingBB(p):
    """trainingBB : TRAINING DOS_PUNTITOS OPEN_BRACKET train_actionsBB CLOSE_BRACKET"""

def p_trainactionsBB(p):
    """trainactionsBB : PREDICt_MOVES OPEN_PAREN CLOSE_PAREN CALCULATE_Q_VALUES OPEN_PAREN CLOSE_PAREN"""

def p_trainingP(p):
    """trainingP : TRAINING DOS_PUNTITOS OPEN_BRACKET train_actionsP CLOSE_BRACKET"""

def p_trainactionsP(p):
    """trainactionsP : FIND_POBABILITIES OPEN_PAREN CLOSE_PAREN PREDICT_MOVES OPEN_PAREN CLOSE_PAREN FIT OPEN_PAREN CLOSE_PAREN"""

def p_calls(p):
    """calls : id DOT EXECUTE OPEN_PAREN CLOSE_PAREN"""

