import MLCodeGenerator as ml
import DQB_lexer as lex
import ply.yacc as yac

tokens = lex.tokens


def p_main(p):
    """main : mainBB"""

def p_mainP(p):
    """mainP : MAIN OPEN_PAREN CLOSE_PAREN OPEN_BRACKET environmentP agentP calls"""
    ml.set_algorithm("PG")
    ml.ini()


def p_mainBB(p):
    """mainBB : MAIN OPEN_PAREN CLOSE_PAREN OPEN_BRACKET environmentBB agentBB calls CLOSE_BRACKET"""



def p_environmentP(p):
    """environmentP : ENVIRONMENT DOS_PUNTITOS PONG"""


def p_environmentBB(p):
    """environmentBB : ENVIRONMENT DOS_PUNTITOS BRICK_BREAKER"""


def p_agentBB(p):
    """agentBB : AGENT DOS_PUNTITOS NAME OPEN_BRACKET modelparamBB network trainingBB CLOSE_BRACKET"""


def p_agentP(p):
    """agentP : NAME OPEN_BRACKET modelparamP network trainingP CLOSE_BRACKET"""

#def p_id(p):
#    """id : CHARACTER"""

def p_modelparamBB(p):
    """modelparamBB : MODEL_PARAMETERS DOS_PUNTITOS OPEN_BRACKET propertiesBB CLOSE_BRACKET"""

def p_propertiesBB(p):
    """propertiesBB : LEARNING_RATE EQUALS FLOAT EPSILON_START EQUALS FLOAT EPSILON_END EQUALS FLOAT EXPLORATION_STEPS EQUALS INT BATCH_SIZE EQUALS INT DISCOUNT_FACTOR EQUALS FLOAT NO_STEPS EQUALS INT ACTION_SIZE EQUALS INT"""


def p_modelparamP(p):
    """modelparamP : MODEL_PARAMETERS DOS_PUNTITOS OPEN_BRACKET propertiesP CLOSE_BRACKET"""


def p_propertiesP(p):
    """propertiesP : LEARNING_RATE EQUALS FLOAT DISCOUNT_FACTOR EQUALS FLOAT"""
    ml.model_parametersPG(p[3], p[6])

def p_network(p):
    """network : NETWORK DOS_PUNTITOS OPEN_BRACKET invocations CLOSE_BRACKET"""

def p_invocations(p):
    """invocations : ADD OPEN_PAREN CONV_LAYER CLOSE_PAREN ADD OPEN_PAREN PREDICTIVE_LAYER CLOSE_PAREN"""


def p_trainingBB(p):
    """trainingBB : TRAINING DOS_PUNTITOS OPEN_BRACKET trainactionsBB CLOSE_BRACKET"""



def p_trainactionsBB(p):
    """trainactionsBB : PREDICT_MOVES OPEN_PAREN CLOSE_PAREN CALCULATE_Q_VALUES OPEN_PAREN CLOSE_PAREN"""



def p_trainingP(p):
    """trainingP : TRAINING DOS_PUNTITOS OPEN_BRACKET trainactionsP CLOSE_BRACKET"""


def p_trainactionsP(p):
    """trainactionsP : FIND_PROBABILITIES OPEN_PAREN CLOSE_PAREN PREDICT_MOVES OPEN_PAREN CLOSE_PAREN FIT OPEN_PAREN CLOSE_PAREN"""
    ml.find_probPG()
    ml.predict_movesPG()
    ml.fitPG()

def p_calls(p):
    """calls : EXECUTE"""
    ml.set_algorithm("QL")
    ml.ini()
    ml.model_parametersQL(".10", "1.0", ".1", "1000000", "16", ".9", "30", "3")
    ml.ConvLayers()
    ml.PredLayers()
    ml.training()
    ml.main()
    ml.predictmovesQL()
    ml.calculateQvalues()
    ml.generate()

def p_error(p):
    print("error")
    exit()

def translate(file):

    try:
        DQB = open(file, "r")
    except IOError:
        print("Error opening File")
        exit()

    fileScript = DQB.read()
    parser = yac.yacc()
    pat = parser.parse(fileScript)


