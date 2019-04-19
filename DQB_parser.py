import MLCodeGenerator as ml
import DQB_lexer as lex
import ply.yacc as yac

tokens = lex.tokens


def p_main(p):
    """main : mainP
            | mainBB"""


def p_mainP(p):
    """mainP : MAIN OPEN_PAREN CLOSE_PAREN OPEN_BRACKET enviromentP agentP calls CLOSE_BRACKET"""
    ml.set_algorithm("PG")
    ml.ini()


def p_mainBB(p):
    """mainBB : MAIN OPEN_PAREN CLOSE_PAREN OPEN_BRACKET enviromentBB agentBB calls CLOSE_BRACKET"""
    ml.set_algorithm("QL")
    ml.ini()


def p_environmentP(p):
    """environmentP : ENVIRONMENT DOS_PUNTITOS PONG"""


def p_environmentBB(p):
    """environmentBB : ENVIRONMENT DOS_PUNTITOS BRICK_BREAKER"""


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

    ml.model_parametersQL(p[3], p[6], p[9], p[12], p[15], p[18], p[21], p[24])

def p_modelparamP(p):
    """modelparamP : MODEL_PARAMETERS DOS_PUNTITOS OPEN_BRACKET propertiesP CLOSE_BRACKET"""


def p_propertiesP(p):
    """propertiesP : LEARNING_RATE EQUALS FLOAT DISCOUNT_FACTOR EQUALS FLOAT"""
    ml.model_parametersPG(p[3], p[6])

def p_network(p):
    """network : NETWORK DOS_PUNITOS OPEN_BRACKET invocations CLOSE_BRACKET"""

def p_invocations(p):
    """invocations : ADD OPEN_PAREN CONV_LAYER CLOSE_PAREN ADD OPEN_PAREN PREDICTIVE_LAYER CLOSE_PAREN"""
    ml.ConvLayers()
    ml.PredLayers()

def p_trainingBB(p):
    """trainingBB : TRAINING DOS_PUNTITOS OPEN_BRACKET train_actionsBB CLOSE_BRACKET"""
    ml.training()


def p_trainactionsBB(p):
    """trainactionsBB : PREDICT_MOVES OPEN_PAREN CLOSE_PAREN CALCULATE_Q_VALUES OPEN_PAREN CLOSE_PAREN"""
    ml.predictmovesQL()
    ml.calculateQvalues()


def p_trainingP(p):
    """trainingP : TRAINING DOS_PUNTITOS OPEN_BRACKET train_actionsP CLOSE_BRACKET"""
    ml.training()

def p_trainactionsP(p):
    """trainactionsP : FIND_POBABILITIES OPEN_PAREN CLOSE_PAREN PREDICT_MOVES OPEN_PAREN CLOSE_PAREN FIT OPEN_PAREN CLOSE_PAREN"""
    ml.find_probPG()
    ml.predict_movesPG()
    ml.fitPG()

def p_calls(p):
    """calls : EXECUTE OPEN_PAREN CLOSE_PAREN"""
    ml.generate()


def translate(file):

    try:
        DQB = open(file, "r")
    except IOError:
        print("Error opening File")
        exit()

    fileScript = DQB.read()
    parser = yac.yacc()
    parser.parse(fileScript)


