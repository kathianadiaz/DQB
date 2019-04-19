import MLCodeGenerator as ml



if __name__ == '__main__':
    ml.set_algorithm("PG")
    ml.ini()
    ml.model_parametersPG(".001 ", " 1" )
    ml.ConvLayers()
    ml.PredLayers()
    ml.training()
    ml.main()
    ml.find_probPG()
    ml.predict_movesPG()
    ml.fitPG()
    ml.generate()