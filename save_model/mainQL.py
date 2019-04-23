import MLCodeGenerator as ml



if __name__ == '__main__':
    ml.set_algorithm("QL")
    ml.ini()
    ml.model_parametersQL(".10", "1.0", ".1", "1000000" , "16", ".9" , "30", "3")
    ml.ConvLayers()
    ml.PredLayers()
    ml.training()
    ml.main()
    ml.predictmovesQL()
    ml.calculateQvalues()
    ml.generate()