import DQB_parser as App
import time


print("DQB blackbox initiated... \n")
print("Processing Code...\n")
time.sleep(0.25)

file = 'script2.txt'


App.translate(file)


App.ml.runAlgorthm()

#except:
#    print("Errors were encountered\n")
