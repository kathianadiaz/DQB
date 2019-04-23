import DQB_parser as App
import time


print("Running \n")
time.sleep(0.25)
print("Processing Code...\n")
time.sleep(0.25)

file = 'DQB_script.txt'


App.translate(file)


App.ml.runAlgorthm()

#except:
#    print("Errors were encountered\n")
