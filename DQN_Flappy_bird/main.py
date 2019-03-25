from run_model import play_game
import os
import sys
sys.path.append("game/")
import re

def find_best_model():
    filelists = []
    for file in os.listdir('./cache/'):
        if file.startswith('best'):
            nums = re.findall('\d+', file)
            filelists.append((nums[1], file))

    best_model_file = max(filelists,key=lambda x:x[0])[1]


    return best_model_file

if __name__ == '__main__':
    best_model = find_best_model()
    play_game(best_model)
