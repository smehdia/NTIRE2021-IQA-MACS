import numpy as np
from tqdm import tqdm
from scipy import stats
from tensorflow.keras.models import load_model

'''
In this script we test the model to see how much error it has for the surrogate ranking model
'''

NUM_TESTS = 1000


if __name__ == "__main__":
    model = load_model('./ranking_model.hdf5')

    diff_total = []
    for i in tqdm(range(NUM_TESTS)):
        y1 = np.random.normal(0, 1, 16)
        y2 = np.random.normal(0, 1, 16)
        rk1 = model.predict(y1.reshape([-1, 16]))[0]
        rk2 = model.predict(y2.reshape([-1, 16]))[0]

        spearman_surrograte = 1 - 6 * (np.square(np.linalg.norm(rk1-rk2)))/(16 * (16 * 16 - 1))

        difference = np.abs(stats.spearmanr(y1, y2)[0] - spearman_surrograte)
        diff_total.extend([difference])

    print('Mean Absolute Error for True Spearman and Surrogate Spearman: ')
    print(np.mean(np.array(diff_total)))
