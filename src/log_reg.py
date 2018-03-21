from sklearn.linear_model import LogisticRegression

from music_genre_mfcc import read_ceps


genre_list = ["classical", "jazz", "country", "pop", "rock", "metal"]


if __name__ == "__main__":
	x, y = read_ceps(genre_list, "genres")

	lr = LogisticRegression()
	lr.fit(x, y)
	y_pred = lr.predict(x)

	print(y_pred)