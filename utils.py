import json

#constants for indexing the json object
STARS = 'stars'
TEXT = 'text'
DATE = 'date'


"""Function to parse a json data point into the review, star pair.
Input is a json object
Returns (review_text, num_stars)"""
def parse_json_to_pair(data):
	return (data[TEXT].replace('\n', ' '), data[STARS])


"""Function to turn a line from the data file into a json object"""
def text_to_json(text):
	return json.loads(text)


def get_test_data():
	data = create_train_data('yelp_data/tiny.json')
	return data

"""Function to generate training data from a json file
Input is a json file of yelp reviews
Returns a list of (data, label) pairs"""
def create_train_data(filename):
	data = []
	with open(filename) as f:
		for line in f:
			json_line = text_to_json(line)
			data.append(parse_json_to_pair(json_line))
	return data
