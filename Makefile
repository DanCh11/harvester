install:
	@ echo "Installing the requirements"
	@ pip install --upgrade pip
	@ pip install -r requirements.txt
	@ python -m spacy download de_core_news_sm

test:
	@ echo "$(BUILD_PRINT)Running the unit tests"