install: install_requirements install_package

install_requirements:
	pip install -r requirements.txt

install_package:
	pip install -e src/

test:
	py.test src/test/