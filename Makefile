.EXPORT_ALL_VARIABLES:
PIPENV_VENV_IN_PROJECT = 1
TEST_PATH = './tests'

help:
	@echo "clean - remove all build, test, coverage and Python artifacts"
	@echo "lint - check style"
	@echo "isort - sort imported packages in a file"
	@echo "test - run tests quickly with the default Python"
	@echo "coverage - check code coverage quickly with the default Python"
	@echo "build - package"
	@echo "run-basic - basic experiment"

clean:
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name '*.pyc' -exec rm -rf {} +
	find . -name '*.pyo' -exec rm -rf {} +
	rm 	--force --recursive build/
	rm 	--force --recursive *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf docs/build/
	rm -rf docs/source/_templates

doc:
	@echo 'Creating documentation'
	# sphinx-quickstart.exe
	# sphinx-apidoc.exe -o docs msp
	cd docs && make html && cd ..
	pipenv lock --requirements > requirements.txt
	# pipenv run pipenv-setup sync

lint:
	flake8 --exit-zeropipenv install pipenv-setup --dev

isort:
	sh -c "isort . --interactive"

test: 
	(	\
		. .venv/bin/activate; \
		python -m unittest discover -s $(TEST_PATH); \
	)
	 
coverage:
	coverage run --source=msp -m unittest discover -s $(TEST_PATH)
	coverage report -m 
	coverage html

build: clean
	pip3 install --upgrade pip
	pip3 install --user pipenv
	pipenv install

test-build: clean
	pip3 install --upgrade pip
	pip3 install --user pipenv
	pipenv install --dev

run-basic:
ifeq "$(action)" "solver"
	python3 bin/main.py --config_path='bin/configs/exact_solver_msp_5_2.yml' --action='run_solver'
else ifeq "$(action)" "train"
	python3 bin/main.py --confilg_path='bin/configs/rl_model_msp_5_2.yml' --action='train_model'
else	
	@echo 'invalid action'
endif